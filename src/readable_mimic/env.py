"""Actor action -> physics step -> state / reward / replay transition."""

from .sim import (
    create_simulation,
    joint_pos_offset_limits,
    initialize_xi,
    randomize_joint_pos_offsets,
)
from . import data_transforms as transforms
import torch
from .task import MAX_STEP_REWARD
from .rotations import so3_exp, mat_to_quat
from .timing import TrainingTimer
from .motion import BIN_FRAMES


class TrackingEnv:
    def __init__(
        self,
        simulation_settings,
        num_envs,
        robot,
        motion,
        task,
        device,
        max_episode_steps=500,
        evaluation=False,
        timer=None,
        backend="mujoco",
        genesis_settings=None,
        failure_reset_fraction=0.0,
        compile_reward=False,
    ):
        self.evaluation = evaluation
        self.num_envs = num_envs
        self.num_joints = robot.num_joints
        self.device = device
        self.timer = timer if timer is not None else TrainingTimer(False, device)
        self.robot_description = robot
        self.motion = motion
        self.task = task
        self.max_episode_steps = max_episode_steps
        self.simulation = create_simulation(
            robot, num_envs, device, simulation_settings, backend, genesis_settings
        )
        self.default_joint_pos = torch.tensor(robot.default_joint_pos_list(), device=device)
        self.action_scale = torch.tensor(robot.action_scale_list(), device=device)
        self.joint_pos_offset_limits = joint_pos_offset_limits(robot, device)
        self.xi = initialize_xi(self, randomize=not evaluation)
        self.motion_frame_index = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.init_f = torch.zeros_like(self.motion_frame_index)
        self.init_root = torch.empty(num_envs, 7, device=device)
        self.episode_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_action = torch.zeros(num_envs, self.num_joints, device=device)
        self.episode_return = torch.zeros(num_envs, device=device)
        self.failure_reset = None
        if failure_reset_fraction and not evaluation:
            from .failure_reset import FailureReset

            self.failure_reset = FailureReset(motion, num_envs, failure_reset_fraction, max_episode_steps)
        self.reset(torch.arange(num_envs, device=device))
        self.observation_noise_limits = transforms.compute_observation_noise_limits(
            robot.observation_noise, robot.num_joints
        ).to(device)

        if compile_reward and torch.device(device).type == "cuda":
            self.reward_and_termination = torch.compile(
                self.reward_and_termination,
                fullgraph=True,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )

    def reward_and_termination(self, state, motion_frame_index, action, last_action, init_root, init_f):
        reward = self.task.reward(
            state, motion_frame_index, action, last_action,
            init_root=init_root, init_f=init_f,
        )
        termination = self.task.termination(state, motion_frame_index)
        return reward, termination

    def state(self):
        return self.simulation.state()

    def sample_motion_frames(self, num_samples):
        bin_indices = torch.randint(self.motion.num_bins, (num_samples,), device=self.device)
        if self.motion.reset_starts is not None:
            starts, counts = self.motion.reset_starts
            offsets = (torch.rand(num_samples, device=self.device) * counts[bin_indices]).long()
            return starts[bin_indices, offsets]
        start_frames = self.motion.bin_start[bin_indices]
        end_frames = torch.minimum(start_frames + BIN_FRAMES, self.motion.clip_end[self.motion.bin_clip[bin_indices]])
        return (
            start_frames + (torch.rand(num_samples, device=self.device) * (end_frames - start_frames)).long()
        )

    def reset(self, env_indices, motion_frames=None):
        num_resets = len(env_indices)
        reset_config = self.robot_description.domain_randomization
        if motion_frames is None:
            motion_frames = self.sample_motion_frames(num_resets)
            if self.failure_reset is not None:
                motion_frames = self.failure_reset.choose(env_indices, motion_frames)
        elif self.failure_reset is not None:
            self.failure_reset.target_bin[env_indices] = -1
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = (
            values.clone() for values in self.motion.ref_state(motion_frames)
        )

        def uniform_noise(limits):
            return torch.stack(
                [torch.empty(num_resets, device=self.device).uniform_(-limit, limit) for limit in limits], -1
            )

        if not self.evaluation:
            base_pos_w += uniform_noise(reset_config.reset_pos)
            base_rot_mat_w = so3_exp(uniform_noise(reset_config.reset_rot)) @ base_rot_mat_w
            base_lin_vel_w += uniform_noise(reset_config.reset_lin_vel)
            base_ang_vel_w += uniform_noise(reset_config.reset_ang_vel)
            joint_pos += torch.empty_like(joint_pos).uniform_(
                -reset_config.reset_joint_pos, reset_config.reset_joint_pos
            )
            joint_vel += torch.empty_like(joint_vel).uniform_(
                -reset_config.reset_joint_vel, reset_config.reset_joint_vel
            )
        joint_pos = joint_pos.clamp(
            self.task.kinematics.joint_pos_lower_limit, self.task.kinematics.joint_pos_upper_limit
        )
        self.simulation.reset(
            env_indices, (base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel)
        )
        self.init_root[env_indices] = torch.cat((base_pos_w, mat_to_quat(base_rot_mat_w)), -1)
        self.init_f[env_indices] = motion_frames
        self.motion_frame_index[env_indices] = motion_frames
        self.episode_step[env_indices] = 0
        self.last_action[env_indices] = 0
        self.episode_return[env_indices] = 0
        if not self.evaluation:
            randomize_joint_pos_offsets(self.xi, env_indices, self.joint_pos_offset_limits)

    def observe(self):
        observation = transforms.compute_observation(
            self.state(),
            self.last_action,
            self.xi,
            self.motion_frame_index,
            self.motion,
            self.default_joint_pos,
        )
        return (
            observation
            if self.evaluation
            else transforms.apply_observation_noise(observation, self.observation_noise_limits)
        )

    def step(self, action):
        state = self.state()
        last_action = self.last_action.clone()
        motion_frame_index = self.motion_frame_index.clone()
        episode_step = self.episode_step.clone()
        joint_pos_offset = self.xi[:, 1 : 1 + self.num_joints]
        target_joint_pos = self.default_joint_pos + joint_pos_offset + self.action_scale * action
        self.simulation.step(target_joint_pos)
        next_state = self.state()
        next_motion_frame_index = motion_frame_index + 1
        self.timer.record("collect_physics")
        reward, termination = self.reward_and_termination(
            next_state, next_motion_frame_index, action, last_action, self.init_root, self.init_f,
        )
        clip_end_frame = self.motion.end_frame(motion_frame_index)
        timeout = (next_motion_frame_index >= clip_end_frame) | (episode_step + 1 >= self.max_episode_steps)
        done = termination | timeout
        self.timer.record("collect_reward")
        transition = dict(
            state=transforms.flatten_state(state),
            next_state=transforms.flatten_state(next_state),
            action=action.clone(),
            last_action=last_action,
            motion_frame_index=motion_frame_index,
            episode_step=episode_step,
            init_root=self.init_root.clone(),
            init_f=self.init_f.clone(),
            xi=self.xi.clone(),
            reward=reward,
            termination=termination,
            done=done,
            transition_cut=done,
        )
        if self.failure_reset is not None:
            self.failure_reset.observe(next_motion_frame_index, termination, done)
        self.motion_frame_index = next_motion_frame_index
        self.episode_step += 1
        self.last_action = action.clone()
        self.episode_return += reward
        completed_env_indices = done.nonzero().flatten()
        if len(completed_env_indices) > 0:
            episode_length = self.episode_step[completed_env_indices].float()
            transition["episode_length"] = episode_length
            transition["normalized_episode_return"] = self.episode_return[completed_env_indices] / (
                episode_length * MAX_STEP_REWARD
            )
            if not self.evaluation:
                self.reset(completed_env_indices)
        self.timer.record("collect_finalize")
        return transition
