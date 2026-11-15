"""Actor action -> MuJoCo step -> state / reward / replay transition."""

from . import data_transforms as transforms
import torch
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from .motion import CONTROL_DT
from .task import MAX_STEP_REWARD
from .rotations import mat_to_quat, quat_to_mat, so3_exp
from .simulation_parameters import initialize_simulation_parameters, randomize_joint_pos_offsets

DECIMATION = 4
PHYSICS_DT = CONTROL_DT / DECIMATION


class EagerSimulation(Simulation):
    def _should_use_cuda_graph(self):
        return False


def make_sim(robot, settings):
    settings = dict(settings)
    settings.pop("control_dt")
    settings.pop("decimation")
    joint_capacity = settings.pop("njmax")
    contact_capacity = settings.pop("nconmax")
    return SimulationCfg(
        njmax=joint_capacity, nconmax=contact_capacity, mujoco=MujocoCfg(timestep=PHYSICS_DT, **settings)
    )


class TrackingEnvironment:
    def __init__(
        self,
        simulation_config,
        num_envs,
        robot,
        motion,
        task,
        device,
        max_episode_steps=500,
        evaluation=False,
    ):
        self.evaluation = evaluation
        self.num_envs = num_envs
        self.num_joints = robot.num_joints
        self.device = device
        self.robot_description = robot
        self.motion = motion
        self.task = task
        self.max_episode_steps = max_episode_steps
        scene_config = SceneCfg(
            num_envs=num_envs,
            env_spacing=0.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot.robot_cfg()},
        )
        self.scene = Scene(scene_config, device=device)
        self.simulation = EagerSimulation(
            num_envs=num_envs, cfg=simulation_config, model=self.scene.compile(), device=device
        )
        self.scene.initialize(
            mj_model=self.simulation.mj_model, model=self.simulation.model, data=self.simulation.data
        )
        self.robot = self.scene["robot"]
        assert list(self.robot.joint_names) == list(robot.joint_names)
        self.default_joint_pos = self.robot.data.default_joint_pos[0].clone()
        self.action_scale = torch.tensor(robot.action_scale_list(), device=device)
        self.simulation_parameters = initialize_simulation_parameters(self, randomize=not evaluation)
        self.motion_frame_index = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episode_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_action = torch.zeros(num_envs, self.num_joints, device=device)
        self.episode_return = torch.zeros(num_envs, device=device)
        self.reset(torch.arange(num_envs, device=device))
        self.observation_noise_limits = transforms.compute_observation_noise_limits(
            robot.observation_noise, robot.num_joints
        ).to(device)

    def state(self):
        robot_indices = self.robot.indexing
        base_free_joint_pos = self.simulation.data.qpos[:, robot_indices.free_joint_q_adr]
        base_free_joint_vel = self.simulation.data.qvel[:, robot_indices.free_joint_v_adr]
        base_pos_w = base_free_joint_pos[:, :3] - self.scene.env_origins
        base_rot_mat_w = quat_to_mat(base_free_joint_pos[:, 3:7])
        base_lin_vel_w = base_free_joint_vel[:, :3].clone()
        base_ang_vel_w = (base_rot_mat_w @ base_free_joint_vel[:, 3:6, None]).squeeze(-1)
        joint_pos = self.robot.data.joint_pos.clone()
        joint_vel = self.robot.data.joint_vel.clone()
        return (base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel)

    def sample_motion_frames(self, num_samples):
        bin_indices = torch.randint(self.motion.num_bins, (num_samples,), device=self.device)
        start_frames = self.motion.bin_start[bin_indices]
        end_frames = torch.minimum(start_frames + 50, self.motion.clip_end[self.motion.bin_clip[bin_indices]])
        return (
            start_frames + (torch.rand(num_samples, device=self.device) * (end_frames - start_frames)).long()
        )

    def reset(self, environment_indices, motion_frames=None):
        num_resets = len(environment_indices)
        reset_config = self.robot_description.domain_randomization
        if motion_frames is None:
            motion_frames = self.sample_motion_frames(num_resets)
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
        self.simulation.reset(environment_indices)
        self.scene.reset(environment_indices)
        base_pose_w = torch.cat(
            (base_pos_w + self.scene.env_origins[environment_indices], mat_to_quat(base_rot_mat_w)), -1
        )
        self.robot.write_root_link_pose_to_sim(base_pose_w, env_ids=environment_indices)
        self.robot.write_root_link_velocity_to_sim(
            torch.cat((base_lin_vel_w, base_ang_vel_w), -1), env_ids=environment_indices
        )
        self.robot.write_joint_position_to_sim(joint_pos, env_ids=environment_indices)
        self.robot.write_joint_velocity_to_sim(joint_vel, env_ids=environment_indices)
        self.robot.set_joint_position_target(joint_pos, env_ids=environment_indices)
        self.motion_frame_index[environment_indices] = motion_frames
        self.episode_step[environment_indices] = 0
        self.last_action[environment_indices] = 0
        self.episode_return[environment_indices] = 0
        if not self.evaluation:
            randomize_joint_pos_offsets(
                self.simulation_parameters, environment_indices, self.robot_description
            )
        self.simulation.forward()

    def observe(self):
        observation = transforms.compute_observation(
            self.state(),
            self.last_action,
            self.simulation_parameters,
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
        joint_pos_offset = self.simulation_parameters[:, 1 : 1 + self.num_joints]
        target_joint_pos = self.default_joint_pos + joint_pos_offset + self.action_scale * action
        self.robot.set_joint_position_target(target_joint_pos)
        for _ in range(DECIMATION):
            self.scene.write_data_to_sim()
            self.simulation.step()
            self.scene.update(PHYSICS_DT)
        next_state = self.state()
        next_motion_frame_index = motion_frame_index + 1
        if self.evaluation:
            termination = self.task.termination(next_state, next_motion_frame_index)
            clip_end_frame = self.motion.clip_end[self.motion.clip_id[motion_frame_index]]
            timeout = (next_motion_frame_index >= clip_end_frame) | (
                episode_step + 1 >= self.max_episode_steps
            )
            self.motion_frame_index = next_motion_frame_index
            self.episode_step += 1
            self.last_action = action.clone()
            return {"done": termination | timeout, "termination": termination}
        reward = self.task.reward(next_state, next_motion_frame_index, action, last_action)
        termination = self.task.termination(next_state, next_motion_frame_index)
        clip_end_frame = self.motion.clip_end[self.motion.clip_id[motion_frame_index]]
        timeout = (next_motion_frame_index >= clip_end_frame) | (episode_step + 1 >= self.max_episode_steps)
        done = termination | timeout
        transition = dict(
            state=transforms.flatten_state(state),
            next_state=transforms.flatten_state(next_state),
            action=action.clone(),
            last_action=last_action,
            motion_frame_index=motion_frame_index,
            episode_step=episode_step,
            simulation_parameters=self.simulation_parameters.clone(),
            reward=reward,
            termination=termination,
            done=done,
            transition_cut=done,
        )
        self.motion_frame_index = next_motion_frame_index
        self.episode_step += 1
        self.last_action = action.clone()
        self.episode_return += reward
        completed_environment_indices = done.nonzero().flatten()
        if len(completed_environment_indices) > 0:
            episode_length = self.episode_step[completed_environment_indices].float()
            transition["episode_length"] = episode_length
            transition["normalized_episode_return"] = self.episode_return[completed_environment_indices] / (
                episode_length * MAX_STEP_REWARD
            )
            if not self.evaluation:
                self.reset(completed_environment_indices)
        return transition
