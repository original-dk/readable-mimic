"""Motion-tracking simulator on mjlab (MuJoCo-Warp).

The environment is a thin batched wrapper: it applies the joint setpoint, steps MuJoCo-Warp, labels
the transition with the analytic reward/termination of :mod:`readable_mimic.mdp`, and resets episodes to
reference-motion states drawn by return-based bin sampling (SuperMimic paper Appendix D.1). Domain randomization
(:mod:`readable_mimic.dr`) and the policy's sensing (:mod:`readable_mimic.sensing`) follow ``RobotSpec``.
"""
from __future__ import annotations
import torch
import mujoco_warp as mjwarp
import warp as wp
from mjlab.entity.data import EntityData
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from .dr import DomainRandomizer, XiLayout
from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .robot import RobotSpec
from .rotations import mat_to_quat, quat_to_mat, so3_exp
from .sensing import ObsPipeline
DECIMATION = 4
PHYSICS_DT = CONTROL_DT / DECIMATION

def make_scene_cfg(robot: RobotSpec, num_envs: int) -> SceneCfg:
    return SceneCfg(num_envs=num_envs, env_spacing=0.0, terrain=TerrainEntityCfg(terrain_type='plane'), entities={'robot': robot.robot_cfg()})

def make_sim(robot: RobotSpec) -> SimulationCfg:
    return SimulationCfg(njmax=robot.njmax, nconmax=robot.nconmax, mujoco=MujocoCfg(timestep=PHYSICS_DT, integrator='implicitfast', cone='pyramidal', solver='newton', iterations=100, ls_iterations=50, tolerance=1e-06, ccd_iterations=35))

class TrackingSim:
    """Batched motion-tracking simulator for the robot of ``robot``."""

    def __init__(self, sim_cfg: SimulationCfg, num_envs: int, robot: RobotSpec, motion: MotionLib, mdp: MDP, device: str, gamma: float=0.99, eval_mode: bool=False, max_episode_steps: int=500, reset_noise: bool=True, randomize: bool | None=None, fail_reset: float=0.0, bin_fix: bool=False):
        """``randomize`` (default: not eval_mode) switches domain randomization, observation noise and
        delay, and pushes; ``eval_mode`` switches evaluation bookkeeping (no auto-reset, episode starts
        exactly on the reference)."""
        self.device = device
        self.scene = Scene(make_scene_cfg(robot, num_envs), device=device)
        self.sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=self.scene.compile(), device=device)
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        self.robot = self.scene['robot']
        assert list(self.robot.joint_names) == list(robot.joint_names), self.robot.joint_names
        self.N = num_envs
        self.J = robot.num_joints
        self.spec = robot
        self.motion, self.mdp, self.gamma = (motion, mdp, gamma)
        self.eval_mode = eval_mode
        self.randomize = not eval_mode if randomize is None else randomize
        self.max_episode_steps = max_episode_steps if not eval_mode else 10 ** 9
        self.reset_noise = reset_noise and (not eval_mode)
        dev = self.device
        self.default_joint_pos = self.robot.data.default_joint_pos[0].clone()
        self.alpha = torch.tensor(robot.action_scale_list(), device=dev)
        self.layout = XiLayout(robot, dev)
        assert self.layout.dim == mdp.xi_dim, (self.layout.dim, mdp.xi_dim)
        self.simulation_parameters = torch.zeros(self.N, self.layout.dim, device=dev)
        self.randomizer = DomainRandomizer(robot, self.layout, self, eval_mode=not self.randomize)
        self.randomizer.startup(self.simulation_parameters)
        self.pipe = ObsPipeline(robot, mdp, self.layout, dev, PHYSICS_DT)
        self.obs_fresh = torch.ones(self.N, dtype=torch.bool, device=dev)
        self._obs = self._obs_state = None
        if robot.sensing.imu is not None:
            m = self.sim.mj_model
            adr = lambda name: int(m.sensor_adr[m.sensor(f'robot/{name}').id])
            self._gyro_adr, self._accel_adr = (adr(robot.sensing.imu.gyro_sensor), adr(robot.sensing.imu.accel_sensor))
        self.push_timer = torch.zeros(self.N, device=dev)
        self.push_enabled = robot.dr.push_interval_s is not None and self.randomize
        if self.push_enabled:
            self.push_timer.uniform_(*robot.dr.push_interval_s)
        self.motion_frame_index = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.last_action = torch.zeros(self.N, self.J, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    @property
    def num_envs(self) -> int:
        return self.N

    def state(self):
        idx = self.robot.indexing
        root_q = self.sim.data.qpos[:, idx.free_joint_q_adr]
        base_lin_vel_w = self.sim.data.qvel[:, idx.free_joint_v_adr]
        base_pos_w = root_q[:, :3] - self.scene.env_origins
        base_rot_mat_w = quat_to_mat(root_q[:, 3:7])
        base_ang_vel_w = (base_rot_mat_w @ base_lin_vel_w[:, 3:6].unsqueeze(-1)).squeeze(-1)
        d = self.robot.data
        return (base_pos_w, base_rot_mat_w, base_lin_vel_w[:, :3].clone(), base_ang_vel_w, d.joint_pos.clone(), d.joint_vel.clone())

    def sample_frames(self, n: int, gate: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        bins = torch.randint(self.motion.num_bins, (n,), device=self.device)
        start = self.motion.bin_start[bins]
        end = torch.minimum(start + 50, self.motion.clip_end[self.motion.bin_clip[bins]])
        frames = start + (torch.rand(n, device=self.device) * (end - start)).long()
        return (frames, bins)

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None=None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m, c = (self.motion, self.spec.dr)
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            u = lambda half: torch.stack([torch.empty(n, device=dev).uniform_(-h, h) for h in half], -1)
            base_pos_w = base_pos_w + u(c.reset_pos)
            base_rot_mat_w = so3_exp(u(c.reset_rot)) @ base_rot_mat_w
            base_lin_vel_w = base_lin_vel_w + u(c.reset_lin_vel)
            base_ang_vel_w = base_ang_vel_w + u(c.reset_ang_vel)
            joint_pos = joint_pos + torch.empty_like(joint_pos).uniform_(-c.reset_joint_pos, c.reset_joint_pos)
            if c.reset_joint_vel > 0:
                joint_vel = joint_vel + torch.empty_like(joint_vel).uniform_(-c.reset_joint_vel, c.reset_joint_vel)
        joint_pos = torch.clamp(joint_pos, self.mdp.kin.joint_pos_lower_limit, self.mdp.kin.joint_pos_upper_limit)
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        pose = torch.cat((base_pos_w + self.scene.env_origins[env_ids], mat_to_quat(base_rot_mat_w)), -1)
        self.robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((base_lin_vel_w, base_ang_vel_w), -1), env_ids=env_ids)
        self.robot.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.motion_frame_index[env_ids] = frames
        self.ep_step.index_fill_(0, env_ids, 0)
        self.last_action.index_fill_(0, env_ids, 0.0)
        self.start_bin[env_ids] = bins
        self.ep_disc_ret.index_fill_(0, env_ids, 0.0)
        self.ep_ret.index_fill_(0, env_ids, 0.0)
        self.randomizer.reset(self.simulation_parameters, env_ids)
        self.obs_fresh.index_fill_(0, env_ids, True)

    def refresh(self):
        """Recompute derived quantities (body poses) so the robot's mjlab data match ``state()`` after a reset."""
        self.sim.forward()

    def _imu_site(self) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.sim.data.sensordata
        return (d[:, self._gyro_adr:self._gyro_adr + 3], d[:, self._accel_adr:self._accel_adr + 3])

    def observe(self) -> torch.Tensor:
        """Policy observation of the current state (training: noisy; eval: noise- and delay-free)."""
        S = self.state()
        if not self.pipe.stateful:
            o = self.mdp.obs(S, self.last_action, self.simulation_parameters, self.motion_frame_index)
            self._obs = self.mdp.add_noise(o) if self.randomize else o
            return self._obs
        if self._obs_state is None or self.obs_fresh.any():
            fresh = self.obs_fresh.unsqueeze(-1)
            Rs = S[1] @ self.pipe.site_rot if self.spec.sensing.imu is not None else S[1]
            gyro = (Rs.transpose(-1, -2) @ S[3].unsqueeze(-1)).squeeze(-1)
            accel = Rs[..., 2, :] * 9.81
            init = self.pipe.init_state(S, self.simulation_parameters, gyro, accel)
            init['prev_raw'] = self.pipe.raw_obs(init, S, self.last_action, self.simulation_parameters, self.motion_frame_index)
            st = init if self._obs_state is None else {k: torch.where(fresh, init[k], x) for k, x in self._obs_state.items()}
        else:
            st = self._obs_state
        raw = self.pipe.raw_obs(st, S, self.last_action, self.simulation_parameters, self.motion_frame_index)
        self._obs, self._obs_state = self.pipe.observe(st, raw, noise=self.randomize, delay=self.randomize)
        self.obs_fresh[:] = False
        return self._obs

    def _push(self, done: torch.Tensor) -> torch.Tensor:
        """Root velocity kicks at random intervals (between steps, after s1 is recorded)."""
        self.push_timer -= CONTROL_DT
        due = (self.push_timer <= 0) & ~done
        ids = due.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            c = self.spec.dr
            _, _, base_lin_vel_w, base_ang_vel_w, _, _ = self.state()
            kick = torch.stack([torch.empty(len(ids), device=self.device).uniform_(-h, h) for h in c.push_velocity], -1)
            vel = torch.cat((base_lin_vel_w[ids], base_ang_vel_w[ids]), -1) + kick
            self.robot.write_root_link_velocity_to_sim(vel, env_ids=ids)
            self.push_timer[ids] = torch.empty(len(ids), device=self.device).uniform_(*c.push_interval_s)
        return due

    def step(self, action: torch.Tensor) -> dict:
        out = self.step_core(action)
        if self.eval_mode:
            self.failed_eval |= out['term'] & ~self.done_eval
            self.done_eval |= out['done']
            return out
        return self.step_tail(out)

    def step_core(self, action: torch.Tensor) -> dict:
        """The step up to the episode ends: no host synchronization (pushes aside), so it can be captured in a CUDA
        graph (train.py --collect_graph). The simulator wrapper rebinds f, ep_step and a_prev."""
        if self.pipe.stateful and self._obs is None:
            self.observe()
        o, o_state = (self._obs, self._obs_state)
        self._obs = None
        S = self.state()
        motion_frame_index = self.motion_frame_index.clone()
        simulation_parameters = self.simulation_parameters.clone()
        last_action = self.last_action.clone()
        ep_step = self.ep_step.clone()
        target = self.default_joint_pos + simulation_parameters[:, 1:1 + self.J] + self.alpha * action
        self.robot.set_joint_position_target(target)
        for _ in range(DECIMATION):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(PHYSICS_DT)
            if self.pipe.stateful:
                gyro, accel = self._imu_site() if self.spec.sensing.imu is not None else (None, None)
                self._obs_state = self.pipe.filter_update(self._obs_state, self.robot.data.joint_vel, gyro, accel, self.simulation_parameters, PHYSICS_DT, noise=self.randomize)
        S1 = self.state()
        f1 = motion_frame_index + 1
        r, terms = self.mdp.reward(S1, f1, action, last_action, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[motion_frame_index]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {'state': flatten_state(S), 'action': action.clone(), 'last_action': last_action, 'motion_frame_index': motion_frame_index, 'next_state': flatten_state(S1), 'reward': r, 'term': term, 'done': done, 'ep_step': ep_step, 'simulation_parameters': simulation_parameters, 'terms': terms, 'transition_cut': done}
        if self.pipe.stateful:
            out['observation'], out['observation_pipeline_state'] = (o, self.pipe.pack(o_state))
        if self.push_enabled:
            out['transition_cut'] = done | self._push(done)
        self.ep_disc_ret += self.gamma ** ep_step.float() * r
        self.ep_ret += r
        self.motion_frame_index = f1
        self.ep_step = ep_step + 1
        self.last_action = action.clone()
        return out

    def step_tail(self, out: dict) -> dict:
        done_indices = out['done'].nonzero(as_tuple=False).squeeze(-1)
        if len(done_indices) > 0:
            episode_length = self.ep_step[done_indices].float()
            out['ep_len'] = episode_length
            out['ep_ret_norm'] = self.ep_ret[done_indices] / (episode_length * MAX_STEP_REWARD)
            self.reset(done_indices)
        return out
