"""MuJoCo/mjlab implementation of the shared state, reset and step interface."""

import torch
import mujoco
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.managers.event_manager import RecomputeLevel

from .sim import DECIMATION, PHYSICS_DT
from .rotations import mat_to_quat, quat_to_mat


class EagerSimulation(Simulation):
    def _should_use_cuda_graph(self):
        return False


class MujocoSim:
    def __init__(self, robot, num_envs, device, settings):
        settings = dict(settings)
        settings.pop("control_dt")
        settings.pop("decimation")
        simulation_config = SimulationCfg(
            njmax=settings.pop("njmax"),
            nconmax=settings.pop("nconmax"),
            mujoco=MujocoCfg(timestep=PHYSICS_DT, **settings),
        )
        self.scene = Scene(
            SceneCfg(
                num_envs=num_envs,
                env_spacing=0.0,
                terrain=TerrainEntityCfg(terrain_type="plane"),
                entities={"robot": robot.robot_cfg()},
            ),
            device=device,
        )
        self.physics = EagerSimulation(
            num_envs=num_envs, cfg=simulation_config, model=self.scene.compile(), device=device
        )
        self.scene.initialize(
            mj_model=self.physics.mj_model, model=self.physics.model, data=self.physics.data
        )
        self.robot = self.scene["robot"]
        assert list(self.robot.joint_names) == list(robot.joint_names)
        self.robot_description = robot
        self.device = device

    def state(self):
        indices = self.robot.indexing
        base_joint_pos = self.physics.data.qpos[:, indices.free_joint_q_adr]
        base_joint_vel = self.physics.data.qvel[:, indices.free_joint_v_adr]
        base_rot_mat_w = quat_to_mat(base_joint_pos[:, 3:7])
        return (
            base_joint_pos[:, :3] - self.scene.env_origins,
            base_rot_mat_w,
            base_joint_vel[:, :3].clone(),
            (base_rot_mat_w @ base_joint_vel[:, 3:6, None]).squeeze(-1),
            self.robot.data.joint_pos.clone(),
            self.robot.data.joint_vel.clone(),
        )

    def reset(self, env_indices, state):
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = state
        self.physics.reset(env_indices)
        self.scene.reset(env_indices)
        base_pose_w = torch.cat(
            (base_pos_w + self.scene.env_origins[env_indices], mat_to_quat(base_rot_mat_w)), -1
        )
        self.robot.write_root_link_pose_to_sim(base_pose_w, env_ids=env_indices)
        self.robot.write_root_link_velocity_to_sim(
            torch.cat((base_lin_vel_w, base_ang_vel_w), -1), env_ids=env_indices
        )
        self.robot.write_joint_position_to_sim(joint_pos, env_ids=env_indices)
        self.robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_indices)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_indices)
        self.physics.forward()

    def step(self, target_joint_pos):
        self.robot.set_joint_position_target(target_joint_pos)
        for _ in range(DECIMATION):
            self.scene.write_data_to_sim()
            self.physics.step()
            self.scene.update(PHYSICS_DT)

    def apply_xi(self, xi, randomize, ground_friction=None):
        physics, robot = self.physics, self.robot_description
        physics.expand_model_fields((
            "body_ipos", "geom_friction", "body_mass", "body_inertia",
            "dof_armature", "dof_damping", "dof_frictionloss",
            "actuator_gainprm", "actuator_biasprm",
            "body_subtreemass", "dof_invweight0", "body_invweight0",
            "tendon_length0", "tendon_invweight0", "actuator_acc0",
        ))
        ground_indices = (
            torch.as_tensor(physics.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE, device=self.device).nonzero().flatten()
        )
        physics.model.geom_friction[:, ground_indices, 0] = (
            ground_friction if ground_friction is not None else
            (robot.domain_randomization.friction_range[0] if randomize else 1.0)
        )
        physics.model.geom_friction[:, self.robot.indexing.geom_ids, 0] = xi[:, :1]
        body_index = self.robot.indexing.body_ids[self.robot.find_bodies(robot.center_of_mass_body)[0][0]]
        physics.model.body_ipos[:, body_index] = (
            physics.get_default_field("body_ipos")[body_index] + xi[:, 1 + robot.num_joints : 4 + robot.num_joints]
        )
        mass, inertia, armature, damping, frictionloss, kp, kd = xi[:, -7:].unbind(-1)
        body_ids = self.robot.indexing.body_ids
        dof_ids = self.robot.indexing.joint_v_adr
        actuator_ids = self.robot.indexing.ctrl_ids
        physics.model.body_mass[:, body_ids] = physics.get_default_field("body_mass")[body_ids] * mass[:, None]
        physics.model.body_inertia[:, body_ids] = (
            physics.get_default_field("body_inertia")[body_ids] * (mass * inertia)[:, None, None]
        )
        physics.model.dof_armature[:, dof_ids] = physics.get_default_field("dof_armature")[dof_ids] * armature[:, None]
        physics.model.dof_damping[:, dof_ids] = physics.get_default_field("dof_damping")[dof_ids] + damping[:, None]
        physics.model.dof_frictionloss[:, dof_ids] = physics.get_default_field("dof_frictionloss")[dof_ids] + frictionloss[:, None]
        gain = physics.get_default_field("actuator_gainprm")[actuator_ids].expand(xi.shape[0], -1, -1).clone()
        bias = physics.get_default_field("actuator_biasprm")[actuator_ids].expand(xi.shape[0], -1, -1).clone()
        gain[..., 0] *= kp[:, None]
        bias[..., 1] *= kp[:, None]
        bias[..., 2] *= kd[:, None]
        physics.model.actuator_gainprm[:, actuator_ids] = gain
        physics.model.actuator_biasprm[:, actuator_ids] = bias
        physics.recompute_constants(RecomputeLevel.set_const)
