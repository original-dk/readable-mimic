"""Genesis implementation; tensors follow the same joint order and world-frame state as MuJoCo."""

import torch

from .sim import CONTROL_DT, DECIMATION
from .rotations import mat_to_quat, quat_to_mat


class GenesisSim:
    def __init__(self, robot, num_envs, device, settings, genesis_settings):
        import genesis as gs

        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            self.device = torch.device("cuda", torch.cuda.current_device())
        if not gs._initialized:
            gs.init(backend=gs.cuda if self.device.type == "cuda" else gs.cpu, logging_level="warning")
        if gs.device != self.device:
            raise ValueError(f"Genesis is already initialized on {gs.device}; requested {self.device}.")
        if robot.has_action_delay:
            raise ValueError("Genesis currently supports robots without action delay.")
        self.robot_description = robot
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=CONTROL_DT, substeps=DECIMATION),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                integrator=gs.integrator.approximate_implicitfast,
                friction_cone=gs.friction_cone.pyramidal,
                iterations=settings["iterations"],
                tolerance=settings["tolerance"],
                ls_iterations=settings["ls_iterations"],
                max_contacts=settings["nconmax"],
                batch_links_info=True,
                batch_dofs_info=True,
                **genesis_settings,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=False,
        )
        self.ground = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=robot.mjcf_path, align=False))
        self.scene.build(n_envs=num_envs, env_spacing=(0.0, 0.0))
        joints = [self.robot.get_joint(name) for name in robot.joint_names]
        self.joint_dof_indices = [joint.dofs_idx_local[0] for joint in joints]
        self.joint_qpos_indices = [joint.qs_idx_local[0] for joint in joints]
        free_joints = [joint for joint in self.robot.joints if joint.type == gs.JOINT_TYPE.FREE]
        if len(free_joints) != 1:
            raise ValueError("Expected one floating base joint in the robot MJCF.")
        self.base_dof_indices = free_joints[0].dofs_idx_local
        self.base_qpos_indices = free_joints[0].qs_idx_local
        self.default_qpos = self.robot.get_qpos().clone()
        self.center_of_mass_index = self.robot.get_link(robot.center_of_mass_body).idx_local
        self.default_center_of_mass = self.robot.get_links_COM([self.center_of_mass_index]).clone()
        self.robot.set_dofs_kp(
            [robot.joint_stiffness[name] for name in robot.joint_names], self.joint_dof_indices
        )
        self.robot.set_dofs_kv(
            [robot.joint_damping[name] for name in robot.joint_names], self.joint_dof_indices
        )
        effort_limits = torch.tensor(
            [robot.joint_effort_limit[name] for name in robot.joint_names], device=self.device
        )
        self.robot.set_dofs_force_range(-effort_limits, effort_limits, self.joint_dof_indices)
        self.robot.set_dofs_armature(
            [robot.joint_armature[name] for name in robot.joint_names], self.joint_dof_indices
        )

    def state(self):
        return (
            self.robot.get_pos().clone(),
            quat_to_mat(self.robot.get_quat()),
            self.robot.get_vel().clone(),
            self.robot.get_ang().clone(),
            self.robot.get_dofs_position(self.joint_dof_indices).clone(),
            self.robot.get_dofs_velocity(self.joint_dof_indices).clone(),
        )

    def reset(self, env_indices, state):
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = state
        qpos = self.default_qpos[env_indices].clone()
        qpos[:, self.base_qpos_indices] = torch.cat((base_pos_w, mat_to_quat(base_rot_mat_w)), -1)
        qpos[:, self.joint_qpos_indices] = joint_pos
        base_ang_vel = (base_rot_mat_w.transpose(-1, -2) @ base_ang_vel_w.unsqueeze(-1)).squeeze(-1)
        self.robot.set_qpos(qpos, envs_idx=env_indices, zero_velocity=True)
        self.robot.set_dofs_velocity(
            torch.cat((base_lin_vel_w, base_ang_vel), -1), self.base_dof_indices, envs_idx=env_indices
        )
        self.robot.set_dofs_velocity(joint_vel, self.joint_dof_indices, envs_idx=env_indices)
        self.robot.control_dofs_position(joint_pos, self.joint_dof_indices, envs_idx=env_indices)

    def step(self, target_joint_pos):
        self.robot.control_dofs_position(target_joint_pos, self.joint_dof_indices)
        self.scene.step()

    def apply_xi(self, xi, randomize):
        robot = self.robot_description
        self.ground.set_friction(robot.domain_randomization.friction_range[0] if randomize else 1.0)
        self.robot.set_friction(1.0)
        self.robot.set_friction_ratio(xi[:, :1].expand(-1, self.robot.n_links))
        self.robot.set_links_COM(
            self.default_center_of_mass + xi[:, None, 1 + robot.num_joints :], [self.center_of_mass_index]
        )
