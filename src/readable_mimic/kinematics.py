"""Differentiable forward kinematics of the robot in torch, built from its MJCF.

The same FK computes tracked-body poses and velocities for the reference motion, for logged
simulator states, and for world-model predictions, so the analytic reward head matches the
reward the simulator transitions are labeled with.
"""

from __future__ import annotations
import mujoco
import numpy as np
import torch
from .robot import RobotSpec
from .rotations import axis_angle_to_mat, mm3, mv3, quat_to_mat


class Kinematics:
    def __init__(self, robot: RobotSpec, device: str):
        if not hasattr(mujoco, "MjModel"):
            raise ImportError(
                f"MuJoCo bindings are incomplete or shadowed (imported from {getattr(mujoco, '__file__', None)}). "
                "Switching extras can remove files shared with mujoco-core. Repair with: "
                "uv sync --extra genesis --reinstall-package mujoco "
                "(use --extra mujoco for the MuJoCo backend). "
                "Also check for a local mujoco.py or mujoco/ directory shadowing the package."
            )
        mujoco_model = mujoco.MjModel.from_xml_path(robot.mjcf_path)
        self.device = device
        tracked_body_names = robot.tracked_bodies
        joint_name_to_index = {
            joint_name: body_index for body_index, joint_name in enumerate(robot.joint_names)
        }
        (
            bodies,
            parent_indices,
            body_joint_indices,
            joint_axes,
            joint_origin_positions,
            joint_origin_quaternions,
        ) = ([], [], [], [], [], [])
        mujoco_to_local_body_index = {}
        for mujoco_body_index in range(1, mujoco_model.nbody):
            name = mujoco_model.body(mujoco_body_index).name
            if mujoco_model.body_jntnum[mujoco_body_index] == 0:
                continue
            mujoco_joint_index = mujoco_model.body_jntadr[mujoco_body_index]
            mujoco_to_local_body_index[mujoco_body_index] = len(bodies)
            bodies.append(name)
            if mujoco_model.jnt_type[mujoco_joint_index] == mujoco.mjtJoint.mjJNT_FREE:
                parent_indices.append(-1)
                body_joint_indices.append(-1)
            else:
                mujoco_parent_body_index = mujoco_model.body_parentid[mujoco_body_index]
                while mujoco_parent_body_index not in mujoco_to_local_body_index:
                    raise RuntimeError(f"fixed parent for {name}")
                parent_indices.append(mujoco_to_local_body_index[mujoco_parent_body_index])
                body_joint_indices.append(joint_name_to_index[mujoco_model.joint(mujoco_joint_index).name])
            joint_axes.append(mujoco_model.jnt_axis[mujoco_joint_index].copy())
            joint_origin_positions.append(mujoco_model.body_pos[mujoco_body_index].copy())
            joint_origin_quaternions.append(mujoco_model.body_quat[mujoco_body_index].copy())
        self.body_names = bodies
        self.parent_indices = parent_indices
        self.body_joint_indices = body_joint_indices
        self.joint_axes = torch.tensor(np.array(joint_axes), dtype=torch.float32, device=device)
        self.joint_origin_pos = torch.tensor(
            np.array(joint_origin_positions), dtype=torch.float32, device=device
        )
        self.joint_origin_rot_mat = quat_to_mat(
            torch.tensor(np.array(joint_origin_quaternions), dtype=torch.float32, device=device)
        )
        self.tracked_body_indices = [bodies.index(joint_name) for joint_name in tracked_body_names]
        self.num_bodies = len(bodies)
        joint_lower_limits, joint_upper_limits = (np.zeros(robot.num_joints), np.zeros(robot.num_joints))
        for joint_index in range(mujoco_model.njnt):
            joint_name = mujoco_model.joint(joint_index).name
            if joint_name in joint_name_to_index:
                (
                    joint_lower_limits[joint_name_to_index[joint_name]],
                    joint_upper_limits[joint_name_to_index[joint_name]],
                ) = mujoco_model.jnt_range[joint_index]
        self.joint_pos_lower_limit = torch.tensor(joint_lower_limits, dtype=torch.float32, device=device)
        self.joint_pos_upper_limit = torch.tensor(joint_upper_limits, dtype=torch.float32, device=device)

    def forward(
        self,
        base_pos_w,
        base_rot_mat_w,
        base_lin_vel_w,
        base_ang_vel_w,
        joint_pos,
        joint_vel,
        all_bodies: bool = False,
    ):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w: root world state.
            joint_pos, joint_vel: joint state (B,J) in joint_names order.
        """
        body_positions_w, body_rotations_w, body_linear_velocities_w, body_angular_velocities_w = (
            [None] * self.num_bodies,
            [None] * self.num_bodies,
            [None] * self.num_bodies,
            [None] * self.num_bodies,
        )
        for body_index in range(self.num_bodies):
            parent_index = self.parent_indices[body_index]
            if parent_index < 0:
                (
                    body_positions_w[body_index],
                    body_rotations_w[body_index],
                    body_linear_velocities_w[body_index],
                    body_angular_velocities_w[body_index],
                ) = (base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w)
                continue
            joint_index = self.body_joint_indices[body_index]
            parent_rot_mat_w = body_rotations_w[parent_index]
            body_pos_w = body_positions_w[parent_index] + mv3(
                parent_rot_mat_w, self.joint_origin_pos[body_index]
            )
            joint_origin_rot_mat_w = mm3(parent_rot_mat_w, self.joint_origin_rot_mat[body_index])
            body_rot_mat_w = mm3(
                joint_origin_rot_mat_w,
                axis_angle_to_mat(self.joint_axes[body_index], joint_pos[..., joint_index]),
            )
            joint_axis_w = mv3(joint_origin_rot_mat_w, self.joint_axes[body_index])
            body_ang_vel_w = (
                body_angular_velocities_w[parent_index]
                + joint_axis_w * joint_vel[..., joint_index : joint_index + 1]
            )
            body_lin_vel_w = body_linear_velocities_w[parent_index] + torch.cross(
                body_angular_velocities_w[parent_index], body_pos_w - body_positions_w[parent_index], dim=-1
            )
            (
                body_positions_w[body_index],
                body_rotations_w[body_index],
                body_linear_velocities_w[body_index],
                body_angular_velocities_w[body_index],
            ) = (body_pos_w, body_rot_mat_w, body_lin_vel_w, body_ang_vel_w)
        selected_body_indices = range(self.num_bodies) if all_bodies else self.tracked_body_indices
        return (
            torch.stack([body_positions_w[body_index] for body_index in selected_body_indices], -2),
            torch.stack([body_rotations_w[body_index] for body_index in selected_body_indices], -3),
            torch.stack([body_linear_velocities_w[body_index] for body_index in selected_body_indices], -2),
            torch.stack([body_angular_velocities_w[body_index] for body_index in selected_body_indices], -2),
        )
