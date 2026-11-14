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
        m = mujoco.MjModel.from_xml_path(robot.mjcf_path)
        self.device = device
        tracked = robot.tracked_bodies
        jname_to_idx = {n: i for i, n in enumerate(robot.joint_names)}
        # moving bodies in tree order (mujoco orders parents before children)
        bodies, parent, joint_idx, axes, offs_p, offs_R = [], [], [], [], [], []
        mj_to_local = {}
        for b in range(1, m.nbody):
            name = m.body(b).name
            if m.body_jntnum[b] == 0:
                continue  # fixed links are merged visually only; none are tracked
            jid = m.body_jntadr[b]
            mj_to_local[b] = len(bodies)
            bodies.append(name)
            if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                parent.append(-1)
                joint_idx.append(-1)
            else:
                pb = m.body_parentid[b]
                while pb not in mj_to_local:  # skip fixed parents
                    raise RuntimeError(f"fixed parent for {name}")
                parent.append(mj_to_local[pb])
                joint_idx.append(jname_to_idx[m.joint(jid).name])
            axes.append(m.jnt_axis[jid].copy())
            offs_p.append(m.body_pos[b].copy())
            offs_R.append(m.body_quat[b].copy())
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(offs_p), dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), dtype=torch.float32, device=device))
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.num_bodies = len(bodies)
        # joint limits (in joint_names order)
        lo, hi = np.zeros(robot.num_joints), np.zeros(robot.num_joints)
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jname_to_idx:
                lo[jname_to_idx[n]], hi[jname_to_idx[n]] = m.jnt_range[j]
        self.joint_pos_lower_limit = torch.tensor(lo, dtype=torch.float32, device=device)
        self.joint_pos_upper_limit = torch.tensor(hi, dtype=torch.float32, device=device)

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
        for i in range(self.num_bodies):
            parent_index = self.parent[i]
            if parent_index < 0:
                (
                    body_positions_w[i],
                    body_rotations_w[i],
                    body_linear_velocities_w[i],
                    body_angular_velocities_w[i],
                ) = base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w
                continue
            j = self.joint_idx[i]
            parent_rot_mat_w = body_rotations_w[parent_index]
            body_pos_w = body_positions_w[parent_index] + mv3(parent_rot_mat_w, self.off_p[i])
            joint_origin_rot_mat_w = mm3(parent_rot_mat_w, self.off_R[i])
            body_rot_mat_w = mm3(joint_origin_rot_mat_w, axis_angle_to_mat(self.axes[i], joint_pos[..., j]))
            joint_axis_w = mv3(joint_origin_rot_mat_w, self.axes[i])
            body_ang_vel_w = (
                body_angular_velocities_w[parent_index] + joint_axis_w * joint_vel[..., j : j + 1]
            )
            body_lin_vel_w = body_linear_velocities_w[parent_index] + torch.cross(
                body_angular_velocities_w[parent_index], body_pos_w - body_positions_w[parent_index], dim=-1
            )
            (
                body_positions_w[i],
                body_rotations_w[i],
                body_linear_velocities_w[i],
                body_angular_velocities_w[i],
            ) = body_pos_w, body_rot_mat_w, body_lin_vel_w, body_ang_vel_w
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([body_positions_w[i] for i in idx], -2),
            torch.stack([body_rotations_w[i] for i in idx], -3),
            torch.stack([body_linear_velocities_w[i] for i in idx], -2),
            torch.stack([body_angular_velocities_w[i] for i in idx], -2),
        )
