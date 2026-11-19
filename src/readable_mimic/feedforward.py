"""Nominal reference inverse dynamics for the Genesis PD-target experiment."""

import mujoco
import numpy as np
import torch


class ReferenceFeedforward:
    def __init__(self, robot):
        self.model = mujoco.MjModel.from_xml_path(robot.mjcf_path)
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(name) for name in robot.joint_names]
        self.qpos_ids = [int(joint.qposadr[0]) for joint in joints]
        self.dof_ids = [int(joint.dofadr[0]) for joint in joints]
        self.stiffness = np.array([robot.joint_stiffness[name] for name in robot.joint_names])
        if np.any(self.stiffness <= 0):
            raise ValueError("Feedforward requires positive joint stiffness")
        if self.model.nq != len(joints) + 7 or self.model.nv != len(joints) + 6:
            raise ValueError("Feedforward requires the floating-base G1 model")
        self.model.dof_armature[self.dof_ids] = [robot.joint_armature[name] for name in robot.joint_names]
        self.mass_matrix = np.empty((self.model.nv, self.model.nv))

    def position_offsets(self, base_pos_w, base_quat_w, base_velocity, joint_pos, joint_vel, acceleration):
        """Evaluate nominal M(q) acceleration + bias along the reference trajectory."""
        output = np.empty(joint_pos.shape, dtype=np.float32)
        for frame in range(len(joint_pos)):
            self.data.qpos[:3] = base_pos_w[frame]
            self.data.qpos[3:7] = base_quat_w[frame]
            self.data.qpos[self.qpos_ids] = joint_pos[frame]
            self.data.qvel[:6] = base_velocity[frame]
            self.data.qvel[self.dof_ids] = joint_vel[frame]
            mujoco.mj_fwdPosition(self.model, self.data)
            mujoco.mj_fwdVelocity(self.model, self.data)
            mujoco.mj_fullM(self.model, self.data, self.mass_matrix)
            torque = self.mass_matrix @ acceleration[frame] + self.data.qfrc_bias
            output[frame] = torque[self.dof_ids] / self.stiffness
        return torch.from_numpy(output)
