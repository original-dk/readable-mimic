"""Current observation and its noise; history is maintained by Trainer."""

from dataclasses import dataclass
import torch
from .rotations import mat_to_6d


@dataclass(frozen=True)
class ObservationNoiseConfig:
    joint_pos_noise: float = 0.01
    joint_vel_noise: float = 0.5
    base_ang_vel_noise: float = 0.2
    projected_gravity_noise: float = 0.05
    ref_base_rot_noise: float = 0.05


def observation_noise_limits(config, num_joints):
    return torch.tensor(
        [config.joint_pos_noise] * num_joints
        + [config.joint_vel_noise] * num_joints
        + [config.base_ang_vel_noise] * 3
        + [config.projected_gravity_noise] * 3
        + [0.0] * num_joints
        + [config.ref_base_rot_noise] * 6
    )


def build_observation(
    state, last_action, simulation_parameters, motion_frame_index, motion, default_joint_pos
):
    _, base_rot_mat_w, _, base_ang_vel_w, joint_pos, joint_vel = state
    world_to_base = base_rot_mat_w.transpose(-1, -2)
    joint_pos_offset = simulation_parameters[..., 1 : 1 + joint_pos.shape[-1]]
    joint_pos_relative = joint_pos - default_joint_pos - joint_pos_offset
    base_ang_vel = (world_to_base @ base_ang_vel_w.unsqueeze(-1)).squeeze(-1)
    projected_gravity = -base_rot_mat_w[..., 2, :]
    ref_base_rot_6d_relative = mat_to_6d(world_to_base @ motion.base_rot_mat_w[motion_frame_index])
    return torch.cat(
        (
            joint_pos_relative,
            joint_vel,
            base_ang_vel,
            projected_gravity,
            last_action,
            ref_base_rot_6d_relative,
        ),
        -1,
    )
