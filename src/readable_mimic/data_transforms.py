"""Shared data conversions for obs, command, history, WM / critic inputs and training targets.

Functions receive their data explicitly; they never read Trainer or TrainingUpdates state.
"""

from __future__ import annotations
import torch
from .rotations import mat_to_6d, mm3, mv3, quat_to_mat, mat_to_quat, yaw_mat
from .sim import default_xi


def compute_state_dim(num_joints: int) -> int:
    return 3 + 4 + 3 + 3 + num_joints + num_joints


def compute_observation_dim(num_joints: int) -> int:
    return 3 * num_joints + 6


def flatten_state(state) -> torch.Tensor:
    base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = state
    return torch.cat(
        (base_pos_w, mat_to_quat(base_rot_mat_w), base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel), -1
    )


def unflatten_state(flat_state: torch.Tensor):
    num_joints = (flat_state.shape[-1] - 13) // 2
    return (
        flat_state[..., 0:3],
        quat_to_mat(flat_state[..., 3:7]),
        flat_state[..., 7:10],
        flat_state[..., 10:13],
        flat_state[..., 13 : 13 + num_joints],
        flat_state[..., 13 + num_joints :],
    )


def compute_invariant_features(state) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = state
    base_heading_rot_mat_w = yaw_mat(base_rot_mat_w)
    world_to_base = base_rot_mat_w.transpose(-1, -2)
    return torch.cat(
        (
            base_pos_w[..., 2:3],
            mat_to_6d(mm3(base_heading_rot_mat_w.transpose(-1, -2), base_rot_mat_w)),
            mv3(world_to_base, base_lin_vel_w),
            mv3(world_to_base, base_ang_vel_w),
            joint_pos,
            joint_vel,
        ),
        -1,
    )


def compute_invariant_feature_dim(num_joints: int) -> int:
    return 1 + 6 + 3 + 3 + num_joints + num_joints


def compute_observation_noise_limits(config, num_joints):
    return torch.tensor(
        [config.joint_pos_noise] * num_joints
        + [config.joint_vel_noise] * num_joints
        + [config.base_ang_vel_noise] * 3
        + [config.projected_gravity_noise] * 3
        + [0.0] * num_joints
    )


def compute_observation(state, last_action, xi, motion_frame_index, motion, default_joint_pos):
    _, base_rot_mat_w, _, base_ang_vel_w, joint_pos, joint_vel = state
    world_to_base = base_rot_mat_w.transpose(-1, -2)
    joint_pos_offset = xi[..., 1 : 1 + joint_pos.shape[-1]]
    joint_pos_relative = joint_pos - default_joint_pos - joint_pos_offset
    base_ang_vel = (world_to_base @ base_ang_vel_w.unsqueeze(-1)).squeeze(-1)
    projected_gravity = -base_rot_mat_w[..., 2, :]
    return torch.cat(
        (
            joint_pos_relative,
            joint_vel,
            base_ang_vel,
            projected_gravity,
            last_action,
        ),
        -1,
    )


def apply_observation_noise(observation: torch.Tensor, noise_limits) -> torch.Tensor:
    return observation + (2 * torch.rand_like(observation) - 1) * noise_limits


def compute_command_dim(num_joints, command_frames) -> int:
    return command_frames * (15 + 2 * num_joints)


def compute_future_motion_frames(motion_frame_index, command_frame_offsets, motion) -> torch.Tensor:
    clip_end_frames = motion.end_frame(motion_frame_index)
    future_frames = motion_frame_index[..., None] + command_frame_offsets
    return torch.minimum(future_frames, clip_end_frames[..., None])


def compute_command(motion_frame_index: torch.Tensor, command_frame_offsets, motion) -> torch.Tensor:
    """Future ref motion frames expressed relative to the current ref heading."""
    future_frames = compute_future_motion_frames(motion_frame_index, command_frame_offsets, motion)
    base_heading_rot_mat_w = yaw_mat(motion.base_rotation(motion_frame_index))
    world_to_heading = base_heading_rot_mat_w.transpose(-1, -2).unsqueeze(1)
    ref_pos, ref_rot, ref_lin_vel, ref_ang_vel, ref_joint_pos, ref_joint_vel = motion.ref_state(future_frames)
    ref_world_to_base = ref_rot.transpose(-1, -2)
    ref_base_pos_offset_heading = mv3(
        world_to_heading,
        ref_pos - motion.base_pos_w[motion_frame_index].unsqueeze(1),
    )
    ref_base_rot_6d_heading = mat_to_6d(mm3(world_to_heading, ref_rot))
    command_features = torch.cat(
        (
            ref_base_pos_offset_heading,
            ref_base_rot_6d_heading,
            mv3(ref_world_to_base, ref_lin_vel),
            mv3(ref_world_to_base, ref_ang_vel),
            ref_joint_pos,
            ref_joint_vel * 0.1,
        ),
        -1,
    )
    return command_features.reshape(motion_frame_index.shape[0], -1)


def compute_actor_return(
    rewards, termination_probabilities, bootstrap_value, discount_factors, bootstrap_discount
):
    survival_after_step = torch.cumprod(1 - termination_probabilities, 0)
    survival_weights = torch.cat((torch.ones_like(rewards[:1]), survival_after_step[:-1]), 0)
    actor_return = (discount_factors * survival_weights * rewards).sum(0)
    final_survival = survival_after_step[-1]
    actor_return += bootstrap_discount * final_survival * bootstrap_value
    return (actor_return, survival_weights, final_survival)


def compute_value_targets(state_values, detached_rewards, detached_terminations, gamma, gae_lambda):
    """GAE(lambda) returns over the imagined transitions; V is (H+1, B), bootstrapped from V[-1]."""
    return_target = state_values[-1]
    targets = []
    for step_index in range(detached_rewards.shape[0] - 1, -1, -1):
        return_target = detached_rewards[step_index] + gamma * (1 - detached_terminations[step_index]) * (
            (1 - gae_lambda) * state_values[step_index + 1] + gae_lambda * return_target
        )
        targets.append(return_target)
    return torch.stack(targets[::-1])


def compute_critic_input(state, motion_frame_index, normalized_xi, command, task, init_root, init_f):
    body = task.body_diff(state, motion_frame_index)
    base = task.base_diff(state, motion_frame_index, init_root, init_f)
    joint_pos, joint_vel = state[4:]
    joint_error = task.motion.joint_position(motion_frame_index) - joint_pos
    return torch.cat((joint_pos, joint_vel, joint_error, *(diff.flatten(-2) for diff in body),
                      *base, normalized_xi, command), -1)


def compute_critic_input_dim(num_joints, num_tracked_bodies, xi_dim, command_dim):
    return 3 * num_joints + 12 * num_tracked_bodies + 12 + xi_dim + command_dim


def compute_observation_history(observation, history_length):
    return (
        observation.unsqueeze(-2)
        .expand(*observation.shape[:-1], history_length, observation.shape[-1])
        .clone()
    )


def update_observation_history(history, observation, reset_mask=None):
    next_history = torch.cat((history[..., 1:, :], observation.unsqueeze(-2)), -2)
    if reset_mask is None:
        return next_history
    initial_history = compute_observation_history(observation, history.shape[-2])
    return torch.where(reset_mask[..., None, None], initial_history, next_history)


def compute_history_ages(ages, episode_steps, history_lookback):
    lookback = torch.minimum(torch.minimum(history_lookback, episode_steps[:, None]), ages[:, None])
    return ages[:, None] - lookback


def compute_replay_observation_history(history, motion, default_joint_pos, noise_limits):
    observation_history = compute_observation(
        unflatten_state(history["state"]),
        history["last_action"],
        history["xi"],
        history["motion_frame_index"],
        motion,
        default_joint_pos,
    )
    return apply_observation_noise(observation_history, noise_limits)


def compute_world_model_training_data(
    world_model,
    initial_state,
    next_state,
    action,
    last_action,
    xi,
    xi_center,
    xi_scale,
):
    normalized_xi = compute_normalized_xi(xi, xi_center, xi_scale)
    return (
        world_model.inputs(initial_state, action, last_action, normalized_xi),
        world_model.targets(initial_state, next_state),
        world_model.increments(initial_state, next_state),
    )


def compute_normalized_xi(xi, center, scale):
    return (xi - center) / scale


def compute_ghost_pose(ref_state, robot_state, follow_base):
    ref_base_pos_w, ref_base_rot_mat_w, _, _, ref_joint_pos, _ = ref_state
    if not follow_base:
        return (ref_base_pos_w, ref_base_rot_mat_w, ref_joint_pos)
    base_pos_w, base_rot_mat_w = robot_state[:2]
    ghost_base_pos_w = torch.cat((base_pos_w[..., :2], ref_base_pos_w[..., 2:3]), -1)
    ref_alignment_rot_mat_w = yaw_mat(base_rot_mat_w @ ref_base_rot_mat_w.transpose(-1, -2))
    ghost_base_rot_mat_w = ref_alignment_rot_mat_w @ ref_base_rot_mat_w
    return (ghost_base_pos_w, ghost_base_rot_mat_w, ref_joint_pos)


def compute_xi_normalization(robot, offset_limits):
    friction_lower, friction_upper = robot.domain_randomization.friction_range
    center = default_xi(robot, offset_limits.device)
    scale = torch.cat(
        (
            offset_limits.new_tensor([(friction_upper - friction_lower) / 2]),
            offset_limits.clamp_min(1e-06),
            offset_limits.new_full((3,), 0.05),
        )
    )
    return (center, scale)
