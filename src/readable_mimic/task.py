"""Tracking rewards and termination conditions shared by real and imagined states."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from .sim import CONTROL_DT
from .rotations import mm3, mv3, rot_angle_sq, yaw_mat

CONTACT_Z = 0.045
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT


@dataclass
class RewardConfig:
    body_pos_std: float = 0.3
    body_rot_std: float = 0.4
    body_lin_vel_std: float = 1.0
    body_ang_vel_std: float = 3.14
    base_pos_std: float = 0.3
    base_rot_std: float = 0.4
    action_rate_weight: float = -0.1
    joint_limit_weight: float = -2.0
    soft_limit_factor: float = 0.9
    termination_height_threshold: float = 0.25
    termination_tilt_threshold: float = 0.8


class TrackingTask:
    def __init__(self, kinematics, motion, config):
        self.kinematics = kinematics
        self.motion = motion
        self.config = config
        joint_limit_center = (kinematics.joint_pos_lower_limit + kinematics.joint_pos_upper_limit) / 2
        joint_limit_half_range = (kinematics.joint_pos_upper_limit - kinematics.joint_pos_lower_limit) / 2
        self.joint_pos_soft_lower_limit = (
            joint_limit_center - joint_limit_half_range * config.soft_limit_factor
        )
        self.joint_pos_soft_upper_limit = (
            joint_limit_center + joint_limit_half_range * config.soft_limit_factor
        )
        self.feet = []

    def reward(
        self,
        state,
        motion_frame_index,
        action,
        last_action,
        return_details: bool = False,
        foot_contact_reward_weight: float = 0.0,
    ):
        """Per-step reward, scaled by CONTROL_DT, compared with the ref motion frame.

        ``foot_contact_reward_weight`` > 0 adds the InstantTrack contact term: a smooth foot-contact indicator (ankle-roll
        origin below 4.5 cm) matched to the reference's contact state, plus low horizontal foot speed while
        the reference foot is down. Evaluation keeps ``foot_contact_reward_weight = 0``, i.e. the paper's reward."""
        reward_config, motion = (self.config, self.motion)
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = state
        body_pos_w, body_rot_mat_w, body_lin_vel_w, body_ang_vel_w = self.kinematics.forward(
            base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel
        )
        ref_base_pos_w, ref_base_rot_mat_w = (
            motion.base_pos_w[motion_frame_index],
            motion.base_rot_mat_w[motion_frame_index],
        )
        ref_alignment_rot_mat_w = yaw_mat(mm3(base_rot_mat_w, ref_base_rot_mat_w.transpose(-1, -2)))
        ref_anchor_pos_w = torch.cat((base_pos_w[..., :2], ref_base_pos_w[..., 2:3]), -1)
        ref_body_pos_w_aligned = ref_anchor_pos_w.unsqueeze(-2) + mv3(
            ref_alignment_rot_mat_w.unsqueeze(-3),
            motion.body_pos_w[motion_frame_index] - ref_base_pos_w.unsqueeze(-2),
        )
        ref_body_rot_mat_w_aligned = mm3(
            ref_alignment_rot_mat_w.unsqueeze(-3), motion.body_rot_mat_w[motion_frame_index]
        )
        body_pos_error = ((ref_body_pos_w_aligned - body_pos_w) ** 2).sum(-1).mean(-1)
        body_rot_error = rot_angle_sq(ref_body_rot_mat_w_aligned, body_rot_mat_w).mean(-1)
        body_lin_vel_error = (
            ((motion.body_lin_vel_w[motion_frame_index] - body_lin_vel_w) ** 2).sum(-1).mean(-1)
        )
        body_ang_vel_error = (
            ((motion.body_ang_vel_w[motion_frame_index] - body_ang_vel_w) ** 2).sum(-1).mean(-1)
        )
        base_pos_error = ((ref_base_pos_w - base_pos_w) ** 2).sum(-1)
        base_rot_error = rot_angle_sq(ref_base_rot_mat_w, base_rot_mat_w)
        tracking_reward = (
            torch.exp(-body_pos_error / reward_config.body_pos_std**2)
            + torch.exp(-body_rot_error / reward_config.body_rot_std**2)
            + torch.exp(-body_lin_vel_error / reward_config.body_lin_vel_std**2)
            + torch.exp(-body_ang_vel_error / reward_config.body_ang_vel_std**2)
            + 0.5 * torch.exp(-base_pos_error / reward_config.base_pos_std**2)
            + 0.5 * torch.exp(-base_rot_error / reward_config.base_rot_std**2)
        )
        action_rate_penalty = ((action - last_action) ** 2).sum(-1)
        joint_limit_penalty = (
            torch.relu(joint_pos - self.joint_pos_soft_upper_limit)
            + torch.relu(self.joint_pos_soft_lower_limit - joint_pos)
        ).sum(-1)
        reward = (
            tracking_reward
            + reward_config.action_rate_weight * action_rate_penalty
            + reward_config.joint_limit_weight * joint_limit_penalty
        ) * CONTROL_DT
        if foot_contact_reward_weight > 0:
            foot_height, ref_foot_height = (
                body_pos_w[..., self.feet, 2],
                motion.body_pos_w[motion_frame_index][..., self.feet, 2],
            )
            foot_contact_probability = torch.sigmoid((CONTACT_Z - foot_height) / 0.005)
            ref_foot_contact_labels = getattr(motion, "ref_foot_contact", None)
            ref_foot_contact = (
                ref_foot_contact_labels[motion_frame_index]
                if ref_foot_contact_labels is not None
                else (ref_foot_height < CONTACT_Z).float()
            )
            foot_horizontal_speed_squared = (body_lin_vel_w[..., self.feet, :2] ** 2).sum(-1)
            foot_contact_reward = (
                1
                - (foot_contact_probability - ref_foot_contact) ** 2
                + ref_foot_contact * torch.exp(-foot_horizontal_speed_squared / 0.3**2)
            )
            reward = reward + foot_contact_reward_weight * foot_contact_reward.mean(-1) * CONTROL_DT
        if return_details:
            tracking_reward = tracking_reward * CONTROL_DT
            return (
                reward,
                {
                    "tracking_reward": tracking_reward,
                    "body_pos_error": body_pos_error,
                    "body_rot_error": body_rot_error,
                    "body_lin_vel_error": body_lin_vel_error,
                    "body_ang_vel_error": body_ang_vel_error,
                    "base_pos_error": base_pos_error,
                    "base_rot_error": base_rot_error,
                    "action_rate_penalty": action_rate_penalty,
                    "joint_limit_penalty": joint_limit_penalty,
                },
            )
        return reward

    def termination(self, state, motion_frame_index) -> torch.Tensor:
        base_pos_w, base_rot_mat_w = (state[0], state[1])
        ref_base_pos_w, ref_base_rot_mat_w = (
            self.motion.base_pos_w[motion_frame_index],
            self.motion.base_rot_mat_w[motion_frame_index],
        )
        base_height_error = (ref_base_pos_w[..., 2] - base_pos_w[..., 2]).abs()
        base_tilt_error = (ref_base_rot_mat_w[..., 2, 2] - base_rot_mat_w[..., 2, 2]).abs()
        return (base_height_error > self.config.termination_height_threshold) | (
            base_tilt_error > self.config.termination_tilt_threshold
        )

    def soft_termination(
        self,
        state,
        motion_frame_index,
        height_transition_width: float = 0.02,
        tilt_transition_width: float = 0.04,
    ) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        base_pos_w, base_rot_mat_w = (state[0], state[1])
        ref_base_pos_w, ref_base_rot_mat_w = (
            self.motion.base_pos_w[motion_frame_index],
            self.motion.base_rot_mat_w[motion_frame_index],
        )
        base_height_error = (ref_base_pos_w[..., 2] - base_pos_w[..., 2]).abs()
        base_tilt_error = (ref_base_rot_mat_w[..., 2, 2] - base_rot_mat_w[..., 2, 2]).abs()
        survival_probability = (
            1
            - torch.sigmoid(
                (base_height_error - self.config.termination_height_threshold) / height_transition_width
            )
        ) * (
            1
            - torch.sigmoid(
                (base_tilt_error - self.config.termination_tilt_threshold) / tilt_transition_width
            )
        )
        return 1 - survival_probability
