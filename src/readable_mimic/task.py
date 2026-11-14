"""Observation, tracking reward, termination and critic features shared by real and imagined states."""

from __future__ import annotations
from .observation import build_observation
from dataclasses import dataclass
import torch
from .kinematics import Kinematics
from .motion import CONTROL_DT, ReferenceMotion
from .rotations import mat_to_6d, mm3, mv3, quat_to_mat, mat_to_quat, rot_angle_sq, yaw_mat
from .observation import ObservationNoiseConfig, observation_noise_limits


def state_dim(num_joints: int) -> int:
    return 3 + 4 + 3 + 3 + num_joints + num_joints


def observation_dim(num_joints: int) -> int:
    return num_joints + num_joints + 3 + 3 + num_joints + 6


def observation_scale(num_joints: int) -> torch.Tensor:
    return torch.cat(
        (
            torch.ones(num_joints),
            torch.full((num_joints,), 0.1),
            torch.full((3,), 0.25),
            torch.ones(3),
            torch.ones(num_joints),
            torch.ones(6),
        )
    )


CONTACT_Z = 0.045
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT


@dataclass
class RewardConfig:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8


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


def invariant_features(state) -> torch.Tensor:
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


def invariant_feature_dim(num_joints: int) -> int:
    return 1 + 6 + 3 + 3 + num_joints + num_joints


class TrackingTask:
    def __init__(
        self,
        kinematics: Kinematics,
        motion: ReferenceMotion,
        default_joint_pos: torch.Tensor,
        num_simulation_parameters: int,
        observation_noise: ObservationNoiseConfig = ObservationNoiseConfig(),
        config: RewardConfig = RewardConfig(),
    ):
        self.kinematics, self.motion, self.config = (kinematics, motion, config)
        self.default_joint_pos = default_joint_pos
        joint_limit_center, joint_limit_half_range = (
            (kinematics.joint_pos_lower_limit + kinematics.joint_pos_upper_limit) / 2,
            (kinematics.joint_pos_upper_limit - kinematics.joint_pos_lower_limit) / 2,
        )
        self.joint_pos_soft_lower_limit, self.joint_pos_soft_upper_limit = (
            joint_limit_center - joint_limit_half_range * config.soft_limit_factor,
            joint_limit_center + joint_limit_half_range * config.soft_limit_factor,
        )
        device = default_joint_pos.device
        num_joints = default_joint_pos.shape[0]
        self.num_joints = num_joints
        self.observation_dim, self.num_simulation_parameters = (
            observation_dim(num_joints),
            num_simulation_parameters,
        )
        self.critic_extra = (
            num_joints + len(kinematics.tracked_body_indices) * 3 + 3 + 6 + self.num_simulation_parameters
        )
        self.observation_noise_limits = observation_noise_limits(observation_noise, num_joints).to(device)
        self.observation_scale = observation_scale(num_joints).to(device)
        self.feet: list[int] = []

    def observation(self, state, last_action, simulation_parameters, motion_frame_index) -> torch.Tensor:
        return build_observation(
            state, last_action, simulation_parameters, motion_frame_index, self.motion, self.default_joint_pos
        )

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return o + (2 * torch.rand_like(o) - 1) * self.observation_noise_limits

    def reward(
        self, state, motion_frame_index, action, last_action, terms: bool = False, w_contact: float = 0.0
    ):
        """Per-step reward (already multiplied by dt) comparing S with reference frame f.

        ``w_contact`` > 0 adds the InstantTrack contact term: a smooth foot-contact indicator (ankle-roll
        origin below 4.5 cm) matched to the reference's contact state, plus low horizontal foot speed while
        the reference foot is down. Evaluation keeps ``w_contact = 0``, i.e. the paper's reward."""
        reward_config, motion = (self.config, self.motion)
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = state
        body_pos_w, body_rot_mat_w, body_lin_vel_w, body_ang_vel_w = self.kinematics.forward(
            base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel
        )
        ref_base_pos_w, ref_base_rot_mat_w = (
            motion.base_pos_w[motion_frame_index],
            motion.base_rot_mat_w[motion_frame_index],
        )
        reference_alignment_rotation = yaw_mat(mm3(base_rot_mat_w, ref_base_rot_mat_w.transpose(-1, -2)))
        reference_anchor_pos_w = torch.cat((base_pos_w[..., :2], ref_base_pos_w[..., 2:3]), -1)
        aligned_ref_body_pos_w = reference_anchor_pos_w.unsqueeze(-2) + mv3(
            reference_alignment_rotation.unsqueeze(-3),
            motion.body_pos_w[motion_frame_index] - ref_base_pos_w.unsqueeze(-2),
        )
        aligned_ref_body_rot_mat_w = mm3(
            reference_alignment_rotation.unsqueeze(-3), motion.body_rot_mat_w[motion_frame_index]
        )
        body_pos_error = ((aligned_ref_body_pos_w - body_pos_w) ** 2).sum(-1).mean(-1)
        body_rotation_error = rot_angle_sq(aligned_ref_body_rot_mat_w, body_rot_mat_w).mean(-1)
        body_lin_vel_error = (
            ((motion.body_lin_vel_w[motion_frame_index] - body_lin_vel_w) ** 2).sum(-1).mean(-1)
        )
        body_ang_vel_error = (
            ((motion.body_ang_vel_w[motion_frame_index] - body_ang_vel_w) ** 2).sum(-1).mean(-1)
        )
        base_pos_error = ((ref_base_pos_w - base_pos_w) ** 2).sum(-1)
        base_rotation_error = rot_angle_sq(ref_base_rot_mat_w, base_rot_mat_w)
        tracking_reward = (
            torch.exp(-body_pos_error / reward_config.body_pos_std**2)
            + torch.exp(-body_rotation_error / reward_config.body_ori_std**2)
            + torch.exp(-body_lin_vel_error / reward_config.body_linvel_std**2)
            + torch.exp(-body_ang_vel_error / reward_config.body_angvel_std**2)
            + 0.5 * torch.exp(-base_pos_error / reward_config.root_pos_std**2)
            + 0.5 * torch.exp(-base_rotation_error / reward_config.root_ori_std**2)
        )
        action_rate_penalty = ((action - last_action) ** 2).sum(-1)
        joint_limit_penalty = (
            torch.relu(joint_pos - self.joint_pos_soft_upper_limit)
            + torch.relu(self.joint_pos_soft_lower_limit - joint_pos)
        ).sum(-1)
        reward = (
            tracking_reward
            + reward_config.w_action_rate * action_rate_penalty
            + reward_config.w_joint_limit * joint_limit_penalty
        ) * CONTROL_DT
        if w_contact > 0:
            foot_height, ref_foot_height = (
                body_pos_w[..., self.feet, 2],
                motion.body_pos_w[motion_frame_index][..., self.feet, 2],
            )
            foot_contact_probability = torch.sigmoid((CONTACT_Z - foot_height) / 0.005)
            reference_contact_labels = getattr(motion, "ref_contact", None)
            ref_foot_contact = (
                reference_contact_labels[motion_frame_index]
                if reference_contact_labels is not None
                else (ref_foot_height < CONTACT_Z).float()
            )
            foot_horizontal_speed_squared = (body_lin_vel_w[..., self.feet, :2] ** 2).sum(-1)
            foot_contact_reward = (
                1
                - (foot_contact_probability - ref_foot_contact) ** 2
                + ref_foot_contact * torch.exp(-foot_horizontal_speed_squared / 0.3**2)
            )
            reward = reward + w_contact * foot_contact_reward.mean(-1) * CONTROL_DT
        if terms:
            return (
                reward,
                {
                    "track": tracking_reward * CONTROL_DT,
                    "e_body_pos": body_pos_error,
                    "e_body_ori": body_rotation_error,
                    "e_root_pos": base_pos_error,
                    "action_rate": action_rate_penalty,
                    "joint_limit": joint_limit_penalty,
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
        return (base_height_error > self.config.term_height) | (base_tilt_error > self.config.term_tilt)

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
            1 - torch.sigmoid((base_height_error - self.config.term_height) / height_transition_width)
        ) * (1 - torch.sigmoid((base_tilt_error - self.config.term_tilt) / tilt_transition_width))
        return 1 - survival_probability

    def critic_features(self, state, motion_frame_index, simulation_parameters) -> torch.Tensor:
        """Critic input: invariant state, goal, sim context, and robot-vs-reference errors."""
        motion = self.motion
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = state
        base_heading_rot_mat_w = yaw_mat(base_rot_mat_w)
        world_to_heading = base_heading_rot_mat_w.transpose(-1, -2)
        ref_base_pos_w, ref_base_rot_mat_w = (
            motion.base_pos_w[motion_frame_index],
            motion.base_rot_mat_w[motion_frame_index],
        )
        body_pos_w = self.kinematics.forward(
            base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel
        )[0]
        reference_alignment_rotation = yaw_mat(mm3(base_rot_mat_w, ref_base_rot_mat_w.transpose(-1, -2)))
        reference_anchor_pos_w = torch.cat((base_pos_w[..., :2], ref_base_pos_w[..., 2:3]), -1)
        aligned_ref_body_pos_w = reference_anchor_pos_w.unsqueeze(-2) + mv3(
            reference_alignment_rotation.unsqueeze(-3),
            motion.body_pos_w[motion_frame_index] - ref_base_pos_w.unsqueeze(-2),
        )
        body_pos_error_heading = mv3(
            world_to_heading.unsqueeze(-3), aligned_ref_body_pos_w - body_pos_w
        ).flatten(-2)
        base_pos_error_heading = mv3(world_to_heading, ref_base_pos_w - base_pos_w)
        ref_base_rot_6d_relative = mat_to_6d(mm3(base_rot_mat_w.transpose(-1, -2), ref_base_rot_mat_w))
        return torch.cat(
            (
                invariant_features(state),
                motion.joint_pos[motion_frame_index] - joint_pos,
                body_pos_error_heading,
                base_pos_error_heading,
                ref_base_rot_6d_relative,
                simulation_parameters,
            ),
            -1,
        )
