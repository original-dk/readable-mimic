"""Tracking rewards and termination conditions shared by real and imagined states."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from .sim import CONTROL_DT
from .rotations import mm3, mv3, mat_to_quat, quat_to_mat, so3_log

CONTACT_Z = 0.045
MAX_STEP_REWARD = (8 * 0.75) * CONTROL_DT


@dataclass
class RewardConfig:
    body_pos_std: float = 0.3
    body_rot_std: float = 0.4
    body_lin_vel_std: float = 1.0
    body_ang_vel_std: float = 3.14
    base_pos_std: float = 0.5
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

    def base_diff(self, S, f, init_root=None, init_f=None):
        """Reference base diffs: initial-frame displacement/rotation, current-base local velocities.

        ``init_root`` is robot xyz+wxyz after reset noise; ``init_f`` selects the expert frame before
        noise. Without context, position/orientation compare absolute poses for standalone playback.
        Rotation diff is 2 * sign(q_err.w) * q_err.xyz, not a rotation vector/geodesic angle.
        Velocity diffs use each current base frame; initial velocities are not subtracted.
        """
        assert (init_root is None) == (init_f is None), "reset pose and expert frame must be supplied together"
        p, R, v, w = S[:4]
        m = self.motion
        rp, rR, ref_v, ref_w = m.ref_state(f)[:4]
        if init_root is None:
            diff_pos, rel_R, ref_rel_R = rp - p, R, rR
        else:
            p0, R0 = init_root[..., :3], quat_to_mat(init_root[..., 3:7])
            rp0, rR0 = m.base_pos_w[init_f], m.base_rotation(init_f)
            R0t, rR0t = R0.transpose(-1, -2), rR0.transpose(-1, -2)
            diff_pos = mv3(rR0t, rp - rp0) - mv3(R0t, p - p0)
            rel_R, ref_rel_R = mm3(R0t, R), mm3(rR0t, rR)
        q_err = mat_to_quat(mm3(rel_R.transpose(-1, -2), ref_rel_R))
        sign = torch.where(q_err[..., :1] < 0, -1.0, 1.0)
        diff_rot = 2 * sign * q_err[..., 1:]
        Rt, rRt = R.transpose(-1, -2), rR.transpose(-1, -2)
        diff_lv = mv3(rRt, ref_v) - mv3(Rt, v)
        diff_av = mv3(rRt, ref_w) - mv3(Rt, w)
        return diff_pos, diff_rot, diff_lv, diff_av

    def hybrid_root(self, R, f):
        """Robot's current yaw with the reference's current roll/pitch; not a reset-frame transform."""
        rR = self.motion.base_rotation(f)

        def heading(A):
            x, y = A[..., 0, 0], A[..., 1, 0]
            valid = x.square() + y.square() > 1e-12
            # At a vertical forward axis yaw is undefined: use zero, keeping a valid rotation and finite gradients.
            return torch.atan2(torch.where(valid, y, 0.0), torch.where(valid, x, 1.0))

        delta = heading(R) - heading(rR)
        c, s = delta.cos(), delta.sin()
        z, one = torch.zeros_like(c), torch.ones_like(c)
        Y = torch.stack((c, -s, z, s, c, z, z, z, one), -1).reshape(R.shape)
        return mm3(Y, rR)

    def body_diff(self, S, f, body_state=None, ref_body_state=None):
        """Link diffs in reference-root / hybrid-root frames, with rotation as a shortest-path 3D rotvec.

        Subtract each own root for positions. Velocities are full link world velocities rotated into
        those frames; do not subtract base velocity. ``body_state`` reuses FK when reward needs contacts.
        All four outputs have shape (..., tracked_links, 3).
        """
        p, R = S[:2]
        bp, bR, bv, bw = self.kinematics.forward(*S) if body_state is None else body_state
        m = self.motion
        rp, rR = m.base_pos_w[f], m.base_rotation(f)
        ref_p, ref_R, ref_v, ref_w = m.ref_body_state(f) if ref_body_state is None else ref_body_state
        rRt = rR.transpose(-1, -2).unsqueeze(-3)
        Ht = self.hybrid_root(R, f).transpose(-1, -2).unsqueeze(-3)
        diff_pos = mv3(rRt, ref_p - rp.unsqueeze(-2)) - mv3(Ht, bp - p.unsqueeze(-2))
        ref_local_R, local_R = mm3(rRt, ref_R), mm3(Ht, bR)
        diff_rot = so3_log(mm3(local_R.transpose(-1, -2), ref_local_R))
        diff_lv = mv3(rRt, ref_v) - mv3(Ht, bv)
        diff_av = mv3(rRt, ref_w) - mv3(Ht, bw)
        return diff_pos, diff_rot, diff_lv, diff_av

    def reward(
        self,
        state,
        motion_frame_index,
        action,
        last_action,
        return_details: bool = False,
        foot_contact_reward_weight: float = 0.0,
        init_root=None,
        init_f=None,
    ):
        """Per-step reward, scaled by CONTROL_DT, compared with the ref motion frame.

        ``foot_contact_reward_weight`` > 0 adds the InstantTrack contact term: a smooth foot-contact indicator (ankle-roll
        origin below 4.5 cm) matched to the reference's contact state, plus low horizontal foot speed while
        the reference foot is down. Evaluation keeps ``foot_contact_reward_weight = 0``, i.e. the paper's reward."""
        reward_config, motion = (self.config, self.motion)
        joint_pos = state[4]
        body_state = self.kinematics.forward(*state)
        body_pos_w, _, body_lin_vel_w, _ = body_state
        ref_body_state = motion.ref_body_state(motion_frame_index)
        body_errors = self.body_diff(state, motion_frame_index, body_state, ref_body_state)
        body_pos_error, body_rot_error, body_lin_vel_error, body_ang_vel_error = (
            diff.square().sum(-1).mean(-1) for diff in body_errors
        )
        base_pos_error, base_rot_error, base_lin_vel_error, base_ang_vel_error = (
            diff.square().sum(-1) for diff in self.base_diff(state, motion_frame_index, init_root, init_f)
        )
        # The reference sigma sums position and rotation over its 14 tracked links.
        link_scale = body_pos_w.shape[-2] / 14
        tracking_reward = 0.75 * (
            torch.exp(-body_pos_error * link_scale / reward_config.body_pos_std**2)
            + torch.exp(-body_rot_error * link_scale / reward_config.body_rot_std**2)
            + torch.exp(-body_lin_vel_error / reward_config.body_lin_vel_std**2)
            + torch.exp(-body_ang_vel_error / reward_config.body_ang_vel_std**2)
            + torch.exp(-base_pos_error / reward_config.base_pos_std**2)
            + torch.exp(-base_rot_error / reward_config.base_rot_std**2)
            + torch.exp(-base_lin_vel_error)
            + torch.exp(-base_ang_vel_error / 3.14**2)
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
                ref_body_state[0][..., self.feet, 2],
            )
            foot_contact_probability = torch.sigmoid((CONTACT_Z - foot_height) / 0.005)
            ref_foot_contact_labels = getattr(motion, "ref_foot_contact", None)
            ref_foot_contact = (
                ref_foot_contact_labels[motion_frame_index].float()
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
                    "base_lin_vel_error": base_lin_vel_error,
                    "base_ang_vel_error": base_ang_vel_error,
                    "base_rot_error": base_rot_error,
                    "action_rate_penalty": action_rate_penalty,
                    "joint_limit_penalty": joint_limit_penalty,
                },
            )
        return reward

    def termination_errors(self, state, motion_frame_index):
        base_pos_w, base_rot_mat_w = (state[0], state[1])
        ref_base_pos_w, ref_base_rot_mat_w = (
            self.motion.base_pos_w[motion_frame_index],
            self.motion.base_rotation(motion_frame_index),
        )
        base_height_error = (ref_base_pos_w[..., 2] - base_pos_w[..., 2]).abs()
        base_tilt_error = (ref_base_rot_mat_w[..., 2, 2] - base_rot_mat_w[..., 2, 2]).abs()
        return base_height_error, base_tilt_error

    def termination(self, state, motion_frame_index) -> torch.Tensor:
        base_height_error, base_tilt_error = self.termination_errors(state, motion_frame_index)
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
        base_height_error, base_tilt_error = self.termination_errors(state, motion_frame_index)
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
