"""Analytic MDP heads shared by the simulator and the world model (SuperMimic paper Appendix A).

State tuple ``S = (p, R, v, w, q, qd)``: root position, rotation matrix, world linear/angular
velocity of the root link frame, joint positions and velocities. ``J`` is the number of joints.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kinematics import Kinematics
from .motion import CONTROL_DT, MotionLib
from .rotations import mat_to_6d, mm3, mv3, quat_to_mat, mat_to_quat, rot_angle_sq, yaw_mat
from .sensing import SensingCfg, obs_noise


def state_dim(J: int) -> int:
    return 3 + 4 + 3 + 3 + J + J


def obs_dim(J: int) -> int:
    return J + J + 3 + 3 + J + 6


def obs_scale(J: int) -> torch.Tensor:
    # normalization of the obs before the policy (qd and angular velocity are large)
    return torch.cat(
        (torch.ones(J), torch.full((J,), 0.1), torch.full((3,), 0.25), torch.ones(3), torch.ones(J), torch.ones(6))
    )


CONTACT_Z = 0.045  # ankle-roll link origin height below which a foot counts as in contact (as in tools/eval_metrics.py)
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT


@dataclass
class RewardCfg:
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


def flatten_state(S) -> torch.Tensor:
    base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = S
    return torch.cat((base_pos_w, mat_to_quat(base_rot_mat_w), base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel), -1)


def unflatten_state(x: torch.Tensor):
    J = (x.shape[-1] - 13) // 2
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13 : 13 + J], x[..., 13 + J :]


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = S
    H = yaw_mat(base_rot_mat_w)
    Rt = base_rot_mat_w.transpose(-1, -2)
    return torch.cat(
        (
            base_pos_w[..., 2:3],
            mat_to_6d(mm3(H.transpose(-1, -2), base_rot_mat_w)),
            mv3(Rt, base_lin_vel_w),
            mv3(Rt, base_ang_vel_w),
            joint_pos,
            joint_vel,
        ),
        -1,
    )


def inv_dim(J: int) -> int:
    return 1 + 6 + 3 + 3 + J + J



class MDP:
    def __init__(self, kin: Kinematics, motion: MotionLib, default_joint_pos: torch.Tensor, xi_dim: int,
                 sensing: SensingCfg = SensingCfg(), cfg: RewardCfg = RewardCfg()):  # fmt: skip
        self.kin, self.motion, self.cfg = kin, motion, cfg
        self.default_joint_pos = default_joint_pos
        mid, half = (kin.joint_pos_lower_limit + kin.joint_pos_upper_limit) / 2, (kin.joint_pos_upper_limit - kin.joint_pos_lower_limit) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        dev = default_joint_pos.device
        J = default_joint_pos.shape[0]
        self.num_j = J
        self.obs_dim, self.xi_dim = obs_dim(J), xi_dim
        self.critic_extra = J + len(kin.tracked_idx) * 3 + 3 + 6 + self.xi_dim
        self.obs_noise = obs_noise(sensing, J).to(dev)
        self.obs_scale = obs_scale(J).to(dev)
        # InstantTrack contact reward (training only): tracked-body indices of the feet, set by the trainer
        self.feet: list[int] = []

    # ---------------- observation ----------------
    def obs(self, S, last_action, simulation_parameters, motion_frame_index) -> torch.Tensor:
        """Noise-free proprioceptive observation (+ reference-relative root orientation)."""
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = S
        Rt = base_rot_mat_w.transpose(-1, -2)
        dq0 = simulation_parameters[..., 1 : 1 + joint_pos.shape[-1]]
        g_b = -base_rot_mat_w[..., 2, :]  # R^T (0,0,-1)
        rel = mat_to_6d(mm3(Rt, self.motion.base_rot_mat_w[motion_frame_index]))
        return torch.cat((joint_pos - self.default_joint_pos - dq0, joint_vel, mv3(Rt, base_ang_vel_w), g_b, last_action, rel), -1)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    # ---------------- reward / termination ----------------
    def reward(self, S, motion_frame_index, action, last_action, terms: bool = False, w_contact: float = 0.0):
        """Per-step reward (already multiplied by dt) comparing S with reference frame f.

        ``w_contact`` > 0 adds the InstantTrack contact term: a smooth foot-contact indicator (ankle-roll
        origin below 4.5 cm) matched to the reference's contact state, plus low horizontal foot speed while
        the reference foot is down. Evaluation keeps ``w_contact = 0``, i.e. the paper's reward."""
        c, m = self.cfg, self.motion
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = S
        bp, bR, bv, bw = self.kin.forward(base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel)
        rp, rR = m.base_pos_w[motion_frame_index], m.base_rot_mat_w[motion_frame_index]
        # reference bodies re-anchored to the robot's root (xy + yaw), as in BeyondMimic
        dR = yaw_mat(mm3(base_rot_mat_w, rR.transpose(-1, -2)))
        anchor = torch.cat((base_pos_w[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + mv3(dR.unsqueeze(-3), m.body_pos_w[motion_frame_index] - rp.unsqueeze(-2))
        ref_bR = mm3(dR.unsqueeze(-3), m.body_rot_mat_w[motion_frame_index])
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((m.body_lin_vel_w[motion_frame_index] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((m.body_ang_vel_w[motion_frame_index] - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - base_pos_w) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, base_rot_mat_w)
        r_track = (
            torch.exp(-e_pos / c.body_pos_std**2)
            + torch.exp(-e_ori / c.body_ori_std**2)
            + torch.exp(-e_lv / c.body_linvel_std**2)
            + torch.exp(-e_av / c.body_angvel_std**2)
            + 0.5 * torch.exp(-e_rp / c.root_pos_std**2)
            + 0.5 * torch.exp(-e_ro / c.root_ori_std**2)
        )
        r_rate = ((action - last_action) ** 2).sum(-1)
        r_lim = (torch.relu(joint_pos - self.soft_hi) + torch.relu(self.soft_lo - joint_pos)).sum(-1)
        r = (r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim) * CONTROL_DT
        if w_contact > 0:
            fz, rz = bp[..., self.feet, 2], m.body_pos_w[motion_frame_index][..., self.feet, 2]
            c_sim = torch.sigmoid((CONTACT_Z - fz) / 0.005)
            rc = getattr(m, "ref_contact", None)  # MotionLib.set_ref_contact (--ref_contact_fix), else the fixed 4.5 cm
            c_ref = rc[motion_frame_index] if rc is not None else (rz < CONTACT_Z).float()
            slip = (bv[..., self.feet, :2] ** 2).sum(-1)
            r_contact = (1 - (c_sim - c_ref) ** 2) + c_ref * torch.exp(-slip / 0.3**2)
            r = r + w_contact * r_contact.mean(-1) * CONTROL_DT
        if terms:
            return r, {
                "track": r_track * CONTROL_DT,
                "e_body_pos": e_pos,
                "e_body_ori": e_ori,
                "e_root_pos": e_rp,
                "action_rate": r_rate,
                "joint_limit": r_lim,
            }
        return r

    def termination(self, S, motion_frame_index) -> torch.Tensor:
        base_pos_w, base_rot_mat_w = S[0], S[1]
        rp, rR = self.motion.base_pos_w[motion_frame_index], self.motion.base_rot_mat_w[motion_frame_index]
        e_z = (rp[..., 2] - base_pos_w[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - base_rot_mat_w[..., 2, 2]).abs()
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def soft_termination(self, S, motion_frame_index, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        base_pos_w, base_rot_mat_w = S[0], S[1]
        rp, rR = self.motion.base_pos_w[motion_frame_index], self.motion.base_rot_mat_w[motion_frame_index]
        e_z = (rp[..., 2] - base_pos_w[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - base_rot_mat_w[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((e_z - self.cfg.term_height) / tz)) * (1 - torch.sigmoid((e_tilt - self.cfg.term_tilt) / tt))
        return 1 - alive

    def critic_features(self, S, motion_frame_index, simulation_parameters) -> torch.Tensor:
        """Critic input: invariant state, goal, sim context, and robot-vs-reference errors."""
        m = self.motion
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = S
        H = yaw_mat(base_rot_mat_w)
        Ht = H.transpose(-1, -2)
        rp, rR = m.base_pos_w[motion_frame_index], m.base_rot_mat_w[motion_frame_index]
        bp = self.kin.forward(base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel)[0]
        dR = yaw_mat(mm3(base_rot_mat_w, rR.transpose(-1, -2)))
        anchor = torch.cat((base_pos_w[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + mv3(dR.unsqueeze(-3), m.body_pos_w[motion_frame_index] - rp.unsqueeze(-2))
        e_body = mv3(Ht.unsqueeze(-3), ref_bp - bp).flatten(-2)
        e_root = mv3(Ht, rp - base_pos_w)
        rel = mat_to_6d(mm3(base_rot_mat_w.transpose(-1, -2), rR))
        return torch.cat((invariant_features(S), m.joint_pos[motion_frame_index] - joint_pos, e_body, e_root, rel, simulation_parameters), -1)
