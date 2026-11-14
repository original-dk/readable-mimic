"""Reference motion library: retargeted clips resampled to 50 Hz, with FK of tracked bodies."""

from __future__ import annotations

import glob
import os
from typing import Callable

import numpy as np
import torch

from .kinematics import Kinematics
from .robot import MotionArrays, RobotSpec
from .rotations import mat_to_6d, mm3, mv3, quat_slerp, quat_to_mat, so3_log, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50  # 1 s bins for return-based reset sampling


GOAL_Q_OFFSET = 15  # index of the joint positions q(f+1) in the goal vector (first future frame: dp 3, rot 6, v 3, w 3, then q)


class MotionLib:
    def __init__(
        self, files: list[str], kin: Kinematics, device: str, load_motion: Callable[[str], MotionArrays]
    ):
        self.device = device
        self.names = [
            os.path.splitext(os.path.basename(motion_frame_index))[0] for motion_frame_index in files
        ]
        ps, qs, js = [], [], []
        lens = []
        for motion_frame_index in files:
            base_pos_w, quat, j, src_fps = load_motion(motion_frame_index)
            base_pos_w, quat, j = (torch.tensor(x, dtype=torch.float32) for x in (base_pos_w, quat, j))
            # resample to 50 Hz
            T_src = base_pos_w.shape[0]
            dur = (T_src - 1) / src_fps
            n = int(np.floor(dur / CONTROL_DT)) + 1
            t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            action = (t - i0.float()).clamp(0, 1)
            base_pos_w = base_pos_w[i0] * (1 - action[:, None]) + base_pos_w[i0 + 1] * action[:, None]
            j = j[i0] * (1 - action[:, None]) + j[i0 + 1] * action[:, None]
            quat = quat_slerp(quat[i0], quat[i0 + 1], action)
            ps.append(base_pos_w), qs.append(quat), js.append(j)
            lens.append(n)
        self.clip_len = torch.tensor(lens, device=device)
        self.clip_start = torch.cumsum(
            torch.cat((torch.zeros(1, dtype=torch.long), torch.tensor(lens[:-1], dtype=torch.long))), 0
        ).to(device)  # dtype: one clip gives an empty (float) tensor
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(sum(lens))
        self.clip_id = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lens)]).to(
            device
        )

        base_pos_w_clips, base_rot_mat_w_clips, base_lin_vel_w_clips, base_ang_vel_w_clips, jq, jqd = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for base_pos_w, quat, j in zip(ps, qs, js):
            base_rot_mat_w = quat_to_mat(quat)
            base_lin_vel_w = _gradient(base_pos_w, CONTROL_DT)
            joint_vel = _gradient(j, CONTROL_DT)
            # angular velocity (world) by central differences of rotations
            base_ang_vel_w = torch.zeros_like(base_pos_w)
            base_ang_vel_w[1:-1] = so3_log(base_rot_mat_w[2:] @ base_rot_mat_w[:-2].transpose(-1, -2)) / (
                2 * CONTROL_DT
            )
            base_ang_vel_w[0] = so3_log(base_rot_mat_w[1] @ base_rot_mat_w[0].T) / CONTROL_DT
            base_ang_vel_w[-1] = so3_log(base_rot_mat_w[-1] @ base_rot_mat_w[-2].T) / CONTROL_DT
            (
                base_pos_w_clips.append(base_pos_w),
                base_rot_mat_w_clips.append(base_rot_mat_w),
                base_lin_vel_w_clips.append(base_lin_vel_w),
                base_ang_vel_w_clips.append(base_ang_vel_w),
                jq.append(j),
                jqd.append(joint_vel),
            )
        cat = lambda xs: torch.cat(xs).to(device)  # noqa: E731
        self.base_pos_w, self.base_rot_mat_w, self.base_lin_vel_w, self.base_ang_vel_w = (
            cat(base_pos_w_clips),
            cat(base_rot_mat_w_clips),
            cat(base_lin_vel_w_clips),
            cat(base_ang_vel_w_clips),
        )
        self.joint_pos, self.joint_vel = cat(jq), cat(jqd)
        with torch.no_grad():
            bp, bR, bv, bw = [], [], [], []
            for s in range(0, self.num_frames, 65536):
                e = min(s + 65536, self.num_frames)
                out = kin.forward(
                    self.base_pos_w[s:e],
                    self.base_rot_mat_w[s:e],
                    self.base_lin_vel_w[s:e],
                    self.base_ang_vel_w[s:e],
                    self.joint_pos[s:e],
                    self.joint_vel[s:e],
                )
                bp.append(out[0]), bR.append(out[1]), bv.append(out[2]), bw.append(out[3])
            self.body_pos_w, self.body_rot_mat_w, self.body_lin_vel_w, self.body_ang_vel_w = (
                torch.cat(bp),
                torch.cat(bR),
                torch.cat(bv),
                torch.cat(bw),
            )
        # per-frame static goal parts: q, qd, root velocities in the root frame
        Rt = self.base_rot_mat_w.transpose(-1, -2)
        self.base_lin_vel = (Rt @ self.base_lin_vel_w.unsqueeze(-1)).squeeze(-1)
        self.base_ang_vel = (Rt @ self.base_ang_vel_w.unsqueeze(-1)).squeeze(-1)
        self.root_gz = self.base_rot_mat_w[
            :, 2, 2
        ]  # -(projected gravity z) of the reference root: g_b = R^T (0,0,-1)
        # 1 s bins
        bin_clip, bin_start = [], []
        for c in range(len(lens)):
            nb = int(np.ceil(lens[c] / BIN_FRAMES))
            for b in range(nb):
                bin_clip.append(c)
                bin_start.append(int(self.clip_start[c]) + b * BIN_FRAMES)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=device)
        for b in range(self.num_bins):
            c = bin_clip[b]
            s = bin_start[b]
            e = min(s + BIN_FRAMES, int(self.clip_end[c]) + 1)
            self.frame_bin[s:e] = b

    @staticmethod
    def from_dir(path: str, robot: RobotSpec, kin: Kinematics, device: str) -> MotionLib:
        files = sorted(glob.glob(os.path.join(path, robot.motion_glob)))
        if not files:
            raise FileNotFoundError(f"No reference motion files found in {path}; run tools/download_data.py.")
        return MotionLib(files, kin, device, robot.load_motion)

    def set_ref_contact(
        self, feet: list[int], margin: float = 0.015, vmax: float = 0.5, joint_pos: float = 0.05
    ) -> None:
        """Speed-branch reference foot contact (num_frames, len(feet)): height below the clip's own floor (the
        q-quantile of that foot's height in the clip) + margin, and horizontal speed below vmax. A fixed 4.5 cm
        threshold misses most stance frames of clips whose retargeted feet sit ~1 cm higher (LAFAN1 subject1:
        floor ~4.5 cm against ~3.5 cm for the others)."""
        z, vh = self.body_pos_w[:, feet, 2], self.body_lin_vel_w[:, feet, :2].norm(dim=-1)
        floor = torch.zeros_like(z)
        for c in range(len(self.names)):
            s = self.clip_id == c
            floor[s] = z[s].quantile(joint_pos, dim=0)
        self.ref_contact = ((z < floor + margin) & (vh < vmax)).float()

    def command_dim(self, K: int) -> int:
        return K * (3 + 6 + 3 + 3 + 2 * self.joint_pos.shape[-1])

    def future_frames(self, motion_frame_index: torch.Tensor, K: int) -> torch.Tensor:
        """Frames f+1..f+K clamped to the clip's last frame, shape (B, K)."""
        end = self.clip_end[self.clip_id[motion_frame_index]]
        fut = motion_frame_index.unsqueeze(-1) + torch.arange(1, K + 1, device=motion_frame_index.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def command(self, motion_frame_index: torch.Tensor, K: int) -> torch.Tensor:
        """Goal features from the reference only: K future frames relative to the current reference frame."""
        fut = self.future_frames(motion_frame_index, K)  # (B,K)
        H = yaw_mat(self.base_rot_mat_w[motion_frame_index])  # (B,3,3)
        Ht = H.transpose(-1, -2).unsqueeze(1)
        dp = mv3(Ht, self.base_pos_w[fut] - self.base_pos_w[motion_frame_index].unsqueeze(1))
        rot = mat_to_6d(mm3(Ht, self.base_rot_mat_w[fut]))
        feats = torch.cat(
            (
                dp,
                rot,
                self.base_lin_vel[fut],
                self.base_ang_vel[fut],
                self.joint_pos[fut],
                self.joint_vel[fut] * 0.1,
            ),
            -1,
        )  # (B,K,15+2J)
        return feats.reshape(motion_frame_index.shape[0], -1)

    def ref_state(self, motion_frame_index: torch.Tensor):
        return (
            self.base_pos_w[motion_frame_index],
            self.base_rot_mat_w[motion_frame_index],
            self.base_lin_vel_w[motion_frame_index],
            self.base_ang_vel_w[motion_frame_index],
            self.joint_pos[motion_frame_index],
            self.joint_vel[motion_frame_index],
        )


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g
