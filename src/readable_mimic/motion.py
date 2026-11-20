"""Reference motion library: retargeted clips resampled to 50 Hz, with FK of tracked bodies."""

from __future__ import annotations
from .sim import CONTROL_DT
import glob
import os
from typing import Callable
import numpy as np
import torch
from .kinematics import Kinematics
from .robot import MotionArrays, RobotSpec
from .rotations import quat_slerp, quat_to_mat, so3_log

BIN_FRAMES = 50
REF_JOINT_POS_COMMAND_OFFSET = 15


class ReferenceMotion:
    def __init__(
        self,
        files: list[str],
        kinematics: Kinematics,
        device: str,
        load_motion: Callable[[str], MotionArrays],
    ):
        self.device = device
        self.names = [
            os.path.splitext(os.path.basename(motion_frame_index))[0] for motion_frame_index in files
        ]
        self.kinematics = kinematics
        base_pos_clips, compact_clips, clip_lengths = [], [], []
        for path in files:
            base_pos_w, base_quat_w, joint_pos, source_frame_rate = load_motion(path)
            base_pos_w, base_quat_w, joint_pos = (
                torch.as_tensor(values, dtype=torch.float32) for values in (base_pos_w, base_quat_w, joint_pos)
            )
            if len(base_pos_w) < 2 or not np.isfinite(source_frame_rate) or source_frame_rate <= 0:
                raise ValueError(f"Invalid reference clip: {path}")
            source_frame_count = base_pos_w.shape[0]
            clip_duration = (source_frame_count - 1) / source_frame_rate
            num_frames = int(np.floor(clip_duration / CONTROL_DT)) + 1
            sample_times = torch.arange(num_frames, dtype=torch.float32) * CONTROL_DT * source_frame_rate
            previous_frame_indices = torch.clamp(sample_times.floor().long(), max=source_frame_count - 2)
            interpolation_weight = (sample_times - previous_frame_indices.float()).clamp(0, 1)
            base_pos_w = (
                base_pos_w[previous_frame_indices] * (1 - interpolation_weight[:, None])
                + base_pos_w[previous_frame_indices + 1] * interpolation_weight[:, None]
            )
            joint_pos = (
                joint_pos[previous_frame_indices] * (1 - interpolation_weight[:, None])
                + joint_pos[previous_frame_indices + 1] * interpolation_weight[:, None]
            )
            base_quat_w = quat_slerp(
                base_quat_w[previous_frame_indices],
                base_quat_w[previous_frame_indices + 1],
                interpolation_weight,
            )
            if num_frames < 2:
                raise ValueError(f"Reference clip has fewer than two frames at 50 Hz: {path}")
            base_rot_mat_w = quat_to_mat(base_quat_w)
            base_lin_vel_w = _gradient(base_pos_w, CONTROL_DT)
            base_ang_vel_w = torch.empty_like(base_pos_w)
            base_ang_vel_w[1:-1] = so3_log(
                base_rot_mat_w[2:] @ base_rot_mat_w[:-2].transpose(-1, -2)
            ) / (2 * CONTROL_DT)
            base_ang_vel_w[0] = so3_log(base_rot_mat_w[1] @ base_rot_mat_w[0].T) / CONTROL_DT
            base_ang_vel_w[-1] = so3_log(base_rot_mat_w[-1] @ base_rot_mat_w[-2].T) / CONTROL_DT
            compact = torch.cat(
                (base_quat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, _gradient(joint_pos, CONTROL_DT)), -1
            ).half()
            if not torch.isfinite(compact).all():
                raise ValueError(f"Reference state exceeds FP16 range or is non-finite: {path}")
            base_pos_clips.append(base_pos_w)
            compact_clips.append(compact)
            clip_lengths.append(num_frames)

        self.num_frames = sum(clip_lengths)
        self.num_joints = (compact_clips[0].shape[-1] - 10) // 2
        self.base_pos_w = torch.empty((self.num_frames, 3), device=device, dtype=torch.float32)
        self.compact_state = torch.empty(
            (self.num_frames, compact_clips[0].shape[-1]), device=device, dtype=torch.float16
        )
        clip_starts, clip_ends, bin_clip, bin_start = [], [], [], []
        start = 0
        # Upload contiguous groups without allocating a second full-size library.
        for first in range(0, len(files), 512):
            last = min(first + 512, len(files))
            stop = start + sum(clip_lengths[first:last])
            self.base_pos_w[start:stop].copy_(torch.cat(base_pos_clips[first:last]))
            self.compact_state[start:stop].copy_(torch.cat(compact_clips[first:last]))
            for clip_index in range(first, last):
                length = clip_lengths[clip_index]
                clip_starts.append(start)
                clip_ends.append(start + length - 1)
                starts = range(start, start + length, BIN_FRAMES)
                bin_start.extend(starts)
                bin_clip.extend([clip_index] * len(starts))
                start += length
        self._clip_bounds = list(zip(clip_starts, clip_ends))
        self.clip_len = torch.tensor(clip_lengths, device=device)
        self.clip_start = torch.tensor(clip_starts, device=device)
        self.clip_end = torch.tensor(clip_ends, device=device)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.reset_starts = None

    def top_failure_bins(self, failure_counts, cumulative_motion_failures):
        counts = failure_counts.cpu().numpy()
        ranked = []
        for clip_index, (name, (start, end)) in enumerate(zip(self.names, self._clip_bounds)):
            clip_counts = counts[start : end + 1]
            total = int(clip_counts.sum())
            if total:
                cumulative_motion_failures[clip_index] += total
                bin_starts = np.arange(0, len(clip_counts), BIN_FRAMES)
                bin_failures = np.add.reduceat(clip_counts, bin_starts)
                peak_bin = int(bin_failures.argmax())
                ranked.append((
                    total, name, bin_starts[peak_bin], bin_failures[peak_bin],
                    cumulative_motion_failures[clip_index],
                ))
        top = sorted(ranked, key=lambda entry: (-entry[0], entry[1]))[:5]
        names = "|".join(name for _, name, _, _, _ in top)
        bins = "|".join(f"{start}:{count}:{cumulative}" for _, _, start, count, cumulative in top)
        return f"{names}@{bins}" if top else ""

    def end_frame(self, motion_frame_index):
        return self.clip_end[torch.searchsorted(self.clip_end, motion_frame_index.contiguous())]

    def base_rotation(self, motion_frame_index):
        quat = self.compact_state[motion_frame_index][..., :4].float()
        return quat_to_mat(torch.nn.functional.normalize(quat, dim=-1))

    def joint_position(self, motion_frame_index):
        return self.compact_state[motion_frame_index][..., 10 : 10 + self.num_joints].float()

    def ref_body_state(self, motion_frame_index):
        return self.kinematics.forward(*self.ref_state(motion_frame_index))

    @staticmethod
    def find_files(path: str, robot: RobotSpec) -> list[str]:
        files = sorted(glob.glob(os.path.join(path, robot.motion_glob)))
        if not files:
            raise FileNotFoundError(f"No reference motion files found in {path}; run tools/download_data.py.")
        return files

    @staticmethod
    def from_dir(path: str, robot: RobotSpec, kinematics: Kinematics, device: str) -> ReferenceMotion:
        return ReferenceMotion(ReferenceMotion.find_files(path, robot), kinematics, device, robot.load_motion)

    def set_ref_foot_contact(
        self, feet: list[int], margin: float = 0.015, vmax: float = 0.5, floor_quantile: float = 0.05
    ) -> None:
        """Speed-branch reference foot contact (num_frames, len(feet)): height below the clip's own floor (the
        q-quantile of that foot's height in the clip) + margin, and horizontal speed below vmax. A fixed 4.5 cm
        threshold misses most stance frames of clips whose retargeted feet sit ~1 cm higher (LAFAN1 subject1:
        floor ~4.5 cm against ~3.5 cm for the others)."""
        self.ref_foot_contact = torch.empty((self.num_frames, len(feet)), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            for start, end in self._clip_bounds:
                heights, speeds = [], []
                for first in range(start, end + 1, 65536):
                    indices = torch.arange(first, min(first + 65536, end + 1), device=self.device)
                    body_pos_w, _, body_lin_vel_w, _ = self.ref_body_state(indices)
                    heights.append(body_pos_w[:, feet, 2])
                    speeds.append(body_lin_vel_w[:, feet, :2].norm(dim=-1))
                foot_height, foot_speed = torch.cat(heights), torch.cat(speeds)
                floor = foot_height.quantile(floor_quantile, dim=0)
                self.ref_foot_contact[start : end + 1] = (foot_height < floor + margin) & (foot_speed < vmax)

    def ref_state(self, motion_frame_index: torch.Tensor):
        state = self.compact_state[motion_frame_index].float()
        rotation = quat_to_mat(torch.nn.functional.normalize(state[..., :4], dim=-1))
        return (
            self.base_pos_w[motion_frame_index],
            rotation,
            state[..., 4:7],
            state[..., 7:10],
            state[..., 10 : 10 + self.num_joints],
            state[..., 10 + self.num_joints :],
        )


def _gradient(values: torch.Tensor, time_step: float) -> torch.Tensor:
    gradient = torch.zeros_like(values)
    gradient[1:-1] = (values[2:] - values[:-2]) / (2 * time_step)
    gradient[0] = (values[1] - values[0]) / time_step
    gradient[-1] = (values[-1] - values[-2]) / time_step
    return gradient
