"""Reference motion library: retargeted clips resampled to 50 Hz, with FK of tracked bodies."""

from __future__ import annotations
import glob
import os
from typing import Callable
import numpy as np
import torch
from .kinematics import Kinematics
from .robot import MotionArrays, RobotSpec
from .rotations import quat_slerp, quat_to_mat, so3_log

CONTROL_DT = 0.02
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
        base_pos_clips, base_quat_clips, joint_pos_clips = ([], [], [])
        clip_lengths = []
        for motion_frame_index in files:
            base_pos_w, base_quat_w, joint_pos, source_frame_rate = load_motion(motion_frame_index)
            base_pos_w, base_quat_w, joint_pos = (
                torch.tensor(values, dtype=torch.float32) for values in (base_pos_w, base_quat_w, joint_pos)
            )
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
            (
                base_pos_clips.append(base_pos_w),
                base_quat_clips.append(base_quat_w),
                joint_pos_clips.append(joint_pos),
            )
            clip_lengths.append(num_frames)
        self.clip_len = torch.tensor(clip_lengths, device=device)
        self.clip_start = torch.cumsum(
            torch.cat((torch.zeros(1, dtype=torch.long), torch.tensor(clip_lengths[:-1], dtype=torch.long))),
            0,
        ).to(device)
        self.clip_end = self.clip_start + self.clip_len - 1
        self.num_frames = int(sum(clip_lengths))
        self.clip_id = torch.cat(
            [
                torch.full((num_frames,), clip_index, dtype=torch.long)
                for clip_index, num_frames in enumerate(clip_lengths)
            ]
        ).to(device)
        (
            base_pos_w_clips,
            base_rot_mat_w_clips,
            base_lin_vel_w_clips,
            base_ang_vel_w_clips,
            joint_position_clips,
            joint_velocity_clips,
        ) = ([], [], [], [], [], [])
        for base_pos_w, base_quat_w, joint_pos in zip(base_pos_clips, base_quat_clips, joint_pos_clips):
            base_rot_mat_w = quat_to_mat(base_quat_w)
            base_lin_vel_w = _gradient(base_pos_w, CONTROL_DT)
            joint_vel = _gradient(joint_pos, CONTROL_DT)
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
                joint_position_clips.append(joint_pos),
                joint_velocity_clips.append(joint_vel),
            )
        concatenate_clips = lambda clips: torch.cat(clips).to(device)
        self.base_pos_w, self.base_rot_mat_w, self.base_lin_vel_w, self.base_ang_vel_w = (
            concatenate_clips(base_pos_w_clips),
            concatenate_clips(base_rot_mat_w_clips),
            concatenate_clips(base_lin_vel_w_clips),
            concatenate_clips(base_ang_vel_w_clips),
        )
        self.joint_pos, self.joint_vel = (
            concatenate_clips(joint_position_clips),
            concatenate_clips(joint_velocity_clips),
        )
        with torch.no_grad():
            (
                body_position_clips,
                body_rotation_clips,
                body_linear_velocity_clips,
                body_angular_velocity_clips,
            ) = ([], [], [], [])
            for start_frame in range(0, self.num_frames, 65536):
                end_frame = min(start_frame + 65536, self.num_frames)
                body_states = kinematics.forward(
                    self.base_pos_w[start_frame:end_frame],
                    self.base_rot_mat_w[start_frame:end_frame],
                    self.base_lin_vel_w[start_frame:end_frame],
                    self.base_ang_vel_w[start_frame:end_frame],
                    self.joint_pos[start_frame:end_frame],
                    self.joint_vel[start_frame:end_frame],
                )
                (
                    body_position_clips.append(body_states[0]),
                    body_rotation_clips.append(body_states[1]),
                    body_linear_velocity_clips.append(body_states[2]),
                    body_angular_velocity_clips.append(body_states[3]),
                )
            self.body_pos_w, self.body_rot_mat_w, self.body_lin_vel_w, self.body_ang_vel_w = (
                torch.cat(body_position_clips),
                torch.cat(body_rotation_clips),
                torch.cat(body_linear_velocity_clips),
                torch.cat(body_angular_velocity_clips),
            )
        world_to_base = self.base_rot_mat_w.transpose(-1, -2)
        self.base_lin_vel = (world_to_base @ self.base_lin_vel_w.unsqueeze(-1)).squeeze(-1)
        self.base_ang_vel = (world_to_base @ self.base_ang_vel_w.unsqueeze(-1)).squeeze(-1)
        bin_clip, bin_start = ([], [])
        for clip_index in range(len(clip_lengths)):
            num_clip_bins = int(np.ceil(clip_lengths[clip_index] / BIN_FRAMES))
            for bin_index in range(num_clip_bins):
                bin_clip.append(clip_index)
                bin_start.append(int(self.clip_start[clip_index]) + bin_index * BIN_FRAMES)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=device)
        for bin_index in range(self.num_bins):
            clip_index = bin_clip[bin_index]
            start_frame = bin_start[bin_index]
            end_frame = min(start_frame + BIN_FRAMES, int(self.clip_end[clip_index]) + 1)
            self.frame_bin[start_frame:end_frame] = bin_index

    @staticmethod
    def from_dir(path: str, robot: RobotSpec, kinematics: Kinematics, device: str) -> ReferenceMotion:
        files = sorted(glob.glob(os.path.join(path, robot.motion_glob)))
        if not files:
            raise FileNotFoundError(f"No reference motion files found in {path}; run tools/download_data.py.")
        return ReferenceMotion(files, kinematics, device, robot.load_motion)

    def set_ref_foot_contact(
        self, feet: list[int], margin: float = 0.015, vmax: float = 0.5, floor_quantile: float = 0.05
    ) -> None:
        """Speed-branch reference foot contact (num_frames, len(feet)): height below the clip's own floor (the
        q-quantile of that foot's height in the clip) + margin, and horizontal speed below vmax. A fixed 4.5 cm
        threshold misses most stance frames of clips whose retargeted feet sit ~1 cm higher (LAFAN1 subject1:
        floor ~4.5 cm against ~3.5 cm for the others)."""
        foot_heights, foot_horizontal_speed = (
            self.body_pos_w[:, feet, 2],
            self.body_lin_vel_w[:, feet, :2].norm(dim=-1),
        )
        floor = torch.zeros_like(foot_heights)
        for clip_index in range(len(self.names)):
            start_frame = self.clip_id == clip_index
            floor[start_frame] = foot_heights[start_frame].quantile(floor_quantile, dim=0)
        self.ref_foot_contact = ((foot_heights < floor + margin) & (foot_horizontal_speed < vmax)).float()

    def ref_state(self, motion_frame_index: torch.Tensor):
        return (
            self.base_pos_w[motion_frame_index],
            self.base_rot_mat_w[motion_frame_index],
            self.base_lin_vel_w[motion_frame_index],
            self.base_ang_vel_w[motion_frame_index],
            self.joint_pos[motion_frame_index],
            self.joint_vel[motion_frame_index],
        )


def _gradient(values: torch.Tensor, time_step: float) -> torch.Tensor:
    gradient = torch.zeros_like(values)
    gradient[1:-1] = (values[2:] - values[:-2]) / (2 * time_step)
    gradient[0] = (values[1] - values[0]) / time_step
    gradient[-1] = (values[-1] - values[-2]) / time_step
    return gradient
