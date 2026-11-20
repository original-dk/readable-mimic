"""Read BVH channels in their declared order and compute human body poses."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class HumanMotion:
    body_names: list[str]
    body_pos_w: np.ndarray
    body_rot_mat_w: np.ndarray
    frame_rate: float


def load_bvh(path, start_frame, stop_frame, output_frame_rate, position_scale, axis_order, *, position_channels_replace_offsets=False):
    """Frame cuts are zero-based [start, stop); output is in metres, Z up.

    Both world vectors and local bone axes are changed to the robot convention.
    BVH Euler rotations are intrinsic: Yrotation Xrotation Zrotation → YXZ.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    hierarchy, motion = text.split("MOTION", 1)
    names, parents, offsets, channels = [], [], [], []
    stack = []
    active = -1
    for line in hierarchy.splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] in ("ROOT", "JOINT"):
            parents.append(active)
            names.append(words[1])
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            active = len(names) - 1
        elif words[:2] == ["End", "Site"]:
            active = -2  # End sites have no channels and are not IK targets.
        elif words[0] == "{":
            stack.append(active)
        elif words[0] == "}":
            stack.pop()
            active = stack[-1] if stack else -1
        elif words[0] == "OFFSET" and active >= 0:
            offsets[active] = list(map(float, words[1:4]))
        elif words[0] == "CHANNELS":
            channels[active] = words[2:]
            if int(words[1]) != len(channels[active]):
                raise ValueError(f"Invalid CHANNELS in {path}")
    header = re.match(r"\s*Frames:\s*(\d+)\s*Frame Time:\s*([\d.eE+-]+)\s*", motion)
    if header is None:
        raise ValueError(f"Missing BVH motion header: {path}")
    num_frames, frame_dt = int(header[1]), float(header[2])
    stop_frame = num_frames if stop_frame is None else stop_frame
    if not 0 <= start_frame < stop_frame <= num_frames or stop_frame - start_frame < 2:
        raise ValueError(f"Invalid frame cut [{start_frame}, {stop_frame}) for {num_frames} frames: {path}")
    source_frame_rate = 1.0 / frame_dt
    stride = round(source_frame_rate / output_frame_rate)
    if stride < 1 or not np.isclose(source_frame_rate / stride, output_frame_rate, rtol=1e-4):
        raise ValueError(f"BVH fps {source_frame_rate:g} must be an integer multiple of {output_frame_rate}")
    values = np.fromstring(motion[header.end() :], sep=" ")
    num_channels = sum(map(len, channels))
    if values.size != num_frames * num_channels or not np.isfinite(values).all():
        raise ValueError(f"Invalid BVH frame data: {path}")
    values = values.reshape(num_frames, num_channels)[start_frame:stop_frame:stride]
    if len(values) < 2:
        raise ValueError(f"Too few output frames: {path}")
    local_pos = np.broadcast_to(np.array(offsets), (len(values), len(names), 3)).copy()
    local_rotation = np.tile(np.eye(3), (len(values), len(names), 1, 1))
    cursor = 0
    for body_index, body_channels in enumerate(channels):
        rotation_columns = []
        rotation_axes = ""
        for channel in body_channels:
            if channel.endswith("position"):
                axis = "XYZ".index(channel[0])
                if position_channels_replace_offsets:
                    local_pos[:, body_index, axis] = values[:, cursor]
                else:
                    local_pos[:, body_index, axis] += values[:, cursor]
            elif channel.endswith("rotation"):
                rotation_axes += channel[0]
                rotation_columns.append(cursor)
            else:
                raise ValueError(f"Unsupported BVH channel: {channel}")
            cursor += 1
        local_rotation[:, body_index] = Rotation.from_euler(
            rotation_axes, values[:, rotation_columns], degrees=True
        ).as_matrix()
    world_pos = np.empty_like(local_pos)
    world_rotation = np.empty_like(local_rotation)
    for body_index, parent_index in enumerate(parents):
        if parent_index < 0:
            world_pos[:, body_index] = local_pos[:, body_index]
            world_rotation[:, body_index] = local_rotation[:, body_index]
        else:
            parent_rotation = world_rotation[:, parent_index]
            world_pos[:, body_index] = world_pos[:, parent_index] + np.einsum(
                "fij,fj->fi", parent_rotation, local_pos[:, body_index]
            )
            world_rotation[:, body_index] = parent_rotation @ local_rotation[:, body_index]
    axes = np.eye(3)[["xyz".index(axis) for axis in axis_order]]
    return HumanMotion(
        names,
        world_pos @ axes.T * position_scale,
        axes @ world_rotation @ axes.T,
        output_frame_rate,
    )
