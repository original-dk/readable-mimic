"""Ground-supported reset starts on the training reference frame grid."""

import mujoco
import numpy as np
import torch

from .motion import BIN_FRAMES
from .retargeting.smoothing import smoothing_model


def foot_heights(model, qpos, foot_body_names):
    """Height of the lowest actual collision sole, not the ankle origin."""
    data = mujoco.MjData(model)
    foot_ids = {model.body(name).id for name in foot_body_names}
    geom_indices = [
        index
        for index in range(model.ngeom)
        if model.geom_bodyid[index] in foot_ids
        and (model.geom_contype[index] or model.geom_conaffinity[index])
    ]
    if not geom_indices:
        raise ValueError("No collision geoms on the G1 feet")
    geom_indices = np.asarray(geom_indices)
    geom_types = model.geom_type[geom_indices]
    sphere_mask = geom_types == mujoco.mjtGeom.mjGEOM_SPHERE
    if not np.all(sphere_mask | (geom_types == mujoco.mjtGeom.mjGEOM_BOX)):
        raise ValueError("Expected sphere or box collision geoms on the G1 feet")
    sizes = model.geom_size[geom_indices]
    heights = []
    for frame in qpos:
        data.qpos[:] = frame
        mujoco.mj_kinematics(model, data)
        rotations = data.geom_xmat[geom_indices].reshape(-1, 3, 3)
        box_radius = np.einsum("ij,ij->i", np.abs(rotations[:, 2]), sizes)
        vertical_radius = np.where(sphere_mask, sizes[:, 0], box_radius)
        heights.append(np.min(data.geom_xpos[geom_indices, 2] - vertical_radius))
    return np.asarray(heights)


def check_start_coverage(heights, clearance, gap_limit):
    # The clip endpoint cannot be a start because the next step needs a reference frame.
    eligible = heights[:-1] <= clearance
    legal = np.flatnonzero(eligible)
    edges = np.flatnonzero(np.diff(np.r_[False, ~eligible, False]))
    excluded = []
    for start, stop in zip(edges[::2], edges[1::2]):
        before = int(start - 1) if start else None
        after = int(stop) if stop < len(eligible) else None
        # Between two legal starts, count their frame-index distance.
        gap = int(after - before if before is not None and after is not None else stop - start)
        excluded.append(dict(
            start_frame=int(start), end_frame=int(stop - 1),
            previous_legal_start=before, next_legal_start=after,
            gap_frames=gap, exceeds_limit=gap > gap_limit,
        ))
    return dict(
        frame_count=len(heights),
        legal_start_count=len(legal),
        excluded_start_count=int((~eligible).sum()),
        excluded_frame_fraction=float((~eligible).sum() / len(heights)),
        last_frame_excluded=len(heights) - 1,
        status="passed" if len(legal) and not any(r["exceeds_limit"] for r in excluded) else "rejected",
        reason="no_legal_start" if not len(legal) else (
            "start_gap_too_long" if any(r["exceeds_limit"] for r in excluded) else None
        ),
        excluded_ranges=excluded,
    )


def clip_foot_heights(motion, start, stop, model, joint_indices, foot_bodies):
    qpos = np.zeros((stop - start, model.nq))
    qpos[:, :3] = motion.base_pos_w[start:stop].cpu().numpy()
    state = motion.compact_state[start:stop].cpu().float().numpy()
    qpos[:, 3:7] = state[:, :4]
    qpos[:, joint_indices] = state[:, 10 : 10 + motion.num_joints]
    return foot_heights(model, qpos, foot_bodies)


def build_reset_starts(motion, robot, config):
    clearance = config.reset_foot_clearance_cm / 100
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError("reset_foot_clearance_cm must be finite and nonnegative")
    gap_limit = config.max_episode_steps * 0.8
    model, joint_indices, _ = smoothing_model(robot.mjcf_path, robot.name)
    tables, rejected = [], []
    leading_bins = 0
    for name, (start, end) in zip(motion.names, motion._clip_bounds):
        heights = clip_foot_heights(motion, start, end + 1, model, joint_indices, config.reset_foot_bodies)
        coverage = check_start_coverage(heights, clearance, gap_limit)
        if coverage["status"] == "rejected":
            gaps = [(gap["start_frame"], gap["end_frame"]) for gap in coverage["excluded_ranges"]
                    if gap["exceeds_limit"]]
            rejected.append(f"{name}: {coverage['reason']} {gaps}")
            continue
        legal = np.flatnonzero(heights[:-1] <= clearance) + start
        previous = legal[:1]
        seen_legal = False
        for bin_start in range(start, end + 1, BIN_FRAMES):
            first, stop = np.searchsorted(legal, [bin_start, bin_start + BIN_FRAMES])
            if stop > first:
                previous = legal[first:stop]
                seen_legal = True
            leading_bins += not seen_legal
            tables.append(previous)
    if rejected:
        raise ValueError("Reset start coverage failed (local 50 Hz frame ranges): " + "; ".join(rejected))
    # Pad with a valid index. Sampling uses the real count, never the padding.
    counts = np.array([len(frames) for frames in tables])
    indices = np.stack([np.pad(frames, (0, BIN_FRAMES - len(frames)), mode="edge") for frames in tables])
    print(f"Reset starts: {len(tables)} bins, {leading_bins} leading bins use the first legal frame", flush=True)
    return (
        torch.as_tensor(indices, device=motion.device),
        torch.as_tensor(counts, device=motion.device),
    )
