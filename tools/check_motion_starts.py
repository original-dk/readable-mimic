"""Check ground-supported random reset coverage on the training frame grid."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from process_motion_csv import foot_heights
from readable_mimic.motion import ReferenceMotion
from readable_mimic.retargeting.smoothing import smoothing_model
from readable_mimic.robot import get_robot
from readable_mimic.sim import CONTROL_DT


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path, help="Override motion_dir from the training config")
    parser.add_argument("--foot-clearance-cm", type=float, help="Override reset_foot_clearance_cm from config")
    parser.add_argument("--episode-fraction", type=float, default=0.8)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.episode_fraction <= 1:
        parser.error("--episode-fraction must be in (0, 1]")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    clearance_cm = args.foot_clearance_cm
    if clearance_cm is None:
        clearance_cm = config.get("reset_foot_clearance_cm", 3.0)
    if not np.isfinite(clearance_cm) or clearance_cm < 0:
        parser.error("Foot clearance must be finite and nonnegative")
    directory = args.motion_dir or args.config.resolve().parent / config["motion_dir"]
    robot = get_robot(config["robot"])
    model, joint_indices, _ = smoothing_model(robot.mjcf_path, robot.name)
    root = Path(__file__).resolve().parents[1]
    foot_bodies = json.loads((root / "retarget_amass.json").read_text(encoding="utf-8"))["foot_bodies"]
    max_episode_steps = config["max_episode_steps"]
    gap_limit = max_episode_steps * args.episode_fraction
    torch.set_num_threads(1)
    motions = []
    for path in ReferenceMotion.find_files(str(directory), robot):
        motion = ReferenceMotion([path], None, "cpu", robot.load_motion)
        qpos = np.zeros((motion.num_frames, model.nq))
        qpos[:, :3] = motion.base_pos_w.numpy()
        qpos[:, 3:7] = motion.compact_state[:, :4].float().numpy()
        qpos[:, joint_indices] = motion.compact_state[:, 10 : 10 + motion.num_joints].float().numpy()
        heights = foot_heights(model, qpos, foot_bodies)
        result = check_start_coverage(heights, clearance_cm / 100, gap_limit)
        motions.append(dict(motion=Path(path).name, **result))
    rejected = [motion["motion"] for motion in motions if motion["status"] == "rejected"]
    total_frames = sum(motion["frame_count"] for motion in motions)
    excluded_starts = sum(motion["excluded_start_count"] for motion in motions)
    report = dict(
        motion_dir=str(directory.resolve()), frame_rate=1 / CONTROL_DT,
        max_episode_steps=max_episode_steps, episode_fraction=args.episode_fraction,
        gap_limit_frames=gap_limit, foot_clearance_cm=clearance_cm,
        total_frames=total_frames, excluded_start_count=excluded_starts,
        excluded_frame_fraction=excluded_starts / total_frames,
        checked_count=len(motions), rejected_count=len(rejected),
        rejected_motions=rejected, motions=motions,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Checked {len(motions)} motions, rejected {len(rejected)}. Report: {args.report}")


if __name__ == "__main__":
    main()
