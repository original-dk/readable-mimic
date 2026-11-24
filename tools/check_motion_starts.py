"""Check ground-supported random reset coverage on the training frame grid."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from readable_mimic.motion_starts import check_start_coverage, clip_foot_heights
from readable_mimic.motion import ReferenceMotion
from readable_mimic.retargeting.smoothing import smoothing_model
from readable_mimic.robot import get_robot
from readable_mimic.sim import CONTROL_DT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path, help="Override motion_dir from the training config")
    parser.add_argument("--foot-clearance-cm", type=float, help="Override reset_foot_clearance_cm from config")
    parser.add_argument("--episode-fraction", type=float, default=0.8)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--move-rejected", action="store_true",
                        help="Move rejected CSVs to a unique folder under the project backup directory")
    args = parser.parse_args(argv)
    if not 0 < args.episode_fraction <= 1:
        parser.error("--episode-fraction must be in (0, 1]")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    clearance_cm = args.foot_clearance_cm
    if clearance_cm is None:
        clearance_cm = config.get("reset_foot_clearance_cm", 3.0)
    if not np.isfinite(clearance_cm) or clearance_cm < 0:
        parser.error("Foot clearance must be finite and nonnegative")
    configured = config["motion_dir"]
    if args.motion_dir is not None:
        directories = [args.motion_dir.resolve()]
    else:
        configured = configured if isinstance(configured, list) else [configured]
        if not configured or any(not isinstance(path, str) or not path.strip() for path in configured):
            parser.error("motion_dir must be a directory or a nonempty list of directories")
        directories = [(args.config.resolve().parent / path).resolve() for path in configured]
    robot = get_robot(config["robot"])
    files = ReferenceMotion.find_files([str(path) for path in directories], robot)
    model, joint_indices, _ = smoothing_model(robot.mjcf_path, robot.name)
    foot_bodies = config.get("reset_foot_bodies", ["left_ankle_roll_link", "right_ankle_roll_link"])
    max_episode_steps = config["max_episode_steps"]
    gap_limit = max_episode_steps * args.episode_fraction
    torch.set_num_threads(1)
    motions = []
    for path in files:
        motion = ReferenceMotion([path], None, "cpu", robot.load_motion)
        heights = clip_foot_heights(motion, 0, motion.num_frames, model, joint_indices, foot_bodies)
        result = check_start_coverage(heights, clearance_cm / 100, gap_limit)
        motions.append(dict(motion=Path(path).name, source_path=str(Path(path).resolve()), **result))
    rejected = [motion["motion"] for motion in motions if motion["status"] == "rejected"]
    total_frames = sum(motion["frame_count"] for motion in motions)
    excluded_starts = sum(motion["excluded_start_count"] for motion in motions)
    report = dict(
        motion_dir=str(directories[0]) if len(directories) == 1 else [str(path) for path in directories],
        frame_rate=1 / CONTROL_DT,
        max_episode_steps=max_episode_steps, episode_fraction=args.episode_fraction,
        gap_limit_frames=gap_limit, foot_clearance_cm=clearance_cm,
        total_frames=total_frames, excluded_start_count=excluded_starts,
        excluded_frame_fraction=excluded_starts / total_frames,
        checked_count=len(motions), rejected_count=len(rejected),
        rejected_motions=rejected, motions=motions,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def write_report():
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_report()
    if args.move_rejected and rejected:
        root = Path(__file__).resolve().parents[1]
        backup_root = root / "backup"
        backup_root.mkdir(exist_ok=True)
        backup = Path(tempfile.mkdtemp(prefix="rejected_motion_starts_", dir=backup_root))
        report["backup_dir"] = str(backup)
        report["moved_count"] = 0
        report["moved_motions"] = []
        write_report()
        try:
            for motion in motions:
                if motion["status"] != "rejected":
                    continue
                source = Path(motion["source_path"])
                destination = backup / source.name
                if destination.exists():
                    raise FileExistsError(f"Backup destination already exists: {destination}")
                shutil.move(str(source), str(destination))
                report["moved_motions"].append(dict(source=str(source), destination=str(destination)))
                report["moved_count"] += 1
        finally:
            write_report()
        print(f"Moved {len(rejected)} rejected motions to {backup}")
    print(f"Checked {len(motions)} motions, rejected {len(rejected)}. Report: {args.report}")


if __name__ == "__main__":
    main()
