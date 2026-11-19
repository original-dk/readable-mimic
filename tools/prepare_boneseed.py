"""Convert BONES-SEED G1 CSVs, then run the shared flat-floor training filter."""

import argparse
import csv
import json
import re
from itertools import islice
from pathlib import Path

import numpy as np
from process_motion_csv import process_directory
from scipy.spatial.transform import Rotation, Slerp

from readable_mimic.robot import get_robot

ROOT = Path(__file__).resolve().parents[1]


def convert_motion(source, joint_names, source_fps):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
        columns = [
            *(f"root_translate{axis}" for axis in "XYZ"),
            *(f"root_rotate{axis}" for axis in "XYZ"),
            *(f"{name}_dof" for name in joint_names),
        ]
        if not header or header[0] != "Frame" or any(header.count(name) != 1 for name in columns):
            raise ValueError(f"Unexpected BONES-SEED G1 header: {source}")
        values = np.loadtxt(stream, delimiter=",", usecols=[header.index(name) for name in columns], ndmin=2)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid motion rows: {source}")
    source_time = np.arange(len(values)) / source_fps
    target_time = np.arange(int(np.floor(source_time[-1] * 30)) + 1) / 30
    if len(target_time) < 2:
        raise ValueError(f"Motion shorter than two frames at 30 Hz: {source}")
    position = np.column_stack([np.interp(target_time, source_time, x) for x in values[:, :3].T]) / 100
    rotation = Rotation.from_euler("xyz", values[:, 3:6], degrees=True)
    quaternion = Slerp(source_time, rotation)(target_time).as_quat()
    joint_pos = np.column_stack([np.interp(target_time, source_time, x) for x in values[:, 6:].T])
    return np.column_stack((position, quaternion, np.deg2rad(joint_pos)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--rules", type=Path, default=ROOT / "motion_filter.json")
    args = parser.parse_args()
    source_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if (
        not source_dir.is_dir()
        or source_dir == output_dir
        or output_dir.is_relative_to(source_dir)
        or source_dir.is_relative_to(output_dir)
    ):
        parser.error("Use an existing input directory and a separate output directory")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    settings = json.loads(args.rules.read_text(encoding="utf-8"))["boneseed"]
    files = list(islice(sorted(source_dir.rglob(args.pattern)), args.limit))
    if not files:
        parser.error("No matching CSV files")
    converted = output_dir / "converted"
    # Use a separate output root for each selection, avoiding stale intermediates.
    manifest = output_dir / "boneseed_selection.json"
    selection = {"input_dir": str(source_dir), "pattern": args.pattern, "limit": args.limit, "settings": settings}
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != selection:
        parser.error("Selection changed. Use a new output directory.")
    converted.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    joint_names = get_robot("g1").joint_names
    names = ["__".join(path.relative_to(source_dir).parts) for path in files]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate flattened BONES-SEED names")
    report_path = output_dir / "boneseed_conversion.csv"
    accepted = set()
    with report_path.open("w", newline="", encoding="utf-8") as stream:
        report = csv.writer(stream)
        report.writerow(["source", "converted", "reason"])
        for index, (source, name) in enumerate(zip(files, names), 1):
            normalized = re.sub(r"[^a-z0-9]+", "_", source.stem.lower())
            reason = next((word for word in settings["terrain_keywords"] if word in normalized), "")
            if reason:
                report.writerow([source.relative_to(source_dir), "", f"terrain_name:{reason}"])
            else:
                values = convert_motion(source, joint_names, settings["source_fps"])
                destination = converted / name
                temporary = destination.with_suffix(".tmp")
                np.savetxt(temporary, values, delimiter=",", fmt="%.8f")
                temporary.replace(destination)
                accepted.add(name)
                report.writerow([source.relative_to(source_dir), name, ""])
            if index == len(files) or index % 500 == 0:
                print(f"[BONES-SEED {index}/{len(files)}] converted {len(accepted)}", flush=True)
    if {path.name for path in converted.glob("*.csv")} != accepted:
        raise ValueError("Converted folder contains stale CSVs. Use a new output directory.")
    if accepted and not args.convert_only:
        process_directory(converted, output_dir / "processed", ROOT / "retarget_amass.json", args.rules)


if __name__ == "__main__":
    main()
