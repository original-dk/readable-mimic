"""Convert trimmed 100STYLE BVH clips into the project's 30 Hz G1 training CSV."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np
from .bvh import load_bvh
from .ik import MotionRetargeter, ground_motion, detect_support_frames
from ..robot import get_robot


def convert_clip(source, output_dir, settings, cuts):
    begin = time.perf_counter()
    output = Path(output_dir) / (source.stem + ".csv")
    fingerprint = hashlib.sha256(
        source.read_bytes()
        + json.dumps(
            {
                key: value
                for key, value in settings.items()
                if key
                not in (
                    "workers",
                    "styles",
                    "movements",
                    "input_dir",
                    "output_dir",
                    "frame_cuts",
                    "mjcf_path",
                )
            },
            sort_keys=True,
        ).encode()
        + Path(settings["mjcf_path"]).read_bytes()
        + json.dumps(cuts).encode()
    ).hexdigest()
    metadata_path = output.with_suffix(".json")
    if output.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("fingerprint") == fingerprint
            and metadata.get("csv_sha256") == hashlib.sha256(output.read_bytes()).hexdigest()
        ):
            return metadata | {"skipped": True}
        raise ValueError(f"Existing output has different source/config: {output}; use a new output_dir")
    human = load_bvh(
        source,
        cuts[0],
        cuts[1],
        settings["output_frame_rate"],
        settings["bvh_position_scale"],
        settings["bvh_axis_order"],
    )
    robot = get_robot(settings["robot"])
    model = mujoco.MjModel.from_xml_path(settings["mjcf_path"])
    retargeter = MotionRetargeter(model, robot, settings["ik"], human.body_names)
    qpos = np.array(
        [
            retargeter.retarget(pos, rotation, initialize=(index == 0))
            for index, (pos, rotation) in enumerate(zip(human.body_pos_w, human.body_rot_mat_w))
        ]
    )
    ground_offset = ground_motion(
        model,
        qpos,
        settings["foot_bodies"],
        settings["ground_clearance"],
        detect_support_frames(human, settings["grounding"]),
    )
    base_start = retargeter.base_qpos_start
    base_pos_w = qpos[:, base_start : base_start + 3]
    base_pos_w[:, :2] -= base_pos_w[0, :2].copy()
    base_quat_wxyz = qpos[:, base_start + 3 : base_start + 7]
    joint_pos = qpos[:, retargeter.joint_qpos_indices]
    joint_ids = np.array([model.joint(name).id for name in robot.joint_names])
    limits = model.jnt_range[joint_ids]
    violation = np.maximum(limits[:, 0] - joint_pos, joint_pos - limits[:, 1]).max()
    if violation > settings["joint_limit_tolerance"]:
        raise ValueError(f"Joint limit violation {violation:g}: {source}")
    raw = np.concatenate((base_pos_w, base_quat_wxyz[:, [1, 2, 3, 0]], joint_pos), axis=1)
    if raw.shape[1] != 36 or not np.isfinite(raw).all():
        raise ValueError(f"Invalid CSV data: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.part")
    np.savetxt(temporary, raw, delimiter=",", fmt="%.8f")
    temporary.replace(output)
    metadata = {
        "name": source.stem,
        "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "fingerprint": fingerprint,
        "source": str(source),
        "source_start_frame": cuts[0],
        "source_stop_frame": cuts[1],
        "frame_rate": human.frame_rate,
        "num_frames": len(raw),
        "duration": (len(raw) - 1) / human.frame_rate,
        "joint_names": list(robot.joint_names),
        "ground_offset": ground_offset,
        "max_joint_speed": float(np.abs(np.diff(joint_pos, axis=0)).max() * human.frame_rate),
        "wall_seconds": round(time.perf_counter() - begin, 3),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def convert_dataset(config_path):
    config_path = Path(config_path).resolve()
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    settings["converter_sha256"] = hashlib.sha256(
        b"".join(
            Path(__file__).with_name(name).read_bytes() for name in ("bvh.py", "ik.py", "convert_100style.py")
        )
    ).hexdigest()
    for key in ("input_dir", "output_dir", "frame_cuts", "mjcf_path"):
        settings[key] = str((config_path.parent / settings[key]).resolve())
    if settings["output_frame_rate"] != 30:
        raise ValueError("The project's CSV loader requires 30 Hz output")
    if not isinstance(settings["workers"], int) or settings["workers"] < 1:
        raise ValueError("workers must be a positive integer")
    with Path(settings["frame_cuts"]).open(encoding="utf-8-sig", newline="") as file:
        rows = {row["STYLE_NAME"]: row for row in csv.DictReader(file)}
    files = sorted(Path(settings["input_dir"]).rglob("*.bvh"))
    if not files:
        raise ValueError(f"No BVH files in {settings['input_dir']}")
    tasks = []
    for source in files:
        style, movement = source.stem.rsplit("_", 1)
        if settings["styles"] and style not in settings["styles"]:
            continue
        if settings["movements"] and movement not in settings["movements"]:
            continue
        row = rows[style]
        cuts = [int(row[f"{movement}_START"]), int(row[f"{movement}_STOP"])]
        tasks.append((source, settings["output_dir"], settings, cuts))
    if not tasks or len({task[0].stem for task in tasks}) != len(tasks):
        raise ValueError("Empty selection or duplicate motion names")
    unknown_styles = set(settings["styles"]) - {task[0].stem.rsplit("_", 1)[0] for task in tasks}
    if unknown_styles:
        raise ValueError(f"Selected styles missing from input: {sorted(unknown_styles)}")
    workers = min(settings["workers"], len(tasks), os.cpu_count() or 1)
    print(f"Retargeting {len(tasks)} clips with {workers} CPU workers → {settings['output_dir']}", flush=True)
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(convert_clip, *task): task[0] for task in tasks}
        for future in as_completed(futures):
            source = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{len(results) + len(failures)}/{len(tasks)}] {source.stem}: {result['num_frames']} frames"
                    f" ({'cached' if result.get('skipped') else str(result['wall_seconds']) + 's'})",
                    flush=True,
                )
            except Exception as error:
                failures.append({"source": str(source), "error": str(error)})
                print(f"FAILED {source.stem}: {error}", flush=True)
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": settings,
        "clips": sorted(results, key=lambda item: item["name"]),
        "failures": failures,
    }
    (output_dir / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Finished: {len(results)} clips, {len(failures)} failures", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} clips failed; see conversion_report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="retarget_100style.json")
    args = parser.parse_args()
    convert_dataset(args.config)


if __name__ == "__main__":
    main()
