"""CSV -> smooth/contact repair -> filter -> CSV-only train. No NPZ or body model needed."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from readable_mimic.motion_starts import foot_heights
from readable_mimic.robot import get_robot
from readable_mimic.retargeting.smoothing import smooth_values, smoothing_model


def ground_motion(model, qpos, foot_body_names, clearance, support_mask=None):
    """Align support poses and interpolate correction through flight."""
    heights = foot_heights(model, qpos, foot_body_names)
    if support_mask is None:
        offset = clearance - heights.min()
        qpos[:, 2] += offset
        return float(offset)
    support_frames = np.flatnonzero(support_mask)
    if len(support_frames) < 2:
        raise ValueError("Fewer than two support frames detected; check source grounding thresholds")
    from scipy.interpolate import PchipInterpolator

    offsets = clearance - heights
    frame_indices = np.arange(len(qpos))
    # Interpolate the correction through flight; do not flatten the jump trajectory.
    correction = PchipInterpolator(support_frames, offsets[support_frames], extrapolate=False)(
        np.clip(frame_indices, support_frames[0], support_frames[-1])
    )
    correction = np.maximum(correction, offsets)  # prevent any foot penetrating below clearance
    qpos[:, 2] += correction
    return {
        "support_frames": int(support_mask.sum()),
        "flight_frames": int((~support_mask).sum()),
        "min_offset": float(correction.min()),
        "max_offset": float(correction.max()),
    }


def csv_support_mask(values, model, joint_indices, foot_bodies, rules, dataset=None):
    """Recover flat-floor support from collision soles in the original CSV."""
    qpos = np.zeros((len(values), model.nq))
    qpos[:, :3] = values[:, :3]
    qpos[:, 3:7] = values[:, [6, 3, 4, 5]]
    qpos[:, joint_indices] = values[:, 7:]
    heights = foot_heights(model, qpos, foot_bodies)
    grounding = rules["grounding"]
    floor_offset = grounding["clearance"] - float(np.quantile(heights, grounding["floor_quantile"]))
    limit = grounding["max_floor_alignment"]
    upward_limit = grounding.get("max_upward_alignment_by_dataset", {}).get(dataset, limit)
    if floor_offset > upward_limit or floor_offset < -limit:
        floor_offset = 0.0
    heights = heights + floor_offset
    support = (heights >= -rules["max_foot_penetration"]) & (heights <= rules["foot_contact_height"])
    return support, floor_offset


def shoulder_limit_frames(values, joint_names, ik_limits, tolerance_degrees):
    longest = 0
    for side in ("left", "right"):
        names = [f"{side}_shoulder_{axis}_joint" for axis in ("pitch", "roll", "yaw")]
        if not all(name in ik_limits for name in names):
            continue
        positions = values[:, [7 + joint_names.index(name) for name in names]]
        distances = np.abs(positions[:, :, None] - np.asarray([ik_limits[name] for name in names]))
        near_limit = distances.min(axis=2) <= np.deg2rad(tolerance_degrees)
        saturated = near_limit.sum(axis=1) >= 2
        edges = np.flatnonzero(np.diff(np.r_[False, saturated, False]))
        longest = max(longest, max(edges[1::2] - edges[::2], default=0))
    return int(longest)


def process_csv(source, output, config, rules, pipeline_sha256):
    started = time.perf_counter()
    source = Path(source)
    output = Path(output)
    content = source.read_bytes()
    source_sha256 = hashlib.sha256(content).hexdigest()
    settings = rules["smoothing"]
    shoulder_check = rules.get("shoulder_limit_check")
    ik_limits = config.get("ik", {}).get("joint_pos_limits", {})
    parameters = dict(
        cutoff_hz=settings["cutoff_hz"],
        filter_order=settings["filter_order"],
        foot_contact_height=rules["foot_contact_height"],
        max_foot_penetration=rules["max_foot_penetration"],
        grounding=rules["grounding"],
        grounding_dataset=output.stem.split("__", 1)[0],
        frame_rate=30,
    )
    if shoulder_check and ik_limits:
        parameters.update(shoulder_limit_check=shoulder_check, ik_limits=ik_limits)
    fingerprint = hashlib.sha256(
        json.dumps(
            dict(
                source=source_sha256,
                pipeline=pipeline_sha256,
                model=config["model_sha256"],
                parameters=parameters,
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    metadata_path = output.with_suffix(".json")
    if output.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata["fingerprint"] == fingerprint
            and metadata["csv_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
        ):
            return metadata | dict(skipped=True)
        raise ValueError(f"Existing output differs; use a new output directory: {output}")
    values = np.loadtxt(io.BytesIO(content), delimiter=",", ndmin=2)
    model, joint_indices, limits = smoothing_model(config["mjcf_path"], config["robot"])
    result = smooth_values(values, settings, limits)
    if np.max(np.abs(np.linalg.norm(values[:, 3:7], axis=1) - 1)) > rules["quaternion_tolerance"]:
        raise ValueError("Input quaternion is not unit length")
    support, floor_offset = csv_support_mask(
        values, model, joint_indices, config["foot_bodies"], rules, parameters["grounding_dataset"]
    )
    correction = None
    if support.sum() >= 2:
        result[:, 2] += floor_offset
        qpos = np.zeros((len(result), model.nq))
        qpos[:, :3] = result[:, :3]
        qpos[:, 3:7] = result[:, [6, 3, 4, 5]]
        qpos[:, joint_indices] = result[:, 7:]
        correction = ground_motion(
            model, qpos, config["foot_bodies"], rules["grounding"]["clearance"], support
        )
        correction["floor_offset"] = floor_offset
        result[:, 2] = qpos[:, 2]
    # No support anchors: retain the trajectory and let the quality gate reject hovering.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.part")
    np.savetxt(temporary, result, delimiter=",", fmt="%.8f")
    temporary.replace(output)
    joint_delta = np.abs(result[:, 7:] - values[:, 7:])
    frame, joint = np.unravel_index(joint_delta.argmax(), joint_delta.shape)
    rotation_delta = (
        Rotation.from_quat(values[:, 3:7]).inv() * Rotation.from_quat(result[:, 3:7])
    ).magnitude()
    robot = get_robot(config["robot"])
    metadata = dict(
        name=output.stem,
        source=str(source),
        source_csv_sha256=source_sha256,
        frame_rate=30,
        num_frames=len(result),
        joint_names=list(robot.joint_names),
        fingerprint=fingerprint,
        csv_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        ground_correction=correction,
        grounding_applied=correction is not None,
        processing_stage="processed",
        postprocessing=dict(
            **{key: value for key, value in parameters.items() if key != "ik_limits"}, support_source="original_csv_collision_soles", support_frames=int(support.sum())
        ),
        changes=dict(
            max_joint_change_degrees=float(np.degrees(joint_delta[frame, joint])),
            max_joint_change_name=robot.joint_names[joint],
            max_joint_change_time=float(frame / 30),
            max_base_position_change_m=np.max(np.abs(result[:, :3] - values[:, :3]), axis=0).tolist(),
            max_base_orientation_change_degrees=float(np.degrees(rotation_delta.max())),
        ),
        processing_seconds=round(time.perf_counter() - started, 3),
    )
    if shoulder_check and ik_limits:
        longest = max(
            shoulder_limit_frames(sequence, robot.joint_names, ik_limits, shoulder_check["tolerance_degrees"])
            for sequence in (values, result)
        )
        metadata["shoulder_limit_seconds"] = longest / parameters["frame_rate"]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def process_directory(input_dir, output_dir, config_path, rules_path, pattern="*.csv"):
    # Same filtering implementation and rules as the existing data preparation command.
    from prepare_training_data import screen
    import readable_mimic.retargeting.smoothing as smoothing_module

    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    in_place_dataset = input_dir == output_dir / "candidates"
    candidates = output_dir / ("processed_candidates" if in_place_dataset else "candidates")
    reports = output_dir / "cache/reports"
    if (
        output_dir.is_relative_to(input_dir)
        or (input_dir.is_relative_to(output_dir) and not in_place_dataset)
        or any(
            target.resolve().is_relative_to(input_dir) or input_dir.is_relative_to(target.resolve())
            for target in (candidates, reports, output_dir / "train")
        )
    ):
        raise ValueError("Input and output directories must be separate, not nested")
    config_path, rules_path = Path(config_path).resolve(), Path(rules_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["mjcf_path"] = str((config_path.parent / config["mjcf_path"]).resolve())
    config["model_sha256"] = hashlib.sha256(Path(config["mjcf_path"]).read_bytes()).hexdigest()
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    settings = rules["smoothing"]
    if not 0 < settings["cutoff_hz"] < 15 or settings["filter_order"] < 1 or settings["workers"] < 1:
        raise ValueError("Invalid smoothing cutoff, order or workers")
    files = sorted(input_dir.rglob(pattern))
    if not files:
        raise ValueError(f"No input CSV matched {pattern}: {input_dir}")
    names = ["__".join(p.relative_to(input_dir).with_suffix("").parts) + ".csv" for p in files]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate flattened filenames")
    candidates.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    if set(p.name for p in candidates.glob("*.csv")) - set(names):
        raise ValueError("Output contains other selections; use a new output directory")
    pipeline_sha256 = hashlib.sha256(
        b"".join(Path(p).read_bytes() for p in [__file__, smoothing_module.__file__])
    ).hexdigest()
    results, failures = [], []
    print(
        f"CSV -> CSV: {len(files)} inputs, {settings['cutoff_hz']} Hz, {settings['workers']} workers",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=settings["workers"]) as executor:
        futures = {
            executor.submit(
                process_csv, str(source), str(candidates / name), config, rules, pipeline_sha256
            ): source
            for source, name in zip(files, names)
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(dict(source=str(futures[future]), error=str(error)))
                print(f"FAILED {futures[future].name}: {error}", flush=True)
            count = len(results) + len(failures)
            if count % 50 == 0 or count == len(files):
                print(f"[{count}/{len(files)}]", flush=True)
    (reports / "processing_report.json").write_text(
        json.dumps(
            dict(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                parameters=rules,
                clips=results,
                failures=failures,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} processing failures; train was not updated. See processing_report.json"
        )
    config["output_dir"] = str(candidates)
    generated_config = reports / "filter_config.json"
    generated_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = screen(generated_config, rules_path, export=True)
    print(
        f"Done: {len(results)} processed; {report['exported_count']} CSVs in {output_dir / 'train'}",
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="Dataset output root containing the train directory")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--config", default="retarget_amass.json")
    parser.add_argument("--rules", default=Path(__file__).resolve().parents[1] / "motion_filter_level_1.json")
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, args.config, args.rules, args.pattern)


if __name__ == "__main__":
    main()
