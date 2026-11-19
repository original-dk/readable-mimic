"""Smooth existing AMASS G1 candidates while preserving source support/flight labels."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

from .ik import detect_support_frames, ground_motion
from .smpl import load_amass
from ..robot import get_robot


def smooth_values(values, settings, joint_limits):
    if values.ndim != 2 or values.shape[1] != 36 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite 36-column G1 CSV with at least two frames")
    sos = butter(settings["filter_order"], settings["cutoff_hz"], fs=30, output="sos")
    padlen = min(3 * (2 * len(sos) + 1), len(values) - 1)
    result = values.copy()
    result[:, :3] = sosfiltfilt(sos, values[:, :3], axis=0, padlen=padlen)
    result[:, 7:] = sosfiltfilt(sos, values[:, 7:], axis=0, padlen=padlen)
    result[:, :2] -= result[0, :2].copy()
    quat = values[:, 3:7].copy()
    for index in range(1, len(quat)):
        if np.dot(quat[index], quat[index - 1]) < 0:
            quat[index] *= -1
    quat = sosfiltfilt(sos, quat, axis=0, padlen=padlen)
    norms = np.linalg.norm(quat, axis=1)
    if norms.min() < 0.5:
        raise ValueError("Rotation changes too fast for this quaternion filter")
    result[:, 3:7] = quat / norms[:, None]
    # A continuous amplitude contraction prevents filter overshoot without hard clipping corners.
    center = np.clip(result[:, 7:].mean(axis=0), joint_limits[:, 0], joint_limits[:, 1])
    delta = result[:, 7:] - center
    positive = np.maximum(delta.max(axis=0), 1e-12)
    negative = np.maximum(-delta.min(axis=0), 1e-12)
    scale = np.minimum(
        1, np.minimum((joint_limits[:, 1] - center) / positive, (center - joint_limits[:, 0]) / negative)
    )
    result[:, 7:] = center + delta * scale
    return result


@lru_cache(maxsize=1)
def smoothing_model(mjcf_path, robot_name):
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    robot = get_robot(robot_name)
    indices = np.array([int(model.joint(name).qposadr[0]) for name in robot.joint_names])
    limits = np.array([model.joint(name).range for name in robot.joint_names])
    return model, indices, limits


def smooth_clip(source, config, settings, code_sha256):
    started = time.perf_counter()
    metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if metadata["csv_sha256"] != source_sha256 or metadata["frame_rate"] != 30:
        raise ValueError(f"Source checksum or frame rate mismatch: {source}")
    source_npz = Path(config["input_dir"]).joinpath(*source.stem.split("__")).with_suffix(".npz")
    source_npz_sha256 = hashlib.sha256(source_npz.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(
        json.dumps(
            dict(
                source_csv=source_sha256,
                source_npz=source_npz_sha256,
                cutoff_hz=settings["cutoff_hz"],
                filter_order=settings["filter_order"],
                code=code_sha256,
                config=config,
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output = Path(settings["output_dir"]) / source.name
    metadata_path = output.with_suffix(".json")
    if output.exists() and metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            previous["fingerprint"] == fingerprint
            and previous["csv_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
        ):
            return previous | dict(skipped=True)
        raise ValueError(f"Different existing smoothed output; use a new output_dir: {output}")
    values = np.loadtxt(source, delimiter=",", ndmin=2)
    model, joint_indices, limits = smoothing_model(config["mjcf_path"], config["robot"])
    result = smooth_values(values, settings, limits)
    human = load_amass(source_npz, config)
    support = detect_support_frames(human, config["grounding"])
    if len(support) != len(values):
        raise ValueError(f"Original support labels and CSV frames differ: {source}")
    qpos = np.zeros((len(values), model.nq))
    qpos[:, :3] = result[:, :3]
    qpos[:, 3:7] = result[:, [6, 3, 4, 5]]
    qpos[:, joint_indices] = result[:, 7:]
    correction = ground_motion(model, qpos, config["foot_bodies"], config["ground_clearance"], support)
    result[:, 2] = qpos[:, 2]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.part")
    np.savetxt(temporary, result, delimiter=",", fmt="%.8f")
    temporary.replace(output)
    joint_delta = np.abs(result[:, 7:] - values[:, 7:])
    frame, joint = np.unravel_index(joint_delta.argmax(), joint_delta.shape)
    rotation_delta = (
        Rotation.from_quat(values[:, 3:7]).inv() * Rotation.from_quat(result[:, 3:7])
    ).magnitude()
    metadata.update(
        source=str(source_npz),
        source_csv=str(source),
        source_csv_sha256=source_sha256,
        source_npz_sha256=source_npz_sha256,
        original_fingerprint=metadata["fingerprint"],
        fingerprint=fingerprint,
        csv_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        ground_correction=correction,
        max_joint_vel=float(np.abs(np.diff(result[:, 7:], axis=0)).max() * 30),
        postprocessing=dict(
            cutoff_hz=settings["cutoff_hz"],
            filter_order=settings["filter_order"],
            orientation="sign_continuous_normalized_quaternion",
            source_support_preserved=True,
        ),
        smoothing_seconds=round(time.perf_counter() - started, 3),
        changes=dict(
            max_joint_change_degrees=float(np.degrees(joint_delta[frame, joint])),
            max_joint_change_name=metadata["joint_names"][joint],
            max_joint_change_time=float(frame / 30),
            max_base_position_change_m=np.max(np.abs(result[:, :3] - values[:, :3]), axis=0).tolist(),
            max_base_orientation_change_degrees=float(np.degrees(rotation_delta.max())),
        ),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def smooth_dataset(config_path, rules_path):
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    settings = rules["smoothing"].copy()
    if not 0 < settings["cutoff_hz"] < 15 or settings["filter_order"] < 1 or settings["workers"] < 1:
        raise ValueError("Invalid smoothing cutoff, order or workers")
    for key in ["input_dir", "output_dir", "mjcf_path"]:
        config[key] = str((root / config[key]).resolve())
    config["body_models"] = {
        name: str((root / path).resolve()) for name, path in config["body_models"].items()
    }
    settings["output_dir"] = str((root / settings["output_dir"]).resolve())
    if Path(settings["output_dir"]) == Path(config["output_dir"]):
        raise ValueError("Smoothed output must be separate from original candidates")
    files = sorted(
        path
        for path in Path(config["output_dir"]).glob("*.csv")
        if path.stem.split("__")[0] in settings["datasets"]
    )
    if not files:
        raise ValueError("No selected candidate CSVs")
    code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(f"Smoothing {len(files)} motions at {settings['cutoff_hz']} Hz", flush=True)
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=settings["workers"]) as executor:
        futures = {executor.submit(smooth_clip, path, config, settings, code_sha256): path for path in files}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(dict(source=str(futures[future]), error=str(error)))
                print(f"FAILED {futures[future].name}: {error}", flush=True)
            count = len(results) + len(failures)
            if count % 50 == 0 or count == len(files):
                print(f"[{count}/{len(files)}]", flush=True)
    reports = Path(settings["output_dir"]).parent / "cache/reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "smoothing_report.json").write_text(
        json.dumps(
            dict(settings=settings, code_sha256=code_sha256, clips=results, failures=failures), indent=2
        ),
        encoding="utf-8",
    )
    config["output_dir"] = settings["output_dir"]
    (reports / "retarget_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Finished: {len(results)} smoothed, {len(failures)} failed", flush=True)
    if failures:
        raise RuntimeError("Some motions failed smoothing; see smoothing_report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="retarget_amass.json")
    parser.add_argument("--rules", default="motion_filter.json")
    args = parser.parse_args()
    smooth_dataset(args.config, args.rules)


if __name__ == "__main__":
    main()
