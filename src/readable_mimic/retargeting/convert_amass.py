"""AMASS / SMPL-family NPZ → G1 CSV, sharing IK and support grounding with 100STYLE."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from .smpl import load_amass
from .ik import MotionRetargeter, detect_support_frames, ground_motion
from ..robot import get_robot


def convert_clip(source, settings):
    start = time.perf_counter()
    name = "__".join(source.relative_to(settings["input_dir"]).with_suffix("").parts)
    output = Path(settings["output_dir"]) / (name + ".csv")
    metadata_path = output.with_suffix(".json")
    fingerprint = hashlib.sha256(source.read_bytes() + settings["conversion_sha256"].encode()).hexdigest()
    if output.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata["fingerprint"] == fingerprint
            and metadata["csv_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
        ):
            return metadata | {"skipped": True}
        raise ValueError(f"Existing output differs; use a new output_dir: {output}")
    human = load_amass(source, settings)
    robot = get_robot(settings["robot"])
    model = mujoco.MjModel.from_xml_path(settings["mjcf_path"])
    retargeter = MotionRetargeter(model, robot, settings["ik"], human.body_names)
    qpos = np.array(
        [
            retargeter.retarget(pos, rot, initialize=index == 0)
            for index, (pos, rot) in enumerate(zip(human.body_pos_w, human.body_rot_mat_w))
        ]
    )
    support = detect_support_frames(human, settings["grounding"])
    correction = ground_motion(model, qpos, settings["foot_bodies"], settings["ground_clearance"], support)
    base_pos_w = qpos[:, :3].copy()
    base_pos_w[:, :2] -= base_pos_w[0, :2].copy()
    joint_pos = qpos[:, retargeter.joint_qpos_indices]
    limits = np.array([model.joint(name).range for name in robot.joint_names])
    if np.maximum(limits[:, 0] - joint_pos, joint_pos - limits[:, 1]).max() > 1e-4:
        raise ValueError(f"Joint limit violation: {source}")
    values = np.concatenate((base_pos_w, qpos[:, [4, 5, 6, 3]], joint_pos), axis=1)
    if values.shape[1] != 36 or not np.isfinite(values).all():
        raise ValueError(f"Invalid G1 output: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.part")
    np.savetxt(temporary, values, delimiter=",", fmt="%.8f")
    temporary.replace(output)
    metadata = dict(
        name=name,
        source=str(source),
        frame_rate=human.frame_rate,
        num_frames=len(values),
        joint_names=list(robot.joint_names),
        ground_correction=correction,
        max_joint_vel=float(np.abs(np.diff(joint_pos, axis=0)).max() * human.frame_rate),
        fingerprint=fingerprint,
        csv_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        wall_seconds=round(time.perf_counter() - start, 3),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def convert_dataset(config_path):
    config_path = Path(config_path).resolve()
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("input_dir", "output_dir", "mjcf_path"):
        settings[key] = str((config_path.parent / settings[key]).resolve())
    settings["body_models"] = {
        gender: str((config_path.parent / path).resolve()) for gender, path in settings["body_models"].items()
    }
    if settings["output_frame_rate"] != 30 or settings["workers"] < 1:
        raise ValueError("G1 CSV requires 30 Hz and workers must be positive")
    settings["code_sha256"] = hashlib.sha256(
        b"".join(
            Path(__file__).with_name(name).read_bytes() for name in ["smpl.py", "ik.py", "convert_amass.py"]
        )
    ).hexdigest()
    for path in [settings["mjcf_path"], *settings["body_models"].values()]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing licensed model/robot file: {path}")
    settings["body_model_sha256"] = {
        gender: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for gender, path in settings["body_models"].items()
    }
    settings["robot_sha256"] = hashlib.sha256(Path(settings["mjcf_path"]).read_bytes()).hexdigest()
    settings["conversion_sha256"] = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in settings.items()
                if key not in ("workers", "datasets", "input_dir", "output_dir", "motion_glob", "screening")
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    files = sorted(Path(settings["input_dir"]).rglob(settings["motion_glob"]))
    if settings["datasets"]:
        files = [
            path for path in files if path.relative_to(settings["input_dir"]).parts[0] in settings["datasets"]
        ]
    if not files:
        raise ValueError("No AMASS motion files matched the config")
    print(f"Retargeting {len(files)} AMASS motions", flush=True)
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=settings["workers"]) as executor:
        futures = {executor.submit(convert_clip, path, settings): path for path in files}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{len(results) + len(failures)}/{len(files)}] {result['name']}: {result['num_frames']} frames",
                    flush=True,
                )
            except Exception as error:
                failures.append(dict(source=str(futures[future]), error=str(error)))
                print(f"FAILED {futures[future]}: {error}", flush=True)
    directory = Path(settings["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "conversion_report.json").write_text(
        json.dumps(dict(config=settings, clips=results, failures=failures), indent=2), encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} AMASS clips failed; see conversion_report.json")
    print(f"Finished: {len(results)} motions, 0 failures", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="retarget_amass.json")
    convert_dataset(parser.parse_args().config)
