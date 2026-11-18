"""Report kinematic/context risks in AMASS candidates; never approve motions for training."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from readable_mimic.robot import get_robot
from readable_mimic.g1.g1_constants import motor_of


def check_values(values, limits, velocity_limits, settings):
    flags = []
    if values.ndim != 2 or values.shape[1] != 36 or len(values) < 2 or not np.isfinite(values).all():
        return ["invalid_csv"], {}
    quat_norm = np.linalg.norm(values[:, 3:7], axis=1)
    if np.max(np.abs(quat_norm - 1)) > settings["quaternion_tolerance"]:
        return ["invalid_quaternion"], {}
    joint_pos = values[:, 7:]
    if (
        np.maximum(limits[:, 0] - joint_pos, joint_pos - limits[:, 1]).max()
        > settings["joint_limit_tolerance"]
    ):
        flags.append("joint_limit")
    joint_vel = np.diff(joint_pos, axis=0) * 30
    velocity_ratio = float(np.max(np.abs(joint_vel) / velocity_limits))
    if velocity_ratio > settings["velocity_limit_multiplier"]:
        flags.append("joint_velocity")
    rotations = Rotation.from_quat(values[:, 3:7]).as_matrix()
    max_tilt = float(np.degrees(np.arccos(np.clip(rotations[:, 2, 2], -1, 1))).max())
    min_height = float(values[:, 2].min())
    if max_tilt > settings["max_base_tilt_degrees"]:
        flags.append("base_tilt")
    if min_height < settings["min_base_height"]:
        flags.append("low_base")
    return flags, dict(
        peak_joint_vel=float(np.abs(joint_vel).max()),
        velocity_limit_ratio=velocity_ratio,
        min_base_height=min_height,
        max_base_tilt_degrees=max_tilt,
    )


def screen(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = config["screening"]
    directory = (config_path.parent / config["output_dir"]).resolve()
    robot = get_robot(config["robot"])
    spec = mujoco.MjSpec.from_file(str(config_path.parent / config["mjcf_path"]))
    spec.worldbody.add_geom(name="screen_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1])
    model = spec.compile()
    data = mujoco.MjData(model)
    ground_id = model.geom("screen_ground").id
    foot_ids = {
        i for i in range(model.ngeom) if model.body(model.geom_bodyid[i]).name in config["foot_bodies"]
    }
    limits = np.array([model.joint(name).range for name in robot.joint_names])
    velocity_limits = np.array([motor_of(name)[1] for name in robot.joint_names])
    results = []
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise ValueError(f"No candidates in {directory}")
    for path in files:
        values = np.loadtxt(path, delimiter=",", ndmin=2)
        flags, metrics = check_values(values, limits, velocity_limits, settings)
        name = path.stem.split("__")[-1]
        words = {word.lower() for word in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", name)}
        if words.intersection(settings["review_keywords"]):
            flags.append("motion_name_context_review")
        if path.stem.split("__")[0] in settings["context_review_datasets"]:
            flags.append("dataset_context_review")
        if metrics:
            for frame in values[:: settings["sample_stride"]]:
                data.qpos[:3] = frame[:3]
                data.qpos[3:7] = frame[[6, 3, 4, 5]]
                data.qpos[7:] = frame[7:]
                mujoco.mj_forward(model, data)
                extra = any(
                    ground_id in c.geom
                    and not any(int(i) in foot_ids for i in c.geom)
                    and c.dist < -settings["unsupported_contact_penetration"]
                    for c in data.contact
                )
                if extra:
                    flags.append("nonfoot_ground_contact")
                    break
        results.append(dict(name=path.stem, status="needs_review", flags=flags, metrics=metrics))
    counts = Counter(flag for result in results for flag in result["flags"])
    report = dict(
        candidate_count=len(results),
        without_detected_flags=sum(not r["flags"] for r in results),
        approved_for_training=0,
        flag_counts=dict(counts),
        motions=results,
    )
    (directory / "screening_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    training_dir = (config_path.parent / settings["training_dir"]).resolve()
    training_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({key: value for key, value in report.items() if key != "motions"}), flush=True)
    print(
        "No motions copied to training; manual review and simulation tracking are still required.", flush=True
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="retarget_amass.json")
    screen(parser.parse_args().config)
