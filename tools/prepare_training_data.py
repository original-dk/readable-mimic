"""Apply the same quality checks to 100STYLE and AMASS; export CSVs only."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from readable_mimic.robot import get_robot
from readable_mimic.g1.g1_constants import motor_of


def check_values(values, limits, velocity_limits, settings):
    if values.ndim != 2 or values.shape[1] != 36 or len(values) < 2 or not np.isfinite(values).all():
        return ["invalid_csv"], {}
    if np.max(np.abs(np.linalg.norm(values[:, 3:7], axis=1) - 1)) > settings["quaternion_tolerance"]:
        return ["invalid_quaternion"], {}
    flags = []
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
    tilt = float(np.degrees(np.arccos(np.clip(rotations[:, 2, 2], -1, 1))).max())
    height = float(values[:, 2].min())
    if tilt > settings["max_base_tilt_degrees"]:
        flags.append("base_tilt")
    if height < settings["min_base_height"]:
        flags.append("low_base")
    return flags, dict(
        peak_joint_vel=float(np.abs(joint_vel).max()),
        velocity_limit_ratio=velocity_ratio,
        min_base_height=height,
        max_base_tilt_degrees=tilt,
    )


def check_ground_support(foot_height, com_height, settings, frame_rate=30):
    """Check flat-floor support and distinguish short ballistic flight from hovering."""
    flags = []
    if foot_height.min() < -settings["max_foot_penetration"]:
        flags.append("foot_ground_penetration")
    airborne = foot_height > settings["foot_contact_height"]
    edges = np.diff(np.r_[False, airborne, False].astype(int))
    intervals = []
    gravity = 9.81
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        duration = (stop - start) / frame_rate
        interval = dict(
            start_time=float(start / frame_rate), end_time=float(stop / frame_rate), duration=float(duration)
        )
        if duration > settings["max_flight_duration"]:
            flags.append("prolonged_flight")
        elif duration > settings["min_flight_check_duration"]:
            if start == 0 or stop == len(foot_height):
                flags.append("unbounded_flight")
            else:
                # In free flight the whole-robot COM follows gravity, regardless of joint motion.
                height = com_height[start - 1 : stop + 1]
                time = np.arange(len(height)) / frame_rate
                linear = np.column_stack([np.ones(len(time)), time])
                coefficients = np.linalg.lstsq(linear, height + 0.5 * gravity * time**2, rcond=None)[0]
                residual = height - (linear @ coefficients - 0.5 * gravity * time**2)
                fit_error = float(np.max(np.abs(residual)))
                interval["ballistic_fit_error"] = fit_error
                if fit_error > settings["max_flight_fit_error"]:
                    flags.append("non_ballistic_flight")
        intervals.append(interval)
    return sorted(set(flags)), dict(
        min_foot_height=float(foot_height.min()),
        max_foot_height=float(foot_height.max()),
        airborne_fraction=float(airborne.mean()),
        longest_flight_duration=max((item["duration"] for item in intervals), default=0.0),
        flight_intervals=intervals,
    )


class GroundContactChecker:
    """Cache robot geometry once; measure actual collision soles at every frame."""

    def __init__(self, model, robot, foot_body_names, settings):
        self.model = model
        self.data = mujoco.MjData(model)
        self.settings = settings
        self.ground_id = model.geom("screen_ground").id
        self.joint_indices = [int(model.joint(name).qposadr[0]) for name in robot.joint_names]
        self.base_body_id = int(
            model.jnt_bodyid[np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]]
        )
        self.foot_indices = np.array(
            [
                i
                for i in range(model.ngeom)
                if model.body(model.geom_bodyid[i]).name in foot_body_names
                and (model.geom_contype[i] or model.geom_conaffinity[i])
            ]
        )
        if not len(self.foot_indices):
            raise ValueError("No foot collision geometry")
        types = model.geom_type[self.foot_indices]
        self.sphere_mask = types == mujoco.mjtGeom.mjGEOM_SPHERE
        if not np.all(self.sphere_mask | (types == mujoco.mjtGeom.mjGEOM_BOX)):
            raise ValueError("Expected sphere or box foot collision geometry")
        self.foot_sizes = model.geom_size[self.foot_indices]

    def check(self, values):
        foot_height = np.empty(len(values))
        com_height = np.empty(len(values))
        nonfoot_contact = False
        for index, frame in enumerate(values):
            self.data.qpos[:3] = frame[:3]
            self.data.qpos[3:7] = frame[[6, 3, 4, 5]]
            self.data.qpos[self.joint_indices] = frame[7:]
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            rotations = self.data.geom_xmat[self.foot_indices].reshape(-1, 3, 3)
            box_radius = np.einsum("ij,ij->i", np.abs(rotations[:, 2]), self.foot_sizes)
            radius = np.where(self.sphere_mask, self.foot_sizes[:, 0], box_radius)
            foot_height[index] = np.min(self.data.geom_xpos[self.foot_indices, 2] - radius)
            com_height[index] = self.data.subtree_com[self.base_body_id, 2]
            if not nonfoot_contact and index % self.settings["sample_stride"] == 0:
                mujoco.mj_collision(self.model, self.data)
                nonfoot_contact = any(
                    self.ground_id in contact.geom
                    and not any(int(i) in self.foot_indices for i in contact.geom)
                    and contact.dist < -self.settings["unsupported_contact_penetration"]
                    for contact in self.data.contact
                )
        flags, metrics = check_ground_support(foot_height, com_height, self.settings)
        if nonfoot_contact:
            flags.append("nonfoot_ground_contact")
        return flags, metrics


def check_metadata(path, values, joint_names, max_smoothing_change_degrees=None):
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    valid = (
        metadata["frame_rate"] == 30
        and metadata["num_frames"] == len(values)
        and metadata["joint_names"] == list(joint_names)
        and metadata["csv_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    flags = [] if valid else ["invalid_metadata"]
    if (
        max_smoothing_change_degrees is not None
        and metadata.get("changes", {}).get("max_joint_change_degrees", 0) > max_smoothing_change_degrees
    ):
        flags.append("large_smoothing_change")
    return flags


def export_training_data(directory, training_dir, results):
    """Synchronize only previously exported, unchanged files; preserve user edits."""
    training_dir.mkdir(parents=True, exist_ok=True)
    if any(not path.is_file() or path.suffix != ".csv" for path in training_dir.iterdir()):
        raise ValueError(f"train must contain CSV files only: {training_dir}")
    report_path = directory.parent / "cache/reports/training_export.json"
    previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    passed = {item["name"] for item in results if not item["flags"]}
    for path in training_dir.glob("*.csv"):
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if previous.get(path.name) != checksum:
            raise ValueError(f"Unmanaged or modified training file; review before replacing: {path}")
    for path in training_dir.glob("*.csv"):
        if path.stem not in passed:
            path.unlink()
    exported = {}
    for name in sorted(passed):
        source = directory / (name + ".csv")
        destination = training_dir / source.name
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        if previous.get(source.name) != checksum or not destination.exists():
            temporary_dir = directory.parent / "cache/train_export_tmp"
            temporary_dir.mkdir(parents=True, exist_ok=True)
            temporary = temporary_dir / (source.name + ".part")
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        exported[source.name] = checksum
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(exported, indent=2), encoding="utf-8")
    return len(exported)


def screen(config_path, rules_path=None, export=False):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rules_path = Path(rules_path).resolve() if rules_path else config_path.parent / "motion_filter.json"
    settings = json.loads(rules_path.read_text(encoding="utf-8"))
    directory = (config_path.parent / config["output_dir"]).resolve()
    dataset_root = directory.parent
    robot = get_robot(config["robot"])
    spec = mujoco.MjSpec.from_file(str(config_path.parent / config["mjcf_path"]))
    spec.worldbody.add_geom(name="screen_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1])
    model = spec.compile()
    ground_checker = GroundContactChecker(model, robot, config["foot_bodies"], settings)
    limits = np.array([model.joint(name).range for name in robot.joint_names])
    velocity_limits = np.array([motor_of(name)[1] for name in robot.joint_names])
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise ValueError(f"No candidates in {directory}")
    results = []
    for path in files:
        try:
            values = np.loadtxt(path, delimiter=",", ndmin=2)
            flags, metrics = check_values(values, limits, velocity_limits, settings)
            flags += check_metadata(
                path, values, robot.joint_names, settings.get("smoothing", {}).get("max_joint_change_degrees")
            )
        except (ValueError, OSError, KeyError) as error:
            results.append(
                dict(name=path.stem, status="rejected", flags=["invalid_candidate"], error=str(error))
            )
            continue
        if path.stem in settings["excluded_motions"]:
            flags.append("excluded_motion")
        words = {
            word.lower() for word in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", path.stem.split("__")[-1])
        }
        if words.intersection(settings["review_keywords"]):
            flags.append("motion_name_context_review")
        if path.stem.split("__")[0] in settings["context_review_datasets"]:
            flags.append("dataset_context_review")
        if metrics:
            ground_flags, ground_metrics = ground_checker.check(values)
            flags += ground_flags
            metrics.update(ground_metrics)
        results.append(
            dict(
                name=path.stem, status="rejected" if flags else "passed_checks", flags=flags, metrics=metrics
            )
        )
    training_dir = dataset_root / "train"
    count = export_training_data(directory, training_dir, results) if export else None
    report = dict(
        candidate_count=len(results),
        without_detected_flags=sum(not item["flags"] for item in results),
        exported_count=count,
        dynamically_verified=False,
        screening_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rules=settings,
        flag_counts=dict(Counter(flag for item in results for flag in item["flags"])),
        motions=results,
    )
    reports = dataset_root / "cache/reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "screening_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps({key: value for key, value in report.items() if key not in ["rules", "motions"]}),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rules")
    parser.add_argument(
        "--export", action="store_true", help="Copy motions passing all common checks into train"
    )
    args = parser.parse_args()
    screen(args.config, args.rules, args.export)


if __name__ == "__main__":
    main()
