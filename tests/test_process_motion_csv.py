"""Exercise CSV-only processing without input JSON, NPZ or SMPL-X assets."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
from readable_mimic.robot import get_robot

ROOT = Path(__file__).parents[1]


def csv_fixture(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    robot = get_robot("g1")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "data/g1_description/g1_29dof.xml"))
    data = mujoco.MjData(model)
    values = np.zeros((60, 36))
    values[:, 2] = 0.8
    values[:, 6] = 1
    values[:, 7:] = robot.default_joint_pos_list()
    data.qpos[:3] = values[0, :3]
    data.qpos[3:7] = [1, 0, 0, 0]
    indices = [int(model.joint(n).qposadr[0]) for n in robot.joint_names]
    data.qpos[indices] = values[0, 7:]
    mujoco.mj_kinematics(model, data)
    feet = [
        i
        for i in range(model.ngeom)
        if model.body(model.geom_bodyid[i]).name in ["left_ankle_roll_link", "right_ankle_roll_link"]
        and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    values[:, 2] -= np.min(data.geom_xpos[feet, 2] - model.geom_size[feet, 0]) + 0.0001
    wrist = robot.joint_names.index("left_wrist_yaw_joint")
    values[:, 7 + wrist] = 0.01 * np.sin(2 * np.pi * 10 * np.arange(len(values)) / 30)
    np.savetxt(source / "CMU__test__stand.csv", values, delimiter=",")
    offset = values.copy()
    offset[:, 2] += 0.03
    np.savetxt(source / "CMU__test__offset.csv", offset, delimiter=",")
    hover = values.copy()
    hover[:, 2] += 0.3
    np.savetxt(source / "CMU__test__hover.csv", hover, delimiter=",")
    penetrating = values.copy()
    penetrating[:, 2] -= 0.3
    np.savetxt(source / "CMU__test__penetrating.csv", penetrating, delimiter=",")
    config = dict(
        robot="g1",
        mjcf_path=str(ROOT / "data/g1_description/g1_29dof.xml"),
        foot_bodies=["left_ankle_roll_link", "right_ankle_roll_link"],
        input_dir="does_not_exist",
        body_models={"neutral": "NO_SMPL_MODEL.npz"},
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    rules = json.loads((ROOT / "motion_filter_level_0.json").read_text())
    rules["smoothing"]["workers"] = 1
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules))
    return source, config_path, rules_path


def run_cli(source, output, config, rules):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/process_motion_csv.py"),
            "--input-dir",
            str(source),
            "--output-dir",
            str(output),
            "--config",
            str(config),
            "--rules",
            str(rules),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_csv_only_pipeline_smooths_filters_preserves_original_and_reuses_cache(tmp_path):
    source, config, rules = csv_fixture(tmp_path)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.csv")}
    output = tmp_path / "output"
    result = run_cli(source, output, config, rules)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(p.name for p in (output / "train").iterdir()) == [
        "CMU__test__offset.csv",
        "CMU__test__stand.csv",
    ]
    report = json.loads((output / "cache/reports/screening_report.json").read_text())
    offset_metadata = json.loads((output / "candidates/CMU__test__offset.json").read_text())
    assert offset_metadata["grounding_applied"] is True
    assert -0.031 < offset_metadata["ground_correction"]["floor_offset"] < -0.029
    hover = next(m for m in report["motions"] if "hover" in m["name"])
    assert "prolonged_flight" in hover["flags"]
    assert hover["metrics"]["min_foot_height"] > 0.2  # Not silently forced to the ground.
    penetrating = next(m for m in report["motions"] if "penetrating" in m["name"])
    assert "foot_ground_penetration" in penetrating["flags"]
    assert penetrating["metrics"]["min_foot_height"] < -0.2
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.csv")}
    assert not list(source.glob("*.json"))
    original = np.loadtxt(source / "CMU__test__stand.csv", delimiter=",")
    processed = np.loadtxt(output / "train/CMU__test__stand.csv", delimiter=",")
    assert np.sqrt(np.mean(np.diff(processed[:, 7:], n=2, axis=0) ** 2)) < np.sqrt(
        np.mean(np.diff(original[:, 7:], n=2, axis=0) ** 2)
    )
    result = run_cli(source, output, config, rules)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "cache/reports/processing_report.json").read_text())
    assert all(m["skipped"] for m in report["clips"])


def test_processing_failure_does_not_export_partial_train(tmp_path):
    source, config, rules = csv_fixture(tmp_path)
    np.savetxt(source / "invalid.csv", np.zeros((2, 2)), delimiter=",")
    output = tmp_path / "output"
    result = run_cli(source, output, config, rules)
    assert result.returncode != 0
    report = json.loads((output / "cache/reports/processing_report.json").read_text())
    assert len(report["failures"]) == 1
    assert not (output / "train").exists()


def test_output_cannot_be_nested_in_input(tmp_path):
    source, config, rules = csv_fixture(tmp_path)
    result = run_cli(source, source / "output", config, rules)
    assert result.returncode != 0
    assert "must be separate" in result.stderr
    assert not (source / "output").exists()



def test_dataset_root_keeps_raw_candidates_separate(tmp_path):
    source, config, rules = csv_fixture(tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source = source.rename(dataset / "candidates")
    original = {path.name: path.read_bytes() for path in source.glob("*.csv")}
    result = run_cli(source, dataset, config, rules)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original
    assert {path.name for path in (dataset / "processed_candidates").glob("*.csv")} == set(original)
    assert (dataset / "train/CMU__test__stand.csv").is_file()
    assert (dataset / "cache/reports/screening_report.json").is_file()
