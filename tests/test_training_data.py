"""Training exports share one gate and never copy candidate metadata."""

from pathlib import Path
import hashlib
import json
import runpy
import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
pytest.importorskip("mink")
MODULE = runpy.run_path(str(ROOT / "tools/prepare_training_data.py"))


def test_export_filters_csv_and_removes_stale_owned_files(tmp_path):
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    for name in ["good", "bad"]:
        (candidates / (name + ".csv")).write_text(name)
        (candidates / (name + ".json")).write_text("{}")
    train = tmp_path / "train"
    export = MODULE["export_training_data"]
    assert (
        export(candidates, train, [dict(name="good", flags=[]), dict(name="bad", flags=["joint_velocity"])])
        == 1
    )
    assert [p.name for p in train.iterdir()] == ["good.csv"]
    assert export(candidates, train, [dict(name="good", flags=["joint_velocity"])]) == 0
    assert not list(train.iterdir())


def test_export_preserves_manually_modified_training_file(tmp_path):
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / "good.csv").write_text("original")
    train = tmp_path / "train"
    export = MODULE["export_training_data"]
    export(candidates, train, [dict(name="good", flags=[])])
    (train / "good.csv").write_text("user edit")
    with pytest.raises(ValueError, match="modified training"):
        export(candidates, train, [])
    assert (train / "good.csv").read_text() == "user edit"


def test_metadata_requires_checksum_rate_frames_and_joint_order(tmp_path):
    path = tmp_path / "motion.csv"
    values = np.zeros((2, 36))
    np.savetxt(path, values, delimiter=",")
    metadata = dict(
        frame_rate=30,
        num_frames=2,
        joint_names=["joint"],
        csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    path.with_suffix(".json").write_text(json.dumps(metadata))
    check = MODULE["check_metadata"]
    assert check(path, values, ["joint"]) == []
    metadata["frame_rate"] = 60
    path.with_suffix(".json").write_text(json.dumps(metadata))
    assert check(path, values, ["joint"]) == ["invalid_metadata"]


def test_shared_gate_keeps_manually_excluded_motion_out_of_train(tmp_path):
    from readable_mimic.robot import get_robot

    robot = get_robot("g1")
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    values = np.zeros((2, 36))
    values[:, 2] = 0.8
    values[:, 6] = 1
    values[:, 7:] = robot.default_joint_pos_list()
    for name in ["accepted", "excluded"]:
        path = candidates / (name + ".csv")
        np.savetxt(path, values, delimiter=",")
        metadata = dict(
            frame_rate=30,
            num_frames=2,
            joint_names=list(robot.joint_names),
            csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        path.with_suffix(".json").write_text(json.dumps(metadata))
    config = dict(
        robot="g1",
        mjcf_path=str(ROOT / "data/g1_description/g1_29dof.xml"),
        output_dir="candidates",
        foot_bodies=["left_ankle_roll_link", "right_ankle_roll_link"],
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    rules = json.loads((ROOT / "motion_filter.json").read_text())
    rules["excluded_motions"] = ["excluded"]
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules))
    report = MODULE["screen"](config_path, rules_path, export=True)
    assert report["exported_count"] == 1
    assert [path.name for path in (tmp_path / "train").iterdir()] == ["accepted.csv"]
    rejected = next(item for item in report["motions"] if item["name"] == "excluded")
    assert rejected["flags"] == ["excluded_motion"]
