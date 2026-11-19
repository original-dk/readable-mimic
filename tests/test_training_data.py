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


def support_rules():
    return json.loads((ROOT / "motion_filter.json").read_text())


def test_ground_support_accepts_walk_and_short_ballistic_jump():
    check = MODULE["check_ground_support"]
    foot = np.zeros(40)
    com = np.full(40, 0.8)
    flags, metrics = check(foot, com, support_rules())
    assert flags == [] and metrics["longest_flight_duration"] == 0
    time = np.arange(13) / 30
    jump = 0.5 * 9.81 * time * (time[-1] - time)
    foot[9:22] = jump
    com[9:22] += jump
    flags, metrics = check(foot, com, support_rules())
    assert flags == []
    assert 0 < metrics["longest_flight_duration"] < 0.6


def test_ground_support_rejects_stairs_hover_and_truncated_flight():
    check = MODULE["check_ground_support"]
    foot = np.r_[np.zeros(30), np.full(240, 0.3), np.zeros(30)]
    flags, metrics = check(foot, np.full(len(foot), 0.8), support_rules())
    assert "prolonged_flight" in flags
    assert metrics["longest_flight_duration"] == 8
    foot = np.r_[np.zeros(5), np.full(12, 0.2), np.zeros(5)]
    flags, _ = check(foot, np.full(len(foot), 0.8), support_rules())
    assert "non_ballistic_flight" in flags
    foot = np.r_[np.full(8, 0.2), np.zeros(5)]
    flags, _ = check(foot, np.full(len(foot), 0.8), support_rules())
    assert "unbounded_flight" in flags


def test_ground_support_rejects_foot_penetration():
    flags, _ = MODULE["check_ground_support"](np.full(30, -0.01), np.full(30, 0.8), support_rules())
    assert "foot_ground_penetration" in flags


def test_ground_checker_uses_collision_sole_and_detects_elevated_pose():
    import mujoco
    from readable_mimic.robot import get_robot

    robot = get_robot("g1")
    spec = mujoco.MjSpec.from_file(str(ROOT / "data/g1_description/g1_29dof.xml"))
    spec.worldbody.add_geom(name="screen_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1])
    model = spec.compile()
    checker = MODULE["GroundContactChecker"](
        model, robot, ["left_ankle_roll_link", "right_ankle_roll_link"], support_rules()
    )
    values = np.zeros((int(support_rules()["max_flight_duration"] * 30) + 5, 36))
    values[:, 2] = 0.8
    values[:, 6] = 1
    values[:, 7:] = robot.default_joint_pos_list()
    _, metrics = checker.check(values)
    values[:, 2] -= metrics["min_foot_height"]
    flags, _ = checker.check(values)
    assert flags == []
    values[:, 2] += 0.2
    flags, metrics = checker.check(values)
    assert "prolonged_flight" in flags
    assert metrics["min_foot_height"] == pytest.approx(0.2)


@pytest.mark.skipif(
    not (ROOT / "data/amass/candidates/CMU__13__13_37_stageii.csv").exists(),
    reason="Licensed local CMU regression clip not available",
)
def test_reported_cmu_stairs_cannot_pass_flat_ground_gate():
    import mujoco
    from readable_mimic.robot import get_robot

    robot = get_robot("g1")
    spec = mujoco.MjSpec.from_file(str(ROOT / "data/g1_description/g1_29dof.xml"))
    spec.worldbody.add_geom(name="screen_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1])
    checker = MODULE["GroundContactChecker"](
        spec.compile(), robot, ["left_ankle_roll_link", "right_ankle_roll_link"], support_rules()
    )
    values = np.loadtxt(ROOT / "data/amass/candidates/CMU__13__13_37_stageii.csv", delimiter=",")
    flags, metrics = checker.check(values)
    assert "prolonged_flight" in flags
    assert metrics["longest_flight_duration"] > 8
    assert metrics["max_foot_height"] > 0.5


def test_metadata_flags_large_smoothing_changes_for_review(tmp_path):
    path = tmp_path / "motion.csv"
    values = np.zeros((2, 36))
    np.savetxt(path, values, delimiter=",")
    metadata = dict(
        frame_rate=30,
        num_frames=2,
        joint_names=["joint"],
        csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        changes=dict(max_joint_change_degrees=31),
    )
    path.with_suffix(".json").write_text(json.dumps(metadata))
    assert MODULE["check_metadata"](path, values, ["joint"], 30) == ["large_smoothing_change"]
    metadata["changes"]["max_joint_change_degrees"] = 29
    path.with_suffix(".json").write_text(json.dumps(metadata))
    assert MODULE["check_metadata"](path, values, ["joint"], 30) == []
