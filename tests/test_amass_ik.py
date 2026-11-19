"""Regression checks for GMR-aligned AMASS IK setup (no licensed motion needed)."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("mink")
from readable_mimic.retargeting.amass_ik import AmassRetargeter, load_retarget_model
from readable_mimic.retargeting.smpl import BODY_NAMES
from readable_mimic.robot import get_robot

ROOT = Path(__file__).parents[1]


def test_amass_uses_gmr_initial_pose_and_offline_limits():
    settings = json.loads((ROOT / "retarget_amass.json").read_text(encoding="utf-8"))["ik"]
    original_settings = json.dumps(settings, sort_keys=True)
    path = ROOT / "data/g1_description/g1_29dof.xml"
    training_model = mujoco.MjModel.from_xml_path(str(path))
    model = load_retarget_model(path, settings)
    solver = AmassRetargeter(model, get_robot("g1"), settings, BODY_NAMES, 1.8)
    np.testing.assert_allclose(solver.configuration.data.qpos[:7], [0, 0, 0.793, 1, 0, 0, 0])
    np.testing.assert_allclose(solver.configuration.data.qpos[solver.joint_qpos_indices], 0)
    np.testing.assert_allclose(model.joint("left_shoulder_pitch_joint").range, [-3.0892, 1.149])
    np.testing.assert_allclose(training_model.joint("left_shoulder_pitch_joint").range, [-3.0892, 2.6704])
    assert json.dumps(settings, sort_keys=True) == original_settings


def test_amass_toe_targets_use_gmr_frame_and_scale_by_height():
    settings = json.loads((ROOT / "retarget_amass.json").read_text(encoding="utf-8"))["ik"]
    model = load_retarget_model(ROOT / "data/g1_description/g1_29dof.xml", settings)
    solver = AmassRetargeter(model, get_robot("g1"), settings, BODY_NAMES, 0.9)
    assert solver.settings["base_position_scale"] == pytest.approx(0.45)
    for side in ["left", "right"]:
        toe = model.body(side + "_toe_link")
        assert toe.parentid[0] == model.body(side + "_ankle_roll_link").id
        np.testing.assert_allclose(toe.pos, [0.1, 0, -0.02])
        for stage in solver.settings["stages"]:
            foot = next(entry for entry in stage if entry["robot_body"] == side + "_toe_link")
            assert foot["position_scale"] == pytest.approx(0.45)


def human_frame_from_robot_pose(model, robot, settings):
    """Build consistent task targets from a reachable robot FK pose."""
    from scipy.spatial.transform import Rotation
    from readable_mimic.retargeting.smpl import BODY_NAMES

    data = mujoco.MjData(model)
    data.qpos[:3] = [0.2, -0.1, 0.85]
    indices = [int(model.joint(name).qposadr[0]) for name in robot.joint_names]
    data.qpos[indices] = robot.default_joint_pos_list()
    mujoco.mj_kinematics(model, data)
    positions = np.zeros((len(BODY_NAMES), 3))
    rotations = np.tile(np.eye(3), (len(BODY_NAMES), 1, 1))
    base = data.xpos[model.body("pelvis").id].copy()
    human_base = base / settings["base_position_scale"]
    for entry in settings["stages"][0]:
        index = BODY_NAMES.index(entry["human_body"])
        body = model.body(entry["robot_body"]).id
        rotation = data.xmat[body].reshape(3, 3)
        offset = Rotation.from_quat(entry["rotation_offset_wxyz"], scalar_first=True).as_matrix()
        positions[index] = (
            human_base
            + (data.xpos[body] - base - rotation @ entry["position_offset"]) / entry["position_scale"]
        )
        rotations[index] = rotation @ offset.T
    return positions, rotations


def test_first_frame_base_initialization_includes_smpl_robot_offset():
    from scipy.spatial.transform import Rotation

    settings = json.loads((ROOT / "retarget_amass.json").read_text())["ik"]
    robot = get_robot("g1")
    model = load_retarget_model(ROOT / "data/g1_description/g1_29dof.xml", settings)
    positions, rotations = human_frame_from_robot_pose(model, robot, settings)
    yaw = Rotation.from_euler("z", 145, degrees=True).as_matrix()
    positions = positions @ yaw.T
    rotations = yaw @ rotations
    solver = AmassRetargeter(model, robot, settings, BODY_NAMES, 1.8)
    solver.initialize_base_pose(positions, rotations)
    np.testing.assert_allclose(
        solver.configuration.data.qpos[:3], yaw @ np.array([0.2, -0.1, 0.85]), atol=1e-12
    )
    actual = Rotation.from_quat(solver.configuration.data.qpos[3:7], scalar_first=True).as_matrix()
    np.testing.assert_allclose(actual, yaw, atol=1e-12)


def test_retarget_joint_solution_is_invariant_to_initial_world_heading():
    from scipy.spatial.transform import Rotation

    settings = json.loads((ROOT / "retarget_amass.json").read_text())["ik"]
    robot = get_robot("g1")
    model = load_retarget_model(ROOT / "data/g1_description/g1_29dof.xml", settings)
    positions, rotations = human_frame_from_robot_pose(model, robot, settings)
    results = []
    for degrees in [0, 145, -85]:
        yaw = Rotation.from_euler("z", degrees, degrees=True).as_matrix()
        current_model = load_retarget_model(ROOT / "data/g1_description/g1_29dof.xml", settings)
        solver = AmassRetargeter(current_model, robot, settings, BODY_NAMES, 1.8)
        pos, rot = positions @ yaw.T, yaw @ rotations
        solver.initialize_base_pose(pos, rot)
        for _ in range(3):
            qpos = solver.retarget(pos, rot)
        results.append(qpos[solver.joint_qpos_indices])
    np.testing.assert_allclose(results[1], results[0], atol=1e-5)
    np.testing.assert_allclose(results[2], results[0], atol=1e-5)
