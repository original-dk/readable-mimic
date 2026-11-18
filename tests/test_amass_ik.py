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
