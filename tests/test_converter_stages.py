"""Converters must preserve IK base height; grounding belongs to CSV postprocessing."""

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from readable_mimic.robot import get_robot
from readable_mimic.retargeting import convert_amass, convert_100style

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("kind", ["amass", "100style"])
def test_converter_does_not_change_ik_height(tmp_path, monkeypatch, kind):
    module = convert_amass if kind == "amass" else convert_100style
    config = json.loads((ROOT / f"retarget_{kind}.json").read_text())
    config["input_dir"] = str(tmp_path / "raw")
    config["output_dir"] = str(tmp_path / "candidates")
    config["mjcf_path"] = str(ROOT / config["mjcf_path"])
    config["conversion_sha256"] = "test"
    source = Path(config["input_dir"]) / ("clip_stageii.npz" if kind == "amass" else "clip.bvh")
    source.parent.mkdir()
    if kind == "amass":
        np.savez(source, betas=np.zeros(16))
    else:
        source.write_text("test source")
    robot = get_robot("g1")
    heights = np.array([0.8, 0.9, 1.1])
    positions = np.zeros((3, 1, 3))
    positions[:, 0, 2] = heights
    human = SimpleNamespace(
        body_names=["pelvis"],
        body_pos_w=positions,
        body_rot_mat_w=np.tile(np.eye(3), (3, 1, 1, 1)),
        frame_rate=30,
    )

    class FakeRetargeter:
        def __init__(self, model, *args):
            self.model = model
            self.base_qpos_start = 0
            self.joint_qpos_indices = [int(model.joint(n).qposadr[0]) for n in robot.joint_names]

        def retarget(self, positions, rotations, initialize=False):
            result = self.model.qpos0.copy()
            result[:3] = positions[0]
            result[self.joint_qpos_indices] = robot.default_joint_pos_list()
            return result

    if kind == "amass":
        monkeypatch.setattr(module, "load_amass", lambda *args: human)
        monkeypatch.setattr(module, "AmassRetargeter", FakeRetargeter)
        metadata = module.convert_clip(source, config)
    else:
        monkeypatch.setattr(module, "load_bvh", lambda *args: human)
        monkeypatch.setattr(module, "MotionRetargeter", FakeRetargeter)
        metadata = module.convert_clip(source, config["output_dir"], config, [0, 3])
    values = np.loadtxt(Path(config["output_dir"]) / (metadata["name"] + ".csv"), delimiter=",")
    np.testing.assert_allclose(values[:, 2], heights)
    assert metadata["grounding_applied"] is False
    assert metadata["processing_stage"] == "retargeted"
    assert "ground_correction" not in metadata and "ground_offset" not in metadata
