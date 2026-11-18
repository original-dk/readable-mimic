import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_missing_mujoco_bindings_explain_repair(monkeypatch):
    from readable_mimic import kinematics

    monkeypatch.setattr(kinematics, "mujoco", SimpleNamespace(__file__=None))
    with pytest.raises(ImportError, match="reinstall-package mujoco"):
        kinematics.Kinematics(None, "cpu")


def test_tf32_settings_allow_inductor_legacy_reads():
    code = """
from types import SimpleNamespace
from unittest.mock import patch
import warnings
import torch
from readable_mimic.trainer import Trainer

for allowed in (True, False):
    config = SimpleNamespace(device="cuda:0", allow_tf32=allowed, num_threads=1,
        seed=0, horizon=2, hist=2, goal_frames=2, robot="tiny")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", UserWarning)
        with patch("readable_mimic.trainer.get_robot", side_effect=RuntimeError("stop_before_gpu_allocation")):
            try:
                Trainer(config)
            except RuntimeError as error:
                assert str(error) == "stop_before_gpu_allocation", str(error)
            else:
                raise AssertionError("Expected initialization to stop before GPU allocation")
    # This exact legacy read is used by PyTorch 2.9 Inductor's pad_mm pass.
    assert torch.backends.cuda.matmul.allow_tf32 is allowed
    assert torch.backends.cudnn.allow_tf32 is allowed
    assert torch.get_float32_matmul_precision() == ("high" if allowed else "highest")
    # Genesis/dependencies can also select matmul precision through this API.
    torch.set_float32_matmul_precision("high")
    assert torch.backends.cuda.matmul.allow_tf32 is True
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_evaluation_config_ignores_training_resume(tmp_path):
    import json
    from pathlib import Path
    from readable_mimic.config import load_config

    config = json.loads((Path(__file__).parents[1] / "config.json").read_text(encoding="utf-8"))
    config["resume"] = "missing_training_checkpoint.pt"
    config["simulator"] = "mujoco"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    evaluation = load_config(path, restore_training_config=False)
    assert evaluation.simulator == "mujoco"
    with pytest.raises(FileNotFoundError):
        load_config(path)
