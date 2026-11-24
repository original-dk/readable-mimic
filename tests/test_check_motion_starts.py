"""Coverage tool accepts the same multi-directory config as training."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from readable_mimic.motion import ReferenceMotion


@pytest.mark.parametrize("multiple", [False, True])
def test_check_motion_directories_and_move_rejected(tmp_path, monkeypatch, multiple):
    spec = importlib.util.spec_from_file_location(
        "check_motion_starts", Path(__file__).parents[1] / "tools/check_motion_starts.py",
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "good.csv").write_text("good")
    (first / "bad_first.csv").write_text("bad")
    (second / "bad_second.csv").write_text("bad")

    class Motion:
        find_files = ReferenceMotion.find_files
        def __init__(self, files, *args):
            self.path = files[0]
            self.num_frames = 120

    monkeypatch.setattr(tool, "ReferenceMotion", Motion)
    monkeypatch.setattr(tool, "get_robot", lambda name: SimpleNamespace(motion_glob="*.csv", mjcf_path="", name=name, load_motion=None))
    monkeypatch.setattr(tool, "smoothing_model", lambda *args: (None, None, None))
    monkeypatch.setattr(tool, "clip_foot_heights", lambda motion, *args: np.full(120, 1.0 if Path(motion.path).name.startswith("bad") else 0.0))
    monkeypatch.setattr(tool, "__file__", str(tmp_path / "tools/check_motion_starts.py"))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(robot="g1", motion_dir=["first", "second"] if multiple else "first", max_episode_steps=500)))
    report = tmp_path / "report.json"
    tool.main(["--config", str(config), "--report", str(report), "--move-rejected"])
    result = json.loads(report.read_text())
    assert result["checked_count"] == (3 if multiple else 2)
    assert result["moved_count"] == (2 if multiple else 1)
    assert (first / "good.csv").exists()
    assert not (first / "bad_first.csv").exists()
    assert (second / "bad_second.csv").exists() is (not multiple)
    for moved in result["moved_motions"]:
        assert Path(moved["destination"]).read_text() == "bad"
        assert not Path(moved["source"]).exists()
