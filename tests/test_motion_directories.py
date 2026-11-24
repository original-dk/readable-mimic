"""Multiple directories through config, motion loading and evaluation selection."""
import json
from pathlib import Path

import numpy as np
import pytest

from test_pipeline import tiny_trainer
from readable_mimic.config import load_config
from readable_mimic.motion import ReferenceMotion
from readable_mimic.evaluate import load_evaluation_motion


def test_multiple_motion_directories_and_evaluation(tiny_trainer, tmp_path):
    trainer = tiny_trainer
    directories = [tmp_path / "first", tmp_path / "second"]
    source = np.loadtxt(tmp_path / "motion.csv", delimiter=",")
    for index, directory in enumerate(directories):
        directory.mkdir()
        np.savetxt(directory / f"clip{index}.csv", source, delimiter=",")
    paths = [str(directory) for directory in directories]
    files = ReferenceMotion.find_files(paths, trainer.robot)
    assert [Path(file).stem for file in files] == ["clip0", "clip1"]
    assert ReferenceMotion.find_files(paths + paths, trainer.robot) == files
    motion = ReferenceMotion.from_dir(paths, trainer.robot, trainer.kinematics, "cpu")
    assert motion.names == ["clip0", "clip1"]
    selected = load_evaluation_motion(paths, trainer.robot, trainer.kinematics, "cpu", "clip1")
    assert selected.names == ["clip1"]
    np.savetxt(directories[1] / "clip0.csv", source, delimiter=",")
    with pytest.raises(ValueError, match="Duplicate motion name"):
        ReferenceMotion.find_files(paths, trainer.robot)


def test_motion_directory_config_paths(tiny_trainer, tmp_path):
    values = dict(vars(tiny_trainer.config))
    values.update(warmup_steps=4, motion_dir=["first", "second"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))
    config = load_config(path)
    assert config.motion_dir == [str((tmp_path / name).resolve()) for name in ["first", "second"]]
    values["motion_dir"] = "first"
    path.write_text(json.dumps(values))
    assert load_config(path).motion_dir == str((tmp_path / "first").resolve())
    for invalid in ([], [""], [1], "", None):
        values["motion_dir"] = invalid
        path.write_text(json.dumps(values))
        with pytest.raises(ValueError, match="motion_dir"):
            load_config(path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        ReferenceMotion.find_files([str(empty)], tiny_trainer.robot)
