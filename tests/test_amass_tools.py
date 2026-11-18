from pathlib import Path
import io
import json
import runpy
import tarfile
import numpy as np
import pytest

pytest.importorskip("mink")
ROOT = Path(__file__).parents[1]


def test_screening_reports_speed_and_tilt_without_approving():
    check = runpy.run_path(str(ROOT / "tools/screen_amass.py"))["check_values"]
    settings = dict(
        quaternion_tolerance=1e-5,
        joint_limit_tolerance=1e-4,
        velocity_limit_multiplier=1,
        min_base_height=0.4,
        max_base_tilt_degrees=50,
    )
    values = np.zeros((3, 36))
    values[:, 2] = 0.75
    values[:, 6] = 1
    limits = np.tile([-2, 2], (29, 1))
    velocity_limits = np.full(29, 10.0)
    flags, _ = check(values, limits, velocity_limits, settings)
    assert not flags
    values[1, 7] = 1
    flags, metrics = check(values, limits, velocity_limits, settings)
    assert "joint_velocity" in flags and metrics["velocity_limit_ratio"] == 3
    values[1, 6] = 0
    flags, _ = check(values, limits, velocity_limits, settings)
    assert flags == ["invalid_quaternion"]


def test_archive_import_rejects_path_escape_and_can_resume(tmp_path):
    prepare = runpy.run_path(str(ROOT / "tools/prepare_amass.py"))["prepare_archives"]
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(input_dir="raw")))
    archives = tmp_path / "archives"
    archives.mkdir()
    for dataset, name in [("ACCAD", "ACCAD/S1/walk_stageii.npz"), ("CNRS", "../escape.npz")]:
        with tarfile.open(archives / (dataset + ".tar.bz2"), "w:bz2") as archive:
            entry = tarfile.TarInfo(name)
            entry.size = 3
            archive.addfile(entry, io.BytesIO(b"npz"))
    manifest, prepared = prepare(config, archives)
    assert manifest["ACCAD"]["status"] == "extracted"
    assert manifest["CNRS"]["status"] == "failed"
    assert not (tmp_path / "escape.npz").exists()
    assert (tmp_path / "raw/ACCAD/S1/walk_stageii.npz").read_bytes() == b"npz"
    manifest, prepared = prepare(config, archives)
    assert "ACCAD" not in prepared
