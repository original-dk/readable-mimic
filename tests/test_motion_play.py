from contextlib import nullcontext
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
from mujoco import viewer


@pytest.mark.parametrize("exit_mode", ["normal", "close", "interrupt"])
def test_kinematic_play_quaternion_and_exit(tmp_path, monkeypatch, exit_mode):
    module = runpy.run_path(str(Path(__file__).parents[1] / "tools/play_motion.py"))
    model_path = tmp_path / "g1.xml"
    joints = "".join(
        f'<body name="link_{index}"><joint name="joint_{index}"/><geom type="sphere" size=".01"/></body>'
        for index in range(29)
    )
    model_path.write_text(
        '<mujoco><worldbody><body name="pelvis"><freejoint/>'
        '<geom type="sphere" size=".1"/>' + joints + "</body></worldbody></mujoco>"
    )
    motion_path = tmp_path / "motion.csv"
    motion = np.zeros((2, 36))
    motion[:, :3] = [[1, 2, 0.8], [2, 3, 0.9]]
    motion[:, 3:7] = [[0, 0, np.sqrt(0.5), np.sqrt(0.5)], [0, 0, 1, 0]]
    motion[:, 7:] = 0.01
    np.savetxt(motion_path, motion, delimiter=",")
    snapshots = []

    class Viewer:
        cam = SimpleNamespace()
        closed = False

        def lock(self):
            return nullcontext()

        def is_running(self):
            return exit_mode != "close"

        def sync(self, state_only):
            assert state_only
            if exit_mode == "interrupt":
                raise KeyboardInterrupt
            snapshots.append((self.data.qpos.copy(), self.data.xmat[1].reshape(3, 3).copy()))

        def close(self):
            self.closed = True

    window = Viewer()

    def launch(model, data):
        window.data = data
        return window

    monkeypatch.setattr(viewer, "launch_passive", launch)
    monkeypatch.setattr(module["time"], "sleep", lambda seconds: None)
    if exit_mode == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            module["play"](motion_path, model_path)
    else:
        module["play"](motion_path, model_path)
    assert window.closed
    if exit_mode == "normal":
        assert len(snapshots) == 2
        np.testing.assert_allclose(snapshots[0][0][:3], [1, 2, 0.8])
        np.testing.assert_allclose(snapshots[0][0][3:7], [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
        np.testing.assert_allclose(snapshots[0][1], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-14)
        np.testing.assert_allclose(snapshots[1][1], np.diag([-1, -1, 1]), atol=1e-14)
        np.testing.assert_allclose(snapshots[1][0][7:], 0.01)
