import json
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")
from test_pipeline import tiny_trainer as tiny_trainer
from readable_mimic.deployment import inputs
from readable_mimic import data_transforms as transforms
from readable_mimic.rotations import so3_exp
from readable_mimic.sim import default_xi


def test_numpy_inputs_match_training(tiny_trainer):
    trainer = tiny_trainer
    state = trainer.motion.ref_state(torch.arange(len(trainer.motion.base_pos_w)))
    motion = dict(zip(
        ("base_pos_w", "base_rot_mat_w", "base_lin_vel_w", "base_ang_vel_w", "joint_pos", "joint_vel"),
        (value.numpy() for value in state),
    ))
    inverse_rotation = state[1].transpose(-1, -2)
    motion["base_lin_vel"] = (inverse_rotation @ state[2].unsqueeze(-1)).squeeze(-1).numpy()
    motion["base_ang_vel"] = (inverse_rotation @ state[3].unsqueeze(-1)).squeeze(-1).numpy()
    end = int(trainer.motion.clip_end[0])
    for frame in (12, end - 1):
        expected = transforms.compute_command(
            torch.tensor([frame]), trainer.command_frame_offsets, trainer.motion
        )
        np.testing.assert_allclose(
            inputs.compute_command(motion, frame, end, trainer.command_frames), expected.numpy(), atol=1e-6
        )
        state = list(trainer.motion.ref_state(torch.tensor([frame])))
        state[1] = so3_exp(torch.tensor([[0.2, -0.3, 0.6]]))
        last_action = torch.tensor([[0.1, -0.2]])
        expected = transforms.compute_observation(
            tuple(state),
            last_action,
            default_xi(trainer.robot, "cpu")[None],
            torch.tensor([frame]),
            trainer.motion,
            trainer.default_joint_pos,
        )
        angular_velocity = (state[1].transpose(-1, -2) @ state[3][..., None])[0, :, 0].numpy()
        actual = inputs.compute_observation(
            state[1][0].numpy(),
            angular_velocity,
            state[4][0].numpy(),
            state[5][0].numpy(),
            last_action[0].numpy(),
            motion["base_rot_mat_w"][frame],
            trainer.default_joint_pos.numpy(),
        )
        np.testing.assert_allclose(actual, expected[0].numpy(), atol=1e-6)


def test_export_onnx_and_play_without_training_imports(tiny_trainer, tmp_path, monkeypatch):
    import onnxruntime as ort
    from readable_mimic.deployment import export_onnx
    from readable_mimic.deployment.play_mujoco import create_model
    import mujoco

    trainer = tiny_trainer
    trainer.config.warmup_steps = 4
    extra_directory = tmp_path / "extra_motions"
    extra_directory.mkdir()
    (extra_directory / "second.csv").write_bytes((tmp_path / "motion.csv").read_bytes())
    trainer.config.motion_dir = [str(tmp_path), str(extra_directory)]
    names = trainer.robot.joint_names
    robot = replace(
        trainer.robot,
        joint_stiffness={name: 5.0 for name in names},
        joint_damping={name: 1.0 for name in names},
        joint_effort_limit={name: 20.0 for name in names},
        joint_armature={name: 0.01 for name in names},
    )
    monkeypatch.setattr(export_onnx, "get_robot", lambda name: robot)
    trainer.actor.observation_normalizer.mean.fill_(0.03)
    trainer.actor.command_normalizer.var.fill_(0.8)
    trainer.actor_average.update_parameters(trainer.actor)
    with torch.no_grad():
        trainer.actor.action_decoder[-1].bias.add_(0.1)
    checkpoint = tmp_path / "last.pt"
    trainer.save(checkpoint)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(vars(trainer.config)), encoding="utf-8")
    model_path = export_onnx.export_policy(checkpoint, config_path, tmp_path / "policy.onnx")
    settings = json.loads(model_path.with_suffix(".json").read_text())
    assert settings["actor_weights"] == "ema"
    with np.load(model_path.with_suffix(".motion.npz")) as archive:
        assert archive["names"].tolist() == ["motion", "second"]
    model = create_model(settings, robot.mjcf_path)
    np.testing.assert_allclose(model.dof_armature[6:], 0.01)
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], 1)
    np.testing.assert_allclose(model.actuator_biasprm, 0)
    np.testing.assert_allclose(model.actuator_forcerange, np.tile([-20, 20], (2, 1)))
    data = mujoco.MjData(model)
    data.ctrl[:] = [5.0, -7.0]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, [5.0, -7.0])

    history = torch.randn(1, trainer.config.hist, trainer.observation_dim)
    command = torch.randn(1, transforms.compute_command_dim(robot.num_joints, trainer.command_frames))
    actual = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"]).run(
        ["action"], {"observation_history": history.numpy(), "command": command.numpy()}
    )[0]
    with torch.no_grad():
        expected = trainer.actor_average.module(history, command).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
    script = """
import sys
from readable_mimic.deployment.play_mujoco import play
import numpy as np
reads = {}
original_getitem = np.lib.npyio.NpzFile.__getitem__
def tracked_read(archive, name):
    reads[name] = reads.get(name, 0) + 1
    return original_getitem(archive, name)
np.lib.npyio.NpzFile.__getitem__ = tracked_read
summaries = play(sys.argv[1], duration=0.1, headless=True)
assert reads and all(count == 1 for count in reads.values()), reads
np.lib.npyio.NpzFile.__getitem__ = original_getitem
assert summaries[0]['policy_steps'] == 5
assert summaries[0]['physics_steps'] == 20
assert summaries[0]['work_ms'] >= 0
assert summaries[0]['rtf'] > 0
import json
from pathlib import Path
delay_path = Path(sys.argv[1]).with_suffix(".dr.json")
delay_path.write_text(json.dumps({"domain_randomization": {"action_delay_ms": 10.0}}))
delayed = play(sys.argv[1], duration=0.1, headless=True, dr_path=delay_path)
assert delayed[0]["policy_steps"] == 5
assert delayed[0]["physics_steps"] == 20
from types import SimpleNamespace
from contextlib import nullcontext
from mujoco import viewer
from readable_mimic.deployment import play_mujoco as playback
class Clock:
    now = 0.0
    def perf_counter(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds + 0.003
clock = Clock()
workload = 0.007
class Viewer:
    cam = SimpleNamespace(lookat=__import__('numpy').zeros(3))
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def is_running(self): return True
    def lock(self): return nullcontext()
    def close(self):
        self.closed = True
        assert not self.data.ctrl.any()
    def sync(self, state_only=False):
        assert state_only
        clock.now += workload
instances = []
def launch(model, data):
    instance = Viewer()
    instance.data = data
    instance.closed = False
    instances.append(instance)
    return instance
viewer.launch_passive = launch
playback.time = clock
paced = play(sys.argv[1], duration=0.1)
assert abs(paced[0]['wall_seconds'] - 0.103) < 1e-6, paced
assert paced[0]['work_ms'] == 7.0
clock = Clock()
workload = 0.035
def unexpected_sleep(seconds):
    raise AssertionError('Do not sleep when the control loop is behind schedule')
clock.sleep = unexpected_sleep
playback.time = clock
busy = play(sys.argv[1], duration=0.1)
assert abs(busy[0]['wall_seconds'] - 0.175) < 1e-6, busy
assert busy[0]['work_ms'] == 35.0
assert all(instance.closed for instance in instances)
Viewer.is_running = lambda self: False
assert play(sys.argv[1], duration=0.1) == []
assert instances[-1].closed
Viewer.is_running = lambda self: True
def interrupt(self, state_only=False):
    raise KeyboardInterrupt
Viewer.sync = interrupt
try:
    play(sys.argv[1], duration=0.1)
except KeyboardInterrupt:
    pass
else:
    raise AssertionError('Expected interruption')
assert instances[-1].closed
playback.play = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
sys.argv = ['play_mujoco']
playback.main()
assert 'torch' not in sys.modules
assert not any(name.startswith(('readable_mimic.env', 'mjlab', 'genesis')) for name in sys.modules)
print('Pure ONNX Runtime + native MuJoCo passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(model_path)],
        timeout=60,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_fixed_deployment_dr(tmp_path):
    import mujoco
    from readable_mimic.deployment.play_mujoco import load_dr, apply_dr

    settings = dict(joint_names=["joint"], friction=1.0, kp_scale=1.0, kd_scale=1.0)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="ground" type="plane" size="1 1 .1"/>'
        '<body name="torso_link"><joint name="joint" armature=".1" damping=".2" frictionloss=".3"/>'
        '<geom type="sphere" size=".1" mass="2"/></body></worldbody></mujoco>'
    )
    mass, inertia = model.body_mass.copy(), model.body_inertia.copy()
    path = tmp_path / "dr.json"
    path.write_text(json.dumps({"domain_randomization": {
        "mass_scale": 1.1, "inertia_scale": 0.9, "armature_scale": 1.2,
        "passive_damping_add": 0.04, "frictionloss_add": 0.08,
        "kp_scale": 0.85, "kd_scale": 0.95, "friction": 0.7,
        "ground_friction": 0.3, "center_of_mass_offset": [0.01, -0.02, 0.03],
        "joint_pos_offset": [0.005], "action_delay_ms": 10.0,
    }}))
    dr = load_dr(path, settings)
    apply_dr(model, settings, dr)
    np.testing.assert_allclose(model.body_mass, mass * 1.1)
    np.testing.assert_allclose(model.body_inertia, inertia * 1.1 * 0.9)
    np.testing.assert_allclose(model.body_ipos[1], [0.01, -0.02, 0.03])
    np.testing.assert_allclose(model.dof_armature, [0.12])
    np.testing.assert_allclose(model.dof_damping, [0.24])
    np.testing.assert_allclose(model.dof_frictionloss, [0.38])
    np.testing.assert_allclose(model.geom_friction[:, 0], [0.3, 0.7])
    assert dr["kp_scale"] == 0.85 and dr["kd_scale"] == 0.95
    for invalid in ({"kp_scale": 0}, {"mass_scale": -1}, {"friction": float("nan")},
                    {"joint_pos_offset": [0, 0]}, {"kp_scale_range": [0.8, 1.2]}, {"action_delay_ms": 1}):
        path.write_text(json.dumps({"domain_randomization": invalid}))
        with pytest.raises(ValueError):
            load_dr(path, settings)
