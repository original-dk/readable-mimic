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
    motion = {
        name: getattr(trainer.motion, name).numpy()
        for name in ("base_pos_w", "base_rot_mat_w", "base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel")
    }
    end = int(trainer.motion.clip_end[0])
    for frame in (12, end - 1):
        expected = transforms.compute_command(
            torch.tensor([frame]), trainer.command_frame_offsets, trainer.motion
        )
        np.testing.assert_allclose(
            inputs.command(motion, frame, end, trainer.command_frames), expected.numpy(), atol=1e-6
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
        actual = inputs.observation(
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
    from readable_mimic.deployment.play_mujoco import create_model, Ghost
    import mujoco

    trainer = tiny_trainer
    trainer.config.warmup_steps = 4
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
    model = create_model(settings, robot.mjcf_path)
    ghost = Ghost(model, np.arange(7), np.arange(7, 9))
    scene = mujoco.MjvScene(model, maxgeom=64)
    qpos_before = model.qpos0.copy()
    with np.load(model_path.with_suffix(".motion.npz")) as motion:
        ghost.update(scene, model.qpos0, motion, 12, True)
    assert scene.ngeom > 0
    np.testing.assert_array_equal(model.qpos0, qpos_before)
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
summaries = play(sys.argv[1], duration=0.1, headless=True)
assert summaries[0]['policy_steps'] == 5
assert summaries[0]['physics_steps'] == 20
assert 'torch' not in sys.modules
assert not any(name.startswith(('readable_mimic.env', 'mjlab', 'genesis')) for name in sys.modules)
print('Pure ONNX Runtime + native MuJoCo passed')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(model_path)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )
    assert result.returncode == 0, result.stdout + result.stderr
