import mujoco
import numpy as np
import pytest
import torch
from test_pipeline import tiny_trainer as tiny_trainer
from readable_mimic import data_transforms as transforms
from readable_mimic.evaluate import (
    GhostOverlay,
    parse_args,
    select_motion_indices,
    select_actor_weights,
    sample_start_frame,
)
from readable_mimic.rotations import so3_exp


def test_cli_selection_and_ema_weights():
    arguments = parse_args(
        [
            "--checkpoint",
            "last.pt",
            "--motions",
            "walk|dance.csv",
            "--duration",
            "12",
            "--ghost-base",
            "reference",
        ]
    )
    assert arguments.duration == 12
    assert arguments.ghost_base == "reference"
    assert select_motion_indices(["dance", "walk"], "walk|dance.csv") == [1, 0]
    with pytest.raises(ValueError, match="Unknown motion"):
        select_motion_indices(["dance"], "walk")
    with pytest.raises(SystemExit):
        parse_args(["--ckpt", "last.pt", "--duration", "0"])
    checkpoint = dict(
        actor={"weight": torch.tensor([1.0])},
        actor_average={"n_averaged": torch.tensor(1), "module.weight": torch.tensor([2.0])},
    )
    torch.testing.assert_close(select_actor_weights(checkpoint)["weight"], torch.tensor([2.0]))
    checkpoint["actor_average"]["n_averaged"].zero_()
    torch.testing.assert_close(select_actor_weights(checkpoint)["weight"], torch.tensor([1.0]))


def test_ghost_pose_and_scene_without_physics_changes(tiny_trainer):
    trainer = tiny_trainer
    ref_state = trainer.motion.ref_state(torch.tensor([12]))
    robot_state = list(ref_state)
    robot_state[0] = torch.tensor([[2.0, 3.0, 0.8]])
    robot_state[1] = so3_exp(torch.tensor([[0.0, 0.0, 1.0]]))
    fixed_pose = transforms.compute_ghost_pose(ref_state, robot_state, False)
    follow_pose = transforms.compute_ghost_pose(ref_state, robot_state, True)
    torch.testing.assert_close(fixed_pose[0], ref_state[0])
    torch.testing.assert_close(follow_pose[0][:, :2], robot_state[0][:, :2])
    torch.testing.assert_close(follow_pose[0][:, 2], ref_state[0][:, 2])
    torch.testing.assert_close(follow_pose[1], robot_state[1])
    model = mujoco.MjModel.from_xml_path(trainer.robot.mjcf_path)
    model_dimensions = (model.nbody, model.ngeom, model.nq, model.nv)
    overlay = GhostOverlay(model, np.arange(7), np.arange(7, 9))
    scene = mujoco.MjvScene(model, maxgeom=32)
    robot_qpos = np.zeros(model.nq)
    robot_qpos[3] = 1
    original_qpos = robot_qpos.copy()
    overlay.update(scene, robot_qpos, follow_pose)
    assert scene.ngeom == model.ngeom
    assert all(scene.geoms[index].rgba[3] < 1 for index in range(scene.ngeom))
    assert model_dimensions == (model.nbody, model.ngeom, model.nq, model.nv)
    np.testing.assert_array_equal(robot_qpos, original_qpos)
    for _ in range(20):
        frame = sample_start_frame(trainer.motion, 0)
        assert int(trainer.motion.clip_start[0]) <= int(frame) < int(trainer.motion.clip_end[0])


def test_playback_duration_and_clip_restart(tiny_trainer, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from readable_mimic.evaluate import play_motion

    trainer = tiny_trainer
    model = mujoco.MjModel.from_xml_path(trainer.robot.mjcf_path)
    clip_end = int(trainer.motion.clip_end[0])
    monkeypatch.setattr(
        "readable_mimic.evaluate.sample_start_frame", lambda motion, index: torch.tensor([clip_end - 1])
    )
    monkeypatch.setattr("readable_mimic.evaluate.time.sleep", lambda seconds: None)

    class PlaybackEnvironment:
        device = "cpu"
        resets = 0

        def __init__(self):
            self.motion_frame_index = torch.tensor([clip_end - 1])
            self.simulation = SimpleNamespace(
                mj_model=model,
                data=SimpleNamespace(qpos=torch.zeros(1, model.nq), qvel=torch.zeros(1, model.nv)),
            )

        def reset(self, indices, frames):
            self.resets += 1
            self.motion_frame_index = frames.clone()

        def observe(self):
            return torch.zeros(1, trainer.observation_dim)

        def state(self):
            return trainer.motion.ref_state(self.motion_frame_index)

        def step(self, action):
            self.motion_frame_index += 1
            # Native MuJoCo qpos is position / quaternion / joints, without velocities.
            state = self.state()
            from readable_mimic.rotations import mat_to_quat

            self.simulation.data.qpos[0] = torch.cat((state[0], mat_to_quat(state[1]), state[4]), -1)[0]
            return {"done": torch.tensor([True])}

    class Viewer:
        sync_count = 0
        user_scn = mujoco.MjvScene(model, maxgeom=32)
        cam = SimpleNamespace(lookat=np.zeros(3))

        def is_running(self):
            return True

        def lock(self):
            return nullcontext()

        def sync(self):
            self.sync_count += 1

    environment = PlaybackEnvironment()
    viewer = Viewer()
    ghost = GhostOverlay(model, np.arange(7), np.arange(7, 9))
    assert play_motion(
        environment,
        trainer.actor,
        trainer.motion,
        0,
        trainer.config.hist,
        trainer.config.goal_frames,
        0.06,
        viewer,
        mujoco.MjData(model),
        ghost,
        True,
    )
    assert viewer.sync_count == 2
    assert environment.resets == 4
