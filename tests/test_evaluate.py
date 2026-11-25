import json
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
    overlay = GhostOverlay(model, np.arange(7), np.arange(7, 9), np.arange(model.ngeom))
    scene = mujoco.MjvScene(model, maxgeom=32)
    robot_qpos = np.zeros(model.nq)
    robot_qpos[3] = 1
    original_qpos = robot_qpos.copy()
    overlay.update(scene, robot_qpos, follow_pose)
    assert scene.ngeom == model.ngeom
    assert all((scene.geoms[index].rgba[3] < 1 for index in range(scene.ngeom)))
    assert model_dimensions == (model.nbody, model.ngeom, model.nq, model.nv)
    np.testing.assert_array_equal(robot_qpos, original_qpos)
    for _ in range(20):
        frame = sample_start_frame(trainer.motion, 0)
        assert int(trainer.motion.clip_start[0]) <= int(frame) < int(trainer.motion.clip_end[0])


@pytest.mark.parametrize("native_rendering", [False, True])
def test_video_frame_count_and_clip_restart(tiny_trainer, monkeypatch, tmp_path, native_rendering):
    from types import SimpleNamespace
    import json
    from readable_mimic.evaluate import render_motion
    from readable_mimic.task import MAX_STEP_REWARD

    trainer = tiny_trainer
    model = mujoco.MjModel.from_xml_path(trainer.robot.mjcf_path)
    clip_end = int(trainer.motion.clip_end[0])
    monkeypatch.setattr(
        "readable_mimic.evaluate.sample_start_frame", lambda motion, index, start_frame=None: torch.tensor([clip_end - 1])
    )

    class PlaybackEnv:
        device = "cpu"
        resets = 0
        task = SimpleNamespace(
            config=trainer.task.config,
            termination_errors=lambda state, index: (torch.tensor([1.0]), torch.tensor([0.0])),
        )

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
            state = self.state()
            from readable_mimic.rotations import mat_to_quat

            self.simulation.data.qpos[0] = torch.cat((state[0], mat_to_quat(state[1]), state[4]), -1)[0]
            return {
                "done": torch.tensor([True]),
                "termination": torch.tensor([self.resets == 1]),
                "reward": torch.tensor([MAX_STEP_REWARD * 0.75]),
                "normalized_episode_return": torch.tensor([0.75]),
            }

    class Renderer:
        scene = mujoco.MjvScene(model, maxgeom=32)

        def update_scene(self, data, camera):
            self.scene.ngeom = 0

        def render(self):
            return np.zeros((32, 32, 3), dtype=np.uint8)

    class Writer:
        frame_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def append_data(self, frame):
            self.frame_count += 1

    class NativeRenderer:
        frames = 0

        def video_writer(self, output_path, fps):
            assert fps == 50
            return writer

        def render(self, robot_state, ghost_pose):
            self.frames += 1
            assert robot_state[0].shape == ghost_pose[0].shape
            return np.zeros((32, 32, 3), dtype=np.uint8)

    native_renderer = NativeRenderer() if native_rendering else None
    writer = Writer()
    monkeypatch.setattr("readable_mimic.evaluate.imageio.get_writer", lambda *args, **kwargs: writer)
    env = PlaybackEnv()
    ghost = GhostOverlay(model, np.arange(7), np.arange(7, 9), np.arange(model.ngeom))
    render_motion(
        env,
        trainer.actor,
        trainer.motion,
        0,
        trainer.config.hist,
        trainer.config.goal_frames,
        0.06,
        None if native_rendering else Renderer(),
        None if native_rendering else mujoco.MjData(model),
        None if native_rendering else ghost,
        True,
        tmp_path / "test.mp4",
        native_renderer=native_renderer,
    )
    assert writer.frame_count == 3
    if native_renderer is not None:
        assert native_renderer.frames == 3
    assert env.resets == 4
    summary = json.loads((tmp_path / "test.json").read_text())
    assert summary["sim_r"] == pytest.approx(0.75)
    assert summary["episode_return"] == 0.75
    assert summary["completed_episodes"] == 3
    assert summary["failure_count"] == 1
    failure = summary["first_failure"]
    assert failure["video_frame"] == 0
    assert failure["ref_frame"] == clip_end - int(trainer.motion.clip_start[0])
    assert failure["reasons"] == ["base_height_error"]


def test_ghost_excludes_ground_on_separate_body():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="terrain"><geom name="ground" type="plane" size="3 3 .1" rgba=".4 .4 .4 1"/></body>
    <body name="robot" pos="0 0 1"><freejoint/><geom name="robot_geom" type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ground_index = model.geom("ground").id
    robot_index = model.geom("robot_geom").id
    assert model.geom_bodyid[ground_index] != 0
    scene = mujoco.MjvScene(model, maxgeom=32)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        mujoco.MjvPerturb(),
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    before_count = scene.ngeom
    before_colors = [scene.geoms[index].rgba.copy() for index in range(before_count)]
    overlay = GhostOverlay(model, np.arange(7), [], [robot_index])
    pose = (torch.tensor([[0.0, 0.0, 1.0]]), torch.eye(3)[None], torch.empty(1, 0))
    overlay.update(scene, data.qpos, pose, append=True)
    assert scene.ngeom == before_count + 1
    assert scene.geoms[before_count].type == mujoco.mjtGeom.mjGEOM_SPHERE
    for index in range(before_count):
        np.testing.assert_array_equal(scene.geoms[index].rgba, before_colors[index])


def test_evaluation_selects_files_before_loading(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from readable_mimic import evaluate

    for name in ("cartwheel.csv", "walk.csv"):
        (tmp_path / name).touch()
    robot = SimpleNamespace(motion_glob="*.csv", load_motion=None)
    selected = []
    # Keep file discovery real and capture the exact constructor input.
    from readable_mimic.motion import ReferenceMotion

    def capture(files, kinematics, device, load_motion):
        selected.extend(files)

    capture.find_files = ReferenceMotion.find_files
    monkeypatch.setattr(evaluate, "ReferenceMotion", capture)
    evaluate.load_evaluation_motion(tmp_path, robot, None, "cpu", "cartwheel")
    assert [Path(path).name for path in selected] == ["cartwheel.csv"]


def test_fixed_evaluation_dr_matches_environment_context(tmp_path):
    from types import SimpleNamespace
    from readable_mimic.evaluate import apply_evaluation_dr
    from readable_mimic.fixed_dr import load_dr

    path = tmp_path / "dr.json"
    path.write_text(json.dumps({"domain_randomization": {
        "friction": 0.7, "ground_friction": 0.4,
        "joint_pos_offset": [0.01, -0.02], "center_of_mass_offset": [0.01, 0.02, -0.03],
        "mass_scale": 1.1, "inertia_scale": 0.95, "armature_scale": 1.2,
        "passive_damping_add": 0.03, "frictionloss_add": 0.07, "kp_scale": 0.9, "kd_scale": 1.15,
    }}))
    dr = load_dr(path, dict(joint_names=["left", "right"], friction=1., kp_scale=1., kd_scale=1.))
    applied = {}
    def apply(xi, randomize, ground_friction):
        applied.update(xi=xi.clone(), randomize=randomize, ground_friction=ground_friction)
    env = SimpleNamespace(xi=torch.zeros(1, 13), num_joints=2,
                          robot_description=SimpleNamespace(has_action_delay=False),
                          simulation=SimpleNamespace(apply_xi=apply))
    apply_evaluation_dr(env, dr)
    torch.testing.assert_close(env.xi, torch.tensor([[
        0.7, 0.01, -0.02, 0.01, 0.02, -0.03, 1.1, 0.95, 1.2, 0.03, 0.07, 0.9, 1.15,
    ]]))
    torch.testing.assert_close(env.xi, applied["xi"])
    assert applied["ground_friction"] == 0.4 and applied["randomize"] is False
    assert parse_args(["--ckpt", "last.pt", "--dr", str(path)]).dr == str(path)
    assert parse_args(["--ckpt", "last.pt"]).dr is None
