from readable_mimic.simulation_parameters import default_simulation_parameters

"CPU integration of the actual FK, motion, replay and gradient updates (no fake WM)."
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
import torch
from readable_mimic.config import load_config
from readable_mimic.task import flatten_state, unflatten_state
from readable_mimic.robot import RobotSpec
from readable_mimic.trainer import Trainer


@pytest.fixture
def tiny_trainer(tmp_path, monkeypatch):
    xml = '<mujoco><worldbody><body name="pelvis"><freejoint/>\n    <geom type="sphere" size="0.05" mass="1"/>\n    <body name="left_ankle_roll_link" pos="0 .1 -.2"><joint name="left_pitch_joint" axis="0 1 0" range="-2 2"/>\n    <geom type="sphere" size=".04" mass=".2"/></body>\n    <body name="right_ankle_roll_link" pos="0 -.1 -.2"><joint name="right_pitch_joint" axis="0 1 0" range="-2 2"/>\n    <geom type="sphere" size=".04" mass=".2"/></body></body></worldbody></mujoco>'
    model_path = tmp_path / "robot.xml"
    model_path.write_text(xml)
    motion = np.zeros((120, 9), dtype=np.float32)
    motion[:, 2] = 0.7
    motion[:, 3] = 1
    motion[:, 7] = 0.1 * np.sin(np.arange(120) * 0.1)
    motion[:, 8] = -0.1 * np.sin(np.arange(120) * 0.1)
    np.savetxt(tmp_path / "motion.csv", motion, delimiter=",")

    def load_motion(path):
        data = np.loadtxt(path, delimiter=",")
        return (data[:, :3], data[:, 3:7], data[:, 7:], 50.0)

    robot = RobotSpec(
        "tiny",
        str(model_path),
        ("left_pitch_joint", "right_pitch_joint"),
        ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link"),
        {".*": 0.0},
        {"left_pitch_joint": 0.2, "right_pitch_joint": 0.2},
        "pelvis",
        ((-0.01, 0.01),) * 3,
        lambda: SimpleNamespace(articulation=None),
        str(tmp_path),
        "*.csv",
        load_motion,
    )
    monkeypatch.setattr("readable_mimic.trainer.get_robot", lambda name: robot)
    config = load_config(Path(__file__).parents[1] / "config.json")
    config.device = "cpu"
    config.num_envs = 4
    config.motion_dir = str(tmp_path)
    config.hist = 3
    config.goal_frames = 2
    config.horizon = 3
    config.wm_window = 2
    config.wm_batch = 4
    config.pi_batch = 4
    config.buffer_steps = 16
    config.wm_hidden = "32,32"
    config.enc_hidden = "32,16"
    config.dec_hidden = "32,16"
    config.critic_epochs = 1
    config.critic_minibatches = 2
    trainer = Trainer(config)
    for step in range(10):
        frames = torch.arange(4) + step + 10
        state = trainer.motion.ref_state(frames)
        next_state = trainer.motion.ref_state(frames + 1)
        parameters = default_simulation_parameters(trainer.robot, trainer.device).expand(4, -1).clone()
        action = torch.zeros(4, 2)
        transition = dict(
            state=flatten_state(state),
            next_state=flatten_state(next_state),
            action=action,
            last_action=action,
            simulation_parameters=parameters,
            reward=trainer.task.reward(next_state, frames + 1, action, action),
            motion_frame_index=frames,
            episode_step=torch.full((4,), step),
            done=torch.zeros(4, dtype=torch.bool),
            transition_cut=torch.zeros(4, dtype=torch.bool),
        )
        trainer.buffer.add(transition)
        observation = trainer.task.observation(state, action, parameters, frames)
        trainer.actor.observation_normalizer.update(observation)
        trainer.actor.command_normalizer.update(trainer.motion.command(frames, config.goal_frames))
        inputs, targets, increments = trainer.world_model_normalization_data(
            state, next_state, action, action, parameters
        )
        trainer.world_model.input_normalizer.update(inputs)
        trainer.world_model.output_normalizer.update(targets)
        trainer.world_model.increment_normalizer.update(increments)
    return trainer


def test_fk_matches_mujoco_and_has_gradients(tiny_trainer):
    trainer = tiny_trainer
    model = mujoco.MjModel.from_xml_path(trainer.robot.mjcf_path)
    data = mujoco.MjData(model)
    state = trainer.motion.ref_state(torch.tensor([12]))
    state = list(state)
    state[4] = state[4].detach().requires_grad_()
    data.qpos[:3] = state[0][0].numpy()
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = state[4][0].detach().numpy()
    data.qvel[:3] = state[2][0].numpy()
    data.qvel[3:6] = state[3][0].numpy()
    data.qvel[6:] = state[5][0].numpy()
    mujoco.mj_forward(model, data)
    positions, rotations, linear, angular = trainer.kinematics.forward(*state)
    for index, name in enumerate(trainer.robot.tracked_bodies):
        body = model.body(name).id
        np.testing.assert_allclose(positions[0, index].detach(), data.xpos[body], atol=1e-06)
        np.testing.assert_allclose(rotations[0, index].detach(), data.xmat[body].reshape(3, 3), atol=1e-06)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        offset = data.xpos[body] - data.xipos[body]
        origin_linear = velocity[3:] + np.cross(velocity[:3], offset)
        np.testing.assert_allclose(angular[0, index].detach(), velocity[:3], atol=1e-06)
        np.testing.assert_allclose(linear[0, index].detach(), origin_linear, atol=1e-06)
    rotations.sum().backward()
    assert torch.isfinite(state[4].grad).all()


def test_full_training_updates_and_checkpoint(tiny_trainer, tmp_path):
    trainer = tiny_trainer
    before_world_model = [value.detach().clone() for value in trainer.world_model.parameters()]
    losses = trainer.update_world_model()
    assert all((torch.isfinite(value).all() for value in losses))
    assert any(
        (not torch.equal(old, new) for old, new in zip(before_world_model, trainer.world_model.parameters()))
    )
    before_actor = [value.detach().clone() for value in trainer.actor.parameters()]
    before_critic = [value.detach().clone() for value in trainer.critic.parameters()]
    frozen_model = [value.detach().clone() for value in trainer.world_model.parameters()]
    metrics = trainer.update_actor()
    assert all((torch.isfinite(value).all() for value in metrics))
    assert any((not torch.equal(old, new) for old, new in zip(before_actor, trainer.actor.parameters())))
    assert any((not torch.equal(old, new) for old, new in zip(before_critic, trainer.critic.parameters())))
    assert all((torch.equal(old, new) for old, new in zip(frozen_model, trainer.world_model.parameters())))
    path = tmp_path / "last.pt"
    trainer.save(path)
    expected = trainer.actor.action_decoder[-1].weight.detach().clone()
    with torch.no_grad():
        trainer.actor.action_decoder[-1].weight.add_(1)
    trainer.load(path)
    torch.testing.assert_close(trainer.actor.action_decoder[-1].weight, expected)


def test_state_and_replay_boundaries(tiny_trainer):
    trainer = tiny_trainer
    state = trainer.motion.ref_state(torch.tensor([10]))
    restored = unflatten_state(flatten_state(state))
    for original, result in zip(state, restored):
        torch.testing.assert_close(original, result)
    trainer.buffer.data["episode_step"].zero_()
    start, history = trainer.buffer.starts_at(torch.tensor([5]), torch.tensor([0]), 3)
    torch.testing.assert_close(history["state"][:, 0], history["state"][:, -1])
    trainer.buffer.data["transition_cut"].fill_(True)
    windows = trainer.buffer.sample_windows(4, 2)
    assert windows["mask"][:, 0].eq(1).all()
    assert windows["mask"][:, 1].eq(0).all()


def test_collect_and_training_loop(tiny_trainer, tmp_path, monkeypatch):
    """Real native MuJoCo CPU collection through the same Trainer orchestration."""
    trainer = tiny_trainer

    class NativeEnvironment:
        def __init__(self):
            self.model = mujoco.MjModel.from_xml_path(trainer.robot.mjcf_path)
            self.model.opt.timestep = 0.005
            self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            self.model.dof_armature[6:] = 0.01
            self.data = [mujoco.MjData(self.model) for _ in range(4)]
            self.episode_step = torch.zeros(4, dtype=torch.long)
            self.motion_frame_index = torch.full((4,), 10, dtype=torch.long)
            self.last_action = torch.zeros(4, 2)
            self.parameters = (
                default_simulation_parameters(trainer.robot, trainer.device).expand(4, -1).clone()
            )
            for data in self.data:
                data.qpos[:3] = [0, 0, 0.7]
                data.qpos[3:7] = [1, 0, 0, 0]

        def state(self):
            from readable_mimic.rotations import quat_to_mat

            positions = torch.tensor(np.stack([data.qpos for data in self.data]), dtype=torch.float32)
            velocities = torch.tensor(np.stack([data.qvel for data in self.data]), dtype=torch.float32)
            rotations = quat_to_mat(positions[:, 3:7])
            angular = (rotations @ velocities[:, 3:6, None]).squeeze(-1)
            return (
                positions[:, :3],
                rotations,
                velocities[:, :3],
                angular,
                positions[:, 7:],
                velocities[:, 6:],
            )

        def observe(self):
            return trainer.task.observation(
                self.state(), self.last_action, self.parameters, self.motion_frame_index
            )

        def step(self, action):
            state = self.state()
            last_action = self.last_action.clone()
            frames = self.motion_frame_index.clone()
            steps = self.episode_step.clone()
            for index, data in enumerate(self.data):
                for _ in range(4):
                    data.qfrc_applied[6:] = 5 * (0.2 * action[index].numpy() - data.qpos[7:]) - data.qvel[6:]
                    mujoco.mj_step(self.model, data)
            next_state = self.state()
            self.motion_frame_index += 1
            self.episode_step += 1
            self.last_action = action.clone()
            return dict(
                state=flatten_state(state),
                next_state=flatten_state(next_state),
                action=action,
                last_action=last_action,
                simulation_parameters=self.parameters,
                motion_frame_index=frames,
                episode_step=steps,
                done=torch.zeros(4, dtype=torch.bool),
                transition_cut=torch.zeros(4, dtype=torch.bool),
                reward=trainer.task.reward(next_state, frames + 1, action, last_action),
            )

    monkeypatch.setattr(
        trainer, "create_environment", lambda: setattr(trainer, "environment", NativeEnvironment())
    )
    trainer.config.warmup_steps = 4
    trainer.config.wm_batches = 1
    trainer.config.iterations = 2
    trainer.config.log_every = 1
    trainer.config.ckpt_every = 1
    trainer.config.log_dir = str(tmp_path / "run")
    trainer.run()
    assert trainer.iteration == 2
    assert (tmp_path / "run/last.pt").is_file()
    assert len((tmp_path / "run/metrics.jsonl").read_text().splitlines()) == 2


def test_simulation_parameters_match_applied_physics(tiny_trainer):
    from readable_mimic.simulation_parameters import (
        initialize_simulation_parameters,
        randomize_joint_pos_offsets,
    )

    trainer = tiny_trainer
    default_body_positions = torch.zeros(3, 3)
    simulation = SimpleNamespace(
        model=SimpleNamespace(geom_friction=torch.zeros(4, 3, 3), body_ipos=torch.zeros(4, 3, 3)),
        mj_model=SimpleNamespace(geom_bodyid=np.array([0, 1, 2])),
        expand_model_fields=lambda names: None,
        get_default_field=lambda name: default_body_positions,
    )
    environment = SimpleNamespace(
        robot_description=trainer.robot,
        simulation=simulation,
        num_envs=4,
        device="cpu",
        robot=SimpleNamespace(
            indexing=SimpleNamespace(geom_ids=torch.tensor([1, 2]), body_ids=torch.tensor([1, 2])),
            find_bodies=lambda name: ([0], [name]),
        ),
    )
    parameters = initialize_simulation_parameters(environment)
    torch.testing.assert_close(simulation.model.geom_friction[:, 1, 0], parameters[:, 0])
    torch.testing.assert_close(simulation.model.geom_friction[:, 2, 0], parameters[:, 0])
    torch.testing.assert_close(simulation.model.body_ipos[:, 1], parameters[:, 3:6])
    before = parameters.clone()
    randomize_joint_pos_offsets(parameters, torch.tensor([1, 3]), trainer.robot)
    torch.testing.assert_close(parameters[[0, 2]], before[[0, 2]])
    torch.testing.assert_close(parameters[:, 0], before[:, 0])
    torch.testing.assert_close(parameters[:, 3:], before[:, 3:])
    assert parameters[:, 1:3].abs().le(0.01).all()


def test_observation_fields_and_context_normalization(tiny_trainer):
    trainer = tiny_trainer
    state = trainer.motion.ref_state(torch.tensor([12]))
    parameters = default_simulation_parameters(trainer.robot, "cpu")[None].clone()
    parameters[:, 1:3] = torch.tensor([[0.004, -0.006]])
    last_action = torch.tensor([[0.2, -0.3]])
    observation = trainer.task.observation(state, last_action, parameters, torch.tensor([12]))
    torch.testing.assert_close(observation[:, :2], state[4] - parameters[:, 1:3])
    torch.testing.assert_close(observation[:, 10:12], last_action)
    normalized = trainer.normalize_parameters(parameters)
    torch.testing.assert_close(normalized[:, 1:3], torch.tensor([[0.4, -0.6]]))
    assert observation.shape == (1, 18)


def test_reward_detail_names_and_total(tiny_trainer):
    from readable_mimic.motion import CONTROL_DT

    trainer = tiny_trainer
    frames = torch.tensor([12, 13])
    state = trainer.motion.ref_state(frames)
    action = torch.tensor([[0.2, -0.1], [0.3, -0.2]])
    last_action = torch.zeros_like(action)
    reward, details = trainer.task.reward(state, frames, action, last_action, return_details=True)
    assert set(details) == {
        "tracking_reward",
        "body_pos_error",
        "body_rot_error",
        "body_lin_vel_error",
        "body_ang_vel_error",
        "base_pos_error",
        "base_rot_error",
        "action_rate_penalty",
        "joint_limit_penalty",
    }
    reward_config = trainer.task.config
    reconstructed_reward = details["tracking_reward"] + CONTROL_DT * (
        reward_config.action_rate_weight * details["action_rate_penalty"]
        + reward_config.joint_limit_weight * details["joint_limit_penalty"]
    )
    torch.testing.assert_close(reward, reconstructed_reward)
    torch.testing.assert_close(reward, trainer.task.reward(state, frames, action, last_action))


def test_data_names_follow_diagram():
    import ast

    source_directory = Path(__file__).parents[1] / "src/readable_mimic"
    forbidden_keys = {"track", "e_body_pos", "e_body_ori", "e_root_pos", "action_rate", "joint_limit"}
    for source_file in source_directory.rglob("*.py"):
        for node in ast.walk(ast.parse(source_file.read_text(encoding="utf-8"))):
            identifier = None
            if isinstance(node, ast.Name):
                identifier = node.id
            elif isinstance(node, ast.arg):
                identifier = node.arg
            elif isinstance(node, ast.Attribute):
                identifier = node.attr
            if identifier:
                assert not identifier.startswith(("reference_", "e_", "root_")), (source_file, identifier)
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        assert key.value not in forbidden_keys, (source_file, key.value)
