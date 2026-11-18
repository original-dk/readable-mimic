from readable_mimic.sim import default_xi
from readable_mimic import data_transforms as transforms

"CPU integration of the actual FK, motion, replay and gradient updates (no fake WM)."
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
import torch
from readable_mimic.config import load_config
from readable_mimic.robot import RobotSpec
from readable_mimic.trainer import Trainer


def assert_checkpoint_values_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_checkpoint_values_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            assert_checkpoint_values_equal(actual_value, expected_value)
    else:
        assert actual == expected


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
        xi = default_xi(trainer.robot, trainer.device).expand(4, -1).clone()
        action = torch.zeros(4, 2)
        transition = dict(
            state=transforms.flatten_state(state),
            next_state=transforms.flatten_state(next_state),
            action=action,
            last_action=action,
            xi=xi,
            reward=trainer.task.reward(next_state, frames + 1, action, action),
            motion_frame_index=frames,
            episode_step=torch.full((4,), step),
            done=torch.zeros(4, dtype=torch.bool),
            transition_cut=torch.zeros(4, dtype=torch.bool),
        )
        trainer.buffer.add(transition)
        observation = transforms.compute_observation(
            state, action, xi, frames, trainer.motion, trainer.default_joint_pos
        )
        trainer.actor.observation_normalizer.update(observation)
        trainer.actor.command_normalizer.update(
            transforms.compute_command(frames, trainer.command_frame_offsets, trainer.motion)
        )
        inputs, targets, increments = transforms.compute_world_model_training_data(
            trainer.world_model,
            state,
            next_state,
            action,
            action,
            xi,
            trainer.xi_center,
            trainer.xi_scale,
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


def test_separately_compiled_world_model_outputs_and_gradients(tiny_trainer):
    updates = tiny_trainer.updates
    world_model = tiny_trainer.world_model
    windows = tiny_trainer.buffer.sample_windows(tiny_trainer.config.wm_batch)
    with torch.no_grad():
        world_model.network[-1].weight.normal_(0, 0.01)
    # CPU AOTAutograd checks both compiled graphs and their composed backward.
    compiled_rollout = torch.compile(updates.world_model_rollout, backend="aot_eager", fullgraph=True, dynamic=False)
    compiled_loss = torch.compile(updates.world_model_loss, backend="aot_eager", fullgraph=True, dynamic=False)
    for _ in range(2):
        expected_states = updates.world_model_rollout(
            windows["state"], windows["action"], windows["last_action"], windows["xi"]
        )
        expected_loss, expected_error = updates.world_model_loss(
            expected_states, windows["next_state"], windows["mask"]
        )
        expected_gradients = torch.autograd.grad(expected_loss, tuple(world_model.parameters()))
        actual_states = compiled_rollout(
            windows["state"], windows["action"], windows["last_action"], windows["xi"]
        )
        actual_loss, actual_error = compiled_loss(actual_states, windows["next_state"], windows["mask"])
        actual_gradients = torch.autograd.grad(actual_loss, tuple(world_model.parameters()))
        for actual, expected in zip(actual_states, expected_states):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_loss, expected_loss)
        torch.testing.assert_close(actual_error, expected_error)
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected)
        # Compiled methods must read current weights and normalizer buffers.
        with torch.no_grad():
            world_model.network[-1].bias.add_(0.001)
            world_model.input_normalizer.mean.add_(0.01)
            world_model.increment_normalizer.var.mul_(1.01)


def test_compiled_actor_step_and_reward_gradients(tiny_trainer):
    updates = tiny_trainer.updates
    start, history = updates.sample_valid_starts(tiny_trainer.config.pi_batch)
    initial_state = transforms.unflatten_state(start["state"])
    xi = start["xi"]
    normalized_xi = transforms.compute_normalized_xi(xi, updates.xi_center, updates.xi_scale)
    initial_history = transforms.compute_replay_observation_history(
        history, updates.motion, updates.default_joint_pos, updates.observation_noise_limits
    )
    initial_command = transforms.compute_command(
        start["motion_frame_index"], updates.command_frame_offsets, updates.motion
    )
    with torch.no_grad():
        updates.world_model.network[-1].weight.normal_(0, 0.01)
    for parameter in updates.world_model.parameters():
        parameter.requires_grad_(False)
    compiled_step = torch.compile(updates.imagine_step, backend="aot_eager", fullgraph=True, dynamic=False)
    compiled_reward = torch.compile(
        updates.reward_and_termination, backend="aot_eager", fullgraph=True, dynamic=False
    )

    def rollout(step, reward):
        state, observation_history, command = initial_state, initial_history, initial_command
        frames, last_action = start["motion_frame_index"], start["last_action"]
        objective = torch.zeros(())
        for _ in range(2):
            previous_action = last_action
            state, last_action, observation_history, frames, command = step(
                state, observation_history, last_action, frames, xi, normalized_xi, command
            )
            rewards, termination = reward(state, frames, last_action, previous_action)
            objective = objective + (rewards * (1 - termination)).sum() + 0.01 * last_action.square().sum()
        gradients = torch.autograd.grad(objective, tuple(updates.actor.parameters()))
        return state, objective, gradients

    torch.manual_seed(47)
    expected_state, expected_objective, expected_gradients = rollout(
        updates.imagine_step, updates.reward_and_termination
    )
    torch.manual_seed(47)
    actual_state, actual_objective, actual_gradients = rollout(compiled_step, compiled_reward)
    for actual, expected in zip(actual_state, expected_state):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_objective, expected_objective)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_full_training_updates_and_checkpoint(tiny_trainer, tmp_path):
    trainer = tiny_trainer
    before_world_model = [value.detach().clone() for value in trainer.world_model.parameters()]
    losses = trainer.updates.update_world_model()
    assert all((torch.isfinite(value).all() for value in losses))
    assert any(
        (not torch.equal(old, new) for old, new in zip(before_world_model, trainer.world_model.parameters()))
    )
    before_actor = [value.detach().clone() for value in trainer.actor.parameters()]
    before_critic = [value.detach().clone() for value in trainer.critic.parameters()]
    frozen_model = [value.detach().clone() for value in trainer.world_model.parameters()]
    metrics = trainer.updates.update_actor()
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
    restored = transforms.unflatten_state(transforms.flatten_state(state))
    for original, result in zip(state, restored):
        torch.testing.assert_close(original, result)
    trainer.buffer.data["episode_step"].zero_()
    start, history = trainer.buffer.starts_at(torch.tensor([5]), torch.tensor([0]))
    torch.testing.assert_close(history["state"][:, 0], history["state"][:, -1])
    trainer.buffer.data["transition_cut"].fill_(True)
    windows = trainer.buffer.sample_windows(4)
    assert windows["mask"][:, 0].eq(1).all()
    assert windows["mask"][:, 1].eq(0).all()


def test_collect_and_training_loop(tiny_trainer, tmp_path, monkeypatch):
    """Real native MuJoCo CPU collection through the same Trainer orchestration."""
    trainer = tiny_trainer

    class NativeEnv:
        def __init__(self):
            self.model = mujoco.MjModel.from_xml_path(trainer.robot.mjcf_path)
            self.model.opt.timestep = 0.005
            self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            self.model.dof_armature[6:] = 0.01
            self.data = [mujoco.MjData(self.model) for _ in range(4)]
            self.episode_step = torch.zeros(4, dtype=torch.long)
            self.motion_frame_index = torch.full((4,), 10, dtype=torch.long)
            self.last_action = torch.zeros(4, 2)
            self.xi = default_xi(trainer.robot, trainer.device).expand(4, -1).clone()
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
            return transforms.compute_observation(
                self.state(),
                self.last_action,
                self.xi,
                self.motion_frame_index,
                trainer.motion,
                trainer.default_joint_pos,
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
                state=transforms.flatten_state(state),
                next_state=transforms.flatten_state(next_state),
                action=action,
                last_action=last_action,
                xi=self.xi,
                motion_frame_index=frames,
                episode_step=steps,
                done=torch.zeros(4, dtype=torch.bool),
                transition_cut=torch.zeros(4, dtype=torch.bool),
                reward=trainer.task.reward(next_state, frames + 1, action, last_action),
            )

    monkeypatch.setattr(trainer, "create_env", lambda: setattr(trainer, "env", NativeEnv()))
    trainer.config.warmup_steps = 4
    trainer.config.wm_batches = 1
    trainer.config.iterations = 2
    trainer.config.log_every = 1
    trainer.config.ckpt_every = 2
    trainer.config.log_dir = str(tmp_path / "run")
    trainer.run()
    assert trainer.iteration == 2
    assert (tmp_path / "run/last.pt").is_file()
    assert len((tmp_path / "run/metrics.jsonl").read_text().splitlines()) == 2
    import json

    rows = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert all(("sim_r" in row and np.isfinite(row["sim_r"]) for row in rows))
    assert all(0 < row["iterations_per_second"] < float("inf") for row in rows)
    stage_names = (
        "prepare", "collect", "world_model", "actor_prepare", "actor_rollout",
        "reward", "actor_return", "actor_backward", "critic", "other",
    )
    for row in rows:
        assert all(row[f"{name}_ms"] >= 0 for name in stage_names)
        assert row["iteration_ms"] == pytest.approx(sum(row[f"{name}_ms"] for name in stage_names))
        for group in ("collect", "world_model", "actor_backward"):
            children = [
                value for name, value in row.items()
                if name.startswith(f"{group}_") and name.endswith("_ms") and name != f"{group}_ms"
            ]
            assert children
            assert row[f"{group}_ms"] == pytest.approx(sum(children))
        assert all(row[f"world_model_{stage}_ms"] >= 0 for stage in (
            "prepare", "forward", "backward", "clip", "optimizer", "other"
        ))
        assert all(row[f"actor_backward_{stage}_ms"] >= 0 for stage in (
            "gradients", "clip", "optimizer", "ema", "other"
        ))
    assert not (tmp_path / "run/checkpoints/iteration_000001.pt").exists()
    assert (tmp_path / "run/checkpoints/iteration_000002.pt").is_file()
    from copy import deepcopy

    resumed_config = deepcopy(trainer.config)
    resumed_config.iterations = 3
    resumed = Trainer(resumed_config)
    resumed.load(tmp_path / "run/last.pt")
    assert resumed.iteration == 2
    assert resumed.buffer.size == 0
    torch.testing.assert_close(resumed.actor.state_dict(), trainer.actor.state_dict())
    assert_checkpoint_values_equal(resumed.actor_optimizer.state_dict(), trainer.actor_optimizer.state_dict())
    assert_checkpoint_values_equal(
        resumed.critic_optimizer.state_dict(), trainer.critic_optimizer.state_dict()
    )
    assert_checkpoint_values_equal(
        resumed.world_model_optimizer.state_dict(), trainer.world_model_optimizer.state_dict()
    )
    monkeypatch.setattr(resumed, "create_env", lambda: setattr(resumed, "env", NativeEnv()))
    resumed.run()
    assert resumed.iteration == 3
    assert (tmp_path / "run/checkpoints/iteration_000003.pt").is_file()
    assert len((tmp_path / "run/metrics.jsonl").read_text().splitlines()) == 3


def test_xi_matches_applied_physics(tiny_trainer):
    pytest.importorskip("mjlab")
    from readable_mimic.sim import initialize_xi, randomize_joint_pos_offsets
    from readable_mimic.sim_mujoco import MujocoSim

    trainer = tiny_trainer
    default_body_positions = torch.zeros(3, 3)
    simulation = SimpleNamespace(
        model=SimpleNamespace(geom_friction=torch.zeros(4, 3, 3), body_ipos=torch.zeros(4, 3, 3)),
        mj_model=SimpleNamespace(geom_bodyid=np.array([3, 1, 2]), geom_type=np.array([mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_SPHERE])),
        expand_model_fields=lambda names: None,
        get_default_field=lambda name: default_body_positions,
    )
    backend = MujocoSim.__new__(MujocoSim)
    backend.physics = simulation
    backend.robot_description = trainer.robot
    backend.device = "cpu"
    backend.robot = SimpleNamespace(
        indexing=SimpleNamespace(geom_ids=torch.tensor([1, 2]), body_ids=torch.tensor([1, 2])),
        find_bodies=lambda name: ([0], [name]),
    )
    env = SimpleNamespace(
        robot_description=trainer.robot,
        simulation=backend,
        num_envs=4,
        device="cpu",
        robot=SimpleNamespace(
            indexing=SimpleNamespace(geom_ids=torch.tensor([1, 2]), body_ids=torch.tensor([1, 2])),
            find_bodies=lambda name: ([0], [name]),
        ),
    )
    xi = initialize_xi(env)
    torch.testing.assert_close(simulation.model.geom_friction[:, 0, 0], torch.full((4,), trainer.robot.domain_randomization.friction_range[0]))
    torch.testing.assert_close(simulation.model.geom_friction[:, 1, 0], xi[:, 0])
    torch.testing.assert_close(simulation.model.geom_friction[:, 2, 0], xi[:, 0])
    torch.testing.assert_close(simulation.model.body_ipos[:, 1], xi[:, 3:6])
    before = xi.clone()
    randomize_joint_pos_offsets(xi, torch.tensor([1, 3]), trainer.joint_pos_offset_limits)
    torch.testing.assert_close(xi[[0, 2]], before[[0, 2]])
    torch.testing.assert_close(xi[:, 0], before[:, 0])
    torch.testing.assert_close(xi[:, 3:], before[:, 3:])
    assert xi[:, 1:3].abs().le(0.01).all()
    evaluation_xi = initialize_xi(env, randomize=False)
    torch.testing.assert_close(evaluation_xi[:, 0], torch.ones(4))
    torch.testing.assert_close(evaluation_xi[:, 1:], torch.zeros(4, 5))
    torch.testing.assert_close(simulation.model.geom_friction[:, :, 0], torch.ones(4, 3))
    torch.testing.assert_close(simulation.model.body_ipos[:, 1], torch.zeros(4, 3))


def test_observation_fields_and_context_normalization(tiny_trainer):
    trainer = tiny_trainer
    state = trainer.motion.ref_state(torch.tensor([12]))
    xi = default_xi(trainer.robot, "cpu")[None].clone()
    xi[:, 1:3] = torch.tensor([[0.004, -0.006]])
    last_action = torch.tensor([[0.2, -0.3]])
    observation = transforms.compute_observation(
        state, last_action, xi, torch.tensor([12]), trainer.motion, trainer.default_joint_pos
    )
    torch.testing.assert_close(observation[:, :2], state[4] - xi[:, 1:3])
    torch.testing.assert_close(observation[:, 10:12], last_action)
    normalized = transforms.compute_normalized_xi(xi, trainer.xi_center, trainer.xi_scale)
    torch.testing.assert_close(normalized[:, 1:3], torch.tensor([[0.4, -0.6]]))
    assert observation.shape == (1, 18)


def test_reward_detail_names_and_total(tiny_trainer):
    from readable_mimic.sim import CONTROL_DT

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
                assert "simulation_parameter" not in identifier, (source_file, identifier)
                assert not identifier.startswith(("reference_", "e_", "root_")), (source_file, identifier)
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        assert key.value not in forbidden_keys, (source_file, key.value)


def test_shared_command_and_history_transforms(tiny_trainer):
    trainer = tiny_trainer
    last_frame = int(trainer.motion.clip_end[0])
    frames = torch.tensor([12, last_frame])
    future_frames = transforms.compute_future_motion_frames(
        frames, trainer.command_frame_offsets, trainer.motion
    )
    torch.testing.assert_close(future_frames, torch.tensor([[13, 14], [last_frame, last_frame]]))
    command = transforms.compute_command(frames, trainer.command_frame_offsets, trainer.motion)
    assert command.shape == (2, transforms.compute_command_dim(2, 2))
    torch.testing.assert_close(command[:, :19][:, 15:17], trainer.motion.joint_pos[future_frames[:, 0]])
    observation = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    history = transforms.compute_observation_history(observation, 3)
    next_observation = observation + 10
    updated = transforms.update_observation_history(history, next_observation, torch.tensor([False, True]))
    torch.testing.assert_close(updated[0], torch.tensor([[1.0, 2.0], [1.0, 2.0], [11.0, 12.0]]))
    torch.testing.assert_close(updated[1], torch.tensor([[13.0, 14.0], [13.0, 14.0], [13.0, 14.0]]))


def test_shared_actor_return_and_value_targets():
    rewards = torch.tensor([[4.0], [5.0]], requires_grad=True)
    terminations = torch.tensor([[0.0], [0.5]])
    bootstrap = torch.tensor([3.0], requires_grad=True)
    actor_return, survival, final_survival = transforms.compute_actor_return(
        rewards, terminations, bootstrap, torch.tensor([[1.0], [0.9]]), 0.9**2
    )
    torch.testing.assert_close(actor_return, torch.tensor([4 + 0.9 * 5 + 0.9**2 * 0.5 * 3]))
    actor_return.sum().backward()
    torch.testing.assert_close(rewards.grad, torch.tensor([[1.0], [0.9]]))
    torch.testing.assert_close(bootstrap.grad, torch.tensor([0.9**2 * 0.5]))
    targets = transforms.compute_value_targets(
        torch.tensor([[1.0], [2.0], [3.0]]), rewards.detach(), terminations, 0.9, 0.5
    )
    torch.testing.assert_close(targets, torch.tensor([[7.7575], [6.35]]))


def test_updater_dependencies_are_explicit_and_transforms_are_centralized():
    import ast

    source_directory = Path(__file__).parents[1] / "src/readable_mimic"
    update_tree = ast.parse((source_directory / "updates.py").read_text(encoding="utf-8"))
    update_class = next((node for node in update_tree.body if isinstance(node, ast.ClassDef)))
    constructor = next((node for node in update_class.body if getattr(node, "name", None) == "__init__"))
    declared = {
        node.attr
        for node in ast.walk(constructor)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and (node.value.id == "self")
        and isinstance(node.ctx, ast.Store)
    }
    methods = {node.name for node in update_class.body if isinstance(node, ast.FunctionDef)}
    used = {
        node.attr
        for node in ast.walk(update_class)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and (node.value.id == "self")
        and isinstance(node.ctx, ast.Load)
    }
    assert used <= declared | methods, used - declared - methods
    trainer_tree = ast.parse((source_directory / "trainer.py").read_text(encoding="utf-8"))
    trainer_class = next((node for node in trainer_tree.body if isinstance(node, ast.ClassDef)))
    assert trainer_class.bases == []
    transform_tree = ast.parse((source_directory / "data_transforms.py").read_text(encoding="utf-8"))
    assert not any((isinstance(node, ast.Name) and node.id == "self" for node in ast.walk(transform_tree)))
    assert not (source_directory / "objective.py").exists()
    assert not (source_directory / "observation.py").exists()


def test_resume_restores_saved_config_and_random_states(tiny_trainer, tmp_path):
    import json
    import random
    from readable_mimic.config import load_config

    trainer = tiny_trainer
    trainer.config.warmup_steps = 4
    trainer.iteration = 7
    trainer.decay_start = 5
    trainer.collection_remainder = 0.5
    trainer.save(tmp_path / "saved.pt")
    expected_torch = torch.rand(3)
    expected_numpy = np.random.rand(3)
    expected_python = random.random()
    current = dict(vars(trainer.config))
    current.update(
        resume="saved.pt",
        lr=0.123,
        wm_hidden="9,9",
        compile_world_model=False,
        compile_actor=False,
        allow_tf32=False,
        simulator="mujoco",
        genesis={"enable_self_collision": False},
        iterations=20,
        ckpt_every=5,
        log_dir="new_run",
        motion_dir=trainer.config.motion_dir,
    )
    (tmp_path / "config.json").write_text(json.dumps(current), encoding="utf-8")
    config = load_config(tmp_path / "config.json")
    assert config.lr == trainer.config.lr
    assert config.wm_hidden == trainer.config.wm_hidden
    assert config.compile_world_model is False
    assert config.compile_actor is False and config.allow_tf32 is False
    assert config.simulator == "mujoco" and config.genesis == {"enable_self_collision": False}
    assert config.iterations == 20 and config.ckpt_every == 5
    assert config.log_dir == str(tmp_path / "new_run")
    resumed = Trainer(config)
    resumed.load(config.resume)
    assert resumed.decay_start == 5 and resumed.collection_remainder == 0.5
    resumed.restore_random_states()
    torch.testing.assert_close(torch.rand(3), expected_torch)
    np.testing.assert_array_equal(np.random.rand(3), expected_numpy)
    assert random.random() == expected_python
    checkpoint = torch.load(tmp_path / "saved.pt", map_location="cpu", weights_only=False)
    assert "buffer" not in checkpoint
    assert checkpoint["config"]["wm_hidden"] == trainer.config.wm_hidden
    assert checkpoint["format_version"] == 2
    resumed.config.iterations = 7
    with pytest.raises(ValueError, match="larger total target"):
        resumed.load(config.resume)


def test_sim_r_uses_all_real_transitions(tiny_trainer):
    from readable_mimic.task import MAX_STEP_REWARD

    trainer = tiny_trainer
    transition = {name: values[0].clone() for name, values in trainer.buffer.data.items()}
    transition["reward"].fill_(0.05)
    trainer.env = SimpleNamespace(
        observe=lambda: torch.zeros(4, trainer.observation_dim),
        episode_step=torch.zeros(4, dtype=torch.long),
        motion_frame_index=transition["motion_frame_index"],
        step=lambda action: transition,
    )
    assert trainer.collect(3) == []
    assert trainer.sim_reward_count == 12
    assert float(trainer.sim_reward_sum) / trainer.sim_reward_count / MAX_STEP_REWARD == pytest.approx(0.5)


def test_resume_cli_overrides_config_without_modifying_file(tiny_trainer, tmp_path, monkeypatch):
    import json
    from readable_mimic.train import main

    trainer = tiny_trainer
    trainer.config.warmup_steps = 4
    checkpoint_path = tmp_path / "saved.pt"
    trainer.save(checkpoint_path)
    current = dict(vars(trainer.config))
    current.update(resume="nonexistent.pt", lr=0.123)
    config_path = tmp_path / "config.json"
    original_text = json.dumps(current)
    config_path.write_text(original_text, encoding="utf-8")
    calls = {}

    class TrainingRun:
        def __init__(self, config):
            calls["config"] = config

        def load(self, path):
            calls["resume"] = path

        def run(self):
            calls["ran"] = True

    monkeypatch.setattr("readable_mimic.train.Trainer", TrainingRun)
    main(["--config", str(config_path), "--resume", str(checkpoint_path)])
    assert calls["resume"] == str(checkpoint_path)
    assert calls["config"].lr == trainer.config.lr
    assert calls["ran"]
    assert config_path.read_text(encoding="utf-8") == original_text
    current.pop("resume")
    config_path.write_text(json.dumps(current), encoding="utf-8")
    main(["--config", str(config_path), "--resume", str(checkpoint_path)])
    assert calls["resume"] == str(checkpoint_path)


def test_fixed_training_constants_are_not_rebuilt(tiny_trainer, monkeypatch):
    import readable_mimic.sim as sim_module
    import readable_mimic.buffer as buffer_module

    trainer = tiny_trainer
    constants = (
        trainer.joint_pos_offset_limits,
        trainer.xi_center,
        trainer.xi_scale,
        trainer.command_frame_offsets,
        trainer.updates.discount_factors,
        trainer.buffer.window_offsets,
        trainer.buffer.history_lookback,
    )
    before = [value.clone() for value in constants]

    def unexpected_initialization(*args, **kwargs):
        raise AssertionError("Fixed constants must not be rebuilt during updates")

    class NoArangeTorch:
        def __getattr__(self, name):
            if name == "arange":
                return unexpected_initialization
            return getattr(torch, name)

    monkeypatch.setattr(sim_module, "joint_pos_offset_limits", unexpected_initialization)
    monkeypatch.setattr(transforms, "compute_xi_normalization", unexpected_initialization)
    monkeypatch.setattr(transforms, "torch", NoArangeTorch())
    monkeypatch.setattr(buffer_module, "torch", NoArangeTorch())
    for _ in range(2):
        trainer.updates.update_world_model()
        metrics = trainer.updates.update_actor()
        assert all((torch.isfinite(value).all() for value in metrics))
    for original, current in zip(before, constants):
        torch.testing.assert_close(original, current, rtol=0, atol=0)
    xi = torch.randn(2, 6, requires_grad=True)
    normalized = transforms.compute_normalized_xi(xi, trainer.xi_center, trainer.xi_scale)
    expected_center = torch.tensor([0.75, 0.0, 0.0, 0.0, 0.0, 0.0])
    expected_scale = torch.tensor([0.45, 0.01, 0.01, 0.05, 0.05, 0.05])
    torch.testing.assert_close(normalized, (xi - expected_center) / expected_scale)
    normalized.sum().backward()
    torch.testing.assert_close(xi.grad, expected_scale.reciprocal().expand_as(xi))
