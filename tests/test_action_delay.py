"""Physical target timing, reset snapshots and delay-conditioned learning."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from test_pipeline import tiny_trainer as tiny_trainer
from readable_mimic import data_transforms as transforms
from readable_mimic.env import TrackingEnv
from readable_mimic.sim import DomainRandomizationConfig, default_xi, delayed_position_target
from readable_mimic.trainer import Trainer


@pytest.fixture
def delayed_trainer(tiny_trainer):
    config = deepcopy(tiny_trainer.config)
    config.domain_randomization["action_delay_max_ms"] = 20.0
    config.grad_optimization = dict(cap=10, start_draw=True, candidates=2)
    trainer = Trainer(config)
    for step in range(10):
        frames = torch.arange(4) + step + 10
        state = trainer.motion.ref_state(frames)
        action = torch.full((4, 2), 0.1)
        xi = default_xi(trainer.robot, "cpu").expand(4, -1).clone()
        xi[:, -1] = torch.arange(4)
        trainer.buffer.add(dict(
            state=transforms.flatten_state(state),
            next_state=transforms.flatten_state(trainer.motion.ref_state(frames + 1)),
            action=action, last_action=action * 0.5, previous_target=state[4] + 0.01,
            xi=xi, reward=torch.zeros(4), motion_frame_index=frames,
            init_f=frames - step,
            init_root=transforms.flatten_state(trainer.motion.ref_state(frames - step))[:, :7],
            episode_step=torch.full((4,), step), done=torch.zeros(4, dtype=torch.bool),
            transition_cut=torch.zeros(4, dtype=torch.bool),
        ))
    return trainer


def test_each_delay_switches_at_physics_step():
    current = torch.tensor([[10.], [20.], [30.], [40.], [50.]])
    previous = -current
    lag = torch.arange(5)
    actual = torch.stack([delayed_position_target(current, previous, lag, i) for i in range(4)])
    for env in range(5):
        for step in range(4):
            torch.testing.assert_close(actual[step, env], previous[env] if step < env else current[env])
    for invalid in (-5, 1, 21, 40, float("nan")):
        with pytest.raises(ValueError, match="action_delay_max_ms"):
            DomainRandomizationConfig(action_delay_max_ms=invalid)


def test_backends_apply_targets_before_each_physics_step():
    from readable_mimic.sim_genesis import GenesisSim
    from readable_mimic.sim_mujoco import MujocoSim

    current = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    previous = -current
    lag = torch.tensor([0, 1, 2, 4])
    for backend_class in (GenesisSim, MujocoSim):
        captured, target = [], {}
        def set_target(value, *args):
            target["value"] = value.clone()
        def step():
            captured.append(target["value"].clone())
        backend = backend_class.__new__(backend_class)
        backend.joint_dof_indices = [0, 1]
        backend.robot = SimpleNamespace(
            control_dofs_position=set_target, set_joint_position_target=set_target,
        )
        backend.scene = SimpleNamespace(
            step=step, write_data_to_sim=lambda: None, update=lambda dt: None,
        )
        backend.physics = SimpleNamespace(step=step)
        backend.step(current, previous, lag)
        assert len(captured) == 4
        for i, actual in enumerate(captured):
            torch.testing.assert_close(actual, delayed_position_target(current, previous, lag, i))


def test_reset_and_replay_preserve_actual_target_and_delay(delayed_trainer, monkeypatch):
    trainer = delayed_trainer
    initial = trainer.motion.ref_state(torch.tensor([10, 11, 12, 13]))

    class Simulation:
        def __init__(self):
            self.values = tuple(value.clone() for value in initial)
        def apply_xi(self, *args):
            pass
        def reset(self, indices, state):
            for actual, value in zip(self.values, state):
                actual[indices] = value
        def state(self):
            return tuple(value.clone() for value in self.values)
        def step(self, target, previous, lag):
            self.previous = previous.clone()
            self.lag = lag.clone()

    simulation = Simulation()
    monkeypatch.setattr("readable_mimic.env.create_simulation", lambda *args: simulation)
    env = TrackingEnv(
        trainer.config.simulation, 4, trainer.robot, trainer.motion, trainer.task, "cpu",
        max_episode_steps=1,
    )
    env.xi[:, -1] = torch.tensor([0., 1., 2., 4.])
    before = env.previous_target.clone()
    selected = torch.tensor([1, 3])
    env.reset(selected, torch.tensor([15, 16]))
    torch.testing.assert_close(env.previous_target[[0, 2]], before[[0, 2]])
    torch.testing.assert_close(env.previous_target[selected], env.state()[4][selected])
    env.xi[:, -1] = torch.tensor([0., 1., 2., 4.])
    transition = env.step(torch.full((4, 2), 0.2))
    torch.testing.assert_close(transition["previous_target"], simulation.previous)
    torch.testing.assert_close(transition["xi"][:, -1].long(), simulation.lag)
    assert transition["done"].all()
    torch.testing.assert_close(env.previous_target, env.state()[4])
    assert env.last_action.eq(0).all()
    trainer.buffer.add(transition)
    start, _ = trainer.buffer.starts_at(torch.tensor([trainer.buffer.size - 1]), torch.tensor([2]))
    torch.testing.assert_close(start["previous_target"][0], simulation.previous[2])
    torch.testing.assert_close(start["xi"][0], transition["xi"][2])


def test_delay_conditioned_world_model_critic_and_gradient_update(delayed_trainer, tmp_path):
    trainer = delayed_trainer
    assert trainer.xi_dim == 14
    assert trainer.world_model.use_last_action
    windows = trainer.buffer.sample_windows(4)
    assert windows["previous_target"].shape == (4, trainer.config.wm_window, 2)
    state = transforms.unflatten_state(windows["state"][:, 0])
    xi = transforms.compute_normalized_xi(windows["xi"][:, 0], trainer.xi_center, trainer.xi_scale)
    target = windows["previous_target"][:, 0]
    inputs = trainer.world_model.inputs(state, windows["action"][:, 0], windows["last_action"][:, 0], xi, target)
    joint_slice = slice(-trainer.xi_dim - 2, -trainer.xi_dim)
    torch.testing.assert_close(inputs[:, joint_slice], target - state[4])
    for _ in range(2):
        assert all(torch.isfinite(x).all() for x in trainer.updates.update_world_model())
        assert all(torch.isfinite(x).all() for x in trainer.updates.update_actor())
    path = tmp_path / "delay.pt"
    trainer.save(path)
    trainer.load(path)


def test_fixed_delay_evaluation_context(delayed_trainer, tmp_path):
    import json
    from readable_mimic.evaluate import apply_evaluation_dr
    from readable_mimic.fixed_dr import load_dr
    trainer = delayed_trainer
    path = tmp_path / "dr.json"
    path.write_text(json.dumps({"domain_randomization": {"action_delay_ms": 15.0}}))
    dr = load_dr(path, dict(joint_names=trainer.robot.joint_names, friction=1., kp_scale=1., kd_scale=1.))
    applied = {}
    def apply(xi, **kwargs):
        applied["xi"] = xi.clone()
    env = SimpleNamespace(
        xi=default_xi(trainer.robot, "cpu")[None].clone(), num_joints=2,
        robot_description=trainer.robot, simulation=SimpleNamespace(apply_xi=apply),
    )
    apply_evaluation_dr(env, dr)
    assert env.xi[0, -1] == 3
    torch.testing.assert_close(env.xi, applied["xi"])
    torch.testing.assert_close(env.xi[0, 6:13], torch.tensor([1., 1., 1., 0., 0., 1., 1.]))


def test_compiled_delayed_world_model_and_critic_inputs(delayed_trainer):
    trainer = delayed_trainer
    windows = trainer.buffer.sample_windows(4)
    args = (windows["state"], windows["action"], windows["last_action"], windows["xi"], windows["previous_target"])
    compiled = torch.compile(trainer.updates.world_model_rollout, backend="aot_eager", fullgraph=True, dynamic=False)
    for value, target in zip(compiled(*args), trainer.updates.world_model_rollout(*args)):
        torch.testing.assert_close(value, target)
    state = transforms.unflatten_state(windows["state"][:, 0])
    frames = torch.tensor([10, 11, 12, 13])
    xi = transforms.compute_normalized_xi(windows["xi"][:, 0], trainer.xi_center, trainer.xi_scale)
    command = transforms.compute_command(frames, trainer.command_frame_offsets, trainer.motion)
    root = transforms.flatten_state(state)[:, :7]
    target = windows["previous_target"][:, 0].detach().requires_grad_()
    compiled_features = torch.compile(trainer.updates.critic_features, backend="aot_eager", fullgraph=True, dynamic=False)
    features = compiled_features(state, frames, xi, command, root, frames, target)
    expected = trainer.updates.critic_features(state, frames, xi, command, root, frames, target)
    torch.testing.assert_close(features, expected)
    gradient, = torch.autograd.grad(features.sum(), target)
    torch.testing.assert_close(gradient, torch.ones_like(target))
