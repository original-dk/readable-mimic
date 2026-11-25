"""Actual Genesis delay and old/new zero-delay stepping comparison."""
from dataclasses import replace
import time

import pytest
import torch

gs = pytest.importorskip("genesis")
from test_pipeline import tiny_trainer as tiny_trainer
from test_action_delay import delayed_trainer as delayed_trainer
from readable_mimic.sim import default_xi
from readable_mimic.sim_genesis import GenesisSim


def test_delayed_genesis_training_and_zero_delay_partition(delayed_trainer, monkeypatch):
    trainer = delayed_trainer
    names = trainer.robot.joint_names
    trainer.robot = replace(
        trainer.robot,
        joint_stiffness={name: 5. for name in names},
        joint_damping={name: 1. for name in names},
        joint_effort_limit={name: 20. for name in names},
        joint_armature={name: .01 for name in names},
    )
    trainer.config.simulator = "genesis"
    trainer.config.simulation = {**trainer.config.simulation, "iterations": 10, "ls_iterations": 10}
    try:
        trainer.create_env()
        env = trainer.env
        simulation = env.simulation
        assert simulation.scene.dt == pytest.approx(.005)
        assert simulation.scene.substeps == 1
        env.xi[:, -1] = torch.tensor([0., 1., 2., 4.])
        old_target = env.previous_target.clone()
        action = torch.full((4, 2), .2)
        new_target = env.default_joint_pos + env.xi[:, 1:3] + env.action_scale * action
        captured = []
        original_control = simulation.robot.control_dofs_position
        def capture(target, *args, **kwargs):
            captured.append(target.clone())
            return original_control(target, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(simulation.robot, "control_dofs_position", capture)
            env.step(action)
        assert len(captured) >= 4
        for step, target in enumerate(captured[:4]):
            expected = torch.where((step < torch.tensor([0, 1, 2, 4]))[:, None], old_target, new_target)
            torch.testing.assert_close(target, expected)
        trainer.collect(4)
        assert all(torch.isfinite(value).all() for value in trainer.updates.update_world_model())
        assert all(torch.isfinite(value).all() for value in trainer.updates.update_actor())
        frames = torch.tensor([10, 11, 12, 13])
        assert torch.isfinite(trainer.updates.rank_reference(frames, 3)).all()

        # Construct the preceding dt=.02/substeps=4 scene using the same model and solver options.
        original_scene = gs.Scene
        def old_scene(*args, **kwargs):
            kwargs["sim_options"] = gs.options.SimOptions(dt=.02, substeps=4)
            return original_scene(*args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(gs, "Scene", old_scene)
            old = GenesisSim(trainer.robot, 4, "cpu", trainer.config.simulation, trainer.config.genesis)
        nominal = default_xi(trainer.robot, "cpu").expand(4, -1).clone()
        nominal[:, 0] = 1.
        simulation.apply_xi(nominal, False)
        old.apply_xi(nominal, False)
        initial = trainer.motion.ref_state(frames)
        ids = torch.arange(4)
        simulation.reset(ids, initial)
        old.reset(ids, initial)
        zero_lag = torch.zeros(4, dtype=torch.long)
        for _ in range(5):
            old.robot.control_dofs_position(initial[4], old.joint_dof_indices)
            old.scene.step()
            simulation.step(initial[4], initial[4], zero_lag)
        simulation.reset(ids, initial)
        old.reset(ids, initial)
        old_seconds = new_seconds = 0.
        previous_target = initial[4]
        for step in range(25):
            target = initial[4] + .03 * torch.sin(torch.tensor(step * .2))
            begin = time.perf_counter()
            old.robot.control_dofs_position(target, old.joint_dof_indices)
            old.scene.step()
            old_seconds += time.perf_counter() - begin
            begin = time.perf_counter()
            simulation.step(target, previous_target, zero_lag)
            new_seconds += time.perf_counter() - begin
            previous_target = target
            for actual, expected in zip(simulation.state(), old.state()):
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        print(f"CPU zero-delay comparison: old={old_seconds:.3f}s new={new_seconds:.3f}s")
    finally:
        gs.destroy()
