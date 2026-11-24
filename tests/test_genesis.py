from dataclasses import replace, asdict
import torch
import pytest

gs = pytest.importorskip("genesis")
from test_pipeline import tiny_trainer as tiny_trainer
from readable_mimic.rotations import so3_exp


@pytest.mark.parametrize("paper_update", [False, True])
def test_real_genesis_state_reset_xi_and_training(tiny_trainer, tmp_path, paper_update):
    trainer = tiny_trainer
    joint_names = trainer.robot.joint_names
    trainer.robot = replace(
        trainer.robot,
        joint_stiffness={name: 5.0 for name in joint_names},
        joint_damping={name: 1.0 for name in joint_names},
        joint_effort_limit={name: 20.0 for name in joint_names},
        joint_armature={name: 0.01 for name in joint_names},
    )
    if paper_update:
        from readable_mimic.grad_optimization import (
            GradientOptimizationConfig, GradientStartSampler, ModelRankedReset,
        )
        trainer.updates.grad_config = GradientOptimizationConfig(
            cap=10, start_draw=True, candidates=2, model_resets=True,
            reset_horizon=3, reset_tail=5, rank_batch=8,
        )
        trainer.config.grad_optimization = asdict(trainer.updates.grad_config)
        trainer.updates.start_sampler = GradientStartSampler(trainer.updates.grad_config, trainer.device)
        trainer.model_reset = ModelRankedReset(trainer.motion, trainer.updates.grad_config)
    trainer.config.simulator = "genesis"
    trainer.config.simulation = {**trainer.config.simulation, "iterations": 10, "ls_iterations": 10}
    try:
        trainer.create_env()
        env = trainer.env
        simulation = env.simulation
        before = tuple(value.clone() for value in env.state())
        selected = torch.tensor([1, 3])
        target_state = [value[selected].clone() for value in before]
        target_state[0][:, :2] = torch.tensor([[0.2, -0.1], [-0.1, 0.3]])
        target_state[1] = so3_exp(torch.tensor([[0.2, -0.3, 0.6], [-0.2, 0.4, -0.5]]))
        target_state[2] = torch.tensor([[0.1, 0.2, 0.3], [-0.3, 0.2, -0.1]])
        target_state[3] = torch.tensor([[0.4, -0.2, 0.1], [-0.1, 0.3, 0.2]])
        simulation.reset(selected, target_state)
        actual = env.state()
        for index, values in enumerate(actual):
            torch.testing.assert_close(values[selected], target_state[index], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(values[[0, 2]], before[index][[0, 2]], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            simulation.robot.get_links_COM([simulation.center_of_mass_index]),
            simulation.default_center_of_mass + env.xi[:, None, 1 + trainer.robot.num_joints : 4 + trainer.robot.num_joints],
        )
        friction_ratios = simulation.scene.sim.rigid_solver.get_geoms_friction_ratio(
            [geom.idx for geom in simulation.robot.geoms]
        )
        torch.testing.assert_close(friction_ratios, env.xi[:, :1].expand_as(friction_ratios))
        torch.testing.assert_close(
            simulation.robot.get_dofs_kp(simulation.joint_dof_indices), 5.0 * env.xi[:, -2:-1].expand(4, 2)
        )
        mass, inertia, armature, damping, frictionloss, kp, kd = env.xi[:, -7:].unbind(-1)
        torch.testing.assert_close(simulation.robot.get_mass(), simulation.default_mass * mass)
        torch.testing.assert_close(
            simulation.robot.get_links_inertia(),
            simulation.default_inertia * (mass * inertia)[:, None, None, None],
        )
        for getter, expected in (
            (simulation.robot.get_dofs_armature, simulation.default_armature * armature[:, None]),
            (simulation.robot.get_dofs_damping, simulation.default_passive_damping + damping[:, None]),
            (simulation.robot.get_dofs_frictionloss, simulation.default_frictionloss + frictionloss[:, None]),
            (simulation.robot.get_dofs_kv, simulation.default_kd * kd[:, None]),
        ):
            torch.testing.assert_close(getter(simulation.joint_dof_indices), expected)
        env.reset(torch.arange(4), torch.full((4,), 12))
        trainer.updates.timer.start_iteration()
        trainer.collect(4)
        world_model_metrics = trainer.updates.update_world_model()
        actor_metrics = trainer.updates.update_actor()
        trainer.updates.timer.end_iteration()
        timings = trainer.updates.timer.report()
        assert timings["collect_physics_ms"] > 0 and timings["collect_finalize_ms"] > 0
        assert all(torch.isfinite(value).all() for value in (*world_model_metrics, *actor_metrics))
        if paper_update:
            assert env.reset_sampler is trainer.model_reset
            ranked = trainer.model_reset.refresh(trainer.updates.rank_reference, 2000)
            assert ranked["model_reset_hard_bins"] > 0
            reset_frames = env.sample_motion_frames(100)
            assert (reset_frames <= trainer.motion.end_frame(reset_frames) - 5).all()
        trainer.save(tmp_path / "genesis.pt")
        saved = torch.load(tmp_path / "genesis.pt", map_location="cpu", weights_only=False)
        assert saved["config"]["simulator"] == "genesis"
        # Evaluation uses the same dynamics interface but deterministic xi/no observation noise.
        from readable_mimic.sim import initialize_xi

        evaluation_xi = initialize_xi(env, randomize=False)
        torch.testing.assert_close(evaluation_xi[:, 0], torch.ones(4))
        torch.testing.assert_close(evaluation_xi[:, 1:6], torch.zeros(4, 5))
        torch.testing.assert_close(evaluation_xi[:, -7:], torch.tensor([1., 1., 1., 0., 0., 1., 1.]).expand(4, -1))
        torch.testing.assert_close(simulation.robot.get_mass(), simulation.default_mass)
        torch.testing.assert_close(simulation.robot.get_links_inertia(), simulation.default_inertia)
    finally:
        gs.destroy()
