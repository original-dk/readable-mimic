from dataclasses import replace
import torch
import pytest

gs = pytest.importorskip("genesis")
from test_pipeline import tiny_trainer as tiny_trainer
from readable_mimic.rotations import so3_exp


def test_real_genesis_state_reset_xi_and_training(tiny_trainer, tmp_path):
    trainer = tiny_trainer
    joint_names = trainer.robot.joint_names
    trainer.robot = replace(
        trainer.robot,
        joint_stiffness={name: 5.0 for name in joint_names},
        joint_damping={name: 1.0 for name in joint_names},
        joint_effort_limit={name: 20.0 for name in joint_names},
        joint_armature={name: 0.01 for name in joint_names},
    )
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
            simulation.default_center_of_mass + env.xi[:, None, 1 + trainer.robot.num_joints :],
        )
        friction_ratios = simulation.scene.sim.rigid_solver.get_geoms_friction_ratio(
            [geom.idx for geom in simulation.robot.geoms]
        )
        torch.testing.assert_close(friction_ratios, env.xi[:, :1].expand_as(friction_ratios))
        torch.testing.assert_close(
            simulation.robot.get_dofs_kp(simulation.joint_dof_indices), torch.full((4, 2), 5.0)
        )
        env.reset(torch.arange(4), torch.full((4,), 12))
        trainer.updates.timer.start_iteration()
        trainer.collect(4)
        world_model_metrics = trainer.updates.update_world_model()
        actor_metrics = trainer.updates.update_actor()
        trainer.updates.timer.end_iteration()
        timings = trainer.updates.timer.report()
        assert timings["collect_physics_ms"] > 0 and timings["collect_finalize_ms"] > 0
        assert all(torch.isfinite(value).all() for value in (*world_model_metrics, *actor_metrics))
        trainer.save(tmp_path / "genesis.pt")
        saved = torch.load(tmp_path / "genesis.pt", map_location="cpu", weights_only=False)
        assert saved["config"]["simulator"] == "genesis"
        # Evaluation uses the same dynamics interface but deterministic xi/no observation noise.
        from readable_mimic.sim import initialize_xi

        evaluation_xi = initialize_xi(env, randomize=False)
        torch.testing.assert_close(evaluation_xi[:, 0], torch.ones(4))
        torch.testing.assert_close(evaluation_xi[:, 1:], torch.zeros(4, 5))
    finally:
        gs.destroy()
