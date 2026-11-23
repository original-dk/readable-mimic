"""Physical parameter readback through the real MuJoCo-Warp model."""
from types import SimpleNamespace
from pathlib import Path

import mujoco
import torch
from mjlab.sim import SimulationCfg

from test_pipeline import tiny_trainer
from readable_mimic.sim import initialize_xi
from readable_mimic.sim_mujoco import EagerSimulation, MujocoSim


def test_full_dr_real_warp_parameter_readback(tiny_trainer):
    trainer = tiny_trainer
    xml = Path(trainer.robot.mjcf_path).read_text().replace(
        "</mujoco>",
        '<actuator><position joint="left_pitch_joint" kp="5" kv="1"/>'
        '<position joint="right_pitch_joint" kp="5" kv="1"/></actuator></mujoco>',
    )
    model = mujoco.MjModel.from_xml_string(xml)
    model.dof_armature[6:] = 0.01
    physics = EagerSimulation(num_envs=2, cfg=SimulationCfg(), model=model, device="cpu")
    backend = MujocoSim.__new__(MujocoSim)
    backend.physics = physics
    backend.robot_description = trainer.robot
    backend.device = "cpu"
    backend.robot = SimpleNamespace(
        indexing=SimpleNamespace(
            body_ids=torch.arange(1, model.nbody),
            geom_ids=torch.arange(model.ngeom),
            joint_v_adr=torch.arange(6, model.nv),
            ctrl_ids=torch.arange(model.nu),
        ),
        find_bodies=lambda name: ([0], [name]),
    )
    env = SimpleNamespace(robot_description=trainer.robot, device="cpu", num_envs=2, simulation=backend)
    xi = initialize_xi(env)
    mass, inertia, armature, damping, frictionloss, kp, kd = xi[:, -7:].unbind(-1)
    for field, expected in (
        ("body_mass", physics.get_default_field("body_mass")[None, 1:] * mass[:, None]),
        ("body_inertia", physics.get_default_field("body_inertia")[None, 1:] * (mass * inertia)[:, None, None]),
        ("dof_armature", 0.01 * armature[:, None].expand(2, 2)),
        ("dof_damping", damping[:, None].expand(2, 2)),
        ("dof_frictionloss", frictionloss[:, None].expand(2, 2)),
    ):
        actual = getattr(physics.model, field)
        actual = actual[:, 1:] if field.startswith("body_") else actual[:, 6:]
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(physics.model.actuator_gainprm[:, :, 0], 5 * kp[:, None].expand(2, 2))
    torch.testing.assert_close(physics.model.actuator_biasprm[:, :, 1], -5 * kp[:, None].expand(2, 2))
    torch.testing.assert_close(physics.model.actuator_biasprm[:, :, 2], -kd[:, None].expand(2, 2))
    torch.testing.assert_close(
        physics.model.body_subtreemass[:, 1], torch.tensor(model.body_subtreemass[1]) * mass,
    )
    physics.forward()
    physics.step()
    assert torch.isfinite(physics.data.qpos).all()
    initialize_xi(env, randomize=False)
    torch.testing.assert_close(physics.model.body_mass[:], physics.get_default_field("body_mass").expand(2, -1))
    torch.testing.assert_close(physics.model.dof_armature[:], physics.get_default_field("dof_armature").expand(2, -1))
