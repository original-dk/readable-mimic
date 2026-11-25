"""Native mjlab/MuJoCo-Warp delay integration."""
from dataclasses import replace

import mujoco
import pytest
import torch

pytest.importorskip("mjlab")
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from test_pipeline import tiny_trainer as tiny_trainer
from test_action_delay import delayed_trainer as delayed_trainer


def test_delayed_mujoco_collection_and_zero_delay(delayed_trainer, monkeypatch):
    trainer = delayed_trainer
    def robot_cfg():
        return EntityCfg(
            init_state=EntityCfg.InitialStateCfg(pos=(0., 0., .7)),
            spec_fn=lambda: mujoco.MjSpec.from_file(trainer.robot.mjcf_path),
            articulation=EntityArticulationInfoCfg(actuators=(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(".*",), stiffness=5., damping=1.,
                    effort_limit=20., armature=.01,
                ),
            )),
        )
    trainer.robot = replace(trainer.robot, robot_cfg=robot_cfg)
    trainer.config.simulator = "mujoco"
    trainer.create_env()
    env = trainer.env
    env.xi[:, -1] = torch.tensor([0., 1., 2., 4.])
    previous = env.previous_target.clone()
    action = torch.full((4, 2), .2)
    current = env.default_joint_pos + env.xi[:, 1:3] + env.action_scale * action
    captured = []
    setter = env.simulation.robot.set_joint_position_target
    def capture(target, *args, **kwargs):
        captured.append(target.clone())
        return setter(target, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(env.simulation.robot, "set_joint_position_target", capture)
        env.step(action)
    assert len(captured) >= 4
    for step, target in enumerate(captured[:4]):
        expected = torch.where((step < torch.tensor([0, 1, 2, 4]))[:, None], previous, current)
        torch.testing.assert_close(target, expected)
    trainer.collect(4)
    assert all(torch.isfinite(value).all() for value in trainer.updates.update_world_model())
    assert all(torch.isfinite(value).all() for value in trainer.updates.update_actor())
    ids = torch.arange(4)
    state = trainer.motion.ref_state(torch.tensor([10, 11, 12, 13]))
    env.simulation.reset(ids, state)
    env.simulation.step(current)
    expected = env.simulation.state()
    env.simulation.reset(ids, state)
    env.simulation.step(current, previous, torch.zeros(4, dtype=torch.long))
    for value, target in zip(env.simulation.state(), expected):
        torch.testing.assert_close(value, target, atol=1e-6, rtol=1e-6)
