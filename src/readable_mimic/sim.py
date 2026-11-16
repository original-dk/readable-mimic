"""Simulation creation, timing, configuration and physical parameter randomization."""

from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from dataclasses import dataclass, field
import re
import torch

CONTROL_DT = 0.02
DECIMATION = 4
PHYSICS_DT = CONTROL_DT / DECIMATION


class EagerSimulation(Simulation):
    def _should_use_cuda_graph(self):
        return False


@dataclass(frozen=True)
class DomainRandomizationConfig:
    friction_range: tuple[float, float] = (0.3, 1.2)
    joint_pos_offset: dict[str, float] = field(default_factory=lambda: {".*": 0.01})
    reset_pos: tuple[float, float, float] = (0.05, 0.05, 0.01)
    reset_rot: tuple[float, float, float] = (0.1, 0.1, 0.2)
    reset_lin_vel: tuple[float, float, float] = (0.5, 0.5, 0.2)
    reset_ang_vel: tuple[float, float, float] = (0.52, 0.52, 0.78)
    reset_joint_pos: float = 0.1
    reset_joint_vel: float = 0.0


def joint_pos_offset_limits(robot, device):
    limits = []
    for joint_name in robot.joint_names:
        limit = 0.0
        for pattern, configured_limit in robot.domain_randomization.joint_pos_offset.items():
            if re.fullmatch(pattern, joint_name):
                limit = configured_limit
        limits.append(limit)
    return torch.tensor(limits, device=device)


def compute_xi_dim(robot):
    return 1 + robot.num_joints + 3


def default_xi(robot, device):
    friction_lower, friction_upper = robot.domain_randomization.friction_range
    return torch.tensor(
        [(friction_lower + friction_upper) / 2] + [0.0] * (robot.num_joints + 3), device=device
    )


def initialize_xi(env, randomize=True):
    """Sample friction / COM once and apply the same values to MuJoCo and replay context."""
    robot = env.robot_description
    simulation = env.simulation
    device = env.device
    xi = default_xi(robot, device).repeat(env.num_envs, 1)
    simulation.expand_model_fields(("body_ipos", "geom_friction"))
    friction = xi[:, :1]
    if randomize:
        friction.uniform_(*robot.domain_randomization.friction_range)
    else:
        friction.fill_(1.0)
    ground_geom_indices = (
        torch.as_tensor(simulation.mj_model.geom_bodyid == 0, device=device).nonzero().flatten()
    )
    robot_geom_indices = env.robot.indexing.geom_ids
    simulation.model.geom_friction[:, ground_geom_indices, 0] = (
        robot.domain_randomization.friction_range[0] if randomize else 1.0
    )
    simulation.model.geom_friction[:, robot_geom_indices, 0] = friction
    center_of_mass_offset = torch.stack(
        [
            torch.empty(env.num_envs, device=device).uniform_(lower, upper)
            for lower, upper in robot.center_of_mass_offset_range
        ],
        -1,
    )
    if not randomize:
        center_of_mass_offset.zero_()
    xi[:, 1 + robot.num_joints :] = center_of_mass_offset
    body_index = env.robot.indexing.body_ids[env.robot.find_bodies(robot.center_of_mass_body)[0][0]]
    simulation.model.body_ipos[:, body_index] = (
        simulation.get_default_field("body_ipos")[body_index] + center_of_mass_offset
    )
    return xi


def randomize_joint_pos_offsets(xi, env_indices, offset_limits):
    num_joints = offset_limits.numel()
    offsets = 2 * torch.rand(len(env_indices), num_joints, device=xi.device) - 1
    xi[env_indices, 1 : 1 + num_joints] = offsets * offset_limits


def create_simulation(robot, num_envs, device, settings):
    settings = dict(settings)
    settings.pop("control_dt")
    settings.pop("decimation")
    joint_capacity = settings.pop("njmax")
    contact_capacity = settings.pop("nconmax")
    simulation_config = SimulationCfg(
        njmax=joint_capacity, nconmax=contact_capacity, mujoco=MujocoCfg(timestep=PHYSICS_DT, **settings)
    )
    scene_config = SceneCfg(
        num_envs=num_envs,
        env_spacing=0.0,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": robot.robot_cfg()},
    )
    scene = Scene(scene_config, device=device)
    simulation = EagerSimulation(
        num_envs=num_envs, cfg=simulation_config, model=scene.compile(), device=device
    )
    scene.initialize(mj_model=simulation.mj_model, model=simulation.model, data=simulation.data)
    return (scene, simulation)
