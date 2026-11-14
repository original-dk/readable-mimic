"""The WM context is [friction, joint_pos_offset, center_of_mass_offset]."""

from dataclasses import dataclass, field
import re
import torch


@dataclass(frozen=True)
class DomainRandomizationConfig:
    friction_range: tuple[float, float] = (0.3, 1.2)
    joint_offset: dict[str, float] = field(default_factory=lambda: {".*": 0.01})
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
        for pattern, configured_limit in robot.domain_randomization.joint_offset.items():
            if re.fullmatch(pattern, joint_name):
                limit = configured_limit
        limits.append(limit)
    return torch.tensor(limits, device=device)


def simulation_parameter_count(robot):
    return 1 + robot.num_joints + 3


def default_simulation_parameters(robot, device):
    friction_lower, friction_upper = robot.domain_randomization.friction_range
    return torch.tensor(
        [(friction_lower + friction_upper) / 2] + [0.0] * (robot.num_joints + 3), device=device
    )


def normalize_simulation_parameters(parameters, robot):
    friction_lower, friction_upper = robot.domain_randomization.friction_range
    center = default_simulation_parameters(robot, parameters.device)
    scales = torch.cat(
        (
            parameters.new_tensor([(friction_upper - friction_lower) / 2]),
            joint_pos_offset_limits(robot, parameters.device).clamp_min(1e-06),
            parameters.new_full((3,), 0.05),
        )
    )
    return (parameters - center) / scales


def initialize_simulation_parameters(environment):
    """Sample friction / COM once and apply the same values to MuJoCo and replay context."""
    robot = environment.robot_description
    simulation = environment.simulation
    device = environment.device
    parameters = default_simulation_parameters(robot, device).repeat(environment.num_envs, 1)
    simulation.expand_model_fields(("body_ipos", "geom_friction"))
    friction = parameters[:, :1]
    friction.uniform_(*robot.domain_randomization.friction_range)
    ground_geom_indices = (
        torch.as_tensor(simulation.mj_model.geom_bodyid == 0, device=device).nonzero().flatten()
    )
    robot_geom_indices = environment.robot.indexing.geom_ids
    simulation.model.geom_friction[:, ground_geom_indices, 0] = robot.domain_randomization.friction_range[0]
    simulation.model.geom_friction[:, robot_geom_indices, 0] = friction
    center_of_mass_offset = torch.stack(
        [
            torch.empty(environment.num_envs, device=device).uniform_(lower, upper)
            for lower, upper in robot.center_of_mass_offset_range
        ],
        -1,
    )
    parameters[:, 1 + robot.num_joints :] = center_of_mass_offset
    body_index = environment.robot.indexing.body_ids[
        environment.robot.find_bodies(robot.center_of_mass_body)[0][0]
    ]
    simulation.model.body_ipos[:, body_index] = (
        simulation.get_default_field("body_ipos")[body_index] + center_of_mass_offset
    )
    return parameters


def randomize_joint_pos_offsets(parameters, environment_indices, robot):
    num_joints = robot.num_joints
    offsets = 2 * torch.rand(len(environment_indices), num_joints, device=parameters.device) - 1
    parameters[environment_indices, 1 : 1 + num_joints] = offsets * joint_pos_offset_limits(
        robot, parameters.device
    )
