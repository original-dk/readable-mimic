"""Shared simulation timing, xi and backend selection; engine code lives in sim_mujoco/sim_genesis."""

from dataclasses import dataclass, field
import re
import math
import torch

CONTROL_DT = 0.02
DECIMATION = 4
PHYSICS_DT = CONTROL_DT / DECIMATION


# Appended after friction, joint offsets and torso COM xyz. Shared across links/joints per environment.
DYNAMICS_RANGES = (
    "mass_scale_range", "inertia_scale_range", "armature_scale_range",
    "passive_damping_add_range", "frictionloss_add_range", "kp_scale_range", "kd_scale_range",
)
DYNAMICS_DEFAULTS = (1., 1., 1., 0., 0., 1., 1.)


@dataclass(frozen=True)
class DomainRandomizationConfig:
    friction_range: tuple[float, float] = (0.3, 1.2)
    joint_pos_offset: dict[str, float] = field(default_factory=lambda: {".*": 0.01})
    mass_scale_range: tuple[float, float] = (0.9, 1.1)
    inertia_scale_range: tuple[float, float] = (0.9, 1.1)
    armature_scale_range: tuple[float, float] = (0.8, 1.2)
    passive_damping_add_range: tuple[float, float] = (0.0, 0.05)
    frictionloss_add_range: tuple[float, float] = (0.0, 0.1)
    kp_scale_range: tuple[float, float] = (0.8, 1.2)
    kd_scale_range: tuple[float, float] = (0.8, 1.2)
    action_delay_max_ms: float = 0.0

    reset_pos: tuple[float, float, float] = (0.05, 0.05, 0.01)
    reset_rot: tuple[float, float, float] = (0.1, 0.1, 0.2)
    reset_lin_vel: tuple[float, float, float] = (0.5, 0.5, 0.2)
    reset_ang_vel: tuple[float, float, float] = (0.52, 0.52, 0.78)
    reset_joint_pos: float = 0.1
    reset_joint_vel: float = 0.0

    def __post_init__(self):
        delay = self.action_delay_max_ms
        lag = round(delay / (PHYSICS_DT * 1000)) if math.isfinite(delay) else -1
        if lag < 0 or lag > DECIMATION or not math.isclose(delay, lag * PHYSICS_DT * 1000):
            raise ValueError("action_delay_max_ms must be one of 0, 5, 10, 15, 20")


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
    return 1 + robot.num_joints + 3 + len(DYNAMICS_RANGES) + int(robot.has_action_delay)


def default_xi(robot, device):
    friction_lower, friction_upper = robot.domain_randomization.friction_range
    return torch.tensor(
        [(friction_lower + friction_upper) / 2] + [0.0] * (robot.num_joints + 3) + list(DYNAMICS_DEFAULTS) + ([0.0] if robot.has_action_delay else []), device=device
    )


def randomize_joint_pos_offsets(xi, env_indices, offset_limits):
    num_joints = offset_limits.numel()
    offsets = 2 * torch.rand(len(env_indices), num_joints, device=xi.device) - 1
    xi[env_indices, 1 : 1 + num_joints] = offsets * offset_limits


def initialize_xi(env, randomize=True):
    robot = env.robot_description
    xi = default_xi(robot, env.device).repeat(env.num_envs, 1)
    if randomize:
        xi[:, 0].uniform_(*robot.domain_randomization.friction_range)
        xi[:, 1 + robot.num_joints : 4 + robot.num_joints] = torch.stack(
            [
                torch.empty(env.num_envs, device=env.device).uniform_(lower, upper)
                for lower, upper in robot.center_of_mass_offset_range
            ],
            -1,
        )
        for index, name in enumerate(DYNAMICS_RANGES, start=4 + robot.num_joints):
            lower, upper = getattr(robot.domain_randomization, name)
            if (
                not math.isfinite(lower) or not math.isfinite(upper)
                or lower > upper or lower < 0 or ("scale" in name and lower <= 0)
            ):
                raise ValueError(f"Invalid domain randomization range {name}: {(lower, upper)}")
            xi[:, index].uniform_(lower, upper)
    else:
        xi[:, 0] = 1.0
    env.simulation.apply_xi(xi, randomize)
    return xi


def delayed_position_target(current, previous, delay_steps, substep):
    """Switch from the preceding command to the current command at its arrival step."""
    return torch.where((substep < delay_steps)[:, None], previous, current)


def create_simulation(robot, num_envs, device, settings, backend="mujoco", genesis_settings=None):
    if backend == "genesis":
        from .sim_genesis import GenesisSim

        return GenesisSim(robot, num_envs, device, settings, genesis_settings or {})
    if backend == "mujoco":
        from .sim_mujoco import MujocoSim

        return MujocoSim(robot, num_envs, device, settings)
    raise ValueError(f"Unknown simulator: {backend}")
