"""Left-right (sagittal-plane) mirror of states, actions and sim context, for world-model data augmentation.

The mirror is the reflection M = diag(1, -1, 1) of the world y axis: positions and linear velocities
map to M x, rotations to M R M, angular velocities (pseudo-vectors) to -M w, and joints to their
left/right partner, with the sign flipped for roll and yaw joints (and the waist's roll and yaw).
"""

from __future__ import annotations

import torch

from .robot import RobotSpec

_FLIP_TYPES = ("roll", "yaw")


class Mirror:
    def __init__(self, robot: RobotSpec, xi_slices: dict, device: str):
        names = list(robot.joint_names)
        perm, sign = [], []
        for n in names:
            partner = n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
            perm.append(names.index(partner))
            sign.append(-1.0 if any(f"_{t}_" in n for t in _FLIP_TYPES) else 1.0)
        self.perm = torch.tensor(perm, device=device)
        self.sign = torch.tensor(sign, device=device)
        self.M = torch.diag(torch.tensor([1.0, -1.0, 1.0], device=device))
        self.xi_slices = xi_slices
        unknown = set(xi_slices) - {"friction", "joint_offset", "com"}
        assert not unknown, f"mirror of sim-context entries {unknown} not defined"

    def joints(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.perm] * self.sign

    def state(self, S):
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = S
        M = self.M
        return (base_pos_w @ M, M @ base_rot_mat_w @ M, base_lin_vel_w @ M, -(base_ang_vel_w @ M), self.joints(joint_pos), self.joints(joint_vel))

    def simulation_parameters(self, simulation_parameters: torch.Tensor) -> torch.Tensor:
        out = simulation_parameters.clone()
        sl = self.xi_slices["joint_offset"]
        out[..., sl] = self.joints(simulation_parameters[..., sl])
        c = self.xi_slices["com"]
        out[..., c.start + 1] = -simulation_parameters[..., c.start + 1]  # lateral COM offset
        return out
