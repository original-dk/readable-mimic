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
    def __init__(self, robot: RobotSpec, device: str):
        joint_names = list(robot.joint_names)
        joint_permutation, joint_signs = ([], [])
        for joint_name in joint_names:
            mirror_joint_name = (
                joint_name.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
            )
            joint_permutation.append(joint_names.index(mirror_joint_name))
            joint_signs.append(
                -1.0 if any((f"_{axis_name}_" in joint_name for axis_name in _FLIP_TYPES)) else 1.0
            )
        self.joint_permutation = torch.tensor(joint_permutation, device=device)
        self.joint_signs = torch.tensor(joint_signs, device=device)
        self.reflection_matrix = torch.diag(torch.tensor([1.0, -1.0, 1.0], device=device))
        self.num_joints = robot.num_joints

    def joints(self, values: torch.Tensor) -> torch.Tensor:
        return values[..., self.joint_permutation] * self.joint_signs

    def state(self, state):
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = state
        reflection_matrix = self.reflection_matrix
        return (
            base_pos_w @ reflection_matrix,
            reflection_matrix @ base_rot_mat_w @ reflection_matrix,
            base_lin_vel_w @ reflection_matrix,
            -(base_ang_vel_w @ reflection_matrix),
            self.joints(joint_pos),
            self.joints(joint_vel),
        )

    def xi(self, xi: torch.Tensor) -> torch.Tensor:
        mirrored_xi = xi.clone()
        joint_pos_offset = xi[..., 1 : 1 + self.num_joints]
        mirrored_xi[..., 1 : 1 + self.num_joints] = self.joints(joint_pos_offset)
        center_of_mass_y_index = 1 + self.num_joints + 1
        mirrored_xi[..., center_of_mass_y_index] *= -1
        return mirrored_xi
