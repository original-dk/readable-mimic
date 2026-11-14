from __future__ import annotations
import torch
import torch.nn as nn
from .task import invariant_feature_dim, invariant_features
from .rotations import mm3, mv3, so3_exp, yaw_mat
from .common import mlp, RunningNorm


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments.

    With ``use_last_action`` the model also sees a_{t-1}: delayed actuators still track the previous
    action's target during the first physics steps of a control step.
    """

    def __init__(
        self,
        num_joints: int,
        num_simulation_parameters: int,
        use_last_action: bool = False,
        hidden_dims=(2048, 2048, 2048),
        structured: bool = False,
        control_time_step: float = 0.02,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.use_last_action = use_last_action
        self.structured, self.control_time_step = (structured, control_time_step)
        self.input_dim = (
            invariant_feature_dim(num_joints)
            + num_joints * (2 if use_last_action else 1)
            + num_simulation_parameters
        )
        self.output_dim = 3 + 3 + 3 + 3 + num_joints + num_joints
        self.network = mlp(
            self.input_dim, list(hidden_dims), self.output_dim, activation=nn.SiLU, layer_norm=True
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.input_normalizer = RunningNorm(self.input_dim)
        self.output_normalizer = RunningNorm(self.output_dim, epsilon=0.0001)
        self.increment_normalizer = RunningNorm(self.output_dim, epsilon=0.0001)

    def inputs(self, state, action, last_action, normalized_simulation_parameters):
        if self.use_last_action:
            return torch.cat(
                (invariant_features(state), action, last_action, normalized_simulation_parameters), -1
            )
        return torch.cat((invariant_features(state), action, normalized_simulation_parameters), -1)

    NEW_VELOCITY_WEIGHT = 0.625

    def _drift(self, state, next_base_lin_vel_w, next_base_ang_vel_w, next_joint_vel):
        """Semi-implicit position update implied by the old and new velocities, in the frames of the
        increments: yaw frame for the root position, body frame for the rotation vector."""
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = state
        base_heading_rot_mat_w = yaw_mat(base_rot_mat_w)
        old_velocity_weight, new_velocity_weight = (
            self.control_time_step * (1 - self.NEW_VELOCITY_WEIGHT),
            self.control_time_step * self.NEW_VELOCITY_WEIGHT,
        )
        return (
            mv3(
                base_heading_rot_mat_w.transpose(-1, -2),
                old_velocity_weight * base_lin_vel_w + new_velocity_weight * next_base_lin_vel_w,
            ),
            mv3(
                base_rot_mat_w.transpose(-1, -2),
                old_velocity_weight * base_ang_vel_w + new_velocity_weight * next_base_ang_vel_w,
            ),
            old_velocity_weight * joint_vel + new_velocity_weight * next_joint_vel,
        )

    def targets(self, state, next_state):
        """Regression targets of the network: the frame-local increments, or with ``structured`` the
        velocity increments plus the position corrections on top of the semi-implicit drift."""
        increments = self.increments(state, next_state)
        if not self.structured:
            return increments
        num_joints = self.num_joints
        (
            delta_base_pos_heading,
            delta_base_rot,
            delta_base_lin_vel,
            delta_base_ang_vel,
            delta_joint_pos,
            delta_joint_vel,
        ) = increments.split((3, 3, 3, 3, num_joints, num_joints), -1)
        base_pos_drift_heading, base_rot_drift, joint_pos_drift = self._drift(
            state, next_state[2], next_state[3], next_state[5]
        )
        return torch.cat(
            (
                delta_base_pos_heading - base_pos_drift_heading,
                delta_base_rot - base_rot_drift,
                delta_base_lin_vel,
                delta_base_ang_vel,
                delta_joint_pos - joint_pos_drift,
                delta_joint_vel,
            ),
            -1,
        )

    @staticmethod
    def increments(state, next_state):
        """Frame-local increments from S to S1."""
        from .rotations import so3_log

        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = state
        (
            next_base_pos_w,
            next_base_rot_mat_w,
            next_base_lin_vel_w,
            next_base_ang_vel_w,
            next_joint_pos,
            next_joint_vel,
        ) = next_state
        base_heading_rot_mat_w = yaw_mat(base_rot_mat_w)
        world_to_base = base_rot_mat_w.transpose(-1, -2)
        return torch.cat(
            (
                mv3(base_heading_rot_mat_w.transpose(-1, -2), next_base_pos_w - base_pos_w),
                so3_log(mm3(world_to_base, next_base_rot_mat_w)),
                mv3(world_to_base, next_base_lin_vel_w - base_lin_vel_w),
                mv3(world_to_base, next_base_ang_vel_w - base_ang_vel_w),
                next_joint_pos - joint_pos,
                next_joint_vel - joint_vel,
            ),
            -1,
        )

    def step(self, state, action, last_action, normalized_simulation_parameters):
        model_input = self.inputs(state, action, last_action, normalized_simulation_parameters)
        normalized_output = self.network(self.input_normalizer(model_input))
        state_increments = normalized_output * self.output_normalizer.std() + self.output_normalizer.mean
        return self.integrate(state, state_increments)

    def integrate(self, state, state_increments):
        """Next state from the denormalized network output ``d`` (the layout of ``targets``)."""
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = state
        base_heading_rot_mat_w = yaw_mat(base_rot_mat_w)
        (
            delta_base_pos_heading,
            delta_base_rot,
            delta_base_lin_vel,
            delta_base_ang_vel,
            delta_joint_pos,
            delta_joint_vel,
        ) = state_increments.split((3, 3, 3, 3, self.num_joints, self.num_joints), -1)
        next_base_lin_vel_w = base_lin_vel_w + mv3(base_rot_mat_w, delta_base_lin_vel)
        next_base_ang_vel_w = base_ang_vel_w + mv3(base_rot_mat_w, delta_base_ang_vel)
        next_joint_vel = joint_vel + delta_joint_vel
        if self.structured:
            base_pos_drift_heading, base_rot_drift, joint_pos_drift = self._drift(
                state, next_base_lin_vel_w, next_base_ang_vel_w, next_joint_vel
            )
            delta_base_pos_heading, delta_base_rot, delta_joint_pos = (
                delta_base_pos_heading + base_pos_drift_heading,
                delta_base_rot + base_rot_drift,
                delta_joint_pos + joint_pos_drift,
            )
        next_base_pos_w = base_pos_w + mv3(base_heading_rot_mat_w, delta_base_pos_heading)
        next_base_rot_mat_w = mm3(base_rot_mat_w, so3_exp(delta_base_rot))
        return (
            next_base_pos_w,
            next_base_rot_mat_w,
            next_base_lin_vel_w,
            next_base_ang_vel_w,
            joint_pos + delta_joint_pos,
            next_joint_vel,
        )

    @staticmethod
    def state_error(predicted_state, state):
        """Frame-local state error components (same layout as ``targets``)."""
        from .rotations import so3_log

        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = state
        (
            pred_base_pos_w,
            pred_base_rot_mat_w,
            pred_base_lin_vel_w,
            pred_base_ang_vel_w,
            pred_joint_pos,
            pred_joint_vel,
        ) = predicted_state
        base_heading_rot_mat_w = yaw_mat(base_rot_mat_w)
        world_to_base = base_rot_mat_w.transpose(-1, -2)
        return torch.cat(
            (
                mv3(base_heading_rot_mat_w.transpose(-1, -2), pred_base_pos_w - base_pos_w),
                so3_log(mm3(world_to_base, pred_base_rot_mat_w)),
                mv3(world_to_base, pred_base_lin_vel_w - base_lin_vel_w),
                mv3(world_to_base, pred_base_ang_vel_w - base_ang_vel_w),
                pred_joint_pos - joint_pos,
                pred_joint_vel - joint_vel,
            ),
            -1,
        )
