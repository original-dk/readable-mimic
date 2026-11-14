from __future__ import annotations
import torch
import torch.nn as nn
from .mdp import inv_dim, invariant_features
from .rotations import mm3, mv3, so3_exp, yaw_mat
from .common import mlp, RunningNorm, run_mlp


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments.

    With ``use_prev_action`` the model also sees a_{t-1}: delayed actuators still track the previous
    action's target during the first physics steps of a control step.
    """

    def __init__(self, num_j: int, xi_dim: int, use_prev_action: bool=False, hidden=(2048, 2048, 2048), contact_modes: bool=False, structured: bool=False, dt: float=0.02):
        """``contact_modes`` (InstantTrack): the last layer predicts the two feet's next-step ground contact
        and one increment head per contact mode (flight, left, right, double support); the increment is the
        mixture of the heads weighted by the predicted mode probabilities, so each head stays smooth within
        its mode instead of one MLP smoothing across touch-down and lift-off."""
        super().__init__()
        self.num_j = num_j
        self.use_prev_action = use_prev_action
        self.contact_modes = contact_modes
        self.structured, self.dt = (structured, dt)
        self.in_dim = inv_dim(num_j) + num_j * (2 if use_prev_action else 1) + xi_dim
        self.out_dim = 3 + 3 + 3 + 3 + num_j + num_j
        n_out = 4 * self.out_dim + 2 if contact_modes else self.out_dim
        self.net = mlp(self.in_dim, list(hidden), n_out, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=0.0001)
        self.inc_norm = RunningNorm(self.out_dim, eps=0.0001)

    def inputs(self, S, action, last_action, normalized_simulation_parameters):
        if self.use_prev_action:
            return torch.cat((invariant_features(S), action, last_action, normalized_simulation_parameters), -1)
        return torch.cat((invariant_features(S), action, normalized_simulation_parameters), -1)
    DRIFT_W = 0.625

    def _drift(self, S, v1, w1, qd1):
        """Semi-implicit position update implied by the old and new velocities, in the frames of the
        increments: yaw frame for the root position, body frame for the rotation vector."""
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = S
        H = yaw_mat(base_rot_mat_w)
        action, b = (self.dt * (1 - self.DRIFT_W), self.dt * self.DRIFT_W)
        return (mv3(H.transpose(-1, -2), action * base_lin_vel_w + b * v1), mv3(base_rot_mat_w.transpose(-1, -2), action * base_ang_vel_w + b * w1), action * joint_vel + b * qd1)

    def targets(self, S, S1):
        """Regression targets of the network: the frame-local increments, or with ``structured`` the
        velocity increments plus the position corrections on top of the semi-implicit drift."""
        inc = self.increments(S, S1)
        if not self.structured:
            return inc
        J = self.num_j
        dp, dth, dv, dw, dq, dqd = inc.split((3, 3, 3, 3, J, J), -1)
        gp, gth, gq = self._drift(S, S1[2], S1[3], S1[5])
        return torch.cat((dp - gp, dth - gth, dv, dw, dq - gq, dqd), -1)

    @staticmethod
    def increments(S, S1):
        """Frame-local increments from S to S1."""
        from .rotations import so3_log
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(base_rot_mat_w)
        Rt = base_rot_mat_w.transpose(-1, -2)
        return torch.cat((mv3(H.transpose(-1, -2), p1 - base_pos_w), so3_log(mm3(Rt, R1)), mv3(Rt, v1 - base_lin_vel_w), mv3(Rt, w1 - base_ang_vel_w), q1 - joint_pos, qd1 - joint_vel), -1)

    def step(self, S, action, last_action, normalized_simulation_parameters):
        return self.step_c(S, action, last_action, normalized_simulation_parameters)[0]

    def step_c(self, S, action, last_action, normalized_simulation_parameters):
        """Next state and next-step foot-contact logits (B, 2), or None without ``contact_modes``."""
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, action, last_action, normalized_simulation_parameters)))
        logits = None
        if self.contact_modes:
            logits = y[..., -2:]
            cl, cr = torch.sigmoid(logits).unbind(-1)
            mode_w = torch.stack(((1 - cl) * (1 - cr), cl * (1 - cr), (1 - cl) * cr, cl * cr), -1)
            y = (y[..., :-2].unflatten(-1, (4, self.out_dim)) * mode_w.unsqueeze(-1)).sum(-2)
        return (self.integrate(S, y * self.out_norm.std() + self.out_norm.mean), logits)

    def integrate(self, S, d):
        """Next state from the denormalized network output ``d`` (the layout of ``targets``)."""
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = S
        H = yaw_mat(base_rot_mat_w)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.num_j, self.num_j), -1)
        v1 = base_lin_vel_w + mv3(base_rot_mat_w, dv)
        w1 = base_ang_vel_w + mv3(base_rot_mat_w, dw)
        qd1 = joint_vel + dqd
        if self.structured:
            gp, gth, gq = self._drift(S, v1, w1, qd1)
            dp, dth, dq = (dp + gp, dth + gth, dq + gq)
        p1 = base_pos_w + mv3(H, dp)
        R1 = mm3(base_rot_mat_w, so3_exp(dth))
        return (p1, R1, v1, w1, joint_pos + dq, qd1)

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        from .rotations import so3_log
        base_pos_w, base_rot_mat_w, base_lin_vel_w, base_ang_vel_w, joint_pos, joint_vel = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(base_rot_mat_w)
        Rt = base_rot_mat_w.transpose(-1, -2)
        return torch.cat((mv3(H.transpose(-1, -2), ph - base_pos_w), so3_log(mm3(Rt, Rh)), mv3(Rt, vh - base_lin_vel_w), mv3(Rt, wh - base_ang_vel_w), qh - joint_pos, qdh - joint_vel), -1)
