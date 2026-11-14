from __future__ import annotations
import torch
import torch.nn as nn
from .common import mlp, RunningNorm


class Actor(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh.

    ``ref_skip`` = (offset, q0, alpha) (--pi_ref_skip): reference-action skip connection,
    a = (q_ref(f+1) - q0) / alpha + dec(...), with q_ref(f+1) read from the raw (unnormalized) goal vector at
    ``offset``; the network only learns the correction to tracking the reference with the PD controller."""

    def __init__(
        self,
        observation_dim: int,
        command_dim: int,
        num_actions: int,
        history_length: int = 10,
        motion_latent_dim: int = 32,
        encoder_hidden_dims=(2048, 1024, 512, 256),
        decoder_hidden_dims=(2048, 1024, 512, 256),
        reference_action_skip=None,
    ):
        super().__init__()
        self.history_length = history_length
        self.ref_joint_pos_offset = None
        if reference_action_skip is not None:
            self.ref_joint_pos_offset = reference_action_skip[0]
            self.register_buffer(
                "default_joint_pos", torch.tensor(reference_action_skip[1], dtype=torch.float32)
            )
            self.register_buffer("action_scale", torch.tensor(reference_action_skip[2], dtype=torch.float32))
        self.command_encoder = mlp(command_dim, list(encoder_hidden_dims), motion_latent_dim)
        self.action_decoder = mlp(
            motion_latent_dim + observation_dim * history_length, list(decoder_hidden_dims), num_actions
        )
        with torch.no_grad():
            self.action_decoder[-1].weight.mul_(0.01)
            self.action_decoder[-1].bias.zero_()
        self.observation_normalizer = RunningNorm(observation_dim)
        self.command_normalizer = RunningNorm(command_dim)

    def forward(
        self,
        observation_history: torch.Tensor,
        command: torch.Tensor,
        motion_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """obs_hist: (B, history_length, obs_dim) noisy observations, newest last. ``z`` (optional) is
        ``encode(goal)`` computed beforehand."""
        normalized_observation_history = self.observation_normalizer(observation_history).flatten(-2)
        if motion_latent is None:
            motion_latent = self.encode(command)
        action = self.action_decoder(torch.cat((motion_latent, normalized_observation_history), -1))
        if self.ref_joint_pos_offset is not None:
            ref_joint_pos = command[
                ..., self.ref_joint_pos_offset : self.ref_joint_pos_offset + action.shape[-1]
            ]
            action = action + (ref_joint_pos - self.default_joint_pos) / self.action_scale
        return action

    def encode(self, command: torch.Tensor) -> torch.Tensor:
        """Motion latent z from the goal features (reference only, independent of the robot state)."""
        return torch.tanh(self.command_encoder(self.command_normalizer(command)))


def actor_kwargs(config: dict, robot) -> dict:
    """Policy keyword arguments of a run with the training arguments ``config``."""
    return {"reference_action_skip": reference_action_skip(config, robot)}


def reference_action_skip(config: dict, robot):
    """Policy(ref_skip=...) argument of a run with the training arguments ``config`` (None without --pi_ref_skip)."""
    if not config.get("pi_ref_skip", False):
        return None
    from .motion import GOAL_Q_OFFSET

    return (GOAL_Q_OFFSET, robot.default_joint_pos_list(), robot.action_scale_list())
