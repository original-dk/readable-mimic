from __future__ import annotations
import torch.nn as nn
from .common import mlp, RunningNorm


class Critic(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = mlp(input_dim, [256, 128], 1)
        self.norm = RunningNorm(input_dim)

    def forward(self, x):
        return self.network(self.norm(x)).squeeze(-1)
