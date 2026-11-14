from __future__ import annotations
import torch
import torch.nn as nn
from .mdp import inv_dim, invariant_features
from .rotations import mm3, mv3, so3_exp, yaw_mat

def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool=False) -> nn.Sequential:
    layers, d = ([], inp)
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)

class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float=200000000.0, eps: float=0.01):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('count', torch.zeros(()))
        self.max_count, self.eps = (max_count, eps)
    no_sync: bool = False

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if not RunningNorm.no_sync and self.count >= self.max_count:
            return
        n = x.shape[0]
        m, base_lin_vel_w = (x.mean(0), x.var(0, unbiased=False))
        tot = self.count + n
        delta = m - self.mean
        mean = self.mean + delta * n / tot
        var = (self.var * self.count + base_lin_vel_w * n + delta ** 2 * self.count * n / tot) / tot
        if RunningNorm.no_sync:
            live = self.count < self.max_count
            self.mean.copy_(torch.where(live, mean, self.mean))
            self.var.copy_(torch.where(live, var, self.var))
            self.count.copy_(torch.where(live, tot, self.count))
        else:
            self.mean.copy_(mean)
            self.var = var
            self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps ** 2)

    def std(self):
        return torch.sqrt(self.var + self.eps ** 2)

def run_mlp(net, inputs):
    return net(inputs)
