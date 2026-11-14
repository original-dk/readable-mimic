from __future__ import annotations
import torch
import torch.nn as nn


def mlp(
    input_dim: int, hidden: list[int], output_dim: int, activation=nn.ELU, layer_norm: bool = False
) -> nn.Sequential:
    layers, layer_input_dim = ([], input_dim)
    for layer_output_dim in hidden:
        layers.append(nn.Linear(layer_input_dim, layer_output_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(layer_output_dim))
        layers.append(activation())
        layer_input_dim = layer_output_dim
    layers.append(nn.Linear(layer_input_dim, output_dim))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, num_features: int, max_count: float = 200000000.0, epsilon: float = 0.01):
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.ones(num_features))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.epsilon = (max_count, epsilon)

    @torch.no_grad()
    def update(self, values: torch.Tensor):
        values = values.reshape(-1, values.shape[-1]).float()
        if self.count >= self.max_count:
            return
        batch_count = values.shape[0]
        batch_mean, batch_variance = (values.mean(0), values.var(0, unbiased=False))
        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        mean = self.mean + delta * batch_count / total_count
        var = (
            self.var * self.count
            + batch_variance * batch_count
            + delta**2 * self.count * batch_count / total_count
        ) / total_count
        self.mean.copy_(mean)
        self.var = var
        self.count = total_count

    def forward(self, values):
        return (values - self.mean) / torch.sqrt(self.var + self.epsilon**2)

    def std(self):
        return torch.sqrt(self.var + self.epsilon**2)
