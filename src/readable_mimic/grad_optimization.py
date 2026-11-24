"""Rollout gradient caps, importance sampling and model-ranked resets.

Implements the three update mechanisms in arXiv:2611.27648v1.
The full-DR environment is retained. Model reset ranking uses nominal dynamics.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn

from .motion import BIN_FRAMES


@dataclass(frozen=True)
class GradientOptimizationConfig:
    cap: float = 0.0
    start_draw: bool = False
    candidates: int = 8
    quantiles: int = 32
    memory: int = 200
    rate_limit: float = 30.0
    model_resets: bool = False
    reset_interval: int = 2000
    reset_horizon: int = 32
    rank_batch: int = 32768
    hard_fraction: float = 0.05
    reset_tail: int = 50

    def __post_init__(self):
        if not math.isfinite(self.cap) or self.cap < 0:
            raise ValueError("grad_optimization.cap must be finite and nonnegative")
        if self.start_draw and self.cap == 0:
            raise ValueError("Gradient start sampling requires a positive rollout cap")
        for name in ("candidates", "quantiles", "memory", "reset_interval", "reset_horizon", "rank_batch"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"grad_optimization.{name} must be a positive integer")
        if not math.isfinite(self.rate_limit) or self.rate_limit < 1:
            raise ValueError("grad_optimization.rate_limit must be finite and at least 1")
        if not 0 < self.hard_fraction <= 1:
            raise ValueError("grad_optimization.hard_fraction must be in (0, 1]")
        if self.reset_tail < self.reset_horizon:
            raise ValueError("reset_tail must cover reset_horizon")


def capped_action_gradients(gradients, importance, cap):
    """Inputs have shape (H,B,A) and (B,). Cap inverse-weighted rollout norms."""
    finite = torch.isfinite(gradients).all(dim=0).all(dim=-1)
    safe = torch.where(finite[None, :, None], gradients, 0.0)
    norms = torch.linalg.vector_norm(safe, dim=(0, 2))
    finite = finite & torch.isfinite(norms)
    weighted_norms = norms * importance
    median = torch.nanmedian(torch.where(finite, weighted_norms, float("nan")))
    median = torch.nan_to_num(median, nan=0.0)
    denominator = weighted_norms.clamp_min(torch.finfo(norms.dtype).tiny)
    weights = (cap * median / denominator).clamp(max=1)
    weights = torch.where(weighted_norms == 0, 1.0, weights)
    weights = torch.where(finite, weights, 0.0).detach()
    safe = torch.where(finite[None, :, None], safe, 0.0)
    result = safe * (importance * weights)[None, :, None]
    metrics = dict(
        action_gradient_median=median.detach(),
        action_gradient_max_ratio=(torch.nan_to_num(weighted_norms, posinf=0.0).max()
                                   / median.clamp_min(1e-12)).detach(),
        gradient_cap_fraction=(weights < 1).float().mean(),
        gradient_cap_weight=weights.mean(),
        action_gradient_invalid_fraction=(~finite).float().mean(),
    )
    observed_norms = torch.where(finite, norms, float("nan"))
    return result.detach(), observed_norms.detach(), metrics


class GradientStartSampler(nn.Module):
    """EMA gradient-norm table indexed by candidate critic-value quantile."""
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.register_buffer("norms", torch.ones(config.quantiles, device=device))
        self.register_buffer("seen", torch.zeros(config.quantiles, dtype=torch.bool, device=device))

    @torch.no_grad()
    def draw(self, values, batch_size):
        order = values.argsort()
        quantiles = torch.empty_like(order)
        quantiles[order] = torch.arange(len(values), device=values.device) * self.config.quantiles // len(values)
        rates = self.norms / self.norms.median().clamp_min(1e-12)
        rates = rates.clamp(1 / self.config.rate_limit, self.config.rate_limit)
        probabilities = rates[quantiles]
        probabilities = probabilities / probabilities.sum()
        selected = torch.multinomial(probabilities, batch_size, replacement=True)
        importance = 1 / (len(values) * probabilities[selected])
        return selected, importance, quantiles[selected]

    @torch.no_grad()
    def observe(self, quantiles, norms):
        finite = torch.isfinite(norms)
        median = torch.nanmedian(torch.where(finite, norms, float("nan")))
        median = torch.nan_to_num(median, nan=0.0).clamp_min(1e-12)
        bounded = torch.where(finite, norms.clamp(max=30 * median), 0.0)
        sums = torch.zeros_like(self.norms).scatter_add_(0, quantiles, bounded)
        counts = torch.zeros_like(self.norms).scatter_add_(0, quantiles, finite.float())
        active = counts > 0
        means = sums / counts.clamp_min(1)
        rate = torch.where(self.seen, 1 / self.config.memory, 1.0)
        updated = self.norms.lerp(means.clamp_min(1e-12), rate)
        self.norms.copy_(torch.where(active, updated, self.norms))
        self.seen.logical_or_(active)


class ModelRankedReset(nn.Module):
    """50% uniform bins, 40% current hard bins, 10% previous hard bins."""
    def __init__(self, motion, config):
        super().__init__()
        self.motion = motion
        self.config = config
        end = motion.clip_end[motion.bin_clip] - config.reset_tail
        if motion.reset_starts is None:
            counts = (torch.minimum(motion.bin_start + BIN_FRAMES - 1, end)
                      - motion.bin_start + 1).clamp_min(0)
            starts = motion.bin_start
        else:
            starts, original_counts = motion.reset_starts
            columns = torch.arange(starts.shape[1], device=starts.device)[None]
            allowed = (columns < original_counts[:, None]) & (starts <= end[:, None])
            counts = allowed.sum(-1)
            # Preserve only legal starts; padded entries sort after every legal frame.
            starts = torch.where(allowed, starts, torch.iinfo(starts.dtype).max).sort(-1).values
        valid = (counts > 0).nonzero().flatten()
        if not len(valid):
            raise ValueError("No legal model-ranked reset starts before the clip tail")
        self.register_buffer("valid_bins", valid, persistent=False)
        self.register_buffer("starts", starts, persistent=False)
        self.register_buffer("counts", counts, persistent=False)
        self.register_buffer("hard", torch.zeros(len(valid), dtype=torch.bool, device=valid.device))
        self.register_buffer("past", torch.zeros_like(self.hard))
        self.register_buffer("last_rank", torch.tensor(-1, device=valid.device, dtype=torch.long))

    def ranking_frames(self):
        if self.starts.ndim == 1:
            return self.starts[self.valid_bins]
        return self.starts[self.valid_bins, 0]

    @torch.no_grad()
    def sample(self, count):
        device = self.valid_bins.device
        uniform = torch.randint(len(self.valid_bins), (count,), device=device)
        hard = self.hard.nonzero().flatten()
        past = self.past.nonzero().flatten()
        choice = torch.rand(count, device=device)
        selected = uniform
        if len(hard):
            hard_draw = hard[torch.randint(len(hard), (count,), device=device)]
            selected = torch.where((choice >= 0.5) & (choice < 0.9), hard_draw, selected)
        if len(past):
            past_draw = past[torch.randint(len(past), (count,), device=device)]
            selected = torch.where(choice >= 0.9, past_draw, selected)
        bins = self.valid_bins[selected]
        offsets = (torch.rand(count, device=device) * self.counts[bins]).long()
        if self.starts.ndim == 1:
            return self.starts[bins] + offsets
        return self.starts[bins, offsets]

    @torch.no_grad()
    def refresh(self, score, iteration):
        frames = self.ranking_frames()
        scores = torch.cat([score(chunk, self.config.reset_horizon)
                            for chunk in frames.split(self.config.rank_batch)])
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite model-ranked reset scores")
        selected = scores.argsort()[:max(1, math.ceil(len(scores) * self.config.hard_fraction))]
        self.past.logical_or_(self.hard)
        self.hard.zero_()
        self.hard[selected] = True
        self.last_rank.fill_(iteration)
        return dict(model_reset_hard_bins=len(selected), model_reset_bins=len(scores),
                    model_reset_min_return=float(scores.min()), model_reset_mean_return=float(scores.mean()))
