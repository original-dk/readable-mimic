"""Differentiable actor return from imagined rewards and critic bootstrap."""

import torch


def actor_objective(rewards, termination_probabilities, bootstrap_value, gamma):
    horizon = rewards.shape[0]
    survival_after_step = torch.cumprod(1 - termination_probabilities, 0)
    survival_weights = torch.cat((torch.ones_like(rewards[:1]), survival_after_step[:-1]), 0)
    discounts = gamma ** torch.arange(horizon, device=rewards.device, dtype=torch.float32)[:, None]
    actor_return = (discounts * survival_weights * rewards).sum(0)
    final_survival = survival_after_step[-1]
    actor_return += gamma**horizon * final_survival * bootstrap_value
    return (actor_return, survival_weights, final_survival)
