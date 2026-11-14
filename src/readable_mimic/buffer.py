"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations
import torch
from .mdp import state_dim


def fields(num_j: int, xi_dim: int, obs_dim: int | None = None, obs_state_dim: int | None = None) -> dict:
    out = {
        "state": (state_dim(num_j), torch.float32),
        "next_state": (state_dim(num_j), torch.float32),
        "action": (num_j, torch.float32),
        "last_action": (num_j, torch.float32),
        "simulation_parameters": (xi_dim, torch.float32),
        "reward": ((), torch.float32),
        "motion_frame_index": ((), torch.long),
        "episode_step": ((), torch.int32),
        "done": ((), torch.bool),
        "transition_cut": ((), torch.bool),
    }
    if obs_dim is not None:
        out["observation"] = (obs_dim, torch.float32)
        out["observation_pipeline_state"] = (obs_state_dim, torch.float32)
    return out


class ReplayBuffer:
    def __init__(
        self,
        T: int,
        N: int,
        device: str,
        num_j: int,
        xi_dim: int,
        obs_dim: int | None = None,
        obs_state_dim: int | None = None,
    ):
        self.T, self.N, self.device = (T, N, device)
        self.fields = fields(num_j, xi_dim, obs_dim, obs_state_dim)
        self.data = {}
        for k, (shape, dt) in self.fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = (0, 0)

    def _randint(self, n, B: int) -> torch.Tensor:
        return torch.randint(0, n, (B,), device=self.device)

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        return (self.ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0 = self._randint(self.size - k + 1, B)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {
            key: self.data[key][rows, e]
            for key in (
                "state",
                "next_state",
                "action",
                "last_action",
                "simulation_parameters",
                "transition_cut",
            )
        }
        d = out["transition_cut"].float()
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["mask"] = mask
        return out

    def sample_starts(self, B: int, hist: int):
        age = self._randint(self.size, B)
        env = torch.randint(0, self.N, (B,), device=self.device)
        return self.starts_at(age, env, hist)

    def starts_at(self, age: torch.Tensor, env: torch.Tensor, hist: int):
        """Start points at the transitions (age, env) (age 0 = oldest) plus their observation history (oldest first)."""
        row = self._row(age)
        episode_step = self.data["episode_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)
        back = torch.minimum(back.unsqueeze(0), episode_step.unsqueeze(-1))
        back = torch.minimum(back, age.unsqueeze(-1))
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        stateful = "observation" in self.data
        start = {
            k: self.data[k][row, env]
            for k in ("state", "last_action", "simulation_parameters", "motion_frame_index")
            + (("observation_pipeline_state",) if stateful else ())
        }
        hist_d = {
            k: self.data[k][hrows, henv]
            for k in ("state", "last_action", "simulation_parameters", "motion_frame_index")
            + (("observation",) if stateful else ())
        }
        return (start, hist_d)
