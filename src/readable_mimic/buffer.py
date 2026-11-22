"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations
from . import data_transforms as transforms
import torch


def fields(num_joints: int, xi_dim: int) -> dict:
    out = {
        "state": (transforms.compute_state_dim(num_joints), torch.float32),
        "next_state": (transforms.compute_state_dim(num_joints), torch.float32),
        "action": (num_joints, torch.float32),
        "last_action": (num_joints, torch.float32),
        "xi": (xi_dim, torch.float32),
        "reward": ((), torch.float32),
        "motion_frame_index": ((), torch.long),
        "init_root": (7, torch.float32),
        "init_f": ((), torch.long),
        "episode_step": ((), torch.int32),
        "done": ((), torch.bool),
        "transition_cut": ((), torch.bool),
    }
    return out


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        num_envs: int,
        device: str,
        num_joints: int,
        xi_dim: int,
        window_length,
        history_length,
    ):
        self.capacity, self.num_envs, self.device = (capacity, num_envs, device)
        self.fields = fields(num_joints, xi_dim)
        self.data = {}
        for field_name, (shape, dtype) in self.fields.items():
            storage_shape = (capacity, num_envs) + ((shape,) if isinstance(shape, int) else shape)
            self.data[field_name] = torch.zeros(storage_shape, dtype=dtype, device=device)
        self.write_index, self.size = (0, 0)
        self.window_length = window_length
        self.window_offsets = torch.arange(window_length, device=device)
        self.history_lookback = torch.arange(history_length - 1, -1, -1, device=device)[None]

    def _randint(self, n, batch_size: int) -> torch.Tensor:
        return torch.randint(0, n, (batch_size,), device=self.device)

    def add(self, transition: dict):
        for field_name in self.fields:
            self.data[field_name][self.write_index] = transition[field_name]
        self.write_index = (self.write_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _row(self, ages: torch.Tensor) -> torch.Tensor:
        return (self.write_index - self.size + ages) % self.capacity

    def sample_windows(self, batch_size: int):
        window_length = self.window_length
        "k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."
        start_ages = self._randint(self.size - window_length + 1, batch_size)
        env_indices = torch.randint(0, self.num_envs, (batch_size,), device=self.device)
        ages = start_ages.unsqueeze(-1) + self.window_offsets
        rows = self._row(ages)
        window_env_indices = env_indices.unsqueeze(-1).expand_as(rows)
        windows = {
            key: self.data[key][rows, window_env_indices]
            for key in (
                "state",
                "next_state",
                "action",
                "last_action",
                "xi",
                "transition_cut",
            )
        }
        cut_mask = windows["transition_cut"].float()
        mask = torch.cumprod(
            torch.cat((torch.ones(batch_size, 1, device=self.device), 1 - cut_mask[:, :-1]), -1), -1
        )
        windows["mask"] = mask
        return windows

    def sample_starts(self, batch_size: int):
        ages = self._randint(self.size, batch_size)
        env_indices = torch.randint(0, self.num_envs, (batch_size,), device=self.device)
        return self.starts_at(ages, env_indices)

    def starts_at(self, ages: torch.Tensor, env_indices: torch.Tensor):
        row_indices = self._row(ages)
        episode_step = self.data["episode_step"][row_indices, env_indices].long()
        history_rows = self._row(transforms.compute_history_ages(ages, episode_step, self.history_lookback))
        fields = ("state", "last_action", "xi", "motion_frame_index")
        start = {field: self.data[field][row_indices, env_indices] for field in fields}
        for field in ("init_root", "init_f"):
            start[field] = self.data[field][row_indices, env_indices]
        history = {field: self.data[field][history_rows, env_indices[:, None]] for field in fields}
        return (start, history)
