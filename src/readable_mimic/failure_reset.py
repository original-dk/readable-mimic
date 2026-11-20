"""Recent failure feedback for a bounded share of training resets."""

import torch

from .motion import BIN_FRAMES
from .sim import CONTROL_DT


class FailureReset:
    def __init__(self, motion, num_envs, fraction, max_episode_steps):
        self.fraction = fraction
        self.bin_start = motion.bin_start
        self.bin_end = torch.minimum(motion.bin_start + BIN_FRAMES - 1, motion.clip_end[motion.bin_clip])
        self.target_bin = torch.full((num_envs,), -1, device=motion.device, dtype=torch.long)
        self.attempts = torch.zeros(motion.num_bins, device=motion.device)
        self.failures = torch.zeros_like(self.attempts)
        self.recent_attempts = torch.zeros_like(self.attempts)
        self.recent_failures = torch.zeros_like(self.attempts)
        self.weight_cdf = torch.zeros_like(self.attempts)
        # A supported start about half a second before the target, within this clip.
        legal = motion.reset_starts[0].flatten().unique(sorted=True)
        clip_start = motion.clip_start[motion.bin_clip]
        lead = round(0.5 / CONTROL_DT)
        before = (torch.searchsorted(legal, self.bin_start - lead, right=True) - 1).clamp_min(0)
        first_in_clip = torch.searchsorted(legal, clip_start)
        self.start_frame = legal[torch.maximum(before, first_in_clip)]
        # Termination tests the next frame. A reset on the target's first
        # evaluable frame would skip it, so require a strictly earlier start.
        first_tested_frame = torch.maximum(self.bin_start, clip_start + 1)
        self.reachable = (self.start_frame < first_tested_frame) & (
            self.bin_end - self.start_frame <= max_episode_steps
        )
        self.stats = torch.zeros(5, dtype=torch.long, device=motion.device)

    def choose(self, env_indices, uniform_frames):
        total = self.weight_cdf[-1]
        draw = torch.rand(len(env_indices), device=uniform_frames.device) * total
        bins = torch.searchsorted(self.weight_cdf, draw, right=True).clamp_max(len(self.bin_start) - 1)
        priority = (total > 0) & (torch.rand_like(draw) < self.fraction)
        self.target_bin[env_indices] = torch.where(priority, bins, -1)
        self.stats[0] += len(env_indices)
        self.stats[1] += priority.sum()
        return torch.where(priority, self.start_frame[bins], uniform_frames)

    def observe(self, frames, termination, done):
        bins = torch.searchsorted(self.bin_start, frames, right=True) - 1
        passed = (frames == self.bin_end[bins]) & ~termination
        self.attempts.index_add_(0, bins, (passed | termination).float())
        self.failures.index_add_(0, bins, termination.float())

        targeted = self.target_bin >= 0
        at_target = targeted & (bins == self.target_bin)
        success = at_target & passed
        failed = at_target & termination
        missed = targeted & done & ~(success | failed)
        self.stats[2] += success.sum()
        self.stats[3] += failed.sum()
        self.stats[4] += missed.sum()
        self.target_bin.masked_fill_(success | done, -1)

    def report(self):
        # One log window is the unit of decay. A bucket improves when it is
        # passed repeatedly, and stale failure evidence fades across windows.
        decay = 0.8
        self.recent_attempts.mul_(decay).add_(self.attempts)
        self.recent_failures.mul_(decay).add_(self.failures)
        weights = self.recent_failures / (self.recent_attempts + 4)
        self.weight_cdf.copy_((weights * self.reachable).cumsum(0))
        self.attempts.zero_()
        self.failures.zero_()
        resets, priority, passed, failed, missed = self.stats.tolist()
        self.stats.zero_()
        completed = passed + failed
        return dict(
            failure_reset_count=priority,
            failure_reset_share=priority / max(resets, 1),
            failure_target_passes=passed,
            failure_target_failures=failed,
            failure_target_misses=missed,
            failure_target_pass_rate=passed / completed if completed else None,
        )
