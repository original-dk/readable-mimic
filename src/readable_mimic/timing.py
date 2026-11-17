"""Optional synchronized wall-clock timing of consecutive training stages."""

import time
import torch


class TrainingTimer:
    def __init__(self, enabled, device):
        self.enabled = enabled
        self.device = torch.device(device)
        self.last_stage_time = None
        self.stage_seconds = {}
        self.iterations = 0

    def timestamp(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def start_iteration(self):
        if self.enabled:
            self.last_stage_time = self.timestamp()

    def record(self, stage):
        if self.last_stage_time is None:
            return
        current_time = self.timestamp()
        self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + (current_time - self.last_stage_time)
        self.last_stage_time = current_time

    def end_iteration(self):
        if self.enabled:
            self.record("other")
            self.last_stage_time = None
            self.iterations += 1

    def report(self):
        if not self.iterations:
            return {}
        metrics = {
            f"{stage}_ms": seconds * 1000 / self.iterations for stage, seconds in self.stage_seconds.items()
        }
        metrics["iteration_ms"] = sum(metrics.values())
        # Parent totals are derived from disjoint stages, never timed a second time.
        for group in ("collect", "world_model", "actor_backward"):
            children = [value for name, value in metrics.items() if name.startswith(f"{group}_")]
            if children:
                metrics[f"{group}_ms"] = sum(children)
        self.stage_seconds.clear()
        self.iterations = 0
        return metrics
