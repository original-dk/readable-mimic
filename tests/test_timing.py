import pytest
import torch

from readable_mimic.timing import TrainingTimer


def test_stage_averages_reset_and_cuda_synchronization(monkeypatch):
    timestamps = iter((0.0, 0.1, 0.3, 1.0, 1.3, 1.5))
    synchronized_devices = []
    monkeypatch.setattr("readable_mimic.timing.time.perf_counter", lambda: next(timestamps))
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized_devices.append)
    timer = TrainingTimer(True, "cuda:1")
    for _ in range(2):
        timer.start_iteration()
        timer.record("collect")
        timer.end_iteration()
    assert timer.report() == pytest.approx({"collect_ms": 200.0, "other_ms": 200.0, "iteration_ms": 400.0})
    assert synchronized_devices == [torch.device("cuda:1")] * 6
    assert timer.report() == {}
    timer.record("outside_iteration")
    assert timer.stage_seconds == {}


def test_disabled_timer_does_not_synchronize_or_read_clock(monkeypatch):
    def unexpected_call(*args):
        raise AssertionError("Disabled timer must not read a clock or synchronize CUDA")

    monkeypatch.setattr("readable_mimic.timing.time.perf_counter", unexpected_call)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_call)
    timer = TrainingTimer(False, "cuda:0")
    timer.start_iteration()
    timer.record("collect")
    timer.end_iteration()
    assert timer.report() == {}


def test_parent_totals_do_not_double_count_children(monkeypatch):
    import itertools

    timestamps = itertools.count()
    monkeypatch.setattr("readable_mimic.timing.time.perf_counter", lambda: next(timestamps))
    timer = TrainingTimer(True, "cpu")
    timer.start_iteration()
    for stage in (
        "collect_actor",
        "collect_physics",
        "world_model_prepare",
        "world_model_prepare",
        "actor_backward_gradients",
        "actor_backward_optimizer",
    ):
        timer.record(stage)
    timer.end_iteration()
    metrics = timer.report()
    assert metrics["collect_ms"] == 2000
    assert metrics["world_model_ms"] == 2000
    assert metrics["actor_backward_ms"] == 2000
    assert metrics["iteration_ms"] == 7000
    assert metrics["iteration_ms"] == sum(
        metrics[name] for name in ("collect_ms", "world_model_ms", "actor_backward_ms", "other_ms")
    )
