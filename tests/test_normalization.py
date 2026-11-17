import torch

from readable_mimic.common import RunningNorm


def test_running_norm_update_and_freeze_without_tensor_boolean(monkeypatch):
    normalizer = RunningNorm(2, max_count=5)
    first_batch = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    second_batch = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
    buffer_ids = tuple(id(value) for value in normalizer.buffers())

    def unexpected_tensor_boolean(*args):
        raise AssertionError("Normalizer update must not convert a tensor to a Python bool")

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "__bool__", unexpected_tensor_boolean)
        normalizer.update(first_batch)
        normalizer.update(second_batch)
        normalizer.update(torch.full((3, 2), float("nan")))
    combined = torch.cat((first_batch, second_batch))
    torch.testing.assert_close(normalizer.mean, combined.mean(0))
    torch.testing.assert_close(normalizer.var, combined.var(0, unbiased=False))
    assert normalizer.count == 5
    assert tuple(id(value) for value in normalizer.buffers()) == buffer_ids
    restored = RunningNorm(2, max_count=5)
    restored.load_state_dict(normalizer.state_dict())
    restored.update(torch.zeros(3, 2))
    torch.testing.assert_close(restored.mean, normalizer.mean)
    torch.testing.assert_close(restored.var, normalizer.var)


def test_running_norm_finishes_batch_crossing_freeze_threshold():
    normalizer = RunningNorm(2, max_count=4)
    values = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    normalizer.update(values[:3])
    normalizer.update(values[3:])
    torch.testing.assert_close(normalizer.mean, values.mean(0))
    torch.testing.assert_close(normalizer.var, values.var(0, unbiased=False))
    assert normalizer.count == 6
    normalizer.update(torch.ones(2, 2))
    assert normalizer.count == 6
