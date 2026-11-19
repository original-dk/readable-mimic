"""Numerical checks for offline smoothing, including large turns and quaternion sign flips."""

import numpy as np
from scipy.spatial.transform import Rotation
from readable_mimic.retargeting.smoothing import smooth_values


def sample_motion():
    time = np.arange(180) / 30
    values = np.zeros((len(time), 36))
    values[:, 2] = 0.8
    values[:, 6] = 1
    return time, values


def test_smoothing_attenuates_high_frequency_without_time_shift():
    time, values = sample_motion()
    slow = 0.3 * np.sin(2 * np.pi * time)
    values[:, 7] = slow + 0.03 * np.sin(2 * np.pi * 10 * time)
    limits = np.tile([-2, 2], (29, 1))
    result = smooth_values(values, dict(cutoff_hz=5, filter_order=2), limits)
    assert np.sqrt(np.mean((result[15:-15, 7] - slow[15:-15]) ** 2)) < 0.005
    assert np.argmax(np.correlate(result[15:-15, 7], slow[15:-15], mode="full")) == len(slow[15:-15]) - 1
    assert result.shape == values.shape


def test_smoothing_preserves_large_turn_and_handles_quaternion_signs():
    _, values = sample_motion()
    yaw = np.linspace(0, 2 * np.pi, len(values))
    values[:, 3:7] = Rotation.from_rotvec(
        np.column_stack([np.zeros_like(yaw), np.zeros_like(yaw), yaw])
    ).as_quat()
    values[::3, 3:7] *= -1
    result = smooth_values(values, dict(cutoff_hz=5, filter_order=2), np.tile([-2, 2], (29, 1)))
    np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1, atol=1e-12)
    error = (
        Rotation.from_rotvec(np.column_stack([np.zeros_like(yaw), np.zeros_like(yaw), yaw])).inv()
        * Rotation.from_quat(result[:, 3:7])
    ).magnitude()
    assert error.max() < 0.01
    assert np.all(np.sum(result[1:, 3:7] * result[:-1, 3:7], axis=1) > 0)


def test_smoothing_limits_are_respected_without_clipping_and_constant_motion_is_unchanged():
    _, values = sample_motion()
    values[:, 7] = np.r_[np.zeros(90), np.ones(90)]
    result = smooth_values(values, dict(cutoff_hz=5, filter_order=2), np.tile([0, 1], (29, 1)))
    assert result[:, 7:].min() >= 0 and result[:, 7:].max() <= 1
    np.testing.assert_allclose(result[:, 2], 0.8)
    short = smooth_values(values[:3], dict(cutoff_hz=5, filter_order=2), np.tile([-2, 2], (29, 1)))
    assert np.isfinite(short).all()
