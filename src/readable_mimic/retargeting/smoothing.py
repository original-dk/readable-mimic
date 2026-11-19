"""Numeric G1 CSV smoothing and cached robot model; no source-motion loading."""

from functools import lru_cache
import mujoco
import numpy as np
from scipy.signal import butter, sosfiltfilt
from ..robot import get_robot


def smooth_values(values, settings, joint_limits):
    if values.ndim != 2 or values.shape[1] != 36 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite 36-column G1 CSV with at least two frames")
    sos = butter(settings["filter_order"], settings["cutoff_hz"], fs=30, output="sos")
    padlen = min(3 * (2 * len(sos) + 1), len(values) - 1)
    result = values.copy()
    result[:, :3] = sosfiltfilt(sos, values[:, :3], axis=0, padlen=padlen)
    result[:, 7:] = sosfiltfilt(sos, values[:, 7:], axis=0, padlen=padlen)
    result[:, :2] -= result[0, :2].copy()
    quat = values[:, 3:7].copy()
    for index in range(1, len(quat)):
        if np.dot(quat[index], quat[index - 1]) < 0:
            quat[index] *= -1
    quat = sosfiltfilt(sos, quat, axis=0, padlen=padlen)
    norms = np.linalg.norm(quat, axis=1)
    if norms.min() < 0.5:
        raise ValueError("Rotation changes too fast for this quaternion filter")
    result[:, 3:7] = quat / norms[:, None]
    # A continuous amplitude contraction prevents filter overshoot without hard clipping corners.
    center = np.clip(result[:, 7:].mean(axis=0), joint_limits[:, 0], joint_limits[:, 1])
    delta = result[:, 7:] - center
    positive = np.maximum(delta.max(axis=0), 1e-12)
    negative = np.maximum(-delta.min(axis=0), 1e-12)
    scale = np.minimum(
        1, np.minimum((joint_limits[:, 1] - center) / positive, (center - joint_limits[:, 0]) / negative)
    )
    result[:, 7:] = center + delta * scale
    return result


@lru_cache(maxsize=1)
def smoothing_model(mjcf_path, robot_name):
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    robot = get_robot(robot_name)
    indices = np.array([int(model.joint(name).qposadr[0]) for name in robot.joint_names])
    limits = np.array([model.joint(name).range for name in robot.joint_names])
    return model, indices, limits
