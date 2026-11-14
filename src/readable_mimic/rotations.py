"""Batched rotation utilities (quaternions are wxyz, rotation matrices act on column vectors)."""

from __future__ import annotations
import torch
import torch.nn.functional as F

ELEMENTWISE = False


def mm3(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A @ B


def mv3(A: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return (A @ vector.unsqueeze(-1)).squeeze(-1)


def quat_to_mat(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = F.normalize(quaternion, dim=-1)
    scalar_component, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - scalar_component * z),
            2 * (x * z + scalar_component * y),
            2 * (x * y + scalar_component * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - scalar_component * x),
            2 * (x * z - scalar_component * y),
            2 * (y * z + scalar_component * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Robust matrix -> quaternion (wxyz, w >= 0)."""
    m00, m01, m02 = (m[..., 0, 0], m[..., 0, 1], m[..., 0, 2])
    m10, m11, m12 = (m[..., 1, 0], m[..., 1, 1], m[..., 1, 2])
    m20, m21, m22 = (m[..., 2, 0], m[..., 2, 1], m[..., 2, 2])
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    cands = torch.stack(
        (
            torch.stack((qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1),
            torch.stack(((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1),
            torch.stack(((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1),
            torch.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1),
        ),
        dim=-2,
    )
    idx = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    quaternion = torch.gather(cands, -2, idx[..., None, None].expand(idx.shape + (1, 4))).squeeze(-2)
    quaternion = F.normalize(quaternion, dim=-1)
    return quaternion * torch.where(quaternion[..., :1] < 0, -1.0, 1.0)


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues for a fixed unit axis (..., 3) and angle (...)."""
    x, y, z = axis.unbind(-1)
    c, s = (torch.cos(angle), torch.sin(angle))
    C = 1 - c
    return torch.stack(
        (
            c + x * x * C,
            x * y * C - z * s,
            x * z * C + y * s,
            y * x * C + z * s,
            c + y * y * C,
            y * z * C - x * s,
            z * x * C - y * s,
            z * y * C + x * s,
            c + z * z * C,
        ),
        dim=-1,
    ).reshape(angle.shape + (3, 3))


def so3_exp(vector: torch.Tensor) -> torch.Tensor:
    """Rotation vector (..., 3) -> matrix, smooth at zero."""
    theta2 = (vector * vector).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-08
    action = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + 1e-12))
    K = skew(vector)
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype).expand(K.shape)
    return eye + action[..., None] * K + b[..., None] * mm3(K, K)


def so3_log(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Matrix -> rotation vector via the quaternion (stable near 0 and pi)."""
    quaternion = mat_to_quat(rotation_matrix)
    return quat_to_rotvec(quaternion)


def quat_to_rotvec(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion * torch.where(quaternion[..., :1] < 0, -1.0, 1.0)
    vector = quaternion[..., 1:]
    sin_half = torch.linalg.norm(vector, dim=-1, keepdim=True)
    half = torch.atan2(sin_half, quaternion[..., :1])
    scale = torch.where(
        sin_half < 1e-06, 2.0 / torch.clamp(quaternion[..., :1], min=1e-06), 2 * half / (sin_half + 1e-12)
    )
    return vector * scale


def skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(vector.shape[:-1] + (3, 3))


def rot_angle_sq(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle between rotations, differentiable (via acos of trace, clamped)."""
    tr = (Ra * Rb).sum((-1, -2))
    cos = torch.clamp((tr - 1) / 2, -1 + 1e-06, 1 - 1e-06)
    return torch.acos(cos) ** 2


def yaw_mat(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Heading (yaw-only) rotation of R, from the projected x axis."""
    fx, fy = (rotation_matrix[..., 0, 0], rotation_matrix[..., 1, 0])
    n = torch.sqrt(fx * fx + fy * fy + 1e-09)
    c, s = (fx / n, fy / n)
    o, one = (torch.zeros_like(c), torch.ones_like(c))
    return torch.stack((c, -s, o, s, c, o, o, o, one), -1).reshape(rotation_matrix.shape[:-2] + (3, 3))


def mat_to_6d(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """First two columns, flattened (6D continuous rotation representation)."""
    return rotation_matrix[..., :, :2].reshape(rotation_matrix.shape[:-2] + (6,))


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    s = torch.sin(theta)
    t = t[..., None]
    w0 = torch.where(s < 1e-06, 1 - t, torch.sin((1 - t) * theta) / (s + 1e-12))
    w1 = torch.where(s < 1e-06, t, torch.sin(t * theta) / (s + 1e-12))
    return F.normalize(w0 * q0 + w1 * q1, dim=-1)
