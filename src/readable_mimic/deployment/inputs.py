"""NumPy deployment inputs matching data_transforms.py, without Torch or xi."""

import numpy as np


def yaw_matrix(rotation):
    x, y = rotation[0, 0], rotation[1, 0]
    norm = np.sqrt(x * x + y * y + 1e-9)
    cosine, sine = x / norm, y / norm
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=np.float32)


def observation(
    base_rot_mat_w, base_ang_vel, joint_pos, joint_vel, last_action, ref_base_rot_mat_w, default_joint_pos
):
    return np.concatenate(
        (
            joint_pos - default_joint_pos,
            joint_vel,
            base_ang_vel,
            -base_rot_mat_w[2, :],
            last_action,
            (base_rot_mat_w.T @ ref_base_rot_mat_w)[:, :2].reshape(6),
        )
    ).astype(np.float32)


def command(motion, frame, end_frame, command_frames):
    future = np.minimum(frame + np.arange(1, command_frames + 1), end_frame)
    world_to_heading = yaw_matrix(motion["base_rot_mat_w"][frame]).T
    return (
        np.concatenate(
            (
                (world_to_heading @ (motion["base_pos_w"][future] - motion["base_pos_w"][frame])[..., None])[
                    ..., 0
                ],
                (world_to_heading @ motion["base_rot_mat_w"][future])[..., :, :2].reshape(command_frames, 6),
                motion["base_lin_vel"][future],
                motion["base_ang_vel"][future],
                motion["joint_pos"][future],
                motion["joint_vel"][future] * 0.1,
            ),
            axis=-1,
        )
        .reshape(1, -1)
        .astype(np.float32)
    )
