"""AMASS body-joint FK from licensed SMPL-family NPZ models; no mesh skinning."""

from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .bvh import HumanMotion

BODY_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


@lru_cache(maxsize=3)
def load_body_model(model_path, num_betas):
    with np.load(model_path, allow_pickle=False) as model:
        regressor = model["J_regressor"][:22].astype(np.float64)
        template = regressor @ model["v_template"]
        directions = model["shapedirs"][:, :, :num_betas].copy()
        if directions.shape[-1] != num_betas:
            raise ValueError("Body model has fewer shape coefficients than the motion")
        joint_shape_directions = np.einsum("jv,vck->jck", regressor, directions)
        parents = model["kintree_table"][0, :22].astype(int)
        parents[0] = -1
    return template, joint_shape_directions, parents


def body_forward_kinematics(rest_joint_pos, parents, local_rot_mat, translation):
    body_pos_w = np.empty((len(translation), 22, 3))
    body_rot_mat_w = np.empty_like(local_rot_mat)
    body_pos_w[:, 0] = rest_joint_pos[0] + translation
    body_rot_mat_w[:, 0] = local_rot_mat[:, 0]
    for index in range(1, 22):
        parent = parents[index]
        if not 0 <= parent < index:
            raise ValueError("Expected SMPL body joints in parent-first order")
        body_rot_mat_w[:, index] = body_rot_mat_w[:, parent] @ local_rot_mat[:, index]
        body_pos_w[:, index] = body_pos_w[:, parent] + np.einsum(
            "fij,j->fi", body_rot_mat_w[:, parent], rest_joint_pos[index] - rest_joint_pos[parent]
        )
    return body_pos_w, body_rot_mat_w


def load_amass(path, settings):
    with np.load(path, allow_pickle=False) as source:
        gender = source["gender"].item()
        gender = gender.decode() if isinstance(gender, bytes) else str(gender)
        gender = gender.lower()
        model_path = settings["body_models"][gender]
        if "surface_model_type" in source:
            model_type = str(source["surface_model_type"].item()).lower()
            if model_type != settings["body_model_type"]:
                raise ValueError(f"Motion model {model_type} does not match {settings['body_model_type']}")
        frame_rate = float(
            source["mocap_frame_rate"] if "mocap_frame_rate" in source else source["mocap_framerate"]
        )
        translation = source["trans"].astype(np.float64)
        betas = source["betas"].astype(np.float64).reshape(-1)
        if "pose_body" in source:
            rotations = np.concatenate((source["root_orient"], source["pose_body"]), axis=1)
        else:
            rotations = source["poses"][:, :66]
    if len(translation) < 2 or rotations.shape != (len(translation), 66):
        raise ValueError(f"Expected at least two frames of 22 body-joint rotations: {path}")
    if (
        not np.isfinite(frame_rate)
        or frame_rate <= 0
        or not all(np.isfinite(v).all() for v in (translation, betas, rotations))
    ):
        raise ValueError(f"Non-finite or invalid AMASS parameters: {path}")
    template, directions, parents = load_body_model(model_path, len(betas))
    rest_joint_pos = template + np.einsum("jck,k->jc", directions, betas)
    source_times = np.arange(len(translation)) / frame_rate
    fps = settings["output_frame_rate"]
    target_times = np.arange(int(np.floor(source_times[-1] * fps)) + 1) / fps
    local_rot_mat = np.stack(
        [
            Slerp(source_times, Rotation.from_rotvec(rotations.reshape(-1, 22, 3)[:, index]))(
                target_times
            ).as_matrix()
            for index in range(22)
        ],
        axis=1,
    )
    translation = np.stack(
        [np.interp(target_times, source_times, translation[:, axis]) for axis in range(3)], axis=1
    )
    body_pos_w, body_rot_mat_w = body_forward_kinematics(rest_joint_pos, parents, local_rot_mat, translation)
    return HumanMotion(BODY_NAMES, body_pos_w, body_rot_mat_w, fps)
