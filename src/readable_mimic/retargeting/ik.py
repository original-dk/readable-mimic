"""Two-stage Mink IK, using body correspondences adapted from GMR (MIT)."""

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class MotionRetargeter:
    def __init__(self, model, robot, settings, human_body_names):
        self.model = model
        self.settings = settings
        self.configuration = mink.Configuration(model)
        self.limits = [mink.ConfigurationLimit(model)]
        self.joint_qpos_indices = np.array([model.joint(name).qposadr[0] for name in robot.joint_names])
        free_joint = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free_joint) != 1 or model.nq != robot.num_joints + 7:
            raise ValueError("Expected the project's floating-base 29-DoF G1 model")
        self.base_qpos_start = int(model.jnt_qposadr[free_joint[0]])
        self.human_root_index = human_body_names.index(settings["human_base_body"])
        self.stages = []
        for stage in settings["stages"]:
            tasks = []
            for entry in stage:
                task = mink.FrameTask(
                    frame_name=entry["robot_body"],
                    frame_type="body",
                    position_cost=entry["position_cost"],
                    orientation_cost=entry["orientation_cost"],
                    lm_damping=settings["lm_damping"],
                )
                rotation_offset = Rotation.from_quat(
                    entry["rotation_offset_wxyz"], scalar_first=True
                ).as_matrix()
                tasks.append((task, human_body_names.index(entry["human_body"]), entry, rotation_offset))
            self.stages.append(tasks)
        self.configuration.data.qpos[self.joint_qpos_indices] = robot.default_joint_pos_list()
        self.configuration.update()

    def retarget(self, body_pos_w, body_rot_mat_w, initialize=False):
        human_base_pos_w = body_pos_w[self.human_root_index]
        base_pos_w = human_base_pos_w * self.settings["base_position_scale"]
        if initialize:
            base_start = self.base_qpos_start
            self.configuration.data.qpos[base_start : base_start + 3] = base_pos_w
            self.configuration.data.qpos[base_start + 3 : base_start + 7] = Rotation.from_matrix(
                body_rot_mat_w[self.human_root_index]
            ).as_quat(scalar_first=True)
            self.configuration.update()
        for stage in self.stages:
            tasks = [entry[0] for entry in stage]
            for task, human_body_index, entry, rotation_offset in stage:
                target_rotation = body_rot_mat_w[human_body_index] @ rotation_offset
                target_pos = (
                    base_pos_w + (body_pos_w[human_body_index] - human_base_pos_w) * entry["position_scale"]
                )
                target_pos += target_rotation @ np.asarray(entry["position_offset"])
                task.set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(target_rotation), target_pos)
                )
            previous_error = np.inf
            for _ in range(self.settings["max_iterations"]):
                velocity = mink.solve_ik(
                    self.configuration,
                    tasks,
                    1.0,
                    solver=self.settings["solver"],
                    damping=self.settings["damping"],
                    limits=self.limits,
                )
                self.configuration.integrate_inplace(velocity, 1.0)
                error = np.linalg.norm(
                    np.concatenate([task.compute_error(self.configuration) for task in tasks])
                )
                if previous_error - error < self.settings["convergence_tolerance"]:
                    break
                previous_error = error
        qpos = self.configuration.data.qpos.copy()
        if not np.isfinite(qpos).all():
            raise ValueError("IK produced non-finite positions")
        return qpos


def ground_motion(model, qpos, foot_body_names, clearance, support_mask=None):
    """Align support poses to collision surfaces and interpolate corrections through flight."""
    data = mujoco.MjData(model)
    foot_ids = {model.body(name).id for name in foot_body_names}
    geom_indices = [
        index
        for index in range(model.ngeom)
        if model.geom_bodyid[index] in foot_ids
        and (model.geom_contype[index] or model.geom_conaffinity[index])
    ]
    if not geom_indices:
        raise ValueError("No collision geoms on the G1 feet")
    geom_indices = np.asarray(geom_indices)
    geom_types = model.geom_type[geom_indices]
    sphere_mask = geom_types == mujoco.mjtGeom.mjGEOM_SPHERE
    if not np.all(sphere_mask | (geom_types == mujoco.mjtGeom.mjGEOM_BOX)):
        raise ValueError("Expected sphere or box collision geoms on the G1 feet")
    sizes = model.geom_size[geom_indices]
    foot_heights = []
    for frame in qpos:
        data.qpos[:] = frame
        mujoco.mj_kinematics(model, data)
        rotations = data.geom_xmat[geom_indices].reshape(-1, 3, 3)
        box_radius = np.einsum("ij,ij->i", np.abs(rotations[:, 2]), sizes)
        vertical_radius = np.where(sphere_mask, sizes[:, 0], box_radius)
        foot_heights.append(np.min(data.geom_xpos[geom_indices, 2] - vertical_radius))
    foot_heights = np.asarray(foot_heights)
    if support_mask is None:
        offset = clearance - foot_heights.min()
        qpos[:, 2] += offset
        return float(offset)
    support_frames = np.flatnonzero(support_mask)
    if len(support_frames) < 2:
        raise ValueError("Fewer than two support frames detected; check source grounding thresholds")
    from scipy.interpolate import PchipInterpolator

    offsets = clearance - foot_heights
    frame_indices = np.arange(len(qpos))
    # Interpolate the correction through flight; do not flatten the jump trajectory.
    correction = PchipInterpolator(support_frames, offsets[support_frames], extrapolate=False)(
        np.clip(frame_indices, support_frames[0], support_frames[-1])
    )
    correction = np.maximum(correction, offsets)  # prevent any foot penetrating below clearance
    qpos[:, 2] += correction
    return {
        "support_frames": int(support_mask.sum()),
        "flight_frames": int((~support_mask).sum()),
        "min_offset": float(correction.min()),
        "max_offset": float(correction.max()),
    }


def detect_support_frames(human, settings):
    """Infer support from source toe height and vertical speed (metres, seconds)."""
    indices = [human.body_names.index(name) for name in settings["human_foot_bodies"]]
    height = human.body_pos_w[:, indices, 2]
    floor = np.quantile(height, settings["floor_quantile"])
    vertical_velocity = np.gradient(height, 1.0 / human.frame_rate, axis=0)
    contact = (height - floor < settings["height_threshold"]) & (
        np.abs(vertical_velocity) < settings["vertical_velocity_threshold"]
    )
    return contact.any(axis=1)
