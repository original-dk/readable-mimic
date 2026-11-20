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
        posture_cost = settings.get("previous_pose_cost", 0)
        self.previous_pose = mink.PostureTask(model, cost=posture_cost) if posture_cost else None
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
            root_rotation = body_rot_mat_w[self.human_root_index]
            for _, index, entry, rotation_offset in self.stages[0]:
                if index == self.human_root_index:
                    root_rotation = root_rotation @ rotation_offset
                    self.configuration.data.qpos[base_start : base_start + 3] += (
                        root_rotation @ np.asarray(entry["position_offset"])
                    )
                    break
            self.configuration.data.qpos[base_start + 3 : base_start + 7] = Rotation.from_matrix(
                root_rotation
            ).as_quat(scalar_first=True)
            self.configuration.update()
        if self.previous_pose is not None:
            self.previous_pose.set_target_from_configuration(self.configuration)
        for stage in self.stages:
            tasks = [entry[0] for entry in stage]
            if self.previous_pose is not None:
                tasks.append(self.previous_pose)
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
