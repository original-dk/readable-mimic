"""AMASS IK using GMR's initialization, limits, scaling and solve procedure.

Adapted from GMR (MIT), copyright 2025 Yanjieze; see THIRD_PARTY_NOTICES.md.
"""

from copy import deepcopy

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .ik import MotionRetargeter


def load_retarget_model(mjcf_path, settings):
    """Add GMR's toe task frames to an offline copy of the training model."""
    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    for name, frame in settings["frames"].items():
        spec.body(frame["parent_body"]).add_body(name=name, pos=frame["position"])
    return spec.compile()


class AmassRetargeter(MotionRetargeter):
    def __init__(self, model, robot, settings, human_body_names, human_height):
        settings = deepcopy(settings)
        ratio = human_height / settings["human_height_assumption"]
        settings["base_position_scale"] *= ratio
        for stage in settings["stages"]:
            for entry in stage:
                entry["position_scale"] *= ratio
        # These are offline IK limits only; the training robot stays unchanged.
        for name, limits in settings["joint_pos_limits"].items():
            if np.any(np.asarray(limits) < model.joint(name).range[0]) or np.any(
                np.asarray(limits) > model.joint(name).range[1]
            ):
                raise ValueError(f"IK limits exceed robot limits: {name}")
            model.joint(name).range[:] = limits
        super().__init__(model, robot, settings, human_body_names)
        initial_qpos = self.configuration.data.qpos.copy()
        initial_qpos[self.joint_qpos_indices] = 0
        initial_qpos[:3] = settings["initial_base_pos_w"]
        initial_qpos[3:7] = [1, 0, 0, 0]
        self.configuration.update(initial_qpos)

    def initialize_base_pose(self, body_pos_w, body_rot_mat_w):
        """Align the initial floating base with the first frame's ROBOT target.

        A fixed world heading can put the shoulder IK on a folded branch before
        the base turns toward the motion. Include the SMPL-to-robot rotation offset.
        """
        _, index, entry, rotation_offset = next(
            target for target in self.stages[0] if target[1] == self.human_root_index
        )
        rotation = body_rot_mat_w[index] @ rotation_offset
        position = body_pos_w[index] * self.settings["base_position_scale"]
        position += rotation @ np.asarray(entry["position_offset"])
        qpos = self.configuration.data.qpos.copy()
        start = self.base_qpos_start
        qpos[start : start + 3] = position
        qpos[start + 3 : start + 7] = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
        self.configuration.update(qpos)

    def retarget(self, body_pos_w, body_rot_mat_w):
        human_base_pos_w = body_pos_w[self.human_root_index]
        base_pos_w = human_base_pos_w * self.settings["base_position_scale"]
        for stage in self.stages:
            tasks = [entry[0] for entry in stage]
            for task, index, entry, rotation_offset in stage:
                target_rotation = body_rot_mat_w[index] @ rotation_offset
                target_pos = base_pos_w + (body_pos_w[index] - human_base_pos_w) * entry["position_scale"]
                target_pos += target_rotation @ np.asarray(entry["position_offset"])
                task.set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(target_rotation), target_pos)
                )
            error = np.linalg.norm(np.concatenate([task.compute_error(self.configuration) for task in tasks]))
            for _ in range(self.settings["max_iterations"]):
                dt = self.model.opt.timestep
                velocity = mink.solve_ik(
                    self.configuration,
                    tasks,
                    dt,
                    solver=self.settings["solver"],
                    damping=self.settings["damping"],
                    limits=self.limits,
                )
                self.configuration.integrate_inplace(velocity, dt)
                next_error = np.linalg.norm(
                    np.concatenate([task.compute_error(self.configuration) for task in tasks])
                )
                if error - next_error <= self.settings["convergence_tolerance"]:
                    break
                error = next_error
        qpos = self.configuration.data.qpos.copy()
        if not np.isfinite(qpos).all():
            raise ValueError("AMASS IK produced non-finite positions")
        return qpos
