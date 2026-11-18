"""Reference ghost geometry, isolated from the playback control loop."""

import mujoco
import numpy as np
from .inputs import rotation_matrix, yaw_matrix


class Ghost:
    def __init__(self, model, base_qpos_indices, joint_qpos_indices):
        self.model, self.data = model, mujoco.MjData(model)
        self.base_qpos_indices, self.joint_qpos_indices = base_qpos_indices, joint_qpos_indices
        self.scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 32)
        self.options, self.camera, self.perturbation = (
            mujoco.MjvOption(),
            mujoco.MjvCamera(),
            mujoco.MjvPerturb(),
        )

    def update(self, target_scene, qpos, motion, frame, follow):
        self.data.qpos[:] = qpos
        position = motion["base_pos_w"][frame].copy()
        rotation = motion["base_rot_mat_w"][frame].copy()
        if follow:
            base = qpos[self.base_qpos_indices]
            rotation = yaw_matrix(rotation_matrix(base[3:]) @ rotation.T) @ rotation
            position[:2] = base[:2]
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, dtype=np.float64).reshape(9))
        self.data.qpos[self.base_qpos_indices] = np.concatenate((position, quaternion))
        self.data.qpos[self.joint_qpos_indices] = motion["joint_pos"][frame]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.options,
            self.perturbation,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        target_scene.ngeom = 0
        for index in range(self.scene.ngeom):
            source = self.scene.geoms[index]
            if source.objtype != mujoco.mjtObj.mjOBJ_GEOM or self.model.geom_bodyid[source.objid] == 0:
                continue
            target = target_scene.geoms[target_scene.ngeom]
            mujoco.mjv_initGeom(
                target,
                source.type,
                source.size.astype(np.float64),
                source.pos.astype(np.float64),
                source.mat.astype(np.float64).reshape(9),
                np.array([0.2, 0.65, 1, 0.3], dtype=np.float32),
            )
            target.dataid, target.modelrbound, target.matid = source.dataid, source.modelrbound, -1
            target.transparent = True
            target_scene.ngeom += 1
