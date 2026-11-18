"""Run an exported ONNX Actor in native single-robot MuJoCo, without training Env or Torch."""

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
import onnxruntime as ort

from .inputs import command, observation, yaw_matrix


def create_model(settings, model_path):
    spec = mujoco.MjSpec.from_file(str(model_path))
    if len(spec.actuators):
        raise ValueError("Use the original robot MJCF without training-generated actuators.")
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity = 0, 1
        geom.friction[0] = settings["friction"]
    spec.add_texture(
        name="ground_checker",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=300,
        height=300,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
    )
    material = spec.add_material(name="ground_material", texuniform=True, texrepeat=[4, 4])
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ground_checker"
    spec.worldbody.add_geom(
        material="ground_material",
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1],
        friction=[settings["friction"], 0.005, 0.0001],
    )
    spec.worldbody.add_light(pos=[0, -3, 5], dir=[0, 0, -1])
    for index, name in enumerate(settings["joint_names"]):
        stiffness = settings["joint_stiffness"][index] * settings["kp_scale"]
        damping = settings["joint_damping"][index] * settings["kd_scale"]
        effort = settings["joint_effort_limit"][index]
        spec.joint(name).armature = settings["joint_armature"][index]
        spec.add_actuator(
            name=f"position_{name}",
            target=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=[stiffness] + [0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=[0, -stiffness, -damping] + [0] * 7,
            forcelimited=True,
            forcerange=[-effort, effort],
        )
    model = spec.compile()
    physics = settings["simulation"]
    model.opt.timestep = settings["control_dt"] / settings["decimation"]
    model.opt.integrator = getattr(mujoco.mjtIntegrator, "mjINT_" + physics["integrator"].upper())
    model.opt.solver = getattr(mujoco.mjtSolver, "mjSOL_" + physics["solver"].upper())
    model.opt.cone = getattr(mujoco.mjtCone, "mjCONE_" + physics["cone"].upper())
    for name in ("iterations", "ls_iterations", "tolerance", "ccd_iterations"):
        setattr(model.opt, name, physics[name])
    return model


def rotation_matrix(quaternion):
    values = np.empty(9)
    mujoco.mju_quat2Mat(values, np.asarray(quaternion, dtype=np.float64))
    return values.reshape(3, 3)


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
        mujoco.mj_forward(self.model, self.data)
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


def play(
    model_path, motions="all", duration=30.0, seed=0, headless=False, ghost_base="follow", mjcf_path=None
):
    path = Path(model_path).resolve()
    settings = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    motion = np.load(path.parent / settings["motion_file"], allow_pickle=False)
    names = motion["names"].tolist()
    requested = names if motions == "all" else [Path(name.strip()).stem for name in motions.split("|")]
    unknown = set(requested) - set(names)
    if unknown:
        raise ValueError(f"Unknown motions: {sorted(unknown)}; available: {names}")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if settings["format_version"] != 1:
        raise ValueError("Unsupported deployment bundle version")
    model = create_model(settings, Path(mjcf_path) if mjcf_path else path.parent / settings["mjcf_path"])
    data = mujoco.MjData(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    joint_qpos_indices = np.array([model.joint(name).qposadr[0] for name in settings["joint_names"]])
    joint_dof_indices = np.array([model.joint(name).dofadr[0] for name in settings["joint_names"]])
    joint_ranges = np.array([model.joint(name).range for name in settings["joint_names"]])
    joint_limited = np.array([bool(model.joint(name).limited[0]) for name in settings["joint_names"]])
    joint_lower = np.where(joint_limited, joint_ranges[:, 0], -np.inf)
    joint_upper = np.where(joint_limited, joint_ranges[:, 1], np.inf)
    free_joint = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joint) != 1:
        raise ValueError("Expected a single floating-base robot")
    base_qpos_start, base_dof_start = model.jnt_qposadr[free_joint[0]], model.jnt_dofadr[free_joint[0]]
    base_qpos_indices = np.arange(base_qpos_start, base_qpos_start + 7)
    default_joint_pos = np.asarray(settings["default_joint_pos"], dtype=np.float32)
    action_scale = np.asarray(settings["action_scale"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    last_action = np.zeros(len(joint_qpos_indices), dtype=np.float32)

    def observe(frame):
        base = data.qpos[base_qpos_indices]
        return observation(
            rotation_matrix(base[3:]),
            data.qvel[base_dof_start + 3 : base_dof_start + 6],
            data.qpos[joint_qpos_indices],
            data.qvel[joint_dof_indices],
            last_action,
            motion["base_rot_mat_w"][frame],
            default_joint_pos,
        )

    def reset(start, end):
        frame = int(rng.integers(start, end))
        mujoco.mj_resetData(model, data)
        quaternion = np.empty(4)
        rotation = motion["base_rot_mat_w"][frame]
        mujoco.mju_mat2Quat(quaternion, rotation.astype(np.float64).reshape(9))
        data.qpos[base_qpos_indices] = np.concatenate((motion["base_pos_w"][frame], quaternion))
        data.qpos[joint_qpos_indices] = np.clip(motion["joint_pos"][frame], joint_lower, joint_upper)
        data.qvel[base_dof_start : base_dof_start + 6] = np.concatenate(
            (motion["base_lin_vel_w"][frame], rotation.T @ motion["base_ang_vel_w"][frame])
        )
        data.qvel[joint_dof_indices] = motion["joint_vel"][frame]
        data.ctrl[:] = data.qpos[joint_qpos_indices]
        last_action.fill(0)
        mujoco.mj_forward(model, data)
        history = np.repeat(observe(frame)[None, None, :], settings["history_length"], axis=1)
        return frame, history

    viewer_context = nullcontext(None)
    if not headless:
        from mujoco import viewer as mujoco_viewer

        viewer_context = mujoco_viewer.launch_passive(model, data)
    summaries = []
    with viewer_context as viewer:
        ghost = (
            Ghost(model, base_qpos_indices, joint_qpos_indices)
            if viewer is not None and ghost_base != "off"
            else None
        )
        for name in requested:
            index = names.index(name)
            start, end = int(motion["clip_start"][index]), int(motion["clip_end"][index])
            frame, history = reset(start, end)
            steps, resets = 0, 0
            for _ in range(math.ceil(duration / settings["control_dt"])):
                wall_start = time.perf_counter()
                inputs = {
                    "observation_history": history.astype(np.float32),
                    "command": command(motion, frame, end, settings["command_frames"]),
                }
                action = session.run(["action"], inputs)[0][0]
                if not np.isfinite(action).all():
                    raise RuntimeError("ONNX Actor produced non-finite actions")
                data.ctrl[:] = default_joint_pos + action_scale * action
                for _ in range(settings["decimation"]):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError("MuJoCo produced a non-finite state")
                frame += 1
                last_action[:] = action
                steps += 1
                base = data.qpos[base_qpos_indices]
                rotation = rotation_matrix(base[3:])
                done = (
                    frame >= end
                    or abs(base[2] - motion["base_pos_w"][frame, 2])
                    > settings["termination_height_threshold"]
                    or abs(rotation[2, 2] - motion["base_rot_mat_w"][frame, 2, 2])
                    > settings["termination_tilt_threshold"]
                )
                if viewer is not None:
                    if not viewer.is_running():
                        return summaries
                    with viewer.lock():
                        if ghost is not None:
                            ghost.update(viewer.user_scn, data.qpos, motion, frame, ghost_base == "follow")
                        viewer.cam.lookat[:] = base[:3]
                    viewer.sync()
                    time.sleep(max(0, settings["control_dt"] - (time.perf_counter() - wall_start)))
                if done:
                    frame, history = reset(start, end)
                    resets += 1
                else:
                    history = np.concatenate((history[:, 1:], observe(frame)[None, None]), axis=1)
            summary = {
                "motion": name,
                "policy_steps": steps,
                "physics_steps": steps * settings["decimation"],
                "resets": resets,
                "simulated_seconds": steps * settings["control_dt"],
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    motion.close()
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="exports/policy.onnx")
    parser.add_argument("--motions", default="all")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--ghost-base", choices=("follow", "reference", "off"), default="follow")
    parser.add_argument("--mjcf", help="Override the bundle MJCF path after moving assets")
    args = parser.parse_args()
    play(args.model, args.motions, args.duration, args.seed, args.headless, args.ghost_base, args.mjcf)


if __name__ == "__main__":
    main()
