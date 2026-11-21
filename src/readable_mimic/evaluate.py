"""Render random-start Actor rollouts to MP4, with an optional base-following ref ghost."""

import os
import sys
import imageio.v2 as imageio
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import mujoco
import numpy as np
import torch
from . import data_transforms as transforms
from .actor import Actor, actor_kwargs
from .config import load_config
from .sim import CONTROL_DT, DomainRandomizationConfig
from .env import TrackingEnv
from .kinematics import Kinematics
from .motion import ReferenceMotion
from .robot import get_robot
from .task import MAX_STEP_REWARD, RewardConfig, TrackingTask
from .rotations import mat_to_quat

if sys.platform.startswith("linux") and (not os.environ.get("DISPLAY")):
    os.environ.setdefault("MUJOCO_GL", "egl")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--ckpt", required=True)
    parser.add_argument("--output-dir", default="videos", help="One MP4 per selected trajectory.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--motion-dir", help="Override the reference motion directory in the config.")
    parser.add_argument("--motions", default="all", help="Exact clip names separated by | (quote the list).")
    parser.add_argument("--duration", type=float, default=30.0, help="Playback seconds per selected clip.")
    parser.add_argument(
        "--ghost-base",
        choices=("follow", "reference"),
        default="follow",
        help="follow: align ghost xy/yaw to robot; reference: retain the ref world trajectory.",
    )
    parser.add_argument(
        "--start-frame", type=int,
        help="Start each reset at this zero-based reference frame (50 Hz) within each clip. Default: random.",
    )
    parser.add_argument("--seed", type=int, default=0)
    if argv is None:
        arguments = parser.parse_args()
    else:
        arguments = parser.parse_args(argv)
    if not math.isfinite(arguments.duration) or arguments.duration <= 0:
        parser.error("--duration must be finite and positive")
    return arguments


def select_motion_indices(names, requested_motions):
    if requested_motions == "all":
        return list(range(len(names)))
    requested_names = [Path(name.strip()).stem for name in requested_motions.split("|")]
    unknown = [name for name in requested_names if name not in names]
    if unknown:
        raise ValueError(f"Unknown motion names: {unknown}. Available: {' | '.join(names)}")
    return [names.index(name) for name in requested_names]


def load_evaluation_motion(motion_dir, robot, kinematics, device, requested_motions):
    files = ReferenceMotion.find_files(motion_dir, robot)
    indices = select_motion_indices([Path(path).stem for path in files], requested_motions)
    return ReferenceMotion([files[index] for index in indices], kinematics, device, robot.load_motion)


def select_actor_weights(checkpoint):
    average = checkpoint.get("actor_average")
    if average is not None and int(average["n_averaged"]) > 0:
        return {
            name.removeprefix("module."): value
            for name, value in average.items()
            if name.startswith("module.")
        }
    return checkpoint["actor"]


def sample_start_frame(motion, clip_index, start_frame=None):
    clip_start = int(motion.clip_start[clip_index])
    clip_end = int(motion.clip_end[clip_index])
    last_start_frame = clip_end - clip_start - 1
    if last_start_frame < 0:
        raise ValueError(f"Motion {motion.names[clip_index]} has fewer than two frames.")
    if start_frame is None:
        return torch.randint(clip_start, clip_end, (1,), device=motion.device)
    if not 0 <= start_frame <= last_start_frame:
        raise ValueError(
            f"{motion.names[clip_index]}: --start-frame {start_frame} outside valid range 0..{last_start_frame}"
        )
    return torch.tensor([clip_start + start_frame], device=motion.device)


class GhostOverlay:
    """Visual geoms only: the ghost never enters the physics model."""

    def __init__(self, model, base_qpos_indices, joint_qpos_indices, robot_geom_indices):
        self.model = model
        self.data = mujoco.MjData(model)
        self.base_qpos_indices = np.asarray(base_qpos_indices, dtype=int)
        self.joint_qpos_indices = np.asarray(joint_qpos_indices, dtype=int)
        self.robot_geom_indices = set(np.asarray(robot_geom_indices, dtype=int).reshape(-1).tolist())
        self.scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 32)
        self.options = mujoco.MjvOption()
        self.camera = mujoco.MjvCamera()
        self.perturbation = mujoco.MjvPerturb()

    def update(self, user_scene, robot_qpos, ghost_pose, append=False):
        base_pos_w, base_rot_mat_w, joint_pos = ghost_pose
        base_quat_w = mat_to_quat(base_rot_mat_w)
        self.data.qpos[:] = robot_qpos
        self.data.qpos[self.base_qpos_indices] = (
            torch.cat((base_pos_w, base_quat_w), -1).cpu().numpy().reshape(-1)
        )
        self.data.qpos[self.joint_qpos_indices] = joint_pos.cpu().numpy().reshape(-1)
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
        if not append:
            user_scene.ngeom = 0
        for geom_index in range(self.scene.ngeom):
            source = self.scene.geoms[geom_index]
            if source.objtype != mujoco.mjtObj.mjOBJ_GEOM or source.objid not in self.robot_geom_indices:
                continue
            if user_scene.ngeom >= user_scene.maxgeom:
                raise RuntimeError("Viewer ghost geometry capacity exceeded.")
            target = user_scene.geoms[user_scene.ngeom]
            mujoco.mjv_initGeom(
                target,
                source.type,
                source.size.astype(np.float64),
                source.pos.astype(np.float64),
                source.mat.astype(np.float64).reshape(9),
                np.array([0.2, 0.65, 1.0, 0.3], dtype=np.float32),
            )
            target.dataid = source.dataid
            target.modelrbound = source.modelrbound
            target.matid = -1
            target.emission = 0.3
            target.transparent = True
            user_scene.ngeom += 1


@torch.no_grad()
def render_motion(
    env,
    actor,
    motion,
    clip_index,
    history_length,
    command_frames,
    duration,
    renderer,
    render_data,
    ghost,
    follow_base,
    output_path,
    native_renderer=None,
    start_frame=None,
):
    command_frame_offsets = torch.arange(1, command_frames + 1, device=env.device)
    env_indices = torch.tensor([0], device=env.device)
    env.reset(env_indices, sample_start_frame(motion, clip_index, start_frame))
    history = transforms.compute_observation_history(env.observe(), history_length)
    camera = mujoco.MjvCamera()
    camera.distance = 3.0
    camera.azimuth = 135.0
    camera.elevation = -15.0
    steps = math.ceil(duration / CONTROL_DT)
    writer_context = (
        imageio.get_writer(
            str(output_path),
            fps=round(1 / CONTROL_DT),
            codec="libx264",
            macro_block_size=1,
            pixelformat="yuv420p",
        )
        if native_renderer is None
        else native_renderer.video_writer(output_path, round(1 / CONTROL_DT))
    )
    reward_sum = torch.zeros((), device=env.device)
    completed_returns = []
    failures = []
    clip_start = int(motion.clip_start[clip_index])
    with writer_context as writer:
        for step_index in range(steps):
            command = transforms.compute_command(env.motion_frame_index, command_frame_offsets, motion)
            transition = env.step(actor(history, command))
            reward_sum += transition["reward"].sum()
            robot_state = env.state()
            if bool(transition["termination"][0]):
                height_error, tilt_error = env.task.termination_errors(robot_state, env.motion_frame_index)
                ref_frame = int(env.motion_frame_index[0]) - clip_start
                errors = dict(base_height_error=float(height_error[0]), base_tilt_error=float(tilt_error[0]))
                thresholds = dict(
                    base_height_error=env.task.config.termination_height_threshold,
                    base_tilt_error=env.task.config.termination_tilt_threshold,
                )
                failure = dict(
                    video_frame=step_index,
                    video_time_s=step_index * CONTROL_DT,
                    ref_frame=ref_frame,
                    ref_time_s=ref_frame * CONTROL_DT,
                    reasons=[name for name, value in errors.items() if value > thresholds[name]],
                    errors=errors,
                    thresholds=thresholds,
                )
                failures.append(failure)
                print(f"{motion.names[clip_index]} failure: {json.dumps(failure)}", flush=True)
            ref_state = motion.ref_state(env.motion_frame_index)
            ghost_pose = transforms.compute_ghost_pose(ref_state, robot_state, follow_base)
            if native_renderer is not None:
                writer.append_data(native_renderer.render(robot_state, ghost_pose))
            else:
                render_data.qpos[ghost.base_qpos_indices] = (
                    torch.cat((robot_state[0], mat_to_quat(robot_state[1])), -1)[0].cpu().numpy()
                )
                render_data.qpos[ghost.joint_qpos_indices] = robot_state[4][0].cpu().numpy()
                mujoco.mj_forward(ghost.model, render_data)
                camera.lookat[:] = robot_state[0][0].cpu().numpy()
                renderer.update_scene(render_data, camera=camera)
                ghost.update(renderer.scene, render_data.qpos, ghost_pose, append=True)
                writer.append_data(renderer.render())
            if bool(transition["done"][0]):
                completed_returns.append(float(transition["normalized_episode_return"][0]))
                env.reset(env_indices, sample_start_frame(motion, clip_index, start_frame))
                history = transforms.compute_observation_history(env.observe(), history_length)
            else:
                history = transforms.update_observation_history(history, env.observe())
            if (step_index + 1) % 250 == 0 or step_index + 1 == steps:
                print(f"{motion.names[clip_index]}: {step_index + 1}/{steps} video frames", flush=True)

    summary = dict(
        motion=motion.names[clip_index],
        video_frames=steps,
        sim_r=float(reward_sum) / (steps * MAX_STEP_REWARD),
        episode_return=sum(completed_returns) / len(completed_returns) if completed_returns else None,
        completed_episodes=len(completed_returns),
        failure_count=len(failures),
        first_failure=failures[0] if failures else None,
        failures=failures,
    )
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "failures"}), flush=True)


def main(argv=None):
    arguments = parse_args(argv)
    config = load_config(arguments.config, restore_training_config=False)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    actor_config = checkpoint["config"]
    if actor_config["robot"] != config.robot:
        raise ValueError("Checkpoint robot does not match config.robot.")
    torch.manual_seed(arguments.seed)
    torch.set_num_threads(config.num_threads)
    robot = replace(
        get_robot(config.robot), domain_randomization=DomainRandomizationConfig(**config.domain_randomization)
    )
    kinematics = Kinematics(robot, config.device)
    motion_dir = arguments.motion_dir if arguments.motion_dir is not None else config.motion_dir
    motion = load_evaluation_motion(motion_dir, robot, kinematics, config.device, arguments.motions)
    history_length, command_frames = (actor_config["hist"], actor_config["goal_frames"])
    actor = Actor(
        transforms.compute_observation_dim(robot.num_joints),
        transforms.compute_command_dim(robot.num_joints, command_frames),
        robot.num_joints,
        history_length=history_length,
        encoder_hidden_dims=tuple(map(int, actor_config["enc_hidden"].split(","))),
        decoder_hidden_dims=tuple(map(int, actor_config["dec_hidden"].split(","))),
        **actor_kwargs(actor_config, robot),
    ).to(config.device)
    actor.load_state_dict(select_actor_weights(checkpoint))
    actor.eval()
    task = TrackingTask(kinematics, motion, RewardConfig(**config.reward))
    env = TrackingEnv(
        config.simulation,
        1,
        robot,
        motion,
        task,
        config.device,
        max_episode_steps=math.ceil(arguments.duration / CONTROL_DT) + 1,
        evaluation=True,
        backend=config.simulator,
        genesis_settings=config.genesis,
    )
    output_directory = Path(arguments.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    def export(renderer=None, render_data=None, ghost=None, native_renderer=None):
        for clip_index in range(len(motion.names)):
            output_path = output_directory / (motion.names[clip_index] + ".mp4")
            render_motion(
                env,
                actor,
                motion,
                clip_index,
                history_length,
                command_frames,
                arguments.duration,
                renderer,
                render_data,
                ghost,
                arguments.ghost_base == "follow",
                output_path,
                native_renderer=native_renderer,
                start_frame=arguments.start_frame,
            )
            print(f"Saved {output_path.resolve()}", flush=True)

    if config.simulator == "genesis":
        from .render_genesis import GenesisRenderer

        with GenesisRenderer(robot) as native_renderer:
            export(native_renderer=native_renderer)
        return

    render_spec = mujoco.MjSpec.from_file(robot.mjcf_path)
    render_spec.add_texture(
        name="render_checker",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        width=300,
        height=300,
    )
    ground_material = render_spec.add_material(
        name="render_ground_material",
        texuniform=True,
        texrepeat=[4, 4],
        reflectance=0.2,
    )
    ground_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "render_checker"
    render_spec.worldbody.add_geom(
        name="render_ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1],
        material="render_ground_material",
    )
    render_spec.worldbody.add_light(pos=[0, -3, 5], dir=[0, 0, -1])
    model = render_spec.compile()
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 960)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 720)
    render_data = mujoco.MjData(model)
    free_joint = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    base_qpos_start = model.jnt_qposadr[free_joint]
    robot_body = model.jnt_bodyid[free_joint]
    robot_body_indices = {int(robot_body)}
    for body_index in range(robot_body + 1, model.nbody):
        if int(model.body_parentid[body_index]) in robot_body_indices:
            robot_body_indices.add(body_index)
    ghost = GhostOverlay(
        model,
        np.arange(base_qpos_start, base_qpos_start + 7),
        [model.joint(name).qposadr[0] for name in robot.joint_names],
        np.flatnonzero(np.isin(model.geom_bodyid, list(robot_body_indices))),
    )
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        export(renderer, render_data, ghost)


if __name__ == "__main__":
    main()
