"""Render random-start Actor rollouts to MP4, with an optional base-following ref ghost."""

import os
import sys

if sys.platform.startswith("linux") and (not os.environ.get("DISPLAY")):
    os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import argparse
from dataclasses import replace
import math
from pathlib import Path
import mujoco
import numpy as np
import torch
from . import data_transforms as transforms
from .actor import Actor, actor_kwargs
from .config import load_config
from .environment import TrackingEnvironment, make_sim
from .kinematics import Kinematics
from .motion import CONTROL_DT, ReferenceMotion
from .robot import get_robot
from .simulation_parameters import DomainRandomizationConfig
from .task import RewardConfig, TrackingTask
from .rotations import mat_to_quat


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--ckpt", required=True)
    parser.add_argument("--output-dir", default="videos", help="One MP4 per selected trajectory.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--motions", default="all", help="Exact clip names separated by | (quote the list).")
    parser.add_argument("--duration", type=float, default=30.0, help="Playback seconds per selected clip.")
    parser.add_argument(
        "--ghost-base",
        choices=("follow", "reference"),
        default="follow",
        help="follow: align ghost xy/yaw to robot; reference: retain the ref world trajectory.",
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


def select_actor_weights(checkpoint):
    average = checkpoint.get("actor_average")
    if average is not None and int(average["n_averaged"]) > 0:
        return {
            name.removeprefix("module."): value
            for name, value in average.items()
            if name.startswith("module.")
        }
    return checkpoint["actor"]


def sample_start_frame(motion, clip_index):
    start_frame = int(motion.clip_start[clip_index])
    end_frame = int(motion.clip_end[clip_index])
    if end_frame <= start_frame:
        raise ValueError(f"Motion {motion.names[clip_index]} has fewer than two frames.")
    return torch.randint(start_frame, end_frame, (1,), device=motion.device)


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
    environment,
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
):
    environment_indices = torch.tensor([0], device=environment.device)
    environment.reset(environment_indices, sample_start_frame(motion, clip_index))
    history = transforms.compute_observation_history(environment.observe(), history_length)
    camera = mujoco.MjvCamera()
    camera.distance = 3.0
    camera.azimuth = 135.0
    camera.elevation = -15.0
    steps = math.ceil(duration / CONTROL_DT)
    with imageio.get_writer(
        str(output_path),
        fps=round(1 / CONTROL_DT),
        codec="libx264",
        macro_block_size=1,
        pixelformat="yuv420p",
    ) as writer:
        for step_index in range(steps):
            command = transforms.compute_command(environment.motion_frame_index, command_frames, motion)
            transition = environment.step(actor(history, command))
            robot_state = environment.state()
            ref_state = motion.ref_state(environment.motion_frame_index)
            ghost_pose = transforms.compute_ghost_pose(ref_state, robot_state, follow_base)
            render_data.qpos[:] = environment.simulation.data.qpos[0].cpu().numpy()
            render_data.qvel[:] = environment.simulation.data.qvel[0].cpu().numpy()
            mujoco.mj_forward(environment.simulation.mj_model, render_data)
            camera.lookat[:] = robot_state[0][0].cpu().numpy()
            renderer.update_scene(render_data, camera=camera)
            ghost.update(renderer.scene, render_data.qpos, ghost_pose, append=True)
            writer.append_data(renderer.render())
            if bool(transition["done"][0]):
                environment.reset(environment_indices, sample_start_frame(motion, clip_index))
                history = transforms.compute_observation_history(environment.observe(), history_length)
            else:
                history = transforms.update_observation_history(history, environment.observe())
            if (step_index + 1) % 250 == 0 or step_index + 1 == steps:
                print(f"{motion.names[clip_index]}: {step_index + 1}/{steps} video frames", flush=True)


def main(argv=None):
    arguments = parse_args(argv)
    config = load_config(arguments.config)
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
    motion = ReferenceMotion.from_dir(config.motion_dir, robot, kinematics, config.device)
    selected_indices = select_motion_indices(motion.names, arguments.motions)
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
    environment = TrackingEnvironment(
        make_sim(robot, config.simulation),
        1,
        robot,
        motion,
        task,
        config.device,
        max_episode_steps=math.ceil(arguments.duration / CONTROL_DT) + 1,
        evaluation=True,
    )
    model = environment.simulation.mj_model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 960)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 720)
    render_data = mujoco.MjData(model)
    indices = environment.robot.indexing
    ghost = GhostOverlay(
        model,
        torch.as_tensor(indices.free_joint_q_adr).cpu().numpy(),
        torch.as_tensor(indices.joint_q_adr).cpu().numpy(),
        torch.as_tensor(indices.geom_ids).cpu().numpy(),
    )
    output_directory = Path(arguments.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        for clip_index in selected_indices:
            output_path = output_directory / (motion.names[clip_index] + ".mp4")
            render_motion(
                environment,
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
            )
            print(f"Saved {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
