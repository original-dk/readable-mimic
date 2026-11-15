"""Random-start Actor playback with a reference ghost in the MuJoCo viewer."""

import argparse
from dataclasses import replace
import math
from pathlib import Path
import time

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

    def __init__(self, model, base_qpos_indices, joint_qpos_indices):
        self.model = model
        self.data = mujoco.MjData(model)
        self.base_qpos_indices = np.asarray(base_qpos_indices, dtype=int)
        self.joint_qpos_indices = np.asarray(joint_qpos_indices, dtype=int)
        self.scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 32)
        self.options = mujoco.MjvOption()
        self.camera = mujoco.MjvCamera()
        self.perturbation = mujoco.MjvPerturb()

    def update(self, user_scene, robot_qpos, ghost_pose):
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
        user_scene.ngeom = 0
        for geom_index in range(self.scene.ngeom):
            source = self.scene.geoms[geom_index]
            if source.objtype != mujoco.mjtObj.mjOBJ_GEOM or self.model.geom_bodyid[source.objid] == 0:
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
def play_motion(
    environment,
    actor,
    motion,
    clip_index,
    history_length,
    command_frames,
    duration,
    viewer,
    viewer_data,
    ghost,
    follow_base,
):
    environment_indices = torch.tensor([0], device=environment.device)
    environment.reset(environment_indices, sample_start_frame(motion, clip_index))
    history = transforms.compute_observation_history(environment.observe(), history_length)
    steps = math.ceil(duration / CONTROL_DT)
    for _ in range(steps):
        if not viewer.is_running():
            return False
        step_started = time.monotonic()
        command = transforms.compute_command(environment.motion_frame_index, command_frames, motion)
        action = actor(history, command)
        transition = environment.step(action)
        ref_state = motion.ref_state(environment.motion_frame_index)
        ghost_pose = transforms.compute_ghost_pose(ref_state, environment.state(), follow_base)
        with viewer.lock():
            viewer_data.qpos[:] = environment.simulation.data.qpos[0].cpu().numpy()
            viewer_data.qvel[:] = environment.simulation.data.qvel[0].cpu().numpy()
            mujoco.mj_forward(environment.simulation.mj_model, viewer_data)
            ghost.update(viewer.user_scn, viewer_data.qpos, ghost_pose)
            viewer.cam.lookat[:] = environment.state()[0][0].cpu().numpy()
        viewer.sync()
        if bool(transition["done"][0]):
            environment.reset(environment_indices, sample_start_frame(motion, clip_index))
            history = transforms.compute_observation_history(environment.observe(), history_length)
        else:
            history = transforms.update_observation_history(history, environment.observe())
        time.sleep(max(0.0, CONTROL_DT - (time.monotonic() - step_started)))
    return True


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
    history_length, command_frames = actor_config["hist"], actor_config["goal_frames"]
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
    viewer_data = mujoco.MjData(model)
    indices = environment.robot.indexing
    ghost = GhostOverlay(
        model,
        torch.as_tensor(indices.free_joint_q_adr).cpu().numpy(),
        torch.as_tensor(indices.joint_q_adr).cpu().numpy(),
    )
    from mujoco import viewer as mujoco_viewer

    with mujoco_viewer.launch_passive(model, viewer_data) as viewer:
        viewer.cam.distance = 3.0
        for clip_index in selected_indices:
            print(
                f"Playing {motion.names[clip_index]} | {arguments.duration:g}s | ghost={arguments.ghost_base}",
                flush=True,
            )
            if not play_motion(
                environment,
                actor,
                motion,
                clip_index,
                history_length,
                command_frames,
                arguments.duration,
                viewer,
                viewer_data,
                ghost,
                arguments.ghost_base == "follow",
            ):
                break


if __name__ == "__main__":
    main()
