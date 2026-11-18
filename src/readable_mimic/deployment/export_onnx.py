"""Export checkpoint Actor, frozen normalizers, motion reference and deployment settings."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from readable_mimic.actor import Actor, actor_kwargs
from readable_mimic.data_transforms import compute_observation_dim, compute_command_dim
from readable_mimic.config import load_config
from readable_mimic.kinematics import Kinematics
from readable_mimic.motion import ReferenceMotion
from readable_mimic.robot import get_robot


def export_policy(checkpoint_path, config_path, output_path):
    config = load_config(config_path, restore_training_config=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_config = checkpoint["config"]
    robot = get_robot(actor_config["robot"])
    num_joints = robot.num_joints
    observation_dim = compute_observation_dim(num_joints)
    command_dim = compute_command_dim(num_joints, actor_config["goal_frames"])
    actor = Actor(
        observation_dim,
        command_dim,
        num_joints,
        history_length=actor_config["hist"],
        encoder_hidden_dims=tuple(map(int, actor_config["enc_hidden"].split(","))),
        decoder_hidden_dims=tuple(map(int, actor_config["dec_hidden"].split(","))),
        **actor_kwargs(actor_config, robot),
    ).eval()
    average = checkpoint.get("actor_average")
    use_average = average is not None and int(average["n_averaged"]) > 0
    weights = (
        {name.removeprefix("module."): value for name, value in average.items() if name.startswith("module.")}
        if use_average
        else checkpoint["actor"]
    )
    actor.load_state_dict(weights)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    history = torch.randn(1, actor_config["hist"], observation_dim)
    command = torch.randn(1, command_dim)
    torch.onnx.export(
        actor,
        (history, command),
        str(output),
        input_names=["observation_history", "command"],
        output_names=["action"],
        opset_version=18,
        dynamo=True,
        verbose=False,
        external_data=False,
    )
    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    for seed in range(3):
        generator = torch.Generator().manual_seed(seed)
        history = torch.randn(history.shape, generator=generator)
        command = torch.randn(command.shape, generator=generator)
        with torch.no_grad():
            expected = actor(history, command).numpy()
        actual = session.run(
            ["action"], {"observation_history": history.numpy(), "command": command.numpy()}
        )[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
    print("Actor verified; preparing motion reference...", flush=True)
    motion = ReferenceMotion.from_dir(config.motion_dir, robot, Kinematics(robot, "cpu"), "cpu")
    motion_path = output.with_suffix(".motion.npz")
    fields = (
        "base_pos_w",
        "base_rot_mat_w",
        "base_lin_vel_w",
        "base_ang_vel_w",
        "base_lin_vel",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "clip_start",
        "clip_end",
    )
    np.savez_compressed(
        motion_path, names=np.array(motion.names), **{name: getattr(motion, name).numpy() for name in fields}
    )
    metadata = {
        "format_version": 1,
        "robot": robot.name,
        "actor_weights": "ema" if use_average else "actor",
        "mjcf_path": os.path.relpath(robot.mjcf_path, output.parent),
        "motion_file": motion_path.name,
        "joint_names": list(robot.joint_names),
        "history_length": actor_config["hist"],
        "command_frames": actor_config["goal_frames"],
        "observation_dim": observation_dim,
        "default_joint_pos": robot.default_joint_pos_list(),
        "action_scale": robot.action_scale_list(),
        "joint_stiffness": [robot.joint_stiffness[name] for name in robot.joint_names],
        "joint_damping": [robot.joint_damping[name] for name in robot.joint_names],
        "joint_effort_limit": [robot.joint_effort_limit[name] for name in robot.joint_names],
        "joint_armature": [robot.joint_armature[name] for name in robot.joint_names],
        "control_dt": 0.02,
        "decimation": 4,
        "friction": 1.0,
        "kp_scale": 1.0,
        "kd_scale": 1.0,
        "simulation": config.simulation,
        "termination_height_threshold": config.reward["termination_height_threshold"],
        "termination_tilt_threshold": config.reward["termination_tilt_threshold"],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Exported {output}; ONNX Runtime matches PyTorch on 3 input batches.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", default="exports/policy.onnx")
    args = parser.parse_args()
    export_policy(args.checkpoint, args.config, args.output)


if __name__ == "__main__":
    main()
