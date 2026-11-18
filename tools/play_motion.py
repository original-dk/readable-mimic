"""Kinematic playback of G1 CSV: base_pos_w, base_quat_xyzw, joint_pos (30 Hz)."""

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def play(motion_path, mjcf_path, fps=30.0, loop=False, headless=False):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    motion = np.loadtxt(motion_path, delimiter=",", ndmin=2)
    if motion.shape[1] != 36 or not np.isfinite(motion).all():
        raise ValueError("Expected 36 finite columns: base_pos_w[3], base_quat_xyzw[4], joint_pos[29]")
    qpos = motion.copy()
    # CSV stores xyzw; the MuJoCo floating joint stores wxyz.
    qpos[:, 3:7] = motion[:, [6, 3, 4, 5]]
    quaternion_norm = np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    if np.any(quaternion_norm < 1e-8):
        raise ValueError("Motion contains a zero quaternion")
    qpos[:, 3:7] /= quaternion_norm

    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1])
    spec.worldbody.add_light(pos=[0, -3, 5], dir=[0, 0, -1])
    model = spec.compile()
    if model.nq != 36 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected the project's floating-base G1 29-DoF MJCF in CSV joint order")
    data = mujoco.MjData(model)
    viewer = None
    try:
        if not headless:
            from mujoco import viewer as mujoco_viewer

            viewer = mujoco_viewer.launch_passive(model, data)
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = int(model.jnt_bodyid[0])
                viewer.cam.distance = 2.5
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -15
        print(f"Playing {Path(motion_path).name}: {len(qpos)} frames at {fps:g} fps", flush=True)
        while True:
            start = time.perf_counter()
            for frame_index, frame_qpos in enumerate(qpos):
                if viewer is not None and not viewer.is_running():
                    return
                data.qpos[:] = frame_qpos
                data.time = frame_index / fps
                mujoco.mj_forward(model, data)
                if viewer is not None:
                    viewer.sync(state_only=True)
                    remaining = start + (frame_index + 1) / fps - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
            if not loop:
                break
    finally:
        if viewer is not None:
            viewer.close()
    print("Playback finished.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion",  default=ROOT / "data/100style_g1/WalkingStickLeft_SR.csv")
    parser.add_argument("--mjcf", type=Path, default=ROOT / "data/g1_description/g1_29dof.xml")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--headless", action="store_true", help="Validate playback without a window or waiting"
    )
    args = parser.parse_args()
    try:
        play(args.motion, args.mjcf, args.fps, args.loop, args.headless)
    except KeyboardInterrupt:
        print("\nPlayback interrupted.", flush=True)


if __name__ == "__main__":
    main()
