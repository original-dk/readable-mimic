"""BVH conventions, IK limits and grounding, independent of external motion assets."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

pytest.importorskip("mink")
from readable_mimic.retargeting.bvh import load_bvh
from readable_mimic.retargeting.ik import MotionRetargeter, ground_motion


def test_bvh_declared_rotation_order_and_frame_cuts(tmp_path):
    path = tmp_path / "sample.bvh"
    path.write_text(
        """HIERARCHY
ROOT Hips
{
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation
 JOINT LeftAnkle
 {
  OFFSET 0 100 0
  CHANNELS 3 Zrotation Xrotation Yrotation
  End Site
  {
   OFFSET 0 20 0
  }
 }
 JOINT RightAnkle
 {
  OFFSET 100 0 0
  CHANNELS 3 Xrotation Yrotation Zrotation
 }
}
MOTION
Frames: 6
Frame Time: 0.016666667
"""
        + "\n".join(f"{frame} 200 300 90 90 0 0 0 0 0 0 0" for frame in range(6))
    )
    motion = load_bvh(path, 1, 5, 30, 0.01, "zxy")
    assert motion.body_names == ["Hips", "LeftAnkle", "RightAnkle"]
    assert motion.body_pos_w.shape == (2, 3, 3)
    np.testing.assert_allclose(motion.body_pos_w[:, 0], [[3, 0.01, 2], [3, 0.03, 2]])
    # Intrinsic Y then X: Ry(90) @ Rx(90), not Rx @ Ry.
    expected_bvh_rotation = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    axes = np.eye(3)[[2, 0, 1]]
    np.testing.assert_allclose(motion.body_rot_mat_w[0, 0], axes @ expected_bvh_rotation @ axes.T, atol=1e-12)
    np.testing.assert_allclose(motion.body_pos_w[0, 1] - motion.body_pos_w[0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(motion.body_pos_w[0, 2] - motion.body_pos_w[0, 0], [-1, 0, 0], atol=1e-12)
    with pytest.raises(ValueError, match="Invalid frame cut"):
        load_bvh(path, 5, 7, 30, 0.01, "zxy")
    with pytest.raises(ValueError, match="integer multiple"):
        load_bvh(path, 0, 6, 50, 0.01, "zxy")


def test_ik_tracks_known_fk_and_keeps_joint_limits():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/><worldbody>
    <body name="pelvis"><freejoint/><geom type="sphere" size=".1"/>
      <body name="arm" pos="0 0 .2"><joint name="shoulder" axis="0 1 0" range="-.6 .6"/>
        <geom type="capsule" size=".03" fromto="0 0 0 0 0 -.3"/>
      </body>
    </body></worldbody></mujoco>""")
    robot = SimpleNamespace(joint_names=["shoulder"], num_joints=1, default_joint_pos_list=lambda: [0.0])
    entries = [
        dict(
            robot_body=name,
            human_body=name,
            position_cost=100,
            orientation_cost=100,
            position_scale=1.0,
            position_offset=[0, 0, 0],
            rotation_offset_wxyz=[1, 0, 0, 0],
        )
        for name in ["pelvis", "arm"]
    ]
    settings = dict(
        human_base_body="pelvis",
        base_position_scale=1.0,
        stages=[entries],
        lm_damping=1,
        solver="daqp",
        damping=0.01,
        max_iterations=30,
        convergence_tolerance=1e-9,
    )
    retargeter = MotionRetargeter(model, robot, settings, ["pelvis", "arm"])
    target = mujoco.MjData(model)
    target.qpos[:3] = [0.3, -0.2, 0.8]
    target.qpos[7] = 0.4
    mujoco.mj_kinematics(model, target)
    result = retargeter.retarget(target.xpos[[1, 2]], target.xmat[[1, 2]].reshape(2, 3, 3), initialize=True)
    np.testing.assert_allclose(result, target.qpos, atol=2e-4)
    target.qpos[7] = 1.2
    mujoco.mj_kinematics(model, target)
    result = retargeter.retarget(target.xpos[[1, 2]], target.xmat[[1, 2]].reshape(2, 3, 3))
    assert -0.60001 <= result[7] <= 0.60001


def test_grounding_uses_collision_surface_and_preserves_vertical_motion():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="foot">
    <freejoint/><geom type="sphere" size=".1" pos="0 0 -.2"/>
    <geom type="box" size=".1 .1 .05" pos="0 0 -.1"/>
    </body></worldbody></mujoco>""")
    qpos = np.tile(model.qpos0, (3, 1))
    qpos[:, 2] = [0.1, 0.4, 0.2]
    difference = np.diff(qpos[:, 2]).copy()
    offset = ground_motion(model, qpos, ["foot"], 0.002)
    assert offset == pytest.approx(0.202)
    np.testing.assert_allclose(qpos[:, 2].min() - 0.3, 0.002)
    np.testing.assert_allclose(np.diff(qpos[:, 2]), difference)


def test_support_grounding_keeps_flight_height():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="foot"><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>'
    )
    qpos = np.tile(model.qpos0, (7, 1))
    qpos[:, 2] = [0.1, 0.12, 0.3, 0.4, 0.3, 0.11, 0.1]
    support = np.array([True, True, False, False, False, True, True])
    original_rotation = qpos[:, 3:].copy()
    ground_motion(model, qpos, ["foot"], -0.0001, support)
    np.testing.assert_allclose(qpos[support, 2] - 0.1, -0.0001, atol=1e-12)
    assert qpos[3, 2] - 0.1 > 0.25
    np.testing.assert_array_equal(qpos[:, 3:], original_rotation)


def test_support_detection_does_not_ground_a_raised_foot():
    from readable_mimic.retargeting.ik import detect_support_frames

    positions = np.zeros((7, 2, 3))
    positions[:, 0, 2] = [0, 0, 0.1, 0.1, 0.1, 0, 0]
    positions[:, 1, 2] = 0.2
    human = SimpleNamespace(body_names=["LeftToe", "RightToe"], body_pos_w=positions, frame_rate=30)
    settings = dict(
        human_foot_bodies=human.body_names,
        floor_quantile=0.02,
        height_threshold=0.04,
        vertical_velocity_threshold=100,
    )
    np.testing.assert_array_equal(
        detect_support_frames(human, settings), [True, True, False, False, False, True, True]
    )
