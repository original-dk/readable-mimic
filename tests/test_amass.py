import numpy as np
import pytest

pytest.importorskip("mink")
from readable_mimic.retargeting.smpl import load_amass


def make_model(tmp_path):
    template = np.zeros((22, 3))
    template[0] = [0.1, 0.2, 0.3]
    template[1] = [1.1, 0.2, 0.3]
    directions = np.zeros((22, 3, 1))
    directions[1, 0, 0] = 0.1
    path = tmp_path / "body.npz"
    np.savez(
        path,
        v_template=template,
        J_regressor=np.eye(22),
        shapedirs=directions,
        kintree_table=np.stack((np.array([-1] + [0] * 21), np.arange(22))),
    )
    return dict(body_models={"neutral": str(path)}, body_model_type="smplx", output_frame_rate=30)


def test_shape_fk_translation_and_noninteger_frame_rate(tmp_path):
    settings = make_model(tmp_path)
    path = tmp_path / "motion.npz"
    translation = np.zeros((200, 3))
    translation[:, 0] = np.arange(200) / 59.94
    np.savez(
        path,
        gender="neutral",
        surface_model_type="smplx",
        betas=[0.5],
        mocap_frame_rate=59.94,
        trans=translation,
        root_orient=np.tile([0, 0, np.pi / 2], (200, 1)),
        pose_body=np.zeros((200, 63)),
    )
    motion = load_amass(path, settings)
    assert motion.body_pos_w.shape == (100, 22, 3)
    assert motion.frame_rate == 30
    np.testing.assert_allclose(motion.body_pos_w[-1, 0], [3.4, 0.2, 0.3])
    np.testing.assert_allclose(motion.body_pos_w[0, 1], [0.1, 1.25, 0.3], atol=1e-12)


def test_rotation_resampling_takes_short_path_across_pi(tmp_path):
    settings = make_model(tmp_path)
    path = tmp_path / "motion.npz"
    poses = np.zeros((2, 165))
    poses[:, 2] = np.radians([170, -170])
    np.savez(path, gender="neutral", betas=[0], mocap_framerate=10, trans=np.zeros((2, 3)), poses=poses)
    motion = load_amass(path, settings)
    assert len(motion.body_pos_w) == 4
    angle = np.radians(170 + 20 / 3)
    np.testing.assert_allclose(
        motion.body_rot_mat_w[1, 0, :2, :2],
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        atol=1e-12,
    )
    settings["body_model_type"] = "smplh"
    # An explicit source model type mismatch must fail rather than silently use another family.
    np.savez(
        path,
        gender="neutral",
        surface_model_type="smplx",
        betas=[0],
        mocap_frame_rate=10,
        trans=np.zeros((2, 3)),
        poses=poses,
    )
    with pytest.raises(ValueError, match="does not match"):
        load_amass(path, settings)
