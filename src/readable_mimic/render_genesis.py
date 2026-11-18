"""Native Genesis camera rendering of robot and non-colliding reference ghost."""

import numpy as np
import torch

from .rotations import mat_to_quat


class GenesisRenderer:
    def __init__(self, robot, resolution=(960, 720)):
        import genesis as gs

        self.resolution = resolution

        # A separate scene keeps visualization entities out of the rollout dynamics.
        self.scene = gs.Scene(
            show_viewer=False, profiling_options=gs.options.ProfilingOptions(show_FPS=False)
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=robot.mjcf_path, align=False, collision=False))
        self.ghost = self.scene.add_entity(
            gs.morphs.MJCF(file=robot.mjcf_path, align=False, collision=False),
            surface=gs.surfaces.Default(color=(0.2, 0.65, 1.0, 0.3)),
        )
        self.camera = self.scene.add_camera(
            res=resolution, pos=(-2.05, 2.05, 1.5), lookat=(0, 0, 0.7), fov=45, GUI=False
        )
        self.scene.build(n_envs=1)
        self.entities = (self.robot, self.ghost)
        self.joint_indices = [
            [entity.get_joint(name).qs_idx_local[0] for name in robot.joint_names] for entity in self.entities
        ]
        self.base_indices = [
            next(joint.qs_idx_local for joint in entity.joints if joint.type == gs.JOINT_TYPE.FREE)
            for entity in self.entities
        ]
        self.qpos = [entity.get_qpos().clone() for entity in self.entities]

    def render(self, robot_state, ghost_pose):
        poses = ((robot_state[0], robot_state[1], robot_state[4]), ghost_pose)
        for index, (entity, pose) in enumerate(zip(self.entities, poses)):
            base_pos_w, base_rot_mat_w, joint_pos = pose
            self.qpos[index][:, self.base_indices[index]] = torch.cat(
                (base_pos_w, mat_to_quat(base_rot_mat_w)), -1
            )
            self.qpos[index][:, self.joint_indices[index]] = joint_pos
            entity.set_qpos(self.qpos[index], zero_velocity=True)
        lookat = robot_state[0][0].detach().cpu().numpy()
        self.camera.set_pose(pos=lookat + np.array([-2.05, 2.05, 0.78]), lookat=lookat)
        rgb, _, _, _ = self.camera.render(rgb=True, force_render=True)
        return np.asarray(rgb).reshape(self.resolution[1], self.resolution[0], 3)

    def video_writer(self, output_path, fps):
        return GenesisVideoWriter(output_path, fps, self.resolution)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.scene.destroy()


class GenesisVideoWriter:
    """Encode in-process so native rendering does not affect ffmpeg child-process handles."""

    def __init__(self, path, fps, resolution):
        self.path, self.fps, self.resolution = str(path), fps, resolution

    def __enter__(self):
        import av

        self.container = av.open(self.path, mode="w")
        self.stream = self.container.add_stream("libx264", rate=self.fps)
        self.stream.width, self.stream.height = self.resolution
        self.stream.pix_fmt = "yuv420p"
        return self

    def append_data(self, rgb):
        import av

        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def __exit__(self, *args):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
