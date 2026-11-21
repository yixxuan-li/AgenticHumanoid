import numpy as np
import mujoco
from humanoid_skill_lab.types import Observation, Pose


class OnboardCamera:
    """Simulated torso RGB-D camera; extrinsics use proprioceptive camera pose.

    Optical coordinates: right/down/forward. Installation is an experimental
    configuration, not a claim of calibrated physical G1 hardware placement.
    """
    def __init__(self, model, width=320, height=240, object_bodies=None):
        self.model = model
        self.object_bodies = {"object": "env_cube", "target": "env_target"} if object_bodies is None else object_bodies
        self.width, self.height = width, height
        self.cid = model.camera("onboard").id
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        f = height / (2*np.tan(np.deg2rad(model.cam_fovy[self.cid])/2))
        self.intrinsics = np.array([[f, 0, (width-1)/2], [0, f, (height-1)/2], [0, 0, 1.]])

    def observe(self, data, mode):
        self.renderer.update_scene(data, camera="onboard")
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        transform = np.eye(4)
        transform[:3, :3] = data.cam_xmat[self.cid].reshape(3, 3) @ np.diag([1., -1, -1])
        transform[:3, 3] = data.cam_xpos[self.cid]
        poses = {}
        if mode == "oracle_debug":
            for name, body in self.object_bodies.items():
                idx = self.model.body(body).id
                poses[name] = Pose(tuple(data.xpos[idx]), tuple(data.xquat[idx]), "world", float(data.time), "oracle")
        return Observation(float(data.time), rgb, self.intrinsics.copy(), transform,
                           {"root_position": data.qpos[:3].tolist(), "joint_position": data.qpos[7:36].tolist()},
                           "oracle" if mode == "oracle_debug" else "camera", depth, poses)

    def close(self):
        self.renderer.close()
