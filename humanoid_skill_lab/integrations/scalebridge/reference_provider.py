"""World-space smooth targets; official components perform full-root conversion."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class ReferenceProvider:
    def __init__(self, metadata):
        self.names = metadata["selected_body_names"]
        self.frame_origin = np.zeros(3)
        self.positions = np.zeros((len(self.names), 3))
        self.rotations = np.tile([1., 0, 0, 0], (len(self.names), 1))
        self.start = self.positions.copy()
        self.goal = self.positions.copy()
        self.start_time = 0.
        self.duration = 1.
        self.rotation_goal = self.rotations.copy()

    def reset(self, positions, rotations):
        self.positions = positions.copy()
        self.rotations = rotations.copy()
        self.start = positions.copy()
        self.goal = positions.copy()
        self.rotation_goal = rotations.copy()
        self.start_time = 0.
        self.duration = 1.

    def command(self, now, body, position, duration, quaternion=None):
        p, q = self.sample(now, [0])
        self.start = p[0]
        self.rotations = q[0]
        self.goal = self.start.copy()
        self.rotation_goal = self.rotations.copy()
        idx = self.names.index(body)
        target = np.asarray(position, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all() or not np.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid reference command")
        # Smoothstep peak velocity is 1.5 * distance / duration.
        if np.linalg.norm(target - self.start[idx]) * 1.5 / duration > .5:
            raise ValueError("Reference speed exceeds 0.5 m/s")
        self.goal[idx] = target
        if quaternion is not None:
            self.rotation_goal[idx] = quaternion
        self.start_time, self.duration = now, duration

    def sample(self, now, future_idx):
        u = np.clip((now + np.asarray(future_idx) * .02 - self.start_time) / self.duration, 0, 1)
        blend = u*u*(3-2*u)
        p = self.start[None] + blend[:, None, None] * (self.goal-self.start)[None]
        q = np.broadcast_to(self.rotations, (len(u), *self.rotations.shape)).copy()
        for i in range(len(self.names)):
            if not np.allclose(self.rotations[i], self.rotation_goal[i]):
                q[:, i] = Slerp([0, 1], Rotation.from_quat([self.rotations[i], self.rotation_goal[i]], scalar_first=True))(blend).as_quat(scalar_first=True)
        return p, q


class ReplayProvider(ReferenceProvider):
    def __init__(self, metadata, path):
        super().__init__(metadata)
        self.motion = np.load(path)
        selected = [metadata["body_names"].index(n) for n in self.names]
        self.replay_p = self.motion["body_pos_w"][:, selected]
        self.replay_q = self.motion["body_quat_w"][:, selected]

    def reset(self, positions, rotations):
        pass

    def sample(self, now, future_idx):
        idx = np.clip(round(now/.02) + np.asarray(future_idx), 0, len(self.replay_p)-1)
        return self.replay_p[idx], self.replay_q[idx]
