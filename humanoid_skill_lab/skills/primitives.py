"""Transparent motion tools. No primitive implements a complete contact task."""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_skill_lab.types import Truth


def start(stage, belief, provider, now, frame="world"):
    primitive, p = stage["primitive"], stage["parameters"]
    body = p.get("hand", "right") + "_wrist_yaw_link"
    current, _ = provider.sample(now, [0])
    rotation = provider.frame_rotation if frame == "episode" else np.eye(3)
    if primitive in {"wait", "observe"}:
        return
    if primitive == "move_root_world":
        target = rotation @ np.asarray(p["position"]) + (provider.frame_origin if frame == "episode" else 0)
        body = "pelvis"
    elif primitive == "set_body_height":
        target = current[0, provider.names.index("pelvis")].copy()
        target[2] = p["height"]
        body = "pelvis"
    elif primitive == "move_hand_relative":
        reference = p["reference"]
        pose = belief.poses.get(reference)
        if pose is None or pose.frame != "world" or now-pose.timestamp > .25 or pose.timestamp > now:
            raise ValueError("Target pose is absent, stale, or in an unsupported frame")
        target = np.asarray(pose.position) + rotation @ np.asarray(p["offset"])
    elif primitive == "move_hand_world":
        target = rotation @ np.asarray(p["position"]) + (provider.frame_origin if frame == "episode" else 0)
    elif primitive == "retract_hand":
        target = current[0, provider.names.index(body)] + rotation @ np.asarray(p["offset"])
    else:
        raise ValueError("Unsupported primitive")
    quaternion = p.get("quaternion_wxyz")
    if quaternion is not None and frame == "episode":
        quaternion = (Rotation.from_matrix(rotation)*Rotation.from_quat(quaternion, scalar_first=True)).as_quat(scalar_first=True)
    provider.command(now, body, target, p["duration"], quaternion)
