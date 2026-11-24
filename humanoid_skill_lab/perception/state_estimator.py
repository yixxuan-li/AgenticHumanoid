"""Known color/geometry RGB-D detector. No simulator access in this module."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_skill_lab.types import BeliefState, Pose, Truth


class StateEstimator:
    def __init__(self, mode, enabled_objects=True, workspace=None):
        if mode not in {"oracle_debug", "onboard_vision"}:
            raise ValueError("Unknown observation mode")
        self.mode = mode
        self.enabled_objects = enabled_objects
        self.workspace = workspace

    def update(self, obs):
        if self.mode == "onboard_vision" and obs.oracle_poses:
            raise ValueError("Oracle poses must never enter onboard perception")
        if not self.enabled_objects:
            if obs.oracle_poses:
                raise ValueError("No task object bindings are declared for this scene")
            return BeliefState(obs.timestamp, {}, {})
        masks = {}
        hsv = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2HSV)
        masks["object"] = (((hsv[:,:,0] < 12) | (hsv[:,:,0] > 170)) & (hsv[:,:,1] > 150) & (hsv[:,:,2] > 60))
        masks["target"] = ((hsv[:,:,0] > 35) & (hsv[:,:,0] < 85) & (hsv[:,:,1] > 120) & (hsv[:,:,2] > 40))
        poses = dict(obs.oracle_poses) if self.mode == "oracle_debug" else {}
        confidence = {}
        for name, mask in masks.items():
            count = int(mask.sum())
            confidence[name] = min(1., count/100.)
            if self.mode == "onboard_vision" and count >= 15 and obs.depth is not None:
                y, x = np.nonzero(mask & np.isfinite(obs.depth) & (obs.depth > .05) & (obs.depth < 3))
                if len(x) < 15:
                    continue
                z = obs.depth[y, x]
                points = np.stack([(x-obs.intrinsics[0,2])*z/obs.intrinsics[0,0],
                                   (y-obs.intrinsics[1,2])*z/obs.intrinsics[1,1], z], axis=1)
                world = points @ obs.world_from_camera[:3,:3].T + obs.world_from_camera[:3,3]
                if self.workspace:
                    area = self.workspace
                    local = (world-np.asarray(area["origin"])) @ Rotation.from_euler("z", area["yaw"]).as_matrix()
                    keep = np.all((local >= area["lower"]) & (local <= area["upper"]), axis=1)
                    world = world[keep]
                    mask[:] = False
                    mask[y[keep], x[keep]] = True
                    confidence[name] = min(1., len(world)/100.)
                    if len(world) < 15:
                        continue
                position = np.median(world, axis=0)
                quaternion = (1., 0, 0, 0)
                # A surface median is biased toward the visible front face.
                # Fit the known upright cube's top face in metric world XY.
                if name == "object":
                    top_z = np.quantile(world[:,2], .9)
                    top = world[world[:,2] >= top_z-.003]
                    if len(top) < 12:
                        continue
                    rectangle = cv2.minAreaRect(top[:,:2].astype(np.float32))
                    if min(rectangle[1]) < .035 or max(rectangle[1]) > .11:
                        confidence[name] *= .3
                        continue
                    position[:2] = rectangle[0]
                    position[2] = top_z-.04
                    quaternion = tuple(Rotation.from_euler("z", rectangle[2], degrees=True).as_quat(scalar_first=True))
                else:
                    # Known flat, axis-aligned target marker. Visible edge
                    # extrema reduce perspective and partial-occlusion bias.
                    plane = world[np.abs(world[:,2]-np.median(world[:,2])) < .003]
                    if len(plane) < 12:
                        continue
                    position[:2] = (plane[:,:2].min(axis=0)+plane[:,:2].max(axis=0))/2
                poses[name] = Pose(tuple(position), quaternion, "world", obs.timestamp, "camera")
        predicates = {"visible(object)": Truth.TRUE if masks["object"].sum() >= 15 else Truth.UNKNOWN,
                      "hand_free(right)": Truth.UNKNOWN, "object_in_region(object,target)": Truth.UNKNOWN}
        if {"object", "target"} <= poses.keys():
            a, b = np.asarray(poses["object"].position), np.asarray(poses["target"].position)
            inside = np.all(np.abs(a[:2]-b[:2]) <= .025) and abs(a[2]-b[2]-.04) < .03
            predicates["object_in_region(object,target)"] = Truth.TRUE if inside else Truth.FALSE
        # Hand occupancy lacks a tactile or gripper detector: keep unknown.
        return BeliefState(obs.timestamp, poses, predicates, confidence)
