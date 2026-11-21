"""Timestamped, source-labeled observations. SI units; rotations use wxyz."""
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class Truth(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    frame: str
    timestamp: float
    source: str
    units: str = "m"

    def __post_init__(self):
        if self.source not in {"camera", "proprioception", "oracle"}:
            raise ValueError("Invalid pose source")
        if not self.frame or self.units != "m" or not np.isfinite([*self.position, *self.quaternion_wxyz, self.timestamp]).all():
            raise ValueError("Pose requires a finite timestamp, explicit frame, and SI units")
        if not np.isclose(np.linalg.norm(self.quaternion_wxyz), 1, atol=1e-4):
            raise ValueError("Quaternion must be normalized")


@dataclass
class Observation:
    timestamp: float
    rgb: np.ndarray
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    proprioception: dict
    source: str
    depth: np.ndarray | None = None
    oracle_poses: dict[str, Pose] = field(default_factory=dict)


@dataclass
class BeliefState:
    timestamp: float
    poses: dict[str, Pose]
    predicates: dict[str, Truth]
    confidence: dict[str, float] = field(default_factory=dict)

    def query(self, atom, now=None, max_age=0.25):
        if now is not None and (now < self.timestamp or now - self.timestamp > max_age):
            return Truth.UNKNOWN
        return self.predicates.get(atom, Truth.UNKNOWN)
