"""Controller-only scoring in imported scenes, without invented task objects."""
import numpy as np


class ControlEvaluator:
    def __init__(self, model, data):
        self.model = model
        self.rows = []
        self.forbidden = []
        self.fallen = False

    def update(self, data, contacts):
        m = self.model
        self.fallen |= bool(data.qpos[2] < .4 or data.xmat[m.body("pelvis").id].reshape(3,3)[2,2] < .5)
        for contact in contacts:
            a, b = contact["body1"], contact["body2"]
            if a.startswith("env_") == b.startswith("env_"):
                continue
            environment, robot = (a,b) if a.startswith("env_") else (b,a)
            if environment.startswith("env_floor") and "ankle" in robot:
                continue
            self.forbidden.append(contact)
        self.rows.append({"timestamp": float(data.time), "root_position": data.qpos[:3].tolist()})

    def report(self, runtime, elapsed, deadline, hold_seconds=1.):
        return {"success": False, "physics_executed": True, "evaluator": "imported_scene_control_probe_v1",
                "fallen": self.fallen, "timeout": runtime.outcome == "timeout" or (runtime.outcome == "running" and elapsed >= deadline-.02),
                "forbidden_contacts": len(self.forbidden), "goal_held": None, "final_inside": None,
                "hand_clear": None, "final_target_distance_m": None, "object_displacement_m": None,
                "runtime_outcome": runtime.outcome, "runtime_reason": runtime.reason,
                "elapsed_sim_seconds": elapsed, "manipulation_success": None}
