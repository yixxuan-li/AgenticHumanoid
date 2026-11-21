"""Trusted MuJoCo evaluator, outside the candidate-editable program surface."""
import numpy as np


class Evaluator:
    def __init__(self, model, data, minimum_height=.4):
        self.model = model
        self.initial_cube = data.xpos[model.body("env_cube").id].copy()
        self.rows = []
        self.forbidden = []
        self.fallen = False
        self.minimum_height = minimum_height

    def update(self, data, contacts):
        m = self.model
        cube = data.xpos[m.body("env_cube").id].copy()
        target = data.xpos[m.body("env_target").id].copy()
        hand = data.xpos[m.body("right_wrist_yaw_link").id].copy()
        half_extent = np.abs(data.xmat[m.body("env_cube").id].reshape(3,3)) @ np.array([.04,.04,.04])
        inside = bool(np.all(np.abs(cube[:2]-target[:2])+half_extent[:2] <= .065)
                      and abs(cube[2]-.575-half_extent[2]) < .025)
        clear = bool(np.linalg.norm(hand-cube) >= .10)
        up = data.xmat[m.body("pelvis").id].reshape(3,3)[2,2]
        self.fallen |= bool(data.qpos[2] < self.minimum_height or up < .5)
        for contact in contacts:
            b1, b2 = contact["body1"], contact["body2"]
            if contact["geom1"] == "env_floor":
                b1 = "env_floor"
            if contact["geom2"] == "env_floor":
                b2 = "env_floor"
            # Environment/environment and robot/robot contacts are not this check.
            if b1.startswith("env_") == b2.startswith("env_"):
                continue
            environment, robot = (b1, b2) if b1.startswith("env_") else (b2, b1)
            if environment == "env_cube" and ("wrist" in robot or "hand" in robot):
                continue
            if environment == "env_floor" and "ankle" in robot:
                continue
            # A plane attached to world has body name 'world', so also inspect geom.
            if "env_floor" in (contact["geom1"], contact["geom2"]) and "ankle" in robot:
                continue
            self.forbidden.append(contact)
        self.rows.append({"timestamp": float(data.time), "inside": inside, "hand_clear": clear,
                          "object_position": cube.tolist(), "hand_position": hand.tolist(),
                          "target_distance_m": float(np.linalg.norm(cube[:2]-target[:2]))})

    def report(self, runtime, elapsed, deadline, hold_seconds=1.):
        final = self.rows[-1]
        hold = [r for r in self.rows if r["timestamp"] >= elapsed-hold_seconds-1e-6]
        held = bool(hold and hold[-1]["timestamp"]-hold[0]["timestamp"] >= hold_seconds-.03
                    and all(r["inside"] and r["hand_clear"] for r in hold))
        success = bool(held and not self.fallen and not self.forbidden and runtime.outcome == "completed")
        return {"success": success, "physics_executed": True, "evaluator": "fixed_tabletop_v1",
                "fallen": self.fallen, "timeout": runtime.outcome == "timeout" or (runtime.outcome == "running" and elapsed >= deadline-.02),
                "forbidden_contacts": len(self.forbidden), "goal_held": held, "final_inside": final["inside"],
                "hand_clear": final["hand_clear"], "final_target_distance_m": final["target_distance_m"],
                "object_displacement_m": float(np.linalg.norm(np.asarray(final["object_position"])-self.initial_cube)),
                "runtime_outcome": runtime.outcome, "runtime_reason": runtime.reason, "elapsed_sim_seconds": elapsed}
