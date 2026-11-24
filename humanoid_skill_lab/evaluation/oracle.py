"""Trusted MuJoCo evaluator, outside the candidate-editable program surface."""
import numpy as np


class Evaluator:
    def __init__(self, model, data, minimum_height=.4, support_height=.575,
                 floor_geoms=("env_floor",), distractors=()):
        self.model = model
        self.initial_cube = data.xpos[model.body("env_cube").id].copy()
        self.rows = []
        self.forbidden = []
        self.fallen = False
        self.minimum_height = minimum_height
        self.support_height = support_height
        self.floor_geoms = set(floor_geoms)
        self.initial_distractors = {name:data.xpos[model.body(name).id].copy() for name in distractors}
        self.distractor_displacement = 0.

    def update(self, data, contacts):
        m = self.model
        cube = data.xpos[m.body("env_cube").id].copy()
        target = data.xpos[m.body("env_target").id].copy()
        hand = data.xpos[m.body("right_wrist_yaw_link").id].copy()
        half_extent = np.abs(data.xmat[m.body("env_cube").id].reshape(3,3)) @ np.array([.04,.04,.04])
        target_rotation = data.xmat[m.body("env_target").id].reshape(3,3)
        local_delta = target_rotation.T @ (cube-target)
        local_half_extent = np.abs(target_rotation.T @ data.xmat[m.body("env_cube").id].reshape(3,3)) @ np.array([.04]*3)
        inside = bool(np.all(np.abs(local_delta[:2])+local_half_extent[:2] <= .065)
                      and abs(cube[2]-self.support_height-half_extent[2]) < .025)
        for name, initial in self.initial_distractors.items():
            self.distractor_displacement = max(self.distractor_displacement, float(np.linalg.norm(data.xpos[m.body(name).id]-initial)))
        clear = bool(np.linalg.norm(hand-cube) >= .10)
        up = data.xmat[m.body("pelvis").id].reshape(3,3)[2,2]
        self.fallen |= bool(data.qpos[2] < self.minimum_height or up < .5)
        for contact in contacts:
            b1, b2 = contact["body1"], contact["body2"]
            if contact["geom1"] in self.floor_geoms:
                b1 = "env_floor"
            if contact["geom2"] in self.floor_geoms:
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
            if self.floor_geoms.intersection([contact["geom1"], contact["geom2"]]) and "ankle" in robot:
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
        success = bool(held and not self.fallen and not self.forbidden and self.distractor_displacement <= .02 and runtime.outcome == "completed")
        return {"success": success, "physics_executed": True,
                "evaluator": "declared_workcell_v1" if self.initial_distractors else "fixed_tabletop_v1",
                "support_height_m": self.support_height, "distractor_max_displacement_m": self.distractor_displacement,
                "fallen": self.fallen, "timeout": runtime.outcome == "timeout" or (runtime.outcome == "running" and elapsed >= deadline-.02),
                "forbidden_contacts": len(self.forbidden), "goal_held": held, "final_inside": final["inside"],
                "hand_clear": final["hand_clear"], "final_target_distance_m": final["target_distance_m"],
                "object_displacement_m": float(np.linalg.norm(np.asarray(final["object_position"])-self.initial_cube)),
                "runtime_outcome": runtime.outcome, "runtime_reason": runtime.reason, "elapsed_sim_seconds": elapsed}
