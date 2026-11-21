"""Feedback stage interpreter. No eval, exec, imports, or simulator handles."""
from .schema import check
from .primitives import start
from humanoid_skill_lab.types import Truth


class SkillRuntime:
    def __init__(self, program, provider):
        self.program = check(program)
        self.provider = provider
        self.stages = {s["id"]: s for s in self.program["stages"]}
        self.stage = None
        self.next_id = program["stages"][0]["id"]
        self.visits = 0
        self.started = 0.
        self.outcome = "running"
        self.reason = None
        self.events = []

    def update(self, now, belief):
        if self.outcome != "running":
            return
        if self.stage is None:
            if self.next_id is None:
                self.outcome = "completed"
                return
            if self.visits >= self.program["max_transitions"]:
                self.outcome, self.reason = "failure", "transition_budget_exhausted"
                return
            self.stage = self.stages[self.next_id]
            self.started = now
            self.visits += 1
            try:
                start(self.stage, belief, self.provider, now, self.program.get("frame", "world"))
            except ValueError as error:
                self.outcome, self.reason = "failure", str(error)
                self.events.append({"timestamp": now, "event": "primitive_rejected", "reason": self.reason})
                return
            self.events.append({"timestamp": now, "event": "stage_started", "stage": self.stage["id"]})
        elapsed = now-self.started
        condition = self.stage.get("success_when")
        if condition and belief.query(condition, now) == Truth.TRUE:
            done, event = True, "condition_satisfied"
        elif not condition and elapsed+1e-8 >= self.stage["parameters"]["duration"]:
            done, event = True, "duration_complete"
        elif elapsed+1e-8 >= self.stage.get("timeout", self.stage["parameters"]["duration"]+1):
            self.events.append({"timestamp": now, "event": "stage_timeout", "stage": self.stage["id"]})
            if "on_timeout" not in self.stage:
                self.outcome, self.reason = "timeout", "stage_timeout"
                return
            self.next_id, self.stage = self.stage["on_timeout"], None
            return
        else:
            return
        if done:
            self.events.append({"timestamp": now, "event": event, "stage": self.stage["id"]})
            self.next_id, self.stage = self.stage.get("next"), None
