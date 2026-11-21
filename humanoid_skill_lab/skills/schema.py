"""Strict JSON DSL. Bounded transitions ensure all programs terminate."""
from copy import deepcopy
import json
import math

PRIMITIVES = {"wait", "move_hand_relative", "move_hand_world", "retract_hand", "set_body_height", "move_root_world", "observe"}


class ValidationError(ValueError):
    pass


def number(x, low, high):
    return type(x) in (int, float) and math.isfinite(x) and low <= x <= high


def check(program):
    def require(condition, message):
        if not condition:
            raise ValidationError(message)
    require(isinstance(program, dict), "Program must be an object")
    required = {"name", "version", "max_transitions", "stages"}
    require(required <= set(program) <= required | {"frame"}, "Unexpected or missing program fields")
    require(isinstance(program.get("frame", "world"), str) and program.get("frame", "world") in {"world", "episode"}, "Unknown program coordinate frame")
    require(isinstance(program["name"], str) and program["name"].isidentifier(), "Invalid skill name")
    require(type(program["version"]) is int and program["version"] >= 1, "Version must be positive integer")
    require(type(program["max_transitions"]) is int and 1 <= program["max_transitions"] <= 32, "Transition budget must be 1..32")
    stages = program["stages"]
    require(isinstance(stages, list) and 1 <= len(stages) <= 16, "Expected 1..16 stages")
    ids = []
    for stage in stages:
        require(isinstance(stage, dict), "Stage must be an object")
        require(set(stage) <= {"id", "primitive", "parameters", "next", "success_when", "on_timeout", "timeout"}, "Unknown stage field")
        require({"id", "primitive", "parameters"} <= set(stage), "Missing stage fields")
        require(isinstance(stage["id"], str) and stage["id"].isidentifier(), "Invalid stage id")
        ids.append(stage["id"])
        primitive = stage["primitive"]
        require(isinstance(primitive, str) and primitive in PRIMITIVES, "Primitive is not allowlisted")
        p = stage["parameters"]
        require(isinstance(p, dict), "Parameters must be an object")
        fields = {"wait": {"duration"}, "observe": {"duration"}, "set_body_height": {"height", "duration"},
                  "move_root_world": {"position", "duration"},
                  "move_hand_relative": {"reference", "offset", "duration", "hand", "quaternion_wxyz"},
                  "move_hand_world": {"position", "duration", "hand", "quaternion_wxyz"},
                  "retract_hand": {"offset", "duration", "hand"}}[primitive]
        require(set(p) <= fields, "Unexpected primitive parameter")
        require(number(p.get("duration"), .1, 12), "Duration must be 0.1..12 s")
        require(number(stage.get("timeout", p["duration"]+1), p["duration"], 15), "Invalid stage timeout")
        if "hand" in fields:
            require(isinstance(p.get("hand", "right"), str) and p.get("hand", "right") in {"left", "right"}, "Unknown hand")
        if primitive in {"move_hand_relative", "retract_hand", "move_hand_world"}:
            key = "position" if primitive == "move_hand_world" else "offset"
            require(isinstance(p.get(key), list) and len(p[key]) == 3 and all(number(x, -1.5, 1.5) for x in p[key]), "Invalid target vector")
        if primitive == "move_hand_relative":
            require(isinstance(p.get("reference"), str) and p.get("reference") in {"object", "target"}, "Reference must bind object or target")
        if primitive == "set_body_height":
            require(number(p.get("height"), .60, .85), "Height outside validated exploration envelope")
        if primitive == "move_root_world":
            v = p.get("position")
            require(isinstance(v, list) and len(v) == 3 and number(v[0], -.1, .3)
                    and number(v[1], -.15, .15) and number(v[2], .60, .85), "Root target outside exploration envelope")
        if "quaternion_wxyz" in p:
            q = p["quaternion_wxyz"]
            require(isinstance(q, list) and len(q) == 4 and all(number(x, -1, 1) for x in q) and abs(sum(x*x for x in q)-1) < 1e-4, "Invalid orientation")
        if "success_when" in stage:
            require(isinstance(stage["success_when"], str) and stage["success_when"] in {"object_in_region(object,target)", "visible(object)", "hand_free(right)"}, "Unknown condition")
    require(len(ids) == len(set(ids)), "Duplicate stage ids")
    for stage in stages:
        for key in ("next", "on_timeout"):
            require(stage.get(key) is None or stage[key] in ids, "Dangling transition")
    # Every visit is bounded by timeout; loops consume the global transition budget.
    require(program["max_transitions"] * max(s.get("timeout", s["parameters"]["duration"]+1) for s in stages) <= 240,
            "Program worst-case duration exceeds 240 seconds")
    return deepcopy(program)


def load(path):
    with open(path) as stream:
        return check(json.load(stream))
