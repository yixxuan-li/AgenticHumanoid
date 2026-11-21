"""Network calls happen in a short-lived shell, after control episodes finish."""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from .gpt_client import GPTError
from humanoid_skill_lab.skills.schema import check, PRIMITIVES
from humanoid_skill_lab.symbolic.knowledge import check_rule
from humanoid_skill_lab.provenance import ROOT


def validate_candidate(candidate, current):
    if not isinstance(candidate, dict) or set(candidate) != {"failure_hypothesis", "skill", "rule", "next_experiment", "expected_observation"}:
        raise ValueError("Unexpected proposal fields")
    for name in ("failure_hypothesis", "next_experiment", "expected_observation"):
        if not isinstance(candidate[name], str) or not candidate[name].strip() or len(candidate[name]) > 10000:
            raise ValueError("Proposal explanations must be bounded nonempty strings")
    program = check(candidate["skill"])
    rule = check_rule(candidate["rule"])
    if program["name"] != current["name"] or program["version"] != current["version"]+1:
        raise ValueError("Candidate must revise the current skill by exactly one version")
    if rule["skill"] != program["name"] or rule["skill_version"] != program["version"]:
        raise ValueError("Rule and program versions disagree")
    if rule["status"] != "candidate" or rule["evidence"]:
        raise ValueError("GPT cannot declare validated rules or manufacture evidence")
    return candidate


def propose(context, llm_config, output, bashrc=False):
    output = Path(output).resolve()
    context_path = output.parent / (output.name+"_context.json")
    context_path.write_text(json.dumps(context, indent=2)+"\n")
    command = [sys.executable, "-m", "humanoid_skill_lab", "llm-worker", "--context", str(context_path),
               "--llm-config", str(Path(llm_config).resolve()), "--output", str(output)]
    env = {k:v for k,v in os.environ.items() if not any(t in k.upper() for t in ("API_KEY", "API_TOKEN", "AUTH_TOKEN"))}
    if bashrc:
        command = ["bash", "-ic", "exec "+shlex.join(command)]
    # Never relay shell stderr: user startup scripts may print sensitive values.
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        output.mkdir(exist_ok=True)
        (output/"worker_error.json").write_text(json.dumps({"type":"Timeout", "usage":"unknown"})+"\n")
        raise GPTError("LLM worker timed out; API usage is unknown") from None
    if result.returncode:
        if (output/"worker_error.json").exists():
            failure = json.loads((output/"worker_error.json").read_text())
            if failure.get("type") == "GPTError":
                raise GPTError(failure["message"])
        raise GPTError("Proposal worker failed; inspect sanitized call.json if present")
    return validate_candidate(json.loads((output/"response.json").read_text()), context["current_skill"])


def interface_description():
    return {"program_fields": ["name", "version", "max_transitions", "stages", "frame (optional): world|episode"], "max_stages": 16, "max_transitions": 32,
            "stage_fields": ["id", "primitive", "parameters", "next", "success_when", "timeout", "on_timeout"],
            "stage_duration_range_s": [.1, 12], "max_stage_timeout_s": 15, "max_program_worst_case_s": 240,
            "primitives": {"wait": ["duration"], "observe": ["duration"],
                           "move_hand_relative": ["reference: object|target", "offset: xyz meters in world axes", "duration", "hand: right|left", "quaternion_wxyz: optional"],
                           "move_hand_world": ["position: xyz meters", "duration", "hand", "quaternion_wxyz: optional"],
                           "retract_hand": ["offset: xyz meters from last reference", "duration", "hand"],
                           "set_body_height": ["height: .60..85 meters", "duration"],
                           "move_root_world": ["position: x=-.1..3, y=-.15..15, z=.60..85 meters", "duration"]},
            "conditions": ["object_in_region(object,target)", "visible(object)", "hand_free(right)"],
            "limitations": ["Target latched on stage entry; feedback conditions checked every 20ms",
                            "Right hand mode 2 also conditions root and left hand; all unmodified reference bodies retain initialized pose",
                            "Smoothstep reference peak speed <= .5m/s", "No fingers in rubber-hand G1 asset",
                            "hand_free is unknown without occupancy sensing", "No hidden push/grasp/drawer primitive"],
            "hand_geometry": {"reference_body": "right_wrist_yaw_link", "orientation_frame": "world wxyz",
                              "tool_offset_direction": "rubber fingers extend along wrist local +x",
                              "rubber_hand_aabb_wrist_frame_m": [[.0415,-.0217,-.0431],[.1734,.0450,.0635]],
                              "warning": "Keeping the initialized downward wrist pitch can place fingers below table even when wrist target is above table. Orientation is a permitted program parameter."}}
