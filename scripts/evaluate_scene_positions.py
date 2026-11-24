"""Freeze a skill, add declared props, and evaluate across scene/workcell poses."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
# Remove credentials before importing the simulation/policy modules.
for _name in tuple(os.environ):
    if any(token in _name.upper() for token in ["API_KEY", "API_TOKEN", "AUTH_TOKEN"]):
        os.environ.pop(_name, None)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_skill_lab.provenance import canonical_hash, digest
from humanoid_skill_lab.skills.schema import load
from humanoid_skill_lab.runner import make_scene, run_episode, write_json
from scene_reconstruction.provenance import scene_fingerprint


def preflight(config, output):
    """Initial geometry audit, not a control rollout or a task success test."""
    import mujoco
    output.mkdir()
    model = mujoco.MjModel.from_xml_path(str(make_scene(output, config)))
    data = mujoco.MjData(model)
    task = config["task_scene"]
    jid = model.body_jntadr[model.body("pelvis").id]
    address = model.jnt_qposadr[jid]
    data.qpos[address:address+7] = [*task["robot_position"], *Rotation.from_euler("z", task["yaw"]).as_quat(scalar_first=True)]
    metadata = json.loads((ROOT/"checkpoints/scalebfm_m_metadata.json").read_text())
    for name, value in zip(metadata["joint_names"], metadata["default_dof_pos"]):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_forward(model, data)
    penetrations = []
    for contact in data.contact:
        if contact.dist >= -.008:
            continue
        names = [model.body(int(model.geom_bodyid[gid])).name for gid in [contact.geom1, contact.geom2]]
        if names[0].startswith("env_") != names[1].startswith("env_"):
            penetrations.append(dict(bodies=names, depth_m=float(-contact.dist)))
        elif "env_cube" in names:
            penetrations.append(dict(bodies=names, depth_m=float(-contact.dist)))
    result = dict(status="blocked_initial_penetration" if penetrations else "ready",
                  initial_penetrations=penetrations, threshold_m=.008,
                  task_objects=["env_cube", "env_target", "env_distractor_0", "env_distractor_1"],
                  physics_executed=False)
    write_json(output/"preflight.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"humanoid_skill_lab/configs/scene_positions.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    original = load(ROOT/config["skill"])
    skill = deepcopy(original)
    skill["frame"] = "episode"
    assert {k:v for k,v in skill.items() if k != "frame"} == {k:v for k,v in original.items() if k != "frame"}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/"suite.json", config)
    write_json(args.output/"frozen_skill.json", skill)
    source = ROOT/config["reconstructed_scene"]
    before = scene_fingerprint(source)
    results = []
    for cell in config["workcells"]:
        for index, position in enumerate(config["positions"]):
            name = cell["id"]+"_"+position["id"]
            episode = dict(seed=2000+len(results), split="cross_scene_test", observation_mode="onboard_vision",
                           seconds=config["seconds"], mode=2,
                           task_scene={**cell["task"], "object_xy":position["object_xy"],
                                       "target_xy":[position["object_xy"][0]+.13, position["object_xy"][1]]})
            if cell["reconstructed"]:
                episode["scene_directory"] = str(source.resolve())
            write_json(args.output/(name+"_config.json"), episode)
            print("CROSS_SCENE start "+name, flush=True)
            audit = preflight(episode, args.output/(name+"_preflight"))
            row = dict(case=name, workcell=cell["id"], position=position["id"],
                       reconstructed=cell["reconstructed"], support_height=cell["task"]["support_height"],
                       yaw=cell["task"]["yaw"], preflight=audit, success=False, physics_executed=False)
            if audit["status"] == "ready":
                try:
                    row["metrics"] = run_episode(skill, episode, args.output/name, args.device, not args.no_video)
                    row["success"] = row["metrics"]["success"]
                    row["physics_executed"] = True
                    row["status"] = "executed"
                except Exception as error:
                    row.update(status="execution_error", error_type=type(error).__name__, error=str(error))
            else:
                row["status"] = audit["status"]
            results.append(row)
            write_json(args.output/"progress.json", dict(results=results))
            print("CROSS_SCENE done "+name+" status="+row["status"]+" success="+str(row["success"]), flush=True)
    assert before == scene_fingerprint(source), "Source reconstruction was modified"
    groups = {}
    for cell in config["workcells"]:
        rows = [r for r in results if r["workcell"] == cell["id"]]
        groups[cell["id"]] = dict(cases=len(rows), executed=sum(r["physics_executed"] for r in rows),
                                  successes=sum(r["success"] for r in rows), blocked=sum(r["status"].startswith("blocked") for r in rows))
    summary = dict(status="completed", frozen_original_skill_sha256=canonical_hash(original),
                   executed_skill_sha256=canonical_hash(skill), frame_port="Same stages and parameters; explicit episode-frame translation/yaw",
                   source_scene_fingerprint=before["sha256"], source_scene_unchanged=True,
                   observation_mode="onboard_vision", parameter_tuning=False, gpt_calls=0,
                   environment_count=2, note="One open workcell and one captured room; placements are not independent reconstructed rooms",
                   cases=len(results), physics_episodes=sum(r["physics_executed"] for r in results),
                   successes=sum(r["success"] for r in results), groups=groups, results=results)
    write_json(args.output/"summary.json", summary)
    print(json.dumps({k:summary[k] for k in ["cases", "physics_episodes", "successes", "groups"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
