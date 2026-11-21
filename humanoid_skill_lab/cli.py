"""Commands deliberately separate LLM networking from simulation execution."""
import argparse
import json
from pathlib import Path
from .provenance import ROOT


def main():
    p = argparse.ArgumentParser(prog="agentic-humanoid")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("scene", "navigate", "view"):
        q = sub.add_parser(name)
        q.add_argument("--scene", type=Path, default=ROOT/"outputs/worldlabs_scene")
        if name == "navigate":
            q.add_argument("--run-name", default="merged_navigation")
            q.add_argument("--device", default="cuda")
            q.add_argument("--video", action="store_true")
            q.add_argument("--dry-run", action="store_true")
        if name == "view":
            q.add_argument("--collision", action="store_true")
            q.add_argument("--fast-exit", action="store_true")
    d = sub.add_parser("doctor")
    d.add_argument("--output", type=Path, default=ROOT / "outputs/skill_lab/environment.json")
    d.add_argument("--tree", type=Path, default=ROOT / "outputs/skill_lab/upstream_tree.json")
    w = sub.add_parser("llm-worker", help=argparse.SUPPRESS)
    w.add_argument("--context", type=Path, required=True)
    w.add_argument("--llm-config", type=Path, required=True)
    w.add_argument("--output", type=Path, required=True)
    for name in ("explore", "evaluate"):
        q = sub.add_parser(name)
        q.add_argument("--config", type=Path, default=ROOT / "humanoid_skill_lab/configs/experiment.json")
        q.add_argument("--output", type=Path, required=True)
        q.add_argument("--device", default="cuda")
        q.add_argument("--no-video", action="store_true")
        if name == "explore":
            q.add_argument("--skill", type=Path, default=ROOT / "humanoid_skill_lab/configs/push.json")
            q.add_argument("--llm-config", type=Path, default=ROOT / "humanoid_skill_lab/configs/llm.json")
            q.add_argument("--bashrc", action="store_true", help="Load LLM worker credentials from interactive bash startup")
        else:
            q.add_argument("--registry", type=Path, required=True)
    q = sub.add_parser("rollback")
    q.add_argument("--registry", type=Path, required=True)
    q.add_argument("--snapshot", required=True)
    q = sub.add_parser("report")
    q.add_argument("--runs", type=Path, default=ROOT/"outputs/skill_lab")
    q.add_argument("--output", type=Path, default=ROOT/"reports")
    q.add_argument("--plot", action="store_true")
    for name in ("baseline", "run", "probe"):
        q = sub.add_parser(name)
        q.add_argument("--output", type=Path, required=True)
        q.add_argument("--skill", type=Path, default=ROOT / "humanoid_skill_lab/configs/push.json")
        q.add_argument("--config", type=Path)
        q.add_argument("--scene", type=Path, help="Previous version's exported scene directory; controller probes only")
        q.add_argument("--device", default="cuda")
        q.add_argument("--mode", type=int, choices=range(8))
        q.add_argument("--seconds", type=float)
        q.add_argument("--observation-mode", choices=("oracle_debug", "onboard_vision"))
        q.add_argument("--no-video", action="store_true")
        if name == "baseline":
            q.add_argument("--task", choices=("replay", "stand"), default="replay")
        if name == "probe":
            q.add_argument("--task", choices=("stand", "reach", "height"), required=True)
    args = p.parse_args()
    if args.command == "llm-worker":
        from .exploration.gpt_client import worker
        worker(args.context, args.llm_config, args.output)
        return
    if args.command == "doctor":
        from .diagnostics import inspect_environment
        report = inspect_environment(args.tree)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
        if not report["ready"]:
            raise SystemExit(2)
        return
    if args.command == "report":
        from .reporting import summarize, plot
        report = summarize(args.runs,args.output)
        if args.plot:
            plot(args.runs,args.output)
        print(json.dumps({k:report[k] for k in ("episode_count", "known_api_tokens", "unknown_usage_calls")}))
        return
    # Before importing torch/MuJoCo, remove credential variables from this process.
    import os
    for name in tuple(os.environ):
        if any(t in name.upper() for t in ("API_KEY", "API_TOKEN", "AUTH_TOKEN")):
            os.environ.pop(name, None)
    if args.command in {"scene", "navigate", "view"}:
        from .integrations.reconstructed_scene import load_scene
        from .integrations.project_commands import navigate, view
        if args.command == "scene":
            print(json.dumps(load_scene(args.scene), indent=2))
        elif args.command == "navigate":
            navigate(args.scene, args.run_name, args.device, args.video, args.dry_run)
        else:
            view(args.scene, args.collision, args.fast_exit)
        return
    if args.command == "rollback":
        from .exploration.promotion import Registry
        Registry(args.registry).select(args.snapshot)
        print(json.dumps({"selected_snapshot": args.snapshot}))
        return
    if args.command in {"explore", "evaluate"}:
        from .exploration.experiment import Experiment, frozen_evaluate
        from .skills.schema import load
        config = json.loads(args.config.read_text())
        if args.command == "explore":
            report = Experiment(config, args.output, args.device, not args.no_video).execute(load(args.skill), args.llm_config, args.bashrc)
        else:
            report = frozen_evaluate(args.registry, config, args.output, args.device, not args.no_video)
        print(json.dumps(report, indent=2))
        return
    from .runner import run_episode
    from .skills.schema import load
    config = {"seed": 0, "split": "development", "observation_mode": "oracle_debug", "seconds": 18., "mode": 2}
    if args.config:
        config.update(json.loads(args.config.read_text()))
    if args.scene:
        config["scene_directory"] = str(args.scene.resolve())
    for key in ("mode", "seconds", "observation_mode"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.command == "baseline":
        config["seconds"] = args.seconds or 8.
    if args.command == "probe":
        filename = {"stand": "stand", "reach": "reach", "height": "body_height"}[args.task]
        args.skill = ROOT / f"humanoid_skill_lab/configs/{filename}.json"
    report = run_episode(load(args.skill), config, args.output, args.device, not args.no_video,
                         args.task if args.command == "baseline" else None,
                         args.task if args.command == "probe" else None)
    print(json.dumps(report, indent=2))
