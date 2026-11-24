"""Save reusable skill JSON with honest selection status and evidence digests.

Only the registry-selected pair is reusable by default. Rejected proposals are
retained separately and cannot become validated through this export operation.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from humanoid_skill_lab.exploration.promotion import Registry
from humanoid_skill_lab.provenance import canonical_hash, digest
from humanoid_skill_lab.skills.schema import check


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run/"summary.json").read_text())
    config = json.loads((run/"experiment.json").read_text())
    snapshot, pair = Registry(run/"registry").current()
    skill = check(pair["skill"])
    accepted = pair["decision"].get("accepted") is True
    status = "validation_selected" if accepted else "unvalidated_baseline"
    evidence = []
    for path in sorted(run.glob("*/metrics.json")):
        manifest = json.loads(path.with_name("manifest.json").read_text())
        metrics = json.loads(path.read_text())
        if manifest["config"]["observation_mode"] != config["observation_mode"]:
            raise ValueError("Observation-mode mismatch in saved experiment evidence")
        if config["observation_mode"] == "onboard_vision" and any(s != "camera" for s in metrics["agent_pose_sources"]):
            raise ValueError("Non-camera object pose found in an onboard experiment")
        evidence.append(dict(episode=path.parent.name, split=manifest["config"]["split"],
                             skill_sha256=manifest["skill_sha256"], metrics_sha256=digest(path),
                             source_hashes=manifest["source_hashes"], project_commit=manifest["project_commit"],
                             checkpoint_sha256=manifest["checkpoint_sha256"],
                             success=metrics["success"], physics_executed=metrics["physics_executed"],
                             agent_pose_sources=metrics["agent_pose_sources"],
                             runtime_outcome=metrics["runtime_outcome"], runtime_reason=metrics["runtime_reason"]))
    test = None
    if args.test:
        frozen = json.loads((args.test/"summary.json").read_text())
        if frozen["frozen_snapshot"] != snapshot or frozen["used_for_tuning"] is not False:
            raise ValueError("Frozen test does not match the selected immutable pair")
        test = {k:frozen[k] for k in ["frozen_snapshot", "successes", "episodes", "used_for_tuning"]}
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/"skill.json", skill)
    write(args.output/"knowledge.json", pair["knowledge"])
    write(args.output/"snapshot.json", {"snapshot": snapshot, **pair})
    write(args.output/"experiment.json", config)
    write(args.output/"evidence.json", evidence)
    write(args.output/"experiment_summary.json", summary)
    candidates = []
    for proposal in sorted(run.glob("proposal_*")):
        response = proposal/"response.json"
        if not response.exists():
            continue
        try:
            raw = json.loads(response.read_text())
            candidate = check(raw["skill"])
        except (ValueError, KeyError, TypeError):
            continue
        folder = args.output/"candidates"/proposal.name
        folder.mkdir(parents=True)
        write(folder/"skill.json", candidate)
        decision = json.loads((proposal/"decision.json").read_text()) if (proposal/"decision.json").exists() else {"accepted": False, "reason": "not_validated"}
        write(folder/"decision.json", decision)
        if (proposal/"knowledge_after_validation.json").exists():
            write(folder/"knowledge.json", json.loads((proposal/"knowledge_after_validation.json").read_text()))
        candidates.append(dict(proposal=proposal.name, program_sha256=canonical_hash(candidate), decision=decision))
    manifest = dict(status=status, observation_mode=config["observation_mode"],
                    task="Controlled tabletop cube push with hand retreat", simulator="MuJoCo / ScaleBridge / frozen ScaleBFM",
                    reconstruction_scene=False, perception_assumptions="Known 8cm cube and colored markers; simulated RGB-D camera",
                    selected_snapshot=snapshot, skill_sha256=canonical_hash(skill),
                    experiment_directory=str(run.relative_to(ROOT)), episodes=len(evidence),
                    new_skill_selected=accepted, api_calls=summary["api_calls"], tokens=summary["tokens"],
                    uses_online_pose_targets=any(s["primitive"] == "move_hand_relative" for s in skill["stages"]),
                    frozen_test=test, candidates=candidates, symbolic_rules_validated=False)
    write(args.output/"manifest.json", manifest)
    (args.output/"README.md").write_text(
        "# Saved onboard experiment skill\n\n"
        f"Status: **{status}**. Observation mode: **{config['observation_mode']}**.\n\n"
        "`skill.json` and `knowledge.json` are the registry-selected pair. Candidate programs and their decisions are retained under `candidates/`. "
        "Saving a file does not establish acquired capability or validated symbolic rules. See `manifest.json` and `evidence.json` for selection and physical execution evidence.\n\n"
        "Replay in the same bounded simulation benchmark:\n\n```bash\n"
        f".venv-skill-lab/bin/humanoid-lab run --skill {args.output.as_posix()}/skill.json --observation-mode onboard_vision --output outputs/skill_lab/new_replay\n```\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
