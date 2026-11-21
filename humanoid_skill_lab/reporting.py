"""Allowlisted experiment summaries built from saved physical results."""
import csv
import json
from pathlib import Path
from .provenance import ROOT, digest
from .runner import write_json


def summarize(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    episodes = []
    for path in sorted(root.rglob("metrics.json")):
        manifest_path = path.parent/"manifest.json"
        if not manifest_path.exists():
            continue
        metrics = json.loads(path.read_text())
        manifest = json.loads(manifest_path.read_text())
        row = {"episode": str(path.parent.relative_to(root)), "split": manifest["config"].get("split"),
               "observation_mode": manifest["config"].get("observation_mode"), "success": metrics["success"],
               "physics_executed": metrics["physics_executed"], "fallen": metrics["fallen"],
               "forbidden_contacts": metrics["forbidden_contacts"], "right_hand_tracking_mean_m": metrics["right_hand_tracking_mean_m"],
               "final_target_distance_m": metrics["final_target_distance_m"], "baseline": metrics.get("baseline"),
               "probe": metrics.get("probe"), "metrics_sha256": digest(path), "manifest_sha256": digest(manifest_path)}
        row["excluded_from_acceptance"] = row["episode"].split("/")[0] in {"replay_001", "replay_002", "replay_003", "push_initial_001", "exploration_001"}
        episodes.append(row)
    calls = []
    for path in sorted(root.rglob("call.json")):
        item = json.loads(path.read_text())
        calls.append({"record": str(path.relative_to(root)), "model": item["model"], "usage": item["usage"],
                      "usage_status": item["usage_status"],
                      "latency_seconds": item["latency_seconds"], "attempts": item["attempts"]})
    summaries = {}
    for path in sorted(root.glob("*/summary.json")):
        item = json.loads(path.read_text())
        summaries[path.parent.name] = {k:item[k] for k in ("status", "api_calls", "tokens", "episodes", "accepted_skill_version", "successes", "used_for_tuning", "test_feedback_used") if k in item}
    report = {"scope": "exploratory engineering runs; not a statistically powered manipulation study",
              "episodes": episodes, "episode_count": len(episodes), "calls": calls, "experiments": summaries,
              "known_api_tokens": sum((c["usage"] or {}).get("total_tokens",0) for c in calls),
              "unknown_usage_calls": sum(c["usage_status"] != "known" for c in calls),
              "excluded_debugging_runs": ["replay_001", "replay_002", "replay_003", "push_initial_001", "exploration_001"]}
    write_json(output/"skill_lab_results.json", report)
    with (output/"skill_lab_episodes.csv").open("w", newline="") as stream:
        if episodes:
            writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
            writer.writeheader()
            writer.writerows(episodes)
    return report


def plot(root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    root, output = Path(root), Path(output)
    # Paired development comparison only; no held-out outcome feedback.
    selected = [root/"exploration_003/round_00_baseline_development_00", root/"exploration_003/round_00_candidate_development_00"]
    if not all((p/"metrics.json").exists() for p in selected):
        return
    fig, axes = plt.subplots(1,2,figsize=(10,3.6))
    names = ["Initial program", "GPT revision"]
    for p,name in zip(selected,names):
        data = np.load(p/"robot_states.npz")
        axes[0].plot(data["timestamp"], data["tracking_error_m"][:,-1]*100, label=name)
    axes[0].set(xlabel="Simulation time (s)",ylabel="Right wrist position error (cm)")
    axes[0].legend()
    metrics = [json.loads((p/"metrics.json").read_text()) for p in selected]
    axes[1].bar(names,[m["forbidden_contacts"] for m in metrics], color=["#8b99a6","#348a89"])
    axes[1].set(ylabel="Forbidden contact samples (200 Hz)",title="Both programs failed the push goal")
    fig.tight_layout()
    fig.savefig(output/"skill_lab_tracking.png",dpi=180)
    plt.close(fig)
