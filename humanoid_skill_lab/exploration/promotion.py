"""Paired validation and atomic, content-addressed program/knowledge snapshots."""
import json
import os
from pathlib import Path
from humanoid_skill_lab.provenance import canonical_hash


def compare(candidate_reports, baseline_reports, minimum=3):
    if len(candidate_reports) != len(baseline_reports) or len(candidate_reports) < minimum:
        return {"accepted": False, "reason": "insufficient_paired_validation"}
    for candidate, baseline in zip(candidate_reports, baseline_reports):
        if candidate["configuration_sha256"] != baseline["configuration_sha256"]:
            raise ValueError("Promotion requires identical paired configurations")
        if not candidate.get("physics_executed") or not baseline.get("physics_executed"):
            return {"accepted": False, "reason": "physics_not_executed"}
        if candidate["fallen"] or candidate["timeout"] or candidate["forbidden_contacts"]:
            return {"accepted": False, "reason": "candidate_safety_or_timeout_failure"}
        if baseline["success"] and not candidate["success"]:
            return {"accepted": False, "reason": "validation_regression"}
    old = sum(r["success"] for r in baseline_reports)
    new = sum(r["success"] for r in candidate_reports)
    return {"accepted": new > old, "reason": "strict_success_improvement" if new > old else "no_success_improvement",
            "baseline_successes": old, "candidate_successes": new, "episodes": len(candidate_reports)}


class Registry:
    def __init__(self, path):
        self.path = Path(path)
        (self.path / "snapshots").mkdir(parents=True, exist_ok=True)

    def commit(self, skill, knowledge, decision, initial=False):
        if not initial and not decision.get("accepted"):
            raise ValueError("Rejected candidates cannot be committed")
        pair = {"skill": skill, "knowledge": knowledge, "decision": decision}
        identifier = canonical_hash(pair)
        path = self.path / "snapshots" / (identifier+".json")
        if not path.exists():
            path.write_text(json.dumps(pair, indent=2)+"\n")
        self.select(identifier)
        return identifier

    def select(self, identifier):
        if len(identifier) != 64 or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("Invalid snapshot identifier")
        path = self.path / "snapshots" / (identifier+".json")
        pair = json.loads(path.read_text())
        if canonical_hash(pair) != identifier:
            raise ValueError("Snapshot integrity check failed")
        temporary = self.path / "current.tmp"
        temporary.write_text(json.dumps({"snapshot": identifier})+"\n")
        os.replace(temporary, self.path / "current.json")

    def current(self):
        identifier = json.loads((self.path / "current.json").read_text())["snapshot"]
        path = self.path / "snapshots" / (identifier+".json")
        pair = json.loads(path.read_text())
        if canonical_hash(pair) != identifier:
            raise ValueError("Snapshot integrity check failed")
        return identifier, pair
