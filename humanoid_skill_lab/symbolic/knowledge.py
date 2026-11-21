"""Rules stay candidates until supporting and contrastive trials are supplied."""
from copy import deepcopy
from .predicates import parse
from humanoid_skill_lab.provenance import canonical_hash


def check_rule(rule):
    required = {"skill", "skill_version", "preconditions", "add_effects", "delete_effects", "status", "evidence"}
    if not isinstance(rule, dict) or set(rule) != required:
        raise ValueError("Rule fields do not match the schema")
    if not isinstance(rule["skill"], str) or type(rule["skill_version"]) is not int or rule["skill_version"] < 1:
        raise ValueError("Invalid skill binding")
    if not isinstance(rule["status"], str) or rule["status"] not in {"candidate", "validated", "revalidation_required"}:
        raise ValueError("Unknown rule status")
    for key in ("preconditions", "add_effects", "delete_effects"):
        if not isinstance(rule[key], list) or not all(isinstance(term,str) for term in rule[key]) or len(rule[key]) != len(set(rule[key])):
            raise ValueError("Rule terms must be unique lists")
        for atom in rule[key]:
            parse(atom)
    if set(rule["add_effects"]) & set(rule["delete_effects"]):
        raise ValueError("Contradictory effects")
    if not isinstance(rule["evidence"], dict):
        raise ValueError("Evidence must be an object")
    return deepcopy(rule)


class Knowledge:
    def __init__(self, rules=()):
        self.rules = [check_rule(r) for r in rules]

    def revise_skill(self, name, version):
        for rule in self.rules:
            if rule["skill"] == name and rule["skill_version"] != version:
                rule["status"] = "revalidation_required"

    def propose(self, rule):
        rule = check_rule(rule)
        rule["status"] = "candidate"
        rule["evidence"] = {}
        self.rules.append(rule)

    def record(self, index, episode_id, split, success):
        if split == "test":
            raise ValueError("Frozen test evidence cannot update knowledge")
        if split not in {"development", "validation"}:
            raise ValueError("Unknown evidence split")
        key = f"{split}_{'success' if success else 'failure'}"
        self.rules[index]["evidence"].setdefault(key, []).append(episode_id)
        if not success and self.rules[index]["status"] == "validated":
            self.rules[index]["status"] = "revalidation_required"

    def validate(self, index, contrastive_trials, coverage, minimum_coverage=.5):
        rule = self.rules[index]
        evidence = rule["evidence"]
        if not evidence.get("validation_success") or evidence.get("validation_failure"):
            raise ValueError("Effect validation requires independent successes without contradictions")
        if not 0 <= coverage <= 1 or coverage < minimum_coverage:
            raise ValueError("Rule excludes too much of the declared task domain")
        # Trusted experiment coordinator supplies matched-pair records, not GPT.
        for precondition in rule["preconditions"]:
            pairs = [t for t in contrastive_trials if t.get("condition") == precondition
                     and t.get("only_changed_condition") is True and t.get("present_success") is True
                     and t.get("absent_success") is False and t.get("split") == "validation"]
            if not pairs:
                raise ValueError("Every precondition needs a distinguishing validation experiment")
        evidence["contrastive_trials"] = deepcopy(contrastive_trials)
        evidence["coverage"] = coverage
        rule["status"] = "validated"

    def snapshot(self):
        return {"rules": deepcopy(self.rules), "sha256": canonical_hash(self.rules)}
