"""Development proposals, paired validation, and frozen test isolation."""
import json
from pathlib import Path
from copy import deepcopy
from .promotion import Registry, compare
from .proposer import propose, interface_description
from humanoid_skill_lab.provenance import canonical_hash, ROOT, digest
from humanoid_skill_lab.runner import run_episode, write_json
from humanoid_skill_lab.symbolic.knowledge import Knowledge


def check_datasets(config):
    for key in ("rounds", "max_api_calls", "max_tokens", "max_episodes", "minimum_validation_episodes"):
        if type(config.get(key)) is not int or config[key] < (0 if key in {"max_api_calls", "max_tokens"} else 1):
            raise ValueError(f"Invalid finite experiment budget: {key}")
    if config["minimum_validation_episodes"] < 3:
        raise ValueError("At least three paired validation configurations are required")
    seen = set()
    for split in ("development", "validation", "test"):
        if not config.get(split):
            raise ValueError("All three data splits must be nonempty")
        for item in config[split]:
            identity = canonical_hash({k:v for k,v in item.items() if k != "seed"})
            if identity in seen:
                raise ValueError("Overlapping episode configurations across data splits")
            seen.add(identity)
            if type(item.get("seed")) is not int:
                raise ValueError("Each configuration needs an integer seed")
    if len(config["validation"]) < config["minimum_validation_episodes"]:
        raise ValueError("Not enough validation configurations")
    if config.get("ablation") not in {"fixed", "skill_only", "symbolic_only", "joint"}:
        raise ValueError("Unknown ablation")


def brief(report):
    fields = ("success", "fallen", "timeout", "forbidden_contacts", "goal_held", "final_inside", "hand_clear",
              "final_target_distance_m", "object_displacement_m", "runtime_outcome", "runtime_reason",
              "right_hand_tracking_mean_m", "root_tracking_mean_m")
    return {k:report[k] for k in fields}


class Experiment:
    def __init__(self, config, output, device="cuda", video=True):
        check_datasets(config)
        self.config, self.output, self.device, self.video = config, Path(output), device, video
        self.output.mkdir(parents=True, exist_ok=False)
        self.episodes, self.api_calls, self.tokens = 0, 0, 0
        self.replayed_proposals = 0
        self.usage_unknown = False
        self.registry = Registry(self.output / "registry")
        self.history = deepcopy(config.get("prior_development_feedback", []))
        write_json(self.output / "experiment.json", config)

    def run_set(self, skill, knowledge, split, label):
        reports, paths = [], []
        for i, item in enumerate(self.config[split]):
            if self.episodes >= self.config["max_episodes"]:
                raise ValueError("Episode budget exhausted")
            cfg = {**item, "split": split, "knowledge": knowledge,
                   "observation_mode": self.config["observation_mode"], "seconds": self.config["episode_seconds"], "mode": 2}
            path = self.output / f"{label}_{split}_{i:02d}"
            self.episodes += 1
            report = run_episode(skill, cfg, path, self.device, self.video)
            report["configuration_sha256"] = canonical_hash({k:v for k,v in cfg.items() if k != "knowledge"})
            write_json(path / "metrics.json", report)
            reports.append(report)
            paths.append(path)
        return reports, paths

    def execute(self, initial_skill, llm_config, bashrc=False):
        from .gpt_client import GPTError
        skill = deepcopy(initial_skill)
        knowledge = {"rules": []}
        self.registry.commit(skill, knowledge, {"reason": "initial_unvalidated_baseline"}, initial=True)
        baseline_validation, _ = self.run_set(skill, knowledge, "validation", "initial")
        write_json(self.output/"baseline_validation.json", {
            "successes": sum(r["success"] for r in baseline_validation),
            "episodes": len(baseline_validation), "program_sha256": canonical_hash(skill),
            "rule_validation_performed": False})
        status = "completed"
        for iteration in range(self.config["rounds"]):
            development, paths = self.run_set(skill, knowledge, "development", f"round_{iteration:02d}_baseline")
            if self.config["ablation"] == "fixed":
                self.history.append({"round": iteration, "development": [brief(r) for r in development], "decision": "fixed_baseline"})
                break
            # Reserve conservative prompt/image overhead and full output budget.
            llm = json.loads(Path(llm_config).read_text())
            reserve = llm.get("max_output_tokens", 5000)+10000
            replay_paths = self.config.get("proposal_replay", [])
            replay_path = Path(replay_paths[iteration]) if iteration < len(replay_paths) else None
            if not replay_path and (self.api_calls >= self.config["max_api_calls"] or self.usage_unknown or self.tokens+reserve > self.config["max_tokens"]):
                status = "api_budget_exhausted"
                break
            # Reserve all episodes before a proposal; do not spend API on work
            # that cannot be independently validated under the remaining budget.
            required = len(self.config["development"])+len(self.config["validation"])
            if self.episodes+required > self.config["max_episodes"]:
                status = "episode_budget_exhausted"
                break
            traces = []
            for path in paths:
                observed = [json.loads(line) for line in (path/"observations.jsonl").read_text().splitlines()]
                state = __import__("numpy").load(path/"robot_states.npz")
                ref = __import__("numpy").load(path/"references.npz")
                ids = list(range(0, len(state["timestamp"]), 50))
                traces.append({"events": [json.loads(line) for line in (path/"events.jsonl").read_text().splitlines()
                                          if "event" in json.loads(line)][:50],
                               "observed_key_states": observed[::20],
                               "tracking": [{"timestamp": float(state["timestamp"][i]),
                                              "right_hand_reference": ref["world_positions"][i,0,-1].tolist(),
                                              "right_hand_actual": state["body_positions"][i,-1].tolist(),
                                              "right_hand_error_m": float(state["tracking_error_m"][i,-1])} for i in ids]})
            frames = sorted((paths[0]/"frames").glob("*.jpg"))
            context = {"task": "Move the full 8cm cube footprint into green target, retract hand >=10cm and hold result for 1 second",
                       "episode_deadline_seconds": self.config["episode_seconds"],
                       "current_skill": skill, "current_knowledge": knowledge, "allowed_interface": interface_description(),
                       "development_results": [brief(r) for r in development], "execution_records": traces,
                       "previous_experiments": self.history,
                       "remaining_budget": {"calls": self.config["max_api_calls"]-self.api_calls,
                                            "tokens": self.config["max_tokens"]-self.tokens,
                                            "episodes": self.config["max_episodes"]-self.episodes},
                       "image_paths": [str(frames[0].resolve()), str(frames[len(frames)//2].resolve()), str(frames[-1].resolve())]}
            # UTF-8 bytes bound text token count conservatively; reserve 4096
            # tokens per low-detail image plus fixed system/message overhead.
            # Stop rather than submitting an unbounded prompt as history grows.
            prompt_upper_bound = len(json.dumps(context, ensure_ascii=False).encode())+4096*len(context["image_paths"])+2048
            attempts_reserved = min(llm.get("retries",0)+1, self.config["max_api_calls"]-self.api_calls)
            if not replay_path and self.tokens+(prompt_upper_bound+llm.get("max_output_tokens",5000))*attempts_reserved > self.config["max_tokens"]:
                status = "token_budget_reservation_failed"
                break
            proposal_path = self.output / f"proposal_{iteration:02d}"
            try:
                if replay_path:
                    from .proposer import validate_candidate
                    proposal_path.mkdir()
                    candidate = validate_candidate(json.loads(replay_path.read_text()), skill)
                    write_json(proposal_path/"response.json", candidate)
                    write_json(proposal_path/"replay.json", {"source": str(replay_path.resolve()), "sha256": digest(replay_path),
                                                             "new_api_call": False, "original_context_not_replaced": True})
                    self.replayed_proposals += 1
                else:
                    self.api_calls += 1
                    candidate = propose(context, llm_config, proposal_path, bashrc)
            except (GPTError, ValueError) as error:
                self.history.append({"round": iteration, "decision": "proposal_rejected", "reason": str(error)})
                call = proposal_path/"call.json"
                if call.exists():
                    call_record = json.loads(call.read_text())
                    self.api_calls += max(0,len(call_record["attempts"])-1)
                    usage = call_record.get("usage")
                    if usage:
                        self.tokens += usage["total_tokens"]
                        if call_record.get("usage_status") != "known":
                            self.usage_unknown = True
                    else:
                        self.usage_unknown = True
                elif (proposal_path/"worker_error.json").exists():
                    # No SDK request was recorded for an initialization failure.
                    self.usage_unknown = json.loads((proposal_path/"worker_error.json").read_text()).get("usage") == "unknown"
                elif proposal_path.exists() and not replay_path:
                    self.usage_unknown = True
                write_json(self.output/"progress.json", self.summary("running"))
                continue
            if not replay_path:
                call_record = json.loads((proposal_path/"call.json").read_text())
                usage = call_record["usage"]
                self.api_calls += max(0,len(call_record["attempts"])-1)
                if usage:
                    self.tokens += usage["total_tokens"]
                    if call_record.get("usage_status") != "known":
                        self.usage_unknown = True
                else:
                    self.usage_unknown = True
            candidate_skill = candidate["skill"] if self.config["ablation"] in {"skill_only", "joint"} else deepcopy(skill)
            proposed_knowledge = Knowledge(knowledge["rules"])
            proposed_knowledge.revise_skill(skill["name"], candidate_skill["version"])
            if self.config["ablation"] in {"symbolic_only", "joint"}:
                rule = deepcopy(candidate["rule"])
                rule["skill_version"] = candidate_skill["version"]
                proposed_knowledge.propose(rule)
            new_knowledge = proposed_knowledge.snapshot()
            candidate_development, _ = self.run_set(candidate_skill, new_knowledge, "development", f"round_{iteration:02d}_candidate")
            candidate_validation, validation_paths = self.run_set(candidate_skill, new_knowledge, "validation", f"round_{iteration:02d}_candidate")
            # Empirical effects may be recorded; preconditions are not declared
            # necessary without separate contrastive validation.
            if proposed_knowledge.rules and self.config["ablation"] in {"symbolic_only", "joint"}:
                for report, path in zip(candidate_validation, validation_paths):
                    proposed_knowledge.record(len(proposed_knowledge.rules)-1, str(path.relative_to(self.output)), "validation", report["success"])
                new_knowledge = proposed_knowledge.snapshot()
            decision = compare(candidate_validation, baseline_validation, self.config["minimum_validation_episodes"])
            from humanoid_skill_lab.symbolic.graph import build_graph
            write_json(proposal_path/"knowledge_after_validation.json", new_knowledge)
            write_json(proposal_path/"graph.json", build_graph(new_knowledge["rules"]))
            write_json(proposal_path/"decision.json", decision)
            row = {"round": iteration, "candidate_version": candidate_skill["version"], "hypothesis": candidate["failure_hypothesis"],
                   "candidate_skill": candidate_skill, "candidate_rule": candidate["rule"],
                   "proposed_distinguishing_experiment": candidate["next_experiment"],
                   "development": [brief(r) for r in candidate_development],
                   "validation_summary": {"successes": sum(r["success"] for r in candidate_validation), "episodes": len(candidate_validation)},
                   "decision": decision}
            self.history.append(row)
            if decision["accepted"]:
                self.registry.commit(candidate_skill, new_knowledge, decision)
                skill, knowledge, baseline_validation = candidate_skill, new_knowledge, candidate_validation
            write_json(self.output/"progress.json", self.summary("running"))
        result = self.summary(status)
        write_json(self.output/"summary.json", result)
        return result

    def summary(self, status):
        identifier, pair = self.registry.current()
        request_attempts = 0
        for call in self.output.glob("proposal_*/call.json"):
            request_attempts += len(json.loads(call.read_text())["attempts"])
        return {"status": status, "api_calls": self.api_calls, "tokens": self.tokens,
                "api_request_attempts": request_attempts,
                "replayed_proposals": self.replayed_proposals,
                "usage_status": "unknown" if self.usage_unknown else "known", "episodes": self.episodes,
                "current_snapshot": identifier, "accepted_skill_version": pair["skill"]["version"],
                "rules_status": [r["status"] for r in pair["knowledge"]["rules"]], "history": self.history,
                "test_feedback_used": False}


def frozen_evaluate(registry_path, experiment_config, output, device="cuda", video=True):
    """No proposal path exists here. One immutable test result per destination."""
    registry = Registry(registry_path)
    identifier, pair = registry.current()
    config = deepcopy(experiment_config)
    check_datasets(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"frozen_pair.json", {"snapshot": identifier, **pair})
    reports = []
    for i, item in enumerate(config["test"]):
        cfg = {**item, "split": "test", "knowledge": pair["knowledge"], "observation_mode": config["observation_mode"],
               "seconds": config["episode_seconds"], "mode": 2}
        reports.append(run_episode(pair["skill"], cfg, output/f"test_{i:02d}", device, video))
    result = {"frozen_snapshot": identifier, "successes": sum(r["success"] for r in reports), "episodes": len(reports),
              "reports": reports, "used_for_tuning": False, "task_composition_validated": False}
    write_json(output/"summary.json", result)
    return result
