"""Credential-owning worker. Never imported by the simulation process."""
import base64
import json
import os
import time
import hashlib
from pathlib import Path


class GPTError(RuntimeError):
    pass


class GPTClient:
    def __init__(self, config):
        from openai import OpenAI
        import httpx
        self.key = os.environ.get(config.get("key_env", "OPENAI_API_KEY")) or os.environ.get("GPT_API_KEY")
        self.model = os.environ.get("OPENAI_MODEL") or config.get("model")
        if not self.key or not self.model:
            raise GPTError("API credentials and a model must be explicitly configured")
        self.client = OpenAI(api_key=self.key, base_url=os.environ.get("OPENAI_BASE_URL") or config.get("base_url"),
                             timeout=config.get("timeout_seconds", 90), max_retries=0,
                             http_client=httpx.Client(trust_env=config.get("trust_env", True)))
        self.config = config

    def propose(self, context, frames, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        system = ("You improve bounded Unitree G1 skill programs from actual ScaleBFM/MuJoCo execution. "
                  "Return one JSON object containing failure_hypothesis, skill, rule, next_experiment, expected_observation. "
                  "Use ONLY the provided skill schema and permitted primitive interface. Do not change evaluator, scene, "
                  "controller, or claim success without physics evidence. Rule status must be candidate and evidence must be the empty JSON object {}, never an array []. "
                  "Rule fields: skill, skill_version, preconditions, add_effects, delete_effects, status, evidence. "
                  "Predicates: visible(?object), hand_free(right), object_in_region(?object,?target), drawer_open(?drawer), holding(right,?object). "
                  "All rule preconditions require contrastive verification; unknown is not false. "
                  "Increment skill version by one. next_experiment and expected_observation must be strings, not arrays or objects. "
                  'Example rule: {"skill":"push_to_region","skill_version":2,"preconditions":["visible(?object)"],'
                  '"add_effects":["object_in_region(?object,?target)"],"delete_effects":[],"status":"candidate","evidence":{}}. '
                  "An unfinished primitive cannot be replaced by teleportation or joint commands.")
        content = [{"type": "text", "text": json.dumps(context, allow_nan=False)}]
        for frame in frames[:4]:
            encoded = base64.b64encode(Path(frame).read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"+encoded, "detail": "low"}})
        started = time.monotonic()
        attempts = []
        response = None
        retries = self.config.get("retries", 0)
        if type(retries) is not int or not 0 <= retries <= 2:
            raise GPTError("Retry count must be an integer from zero to two")
        attempt_limit = min(retries+1, context.get("remaining_budget", {}).get("calls", retries+1))
        for attempt in range(attempt_limit):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                    response_format={"type": "json_object"}, max_completion_tokens=self.config.get("max_output_tokens", 5000),
                    reasoning_effort=self.config.get("reasoning_effort", "low"))
                attempts.append({"attempt": attempt+1, "status": "received"})
                break
            except Exception as error:
                attempts.append({"attempt": attempt+1, "status": "api_error", "type": type(error).__name__,
                                 "http_status": getattr(error, "status_code", None)})
        usage = response.usage.model_dump() if response and response.usage else None
        metadata = {"model": self.model, "credentials_configured": bool(self.key), "attempts": attempts,
                    "latency_seconds": time.monotonic()-started, "usage": usage,
                    "usage_status": ("partial_unknown" if len(attempts)>1 else "known") if usage else "unknown",
                    "candidate_version": context["current_skill"]["version"]+1,
                    "prompt_sha256": hashlib.sha256(json.dumps({"system":system,"content":content},sort_keys=True).encode()).hexdigest()}
        (output/"call.json").write_text(json.dumps(metadata, indent=2)+"\n")
        if response is None:
            raise GPTError("API request failed; sanitized call metadata was saved")
        raw = response.choices[0].message.content or ""
        if self.key in raw:
            raise GPTError("Response contained credential material; response discarded")
        (output/"response.json").write_text(raw)
        try:
            candidate = json.loads(raw)
        except json.JSONDecodeError:
            raise GPTError("Model returned invalid JSON") from None
        return candidate


def worker(context_path, config_path, output):
    config = json.loads(Path(config_path).read_text())
    context = json.loads(Path(context_path).read_text())
    frames = context.pop("image_paths", [])
    try:
        GPTClient(config).propose(context, frames, output)
    except GPTError as error:
        # Only our sanitized errors are emitted. SDK error bodies are discarded.
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output)/"worker_error.json").write_text(json.dumps({"type": "GPTError", "message": str(error)})+"\n")
        raise SystemExit(str(error)) from None
    except Exception as error:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output)/"worker_error.json").write_text(json.dumps({"type": type(error).__name__, "status": "worker_error"})+"\n")
        raise SystemExit("LLM worker initialization or response handling failed") from None
