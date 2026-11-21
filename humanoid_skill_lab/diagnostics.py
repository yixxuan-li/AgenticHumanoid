"""Fail-closed dependency, hash, and source provenance checks."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from .provenance import ROOT, UPSTREAM_COMMIT, CHECKPOINT_SHA, METADATA_SHA, digest, verify_upstream


def inspect_environment(tree_path=None):
    import torch
    checks = {}
    for name in ("numpy", "scipy", "torch", "mujoco", "hydra-core", "loguru", "openai", "imageio", "pytest"):
        try:
            checks[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            checks[name] = None
    assets = {}
    for path, expected in (("checkpoints/scalebfm_m.pt", CHECKPOINT_SHA), ("checkpoints/scalebfm_m_metadata.json", METADATA_SHA)):
        p = ROOT / path
        assets[path] = {"exists": p.exists(), "sha256": digest(p) if p.exists() else None, "expected_sha256": expected}
    verification = {"status": "not_checked", "mismatches": [], "verified_files": 0}
    try:
        pinned = {"status": "verified", "files": verify_upstream()}
    except RuntimeError as error:
        pinned = {"status": "failed", "reason": str(error)}
    if tree_path and Path(tree_path).exists():
        tree = json.loads(Path(tree_path).read_text())
        verification["tree_commit_matches"] = tree.get("sha") == UPSTREAM_COMMIT
        known = {x["path"]: x["sha"] for x in tree["tree"] if x["type"] == "blob"}
        for p in (ROOT / "vendor/ScaleBFM").rglob("*"):
            if not p.is_file() or p.suffix not in {".py", ".xml", ".yaml", ".npz"} or "__pycache__" in p.parts:
                continue
            relative = str(p.relative_to(ROOT / "vendor/ScaleBFM"))
            content = p.read_bytes()
            git_sha = hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest()
            if relative not in known or git_sha != known[relative]:
                verification["mismatches"].append(relative)
            else:
                verification["verified_files"] += 1
        verification["status"] = "verified" if verification["tree_commit_matches"] and not verification["mismatches"] else "mismatch"
    return {"packages": checks, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "assets": assets, "upstream_commit": UPSTREAM_COMMIT, "source_verification": verification,
            "pinned_runtime_files": pinned,
            "ready": all(v["sha256"] == v["expected_sha256"] for v in assets.values()) and all(checks.values()) and pinned["status"] == "verified"}
