"""Public, allowlisted provenance only. Never snapshot process environments."""
import hashlib
import json
import subprocess
from pathlib import Path

UPSTREAM_COMMIT = "abd6f17c02fe0baabc14709feb8d9ea4959aa621"
CHECKPOINT_SHA = "88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3"
METADATA_SHA = "3024f18136314fade2676e22d6eafb1a8b2befa388da55b0885a53fcb1368d16"
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "unavailable"


def source_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "humanoid_skill_lab").rglob("*"))
            if p.suffix in {".py", ".json", ".xml"} and "__pycache__" not in p.parts}


def verify_upstream():
    pin = json.loads((ROOT/"humanoid_skill_lab/configs/upstream_files.json").read_text())
    if pin["commit"] != UPSTREAM_COMMIT:
        raise RuntimeError("Upstream source pin disagrees with selected commit")
    for relative, expected in pin["git_blob_sha1"].items():
        path = ROOT/"vendor/ScaleBFM"/relative
        if not path.exists():
            raise RuntimeError(f"Missing pinned upstream file: {relative}; run scripts/bootstrap_sources.py")
        content = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Pinned upstream file differs from reviewed commit: {relative}")
    return len(pin["git_blob_sha1"])
