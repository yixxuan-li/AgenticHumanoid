"""Reuse the maintained Atlas export and its existing provenance contract."""
import json
from pathlib import Path
import numpy as np
from scene_reconstruction.provenance import scene_fingerprint


def load_scene(directory):
    directory = Path(directory).expanduser().resolve()
    for filename in ("manifest.json", "scene.xml", "scene.usda"):
        if not (directory/filename).is_file():
            raise ValueError(f"Reconstructed scene is missing {filename}: {directory}")
    manifest = json.loads((directory/"manifest.json").read_text())
    start = manifest.get("navigation", {}).get("start")
    if not isinstance(start, list) or len(start) != 2 or not np.isfinite(start).all():
        raise ValueError("Scene needs an explicit finite navigation start in meters")
    return {"directory": str(directory), "xml": str(directory/"scene.xml"),
            "identity": scene_fingerprint(directory), "robot_start": [*start, .76],
            "scale_status": manifest.get("scale_status", "unknown"),
            "limitations": manifest.get("limitations", []),
            "manipulation_validated": False,
            "declared_pick_profiles": bool(manifest.get("benchmarks", {}).get("pick", {}).get("controller_profile"))}
