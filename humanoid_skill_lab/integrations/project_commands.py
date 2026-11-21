"""Bridge to the previous version's maintained scene/navigation entry points."""
import re
import subprocess
import sys
from pathlib import Path
from humanoid_skill_lab.provenance import ROOT, verify_upstream
from .reconstructed_scene import load_scene


def navigate(scene, run_name, device="cuda", video=False, dry_run=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Run name must contain only letters, numbers, underscores or hyphens")
    context = load_scene(scene)
    verify_upstream()
    command = [sys.executable, str(ROOT/"scripts/run_skill_agent.py"), "--task", "navigate",
               "--scene", context["directory"], "--run-name", run_name, "--device", device]
    if video:
        command.append("--video")
    if dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=ROOT, check=True)


def view(scene, collision=False, fast_exit=False):
    context = load_scene(scene)
    command = [sys.executable, str(ROOT/"scripts/view_scene.py"), "--scene", context["xml"]]
    if collision:
        command.append("--collision")
    if fast_exit:
        command.append("--fast-exit")
    subprocess.run(command, cwd=ROOT, check=True)
