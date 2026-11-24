"""Package the native figure sources and verify an extracted copy."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT/"outputs/figures_v5"
    report = json.loads((out/"provenance.json").read_text())
    assert json.loads((out/"validation.json").read_text())["status"] == "passed"
    files = set()
    for name in ["render_research_teaser_v5.py", "figure_layout.py", "draw_research_figures_v5.py",
                 "validate_research_figures_v5.py", "package_research_figures_v5.py"]:
        files.add(ROOT/"scripts"/name)
    for name in ["requirements-figures.txt", "docs/FIGURES_V5.md",
                 "scene_reconstruction/__init__.py", "scene_reconstruction/provenance.py",
                 "checkpoints/scalebfm_m_metadata.json"]:
        files.add(ROOT/name)
    for name in report["scene_fingerprint"]["files"]:
        files.add(ROOT/"outputs/worldlabs_scene"/name)
    files.add(ROOT/"outputs/worldlabs_scene/photo_ownership.json")
    files.update(p for p in (ROOT/"vendor/unitree_g1").rglob("*") if p.is_file())
    for name in list(report["assets"])+["captured_13s.png", "reconstructed_room.png"]:
        files.add(out/"assets"/name)
    for stem in ["teaser_v5", "system_v5"]:
        files.update(out/(stem+"."+extension) for extension in ["png", "pdf", "svg"])
    files.update(out/name for name in ["render_provenance.json", "provenance.json", "validation.json"])
    files.update(out/"scene"/name for name in ["composed_1.xml", "composed_4.xml", "hero_anchors.json", "hero_poses.npz"])
    files.update((out/"scene").glob("*_pose.npz"))
    manifest = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    note = (ROOT/"docs/FIGURES_V5.md").read_bytes()
    manifest["README.md"] = hashlib.sha256(note).hexdigest()
    (out/"package_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    target = ROOT/"deliverables/figures_v5.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT))
        archive.writestr("README.md", note)
        archive.write(out/"package_manifest.json", "outputs/figures_v5/package_manifest.json")
    with tempfile.TemporaryDirectory(prefix="agentic-humanoid-figures-v5-") as directory:
        extracted = Path(directory)
        with zipfile.ZipFile(target) as archive:
            assert archive.testzip() is None
            archive.extractall(extracted)
        for name, digest in manifest.items():
            assert hashlib.sha256((extracted/name).read_bytes()).hexdigest() == digest, name
        # Loads both native MJCF models against the extracted robot/room assets.
        completed = subprocess.run([sys.executable, "scripts/validate_research_figures_v5.py"],
                                   cwd=extracted, capture_output=True, text=True, check=True)
        assert json.loads(completed.stdout)["status"] == "passed"
    result = dict(archive=str(target.relative_to(ROOT)), bytes=target.stat().st_size,
                  sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  archived_files=len(manifest)+1, crc_check="passed",
                  extracted_hashes="passed", extracted_model_and_artifact_validation="passed")
    (out/"package_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
