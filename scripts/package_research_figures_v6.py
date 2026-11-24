"""Bundle native Gaussian figure assets and check the relocated USD references."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/figures_v6"


def main():
    provenance = json.loads((OUT/"provenance.json").read_text())
    assert json.loads((OUT/"validation.json").read_text())["status"] == "passed"
    files = set()
    for name in ["convert_atlas_spz_nurec.py", "export_figure_foregrounds_usd.py", "render_gaussian_isaac.py",
                 "render_gaussian_figure_views.py", "figure_layout.py", "draw_research_figures_v5.py", "draw_research_figures_v6.py",
                 "validate_research_figures_v6.py", "package_research_figures_v6.py", "run_isaac.sh"]:
        files.add(ROOT/"scripts"/name)
    for name in ["requirements-figures.txt", "requirements-gaussian-figures.txt", "docs/FIGURES_V6.md",
                 "outputs/figures_v5/assets/captured_13s.png", "outputs/worldlabs_scene/manifest.json",
                 "outputs/worldlabs_viewer/cameras.json", provenance["conversion"]["source"],
                 "vendor/3dgrut_native/threedgrut/export/usd/nurec/templates.py",
                 "vendor/3dgrut_native/SOURCE.json", "vendor/3dgrut_native/LICENSE"]:
        files.add(ROOT/name)
    files.update(p for p in (ROOT/"vendor/unitree_g1").rglob("*") if p.is_file())
    for name in ["LICENSE", "load-spz.cc", "splat-types.h", "splat-utils.h"]:
        files.add(ROOT/"vendor/spz-reference"/name)
    for name in provenance["assets"]:
        files.add(OUT/"assets"/name)
    for stem in ["teaser_v6", "system_v6"]:
        files.update(OUT/(stem+"."+ext) for ext in ["png", "pdf", "svg"])
    files.update(OUT/name for name in ["provenance.json", "validation.json", "native_render_jobs.json"])
    files.update(OUT/(name+"_render.json") for name in provenance["views"])
    files.update(OUT/"scene"/(name+"_isaac.usda") for name in provenance["views"])
    files.update((OUT/"scene").glob("*_foreground.usdc"))
    files.update((OUT/"scene").glob("*_foreground.json"))
    files.update(OUT/"scene"/name for name in ["room_nurec.usda", "room_nurec.nurec", "room_nurec.json"])
    manifest = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    readme = (ROOT/"docs/FIGURES_V6.md").read_bytes()
    manifest["README.md"] = hashlib.sha256(readme).hexdigest()
    (OUT/"package_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    target = ROOT/"deliverables/figures_v6.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT))
        archive.writestr("README.md", readme)
        archive.write(OUT/"package_manifest.json", "outputs/figures_v6/package_manifest.json")
    with tempfile.TemporaryDirectory(prefix="agentic-gaussian-figure-") as directory:
        extracted = Path(directory)
        with zipfile.ZipFile(target) as archive:
            assert archive.testzip() is None
            archive.extractall(extracted)
        for name, digest in manifest.items():
            assert hashlib.sha256((extracted/name).read_bytes()).hexdigest() == digest, name
        result = subprocess.run([sys.executable, "scripts/validate_research_figures_v6.py"],
                                cwd=extracted, capture_output=True, text=True, check=True)
        assert json.loads(result.stdout)["status"] == "passed"
    report = dict(archive=str(target.relative_to(ROOT)), bytes=target.stat().st_size,
                  sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  crc="passed", extracted_hashes="passed", relocated_native_usd_validation="passed",
                  contains_isaac_runtime=False, optional_pose_reexport_requires_v5_bundle=True)
    (OUT/"package_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
