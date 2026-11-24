"""Run independent native Isaac render jobs sequentially on one GPU."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/figures_v6"


def main():
    jobs = []
    for name in ["posture", "reach", "push", "open_drawer", "place", "close_drawer"]:
        camera = "drawer_detail" if name in {"open_drawer", "place", "close_drawer"} else "table_detail"
        jobs.append((name, name+"_foreground", camera, 1100, 650))
    jobs += [("onboard_pick", "pick_camera_foreground", "g0_onboard", 900, 600),
             ("onboard_place", "place_camera_foreground", "g3_onboard", 900, 600),
             ("room_probe", None, None, 1600, 1000)]
    records = []
    for name, foreground, camera, width, height in jobs:
        command = [str(ROOT/"scripts/run_isaac.sh"), "scripts/render_gaussian_isaac.py",
                   "--name", name, "--width", str(width), "--height", str(height), "--frames", "65", "--view", "hero", "--proxy"]
        if foreground:
            command += ["--foreground", str(OUT/"scene"/(foreground+".usdc")), "--camera", camera]
        print("NATIVE_RENDER start "+name, flush=True)
        log = OUT/(name+"_isaac.log")
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        records.append(dict(view=name, returncode=result.returncode, log=log.name))
        print("NATIVE_RENDER done "+name+" status="+str(result.returncode), flush=True)
        if result.returncode:
            print(log.read_text()[-4000:], flush=True)
            raise RuntimeError("Isaac render failed: "+name)
    (OUT/"native_render_jobs.json").write_text(json.dumps(records, indent=2)+"\n")


if __name__ == "__main__":
    main()
