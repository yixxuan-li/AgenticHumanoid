"""Check native figure provenance, portable models, authored states, and exports.

These checks establish artifact consistency only. They do not evaluate control,
contact, grasping, navigation, or task success.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scene_reconstruction.provenance import scene_fingerprint


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/figures_v5")
    args = parser.parse_args()
    out = args.output.resolve()
    report = json.loads((out/"provenance.json").read_text())
    checks = {}
    actual = scene_fingerprint(ROOT/"outputs/worldlabs_scene")
    assert actual == report["scene_fingerprint"]
    assert actual["sha256"] == "d991e9181b6d9d428171375e2ab7ecdafa795f6f0cabbbd90eeb2581873ee2fa"
    checks["source_reconstruction_unchanged"] = True
    robot = ROOT/"vendor/unitree_g1"
    assert sha(robot/"g1_29dof_with_hand_rev_1_0.urdf") == report["robot_urdf_sha256"]
    for name, digest in report["robot_mesh_sha256"].items():
        assert sha(robot/"meshes"/name) == digest, name
    assert len(report["robot_mesh_sha256"]) == 49
    checks["official_urdf_and_49_mesh_hashes"] = True
    for name, digest in report["assets"].items():
        path = out/"assets"/name
        assert sha(path) == digest, name
        pixels = np.asarray(Image.open(path).convert("RGB"))
        assert pixels.std() > 10, f"Blank render: {name}"
    checks["native_render_hashes_and_nonblank_images"] = True
    model_records = []
    room_model = mujoco.MjModel.from_xml_path(str(ROOT/"outputs/worldlabs_scene/scene.xml"))
    for count in [1, 4]:
        model = mujoco.MjModel.from_xml_path(str(out/f"scene/composed_{count}.xml"))
        # Preserve any stateful props already in the source reconstruction.
        assert model.nq == room_model.nq+count*57+1
        states = [out/"scene/hero_poses.npz"] if count == 4 else sorted((out/"scene").glob("*_pose.npz"))
        assert len(states) == (1 if count == 4 else 6)
        for path in states:
            state = np.load(path)
            assert state["qpos"].shape == (model.nq,)
            assert np.isfinite(state["qpos"]).all()
            for jid in np.flatnonzero(model.jnt_limited):
                value = state["qpos"][model.jnt_qposadr[jid]]
                lower, upper = model.jnt_range[jid]
                assert lower-1e-6 <= value <= upper+1e-6, (path.name, model.joint(jid).name, value)
            data = mujoco.MjData(model)
            data.qpos[:] = state["qpos"]
            mujoco.mj_forward(model, data)
            assert np.isfinite(data.xpos).all()
        model_records.append(dict(robot_instances=count, nq=model.nq, pose_files=len(states)))
    checks["portable_mjcf_compile_and_joint_bounded_authored_states"] = model_records
    errors = report["hero_hand_target_error_m"] + [r["hand_target_error_m"] for r in report["gallery_poses"] if r["hand_target_error_m"] is not None]
    assert max(errors) < .01, errors
    checks["authored_palm_target_max_residual_m"] = max(errors)
    for stem, dimensions in [("teaser_v5", (4200, 1500)), ("system_v5", (4200, 2310))]:
        with Image.open(out/f"{stem}.png") as image:
            assert image.size == dimensions
        for extension in ["pdf", "svg"]:
            assert (out/f"{stem}.{extension}").stat().st_size > 10000
        tree = ET.parse(out/f"{stem}.svg")
        assert tree.findall(".//{http://www.w3.org/2000/svg}text"), "Missing editable text"
        assert tree.findall(".//{http://www.w3.org/2000/svg}image"), "Missing render images"
    checks["png_dimensions_pdf_svg_and_editable_typography"] = True
    assert report["engine"] == "MuJoCo"
    assert report["image_generation_used"] is False
    assert report["executed_policy"] is False
    assert report["physics_rollout"] is False
    assert report["skill_success_claims"] is False
    checks["illustration_and_execution_provenance_separated"] = True
    result = dict(status="passed", scope="Figure artifacts and authored pose geometry only; no controller or task validation",
                  checks=checks)
    (out/"validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
