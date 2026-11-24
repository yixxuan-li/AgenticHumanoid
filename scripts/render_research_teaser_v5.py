"""Render authored G1 poses in the reconstructed office with native MuJoCo.

This is figure authoring, not a dynamics rollout or a skill benchmark. The
official Unitree URDF is imported by MuJoCo, including its articulated hands.
The source reconstruction is read only. Task props and a drawer are authored
additions, explicitly recorded in the provenance alongside each pose.
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
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scene_reconstruction.provenance import scene_fingerprint


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def look_axes(position, target):
    forward = np.asarray(target) - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    return np.r_[right, np.cross(right, forward)]


def box(body, name, pos, size, color, mass=0):
    return body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                         pos=pos, size=size, rgba=color, mass=mass)


def build(out, count):
    source = ROOT / "outputs/worldlabs_scene/scene.xml"
    scene = mujoco.MjSpec.from_file(str(source))
    for asset in scene.meshes:
        if asset.file:
            asset.file = str(source.parent / asset.file)
    for asset in scene.textures:
        if asset.file:
            asset.file = str(source.parent / asset.file)
    scene.meshdir = scene.texturedir = ""
    scene.visual.global_.offwidth = 2200
    scene.visual.global_.offheight = 1500
    scene.visual.headlight.ambient = [.35, .35, .35]
    scene.visual.headlight.diffuse = [.6, .6, .6]
    scene.visual.headlight.specular = [.15, .15, .15]
    scene.worldbody.add_light(pos=[2, 1, 3], dir=[0, 1, -2],
                              diffuse=[.6, .6, .6], castshadow=True)
    urdf = ROOT / "vendor/unitree_g1/g1_29dof_with_hand_rev_1_0.urdf"
    for index in range(count):
        robot = mujoco.MjSpec.from_file(str(urdf))
        robot.meshdir = ""
        for mesh in robot.meshes:
            mesh.file = str(urdf.parent / mesh.file)
        robot.body("pelvis").add_freejoint(name="floating_root")
        # Preserve genuine link geometry; suppress duplicated collision visuals.
        for geom in robot.geoms:
            if geom.group == 3:
                geom.rgba[3] = 0
        for side in ["left", "right"]:
            robot.body(f"{side}_hand_palm_link").add_site(
                name=f"{side}_palm_center", pos=[.06, 0, 0], size=[.001]*3,
                rgba=[0, 0, 0, 0])
        # Illustrative optical mount on the URDF torso, not hardware calibration.
        eye = np.array([.085, 0, .43])
        robot.body("torso_link").add_camera(
            name="onboard", pos=eye, xyaxes=look_axes(eye, eye+[1, 0, -.7]), fovy=75)
        scene.attach(robot, prefix=f"g{index}_", frame=scene.worldbody.add_frame())
    # A compact task drawer is an authored prop in the existing office, not
    # recovered articulation. It occupies open floor beside the back workbench.
    drawer_base = scene.worldbody.add_body(name="authored_drawer_base", pos=[5.45, 4.75, 0],
                                           quat=[np.sqrt(.5), 0, 0, -np.sqrt(.5)])
    light = [.77, .81, .82, 1]
    dark = [.23, .29, .32, 1]
    box(drawer_base, "cabinet_left", [-.23, 0, .37], [.025, .25, .37], dark)
    box(drawer_base, "cabinet_right", [.23, 0, .37], [.025, .25, .37], dark)
    box(drawer_base, "cabinet_back", [0, .23, .37], [.23, .02, .37], dark)
    box(drawer_base, "cabinet_top", [0, 0, .75], [.255, .27, .025], light)
    box(drawer_base, "cabinet_bottom", [0, 0, .48], [.23, .25, .02], dark)
    drawer = drawer_base.add_body(name="authored_drawer", pos=[0, 0, 0])
    drawer.add_joint(name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE,
                     axis=[0, -1, 0], range=[0, .24], limited=True, damping=5)
    box(drawer, "drawer_floor", [0, 0, .52], [.20, .22, .012], light, 1)
    box(drawer, "drawer_front", [0, -.245, .615], [.23, .015, .10], light, .3)
    box(drawer, "drawer_back", [0, .21, .58], [.20, .012, .05], light, .1)
    for side in [-1, 1]:
        box(drawer, f"drawer_side_{side}", [side*.205, 0, .58], [.012, .22, .05], light, .1)
    box(drawer, "drawer_handle", [0, -.28, .65], [.09, .018, .012], [.12, .16, .18, 1], .05)
    for index in range(count):
        body = scene.worldbody.add_body(name=f"cube_{index}", pos=[3.28, 2.8, .78])
        body.add_freejoint(name=f"cube_free_{index}")
        box(body, f"cube_geom_{index}", [0, 0, 0], [.035]*3, [.95, .38, .08, 1], .08)
    if count > 1:
        # Dashed narrative route, authored for the figure, not a planned path.
        points = [[2.96, 2.70], [2.30, 3.70], [2.30, 4.7], [3.65, 4.95], [4.75, 4.90]]
        segment = 0
        for a, b in zip(points[:-1], points[1:]):
            a, b = np.asarray(a), np.asarray(b)
            length = np.linalg.norm(b-a)
            direction = (b-a)/length
            for t in np.arange(0, length, .17):
                p, q = a+direction*t, a+direction*min(t+.085, length)
                scene.worldbody.add_geom(name=f"narrative_route_{segment}",
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[.012, 0, 0],
                    fromto=[*p, .018, *q, .018], rgba=[.98, .40, .06, 1],
                    contype=0, conaffinity=0, group=2)
                segment += 1
    # Authored task region on the existing reconstructed central table.
    box(scene.worldbody, "target_region", [3.61, 2.82, .745], [.12, .12, .002], [.12, .68, .58, .65])
    scene.worldbody.add_camera(name="hero", pos=[5.7, 2.0, 2.3],
        xyaxes=look_axes(np.array([5.7, 2.0, 2.3]), [3.5, 3.9, .70]), fovy=70)
    scene.worldbody.add_camera(name="table_detail", pos=[1.60, 1.85, 1.65],
        xyaxes=look_axes(np.array([1.60, 1.85, 1.65]), [3.12, 2.8, .70]), fovy=53)
    scene.worldbody.add_camera(name="drawer_detail", pos=[3.9, 3.65, 1.65],
        xyaxes=look_axes(np.array([3.9, 3.65, 1.65]), [5.04, 4.79, .70]), fovy=53)
    model = scene.compile()
    # MuJoCo imports URDF visual shapes into group 1 and collisions into 0.
    # Render only visuals to avoid identical surfaces and collision simplifiers.
    model.vis.quality.shadowsize = 4096
    xml = ET.fromstring(scene.to_xml())
    for asset in xml.findall("./asset/*"):
        if asset.get("file") and Path(asset.get("file")).is_absolute():
            asset.set("file", os.path.relpath(asset.get("file"), out / "scene"))
    ET.indent(xml)
    ET.ElementTree(xml).write(out / "scene" / f"composed_{count}.xml", encoding="unicode")
    return model


def setjoint(model, data, name, value):
    joint = model.joint(name)
    data.qpos[model.jnt_qposadr[joint.id]] = value


def pose(model, data, index, xy, yaw=0., crouch=0):
    prefix = f"g{index}_"
    meta = json.loads((ROOT / "checkpoints/scalebfm_m_metadata.json").read_text())
    for name, value in zip(meta["joint_names"], meta["default_dof_pos"]):
        setjoint(model, data, prefix+name, value)
    for side in ["left", "right"]:
        for joint, value in [("hip_pitch", -.12-crouch*.6), ("knee", .28+crouch*1.2),
                             ("ankle_pitch", -.16-crouch*.6), ("shoulder_pitch", .18),
                             ("elbow", .45)]:
            setjoint(model, data, prefix+f"{side}_{joint}_joint", value)
    root = model.jnt_qposadr[model.joint(prefix+"floating_root").id]
    data.qpos[root:root+7] = [*xy, .8, np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
    mujoco.mj_forward(model, data)
    zmins = []
    for gid in range(model.ngeom):
        name = model.body(model.geom_bodyid[gid]).name
        if name.startswith(prefix) and "ankle_roll" in name and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
            zmins.append((vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])[:, 2].min())
    data.qpos[root+2] -= min(zmins)
    mujoco.mj_forward(model, data)


def hand_target(model, data, index, target, side="right"):
    """Joint-bounded kinematic authoring; no controller execution is claimed."""
    names = [f"g{index}_{side}_{part}_joint" for part in
             ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]]
    joints = [model.joint(name).id for name in names]
    addresses = model.jnt_qposadr[joints]
    initial = data.qpos[addresses].copy()
    limits = model.jnt_range[joints]
    site = model.site(f"g{index}_{side}_palm_center").id
    def objective(values):
        data.qpos[addresses] = values
        mujoco.mj_forward(model, data)
        return np.r_[(data.site_xpos[site]-target)*8, (values-initial)*.035]
    solution = least_squares(objective, np.clip(initial, limits[:, 0]+1e-5, limits[:, 1]-1e-5),
                             bounds=(limits[:, 0], limits[:, 1]), max_nfev=150)
    data.qpos[addresses] = solution.x
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(data.site_xpos[site]-target))


def cube(model, data, index, position):
    address = model.jnt_qposadr[model.joint(f"cube_free_{index}").id]
    data.qpos[address:address+7] = [*position, 1, 0, 0, 0]


def camera(lookat, distance, azimuth, elevation):
    result = mujoco.MjvCamera()
    result.lookat = lookat
    result.distance = distance
    result.azimuth = azimuth
    result.elevation = elevation
    return result


def render(model, data, path, view, width=1800, height=1100):
    mujoco.mj_forward(model, data)
    options = mujoco.MjvOption()
    options.geomgroup[0] = 0  # URDF collision duplicates
    options.geomgroup[1] = 1  # genuine URDF visual meshes
    options.geomgroup[3] = 0  # reconstructed invisible physics proxies
    # Room and authored primitives live in group 0, so restore those at the
    # model level to visible group 2; only robot collision geometry stays hidden.
    for gid in range(model.ngeom):
        if not model.body(model.geom_bodyid[gid]).name.startswith("g"):
            model.geom_group[gid] = 2
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=view, scene_option=options)
        Image.fromarray(renderer.render()).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/figures_v5")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    for folder in ["assets", "scene"]:
        (out/folder).mkdir(parents=True, exist_ok=True)
    before = scene_fingerprint(ROOT/"outputs/worldlabs_scene")
    room_model = mujoco.MjModel.from_xml_path(str(ROOT/"outputs/worldlabs_scene/scene.xml"))
    room_data = mujoco.MjData(room_model)
    render(room_model, room_data, out/"assets/room_probe.png", "overview", 1200, 800)
    model = build(out, 4)
    data = mujoco.MjData(model)
    phases = [([2.96, 2.70], 0, [3.28, 2.56, .81]),
              ([2.30, 3.70], np.pi/3, [2.56, 3.89, .91]),
              ([3.65, 4.95], 0, [3.96, 4.80, .91]),
              ([4.75, 4.9], 0, [5.04, 4.75, .69])]
    errors = []
    for index, (xy, yaw, target) in enumerate(phases):
        pose(model, data, index, xy, yaw, .15 if index == 3 else 0)
        errors.append(hand_target(model, data, index, target))
        cube(model, data, index, np.array(target)+[0, 0, .025])
        for gid in range(model.ngeom):
            if model.body(model.geom_bodyid[gid]).name.startswith(f"g{index}_") and model.geom_group[gid] == 1:
                model.geom_rgba[gid, 3] = [.28, .44, .67, 1][index]
    setjoint(model, data, "drawer_slide", .23)
    render(model, data, out/"assets/hero_sequence.png", "hero", 2200, 1350)
    # Exact projection anchors let the layout put labels beside each exposure.
    cid = model.camera("hero").id
    f = 675/np.tan(np.deg2rad(model.cam_fovy[cid])/2)
    anchors = []
    for xy, _, _ in phases:
        local = (np.r_[xy, .05]-data.cam_xpos[cid]) @ data.cam_xmat[cid].reshape(3, 3)
        anchors.append([float(1100+f*local[0]/-local[2]), float(675-f*local[1]/-local[2])])
    (out/"scene/hero_anchors.json").write_text(json.dumps(anchors, indent=2)+"\n")
    for i, view in enumerate([camera([3, 3.5, .7], 6.5, -115, -35), camera([3, 3.5, .7], 5.5, -70, -30)]):
        if args.probe:
            render(model, data, out/f"assets/probe_{i}.png", view, 1400, 900)
    np.savez_compressed(out/"scene/hero_poses.npz", qpos=data.qpos, geom_rgba=model.geom_rgba, geom_group=model.geom_group)
    saved_rgba = model.geom_rgba.copy()
    # An onboard frame shows its own exposure, not the temporal ghost clones.
    for active, filename in [(0, "onboard_pick"), (3, "onboard_place")]:
        for gid in range(model.ngeom):
            name = model.body(model.geom_bodyid[gid]).name
            if name.startswith("g"):
                model.geom_rgba[gid, 3] = float(name.startswith(f"g{active}_"))
        render(model, data, out/f"assets/{filename}.png", f"g{active}_onboard", 900, 600)
    model.geom_rgba[:] = saved_rgba
    # Distinct single-robot authored poses, all in the same reconstructed room.
    model = build(out, 1)
    data = mujoco.MjData(model)
    tiles = []
    configs = [
        ("posture", [2.96, 2.78], 0, .55, None, 0, [3.40, 2.8, .79]),
        ("reach", [2.96, 2.78], 0, 0, [3.28, 2.65, .85], 0, [3.34, 2.65, .79]),
        ("push", [2.96, 2.78], 0, .1, [3.31, 2.65, .79], 0, [3.40, 2.65, .78]),
        ("open_drawer", [4.66, 4.90], 0, .40, [4.94, 4.75, .65], .23, [5.15, 4.75, .57]),
        ("place", [4.75, 4.90], 0, .25, [5.04, 4.75, .70], .23, [5.08, 4.75, .68]),
        ("close_drawer", [4.85, 4.90], 0, .40, [5.12, 4.75, .65], .05, [5.35, 4.75, .57]),
    ]
    records = []
    for name, xy, yaw, crouch, target, slide, object_pos in configs:
        pose(model, data, 0, xy, yaw, crouch)
        error = hand_target(model, data, 0, target) if target is not None else None
        setjoint(model, data, "drawer_slide", slide)
        cube(model, data, 0, object_pos)
        if "drawer" in name or name == "place":
            view = "drawer_detail"
        else:
            view = "table_detail"
        path = out/f"assets/{name}.png"
        render(model, data, path, view, 1100, 650)
        tiles.append(Image.open(path).copy())
        np.savez_compressed(out/f"scene/{name}_pose.npz", qpos=data.qpos)
        records.append(dict(name=name, target_m=target, hand_target_error_m=error,
                            drawer_opening_m=slide, authored=True))
    gallery = Image.new("RGB", (2200, 1950), "white")
    for i, tile in enumerate(tiles):
        row, col = divmod(i, 2)
        gallery.paste(tile, (1100*col, 650*row))
    gallery.save(out/"assets/skill_gallery.png")
    after = scene_fingerprint(ROOT/"outputs/worldlabs_scene")
    assert before == after, "Source reconstruction changed"
    urdf = ROOT/"vendor/unitree_g1/g1_29dof_with_hand_rev_1_0.urdf"
    report = dict(engine="MuJoCo", engine_version=mujoco.__version__,
        robot_source=json.loads((urdf.parent/"SOURCE.json").read_text()),
        robot_urdf_sha256=sha(urdf), robot_import="MjSpec.from_file(official URDF)",
        robot_mesh_sha256={p.name:sha(p) for p in sorted((urdf.parent/"meshes").glob("*.STL"))},
        robot_dofs="29 body + 14 finger + floating root", scene_source="outputs/worldlabs_scene",
        scene_fingerprint=before, scene_unchanged=True,
        pose_method="Authored joint-bounded inverse kinematics, direct qpos for illustration only",
        executed_policy=False, physics_rollout=False, skill_success_claims=False,
        image_generation_used=False, camera_calibration="Illustrative torso-fixed RGB mount, not calibrated hardware",
        added_props=["Orange cube", "Table target region", "Authored articulated drawer beside whiteboard wall"],
        drawer_recovered_from_capture=False, narrative_route="Authored illustration, not collision-validated planning",
        hero_hand_target_error_m=errors,
        gallery_poses=records,
        assets={name+".png":sha(out/"assets"/(name+".png")) for name in
                ["hero_sequence", "onboard_pick", "onboard_place", "room_probe", "skill_gallery"]
                +[row[0] for row in configs]})
    (out/"render_provenance.json").write_text(json.dumps(report, indent=2)+"\n")
    print(out/"assets/hero_sequence.png")


if __name__ == "__main__":
    main()
