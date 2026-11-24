"""Native Isaac Sim NuRec render probe; no mesh-room fallback or compositing."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=Path, default=ROOT/"outputs/figures_v6/scene/room_nurec.usda")
parser.add_argument("--output", type=Path, default=ROOT/"outputs/figures_v6/assets")
parser.add_argument("--foreground", type=Path)
parser.add_argument("--frames", type=int, default=100)
parser.add_argument("--width", type=int, default=1600)
parser.add_argument("--height", type=int, default=1000)
parser.add_argument("--view", choices=["source", "hero"], default="source")
parser.add_argument("--camera", help="Fixed camera name from the foreground pose metadata")
parser.add_argument("--name", default="isaac_gaussian_probe")
parser.add_argument("--proxy", action="store_true", help="Use aligned matte room meshes for native depth/shadow compositing")
parser.add_argument("--proxy-mode", choices=["table", "mesh"], default="table")
args = parser.parse_args()
if not args.scene.is_file():
    raise FileNotFoundError(args.scene)
from isaacsim import SimulationApp
app = SimulationApp(dict(headless=True, width=args.width, height=args.height,
                         renderer="RaytracedLighting", multi_gpu=False))


def main():
    import carb
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
    from PIL import Image
    settings = carb.settings.get_settings()
    for name, value in {
        "/rtx/post/histogram/enabled": False,
        "/rtx/post/registeredCompositing/invertToneMap": True,
        "/rtx/post/registeredCompositing/invertColorCorrection": True,
        "/rtx/post/tonemap/op": 2,
        "/rtx/raytracing/fractionalCutoutOpacity": False,
        "/rtx/matteObject/visibility/secondaryRays": True,
        "/rtx/rtpt/gaussian/skipTonemapping/enabled": False,
    }.items():
        settings.set(name, value)
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    room = stage.DefinePrim("/World/Reconstruction")
    room.GetReferences().AddReference(str(args.scene.resolve()))
    if args.proxy:
        proxy = stage.DefinePrim("/World/RoomDepthProxy")
        targets = []
        if args.proxy_mode == "mesh":
            proxy.GetReferences().AddReference(str(ROOT/"outputs/worldlabs_scene/scene.usda"), "/World/AtlasVisual")
            for prim in Usd.PrimRange(proxy):
                if prim.IsA(UsdGeom.Mesh):
                    prim.CreateAttribute("primvars:isMatteObject", Sdf.ValueTypeNames.Bool).Set(True)
                    UsdShade.MaterialBindingAPI.Apply(prim).UnbindAllBindings()
                    targets.append(prim.GetPath())
        else:
            # Approximate tabletop occlusion, using the shared calibrated plane.
            table = UsdGeom.Cube.Define(stage, "/World/RoomDepthProxy/Table")
            table.CreateSizeAttr(2.)
            table.AddTranslateOp().Set(Gf.Vec3d(3.6432048445, 3.2605281951, .7275))
            table.AddRotateZOp().Set(-1.421004)
            table.AddScaleOp().Set(Gf.Vec3d(.41871473, .837231935, .0125))
            table.GetPrim().CreateAttribute("primvars:isMatteObject", Sdf.ValueTypeNames.Bool).Set(True)
            table.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
            targets.append(table.GetPath())
        stage.GetPrimAtPath("/World/Reconstruction/RoomGaussians").GetRelationship("proxy").SetTargets(targets)
    if args.foreground:
        stage.DefinePrim("/World/Foreground").GetReferences().AddReference(str(args.foreground.resolve()))
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(650.)
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateClippingRangeAttr(Gf.Vec2f(.03, 100.))
    camera.CreateFocalLengthAttr(24.)
    fovy = 70.
    if args.camera:
        if not args.foreground:
            raise ValueError("A named pose camera requires --foreground")
        camera_info = json.loads(args.foreground.with_suffix(".json").read_text())["cameras"][args.camera]
        fovy = camera_info["vertical_fov_degrees"]
    vertical = 2*24*np.tan(np.deg2rad(fovy)/2)
    camera.CreateVerticalApertureAttr(float(vertical))
    camera.CreateHorizontalApertureAttr(float(vertical*args.width/args.height))
    if args.camera:
        eye = np.array(camera_info["position"])
        basis = np.array(camera_info["rotation"])
        target, up = eye-basis[:, 2], basis[:, 1]
    elif args.view == "source":
        from scipy.spatial.transform import Rotation
        source = json.loads((ROOT/"outputs/worldlabs_viewer/cameras.json").read_text())[0]["camera"]["extrinsics"]
        transform = json.loads((ROOT/"outputs/worldlabs_scene/manifest.json").read_text())["worldlabs_native_to_sim"]
        rotation = np.array(transform["rotation"])
        eye = rotation @ np.array(source["position"])*transform["scale"]
        eye += [*transform["translation_xy"], -transform["floor_y"]*transform["scale"]]
        basis = rotation @ Rotation.from_quat(source["quaternion"]).as_matrix()
        target, up = eye-basis[:, 2], basis[:, 1]
    else:
        eye, target, up = np.array([5.0, 2.3, 2.0]), np.array([3.5, 3.9, .7]), np.array([0, 0, 1])
    matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)),
                                   Gf.Vec3d(*map(float, up))).GetInverse()
    camera.AddTransformOp().Set(matrix)
    product = rep.create.render_product(str(camera.GetPath()), (args.width, args.height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    args.output.mkdir(parents=True, exist_ok=True)
    saved_path = args.output.parent/"scene"/(args.name+"_isaac.usda")
    stage.GetRootLayer().Export(str(saved_path))
    saved = Usd.Stage.Open(str(saved_path))
    saved.GetPrimAtPath("/World/Reconstruction").GetReferences().ClearReferences()
    saved.GetPrimAtPath("/World/Reconstruction").GetReferences().AddReference("./"+args.scene.name)
    if args.foreground:
        saved.GetPrimAtPath("/World/Foreground").GetReferences().ClearReferences()
        saved.GetPrimAtPath("/World/Foreground").GetReferences().AddReference("./"+args.foreground.name)
    saved.GetRootLayer().Save()
    for frame in range(args.frames):
        app.update()
        if frame % 20 == 0:
            print(f"GAUSSIAN_RENDER warmup={frame}", flush=True)
    rep.orchestrator.step(rt_subframes=8, pause_timeline=True)
    pixels = np.asarray(rgb.get_data())
    if pixels.size == 0:
        raise RuntimeError("Isaac RGB annotator returned no image")
    Image.fromarray(pixels[:, :, :3]).save(args.output/(args.name+".png"))
    if pixels[:, :, :3].std() < 10:
        raise RuntimeError("Native Gaussian render is blank; do not report it as a successful render")
    report = dict(renderer="Isaac Sim 5.1 RTX / NuRec", room_representation="Native Gaussian volume",
                  room_mesh_appearance=False, room_mesh_depth_shadow_proxy=args.proxy,
                  depth_proxy_type=args.proxy_mode if args.proxy else None,
                  scene=str(args.scene), foreground=str(args.foreground) if args.foreground else None,
                  camera_position=list(map(float, eye)), camera_target=list(map(float, target)),
                  vertical_fov_degrees=fovy,
                  resolution=[args.width, args.height], rgb_mean=float(pixels[:, :, :3].mean()),
                  rgb_std=float(pixels[:, :, :3].std()), authored_poses=bool(args.foreground), policy_executed=False)
    (args.output.parent/(args.name+"_render.json")).write_text(json.dumps(report, indent=2)+"\n")
    print("GAUSSIAN_RENDER "+json.dumps(report), flush=True)


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
else:
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
