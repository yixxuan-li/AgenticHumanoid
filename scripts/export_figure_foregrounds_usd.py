"""Export URDF-derived authored pose snapshots for native Isaac rendering.

MuJoCo supplies forward kinematics and the original URDF visual triangles;
Isaac Sim renders the resulting USD. No reconstructed room mesh is exported.
These are static figure assets, not an imported PhysX articulation or rollout.
"""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/figures_v6/scene"


def export(name, count, state_name, active=None):
    model = mujoco.MjModel.from_xml_path(str(ROOT/f"outputs/figures_v5/scene/composed_{count}.xml"))
    data = mujoco.MjData(model)
    state = np.load(ROOT/"outputs/figures_v5/scene"/state_name)
    data.qpos[:] = state["qpos"]
    if "geom_rgba" in state:
        model.geom_rgba[:] = state["geom_rgba"]
    mujoco.mj_forward(model, data)
    path = OUT/(name+".usdc")
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, "/Foreground")
    stage.SetDefaultPrim(root.GetPrim())
    materials = {}
    def material(rgba):
        key = tuple(map(float, rgba))
        if key not in materials:
            mat = UsdShade.Material.Define(stage, f"/Foreground/Materials/M{len(materials)}")
            shader = UsdShade.Shader.Define(stage, str(mat.GetPath())+"/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgba[:3])))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.47)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(.12)
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(rgba[3]))
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            materials[key] = mat
        return materials[key]
    exported = []
    for gid in range(model.ngeom):
        body_name = model.body(model.geom_bodyid[gid]).name
        geom_name = model.geom(gid).name or f"visual_{gid}"
        is_robot = body_name.startswith("g") and model.geom_group[gid] == 1
        is_prop = body_name.startswith(("authored_", "cube_")) or geom_name.startswith(("narrative_route_", "target_region"))
        if not (is_robot or is_prop):
            continue
        if active is not None and is_robot and not body_name.startswith(f"g{active}_"):
            continue
        if active is not None and body_name.startswith("cube_") and body_name != f"cube_{active}":
            continue
        rgba = model.geom_rgba[gid].copy()
        # Opaque temporal exposures avoid legacy NuRec/matte transparency artifacts.
        if active is None and count > 1 and is_robot:
            rgba[3] = 1.
        if active is not None and is_robot:
            rgba[3] = 1.
        if rgba[3] == 0:
            continue
        geom_path = "/Foreground/G"+str(gid)
        kind = model.geom_type[gid]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
            faces = model.mesh_face[model.mesh_faceadr[mid]:model.mesh_faceadr[mid]+model.mesh_facenum[mid]]
            geom = UsdGeom.Mesh.Define(stage, geom_path)
            geom.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
            geom.CreateFaceVertexCountsAttr([3]*len(faces))
            geom.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
            geom.CreateSubdivisionSchemeAttr("none")
            geom.CreateDoubleSidedAttr(True)
            local_scale = np.ones(3)
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            geom = UsdGeom.Cube.Define(stage, geom_path)
            geom.CreateSizeAttr(2.)
            local_scale = model.geom_size[gid]
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            geom = UsdGeom.Capsule.Define(stage, geom_path)
            geom.CreateRadiusAttr(float(model.geom_size[gid, 0]))
            geom.CreateHeightAttr(float(2*model.geom_size[gid, 1]))
            local_scale = np.ones(3)
        else:
            raise ValueError(f"Unsupported foreground geometry: {geom_name}, {kind}")
        matrix = np.eye(4)
        matrix[:3, :3] = data.geom_xmat[gid].reshape(3, 3) @ np.diag(local_scale)
        matrix[:3, 3] = data.geom_xpos[gid]
        geom.AddTransformOp().Set(Gf.Matrix4d(*map(float, matrix.T.reshape(-1))))
        UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(material(rgba))
        geom.GetPrim().SetCustomDataByKey("sourceGeometry", geom_name)
        exported.append(geom_name)
    cameras = {}
    for cid in range(model.ncam):
        camera_name = model.camera(cid).name
        if camera_name == "env_overview":
            continue
        cameras[camera_name] = dict(position=data.cam_xpos[cid].tolist(),
                                   rotation=data.cam_xmat[cid].reshape(3,3).tolist(),
                                   vertical_fov_degrees=float(model.cam_fovy[cid]))
    stage.GetRootLayer().Save()
    result = dict(asset=path.name, robot_source="Official Unitree G1 URDF used in figure version 5",
                  pose_source=state_name, exported_geometries=len(exported), room_mesh_exported=False,
                  method="Static visual snapshots from URDF-derived forward kinematics", physics_articulation=False,
                  cameras=cameras, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")
    print(name, len(exported))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    export("hero_foreground", 4, "hero_poses.npz")
    export("pick_camera_foreground", 4, "hero_poses.npz", active=0)
    export("place_camera_foreground", 4, "hero_poses.npz", active=3)
    for name in ["posture", "reach", "push", "open_drawer", "place", "close_drawer"]:
        export(name+"_foreground", 1, name+"_pose.npz")


if __name__ == "__main__":
    main()
