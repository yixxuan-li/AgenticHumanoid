"""Convert the existing degree-zero Atlas SPZ to native Isaac Sim NuRec USDZ.

The SPZ v2 quantization follows Niantic's published format. The original file
is read only. NVIDIA's pinned 3DGRUT template functions build the payload;
the adapter authors the matching NuRec USD fields.
This is a format conversion, without reconstruction or Gaussian retraining.
"""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any, Dict

import numpy as np
import msgpack
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def decode_v2(path):
    raw = gzip.decompress(path.read_bytes())
    magic, version, count, degree, fractional, flags, _ = struct.unpack_from("<IIIBBBB", raw)
    if magic != 0x5053474e or version != 2 or degree != 0:
        raise ValueError("This converter requires SPZ v2 with SH degree zero; never discard higher SH bands")
    if len(raw) != 16+count*19:
        raise ValueError("Unexpected SPZ payload length")
    offset = 16
    def read(columns):
        nonlocal offset
        result = np.frombuffer(raw, np.uint8, count*columns, offset).reshape(count, columns)
        offset += count*columns
        return result
    packed = read(9).reshape(count, 3, 3).astype(np.int32)
    fixed = packed[:, :, 0] | (packed[:, :, 1] << 8) | (packed[:, :, 2] << 16)
    fixed = (fixed ^ 0x800000)-0x800000
    positions = fixed.astype(np.float32)/2**fractional
    alpha = read(1).astype(np.float32)/255.
    dc = (read(3).astype(np.float32)/255.-.5)/.15
    scales = read(3).astype(np.float32)/16.-10.
    xyz = read(3).astype(np.float32)/127.5-1.
    quat = np.c_[xyz, np.sqrt(np.maximum(0., 1.-(xyz*xyz).sum(axis=1)))]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return positions, alpha, dc, scales, quat, dict(version=version, count=count, sh_degree=degree,
                                                   fractional_bits=fractional, flags=flags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT/"outputs/worldlabs/splats/refinedSplats_2000000.spz")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/figures_v6/scene/room_nurec.usda")
    args = parser.parse_args()
    positions, alpha, dc, logscale, xyzw, header = decode_v2(args.input)
    manifest = json.loads((ROOT/"outputs/worldlabs_scene/manifest.json").read_text())
    transform = manifest["worldlabs_native_to_sim"]
    rotation = np.array(transform["rotation"])
    scale = transform["scale"]
    positions = positions @ rotation.T*scale
    positions += [*transform["translation_xy"], -transform["floor_y"]*scale]
    xyzw = (Rotation.from_matrix(rotation)*Rotation.from_quat(xyzw)).as_quat().astype(np.float32)
    logscale += np.log(scale)
    # Finite logits avoid infinities at quantized opacity endpoints.
    finite_alpha = np.clip(alpha, 1e-6, 1-1e-6)
    density = np.log(finite_alpha/(1-finite_alpha))
    vendor = ROOT/"vendor/3dgrut_native"
    # Load NVIDIA's pure template functions verbatim, without importing training
    # infrastructure or camera packages that are irrelevant to format conversion.
    template_path = vendor/"threedgrut/export/usd/nurec/templates.py"
    tree = ast.parse(template_path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_fill_state_dict_tensors", "fill_3dgut_template"}]
    assert len(functions) == 2
    namespace = dict(np=np, Any=Any, Dict=Dict)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(template_path), "exec"), namespace)
    template = namespace["fill_3dgut_template"](
        positions=positions.astype(np.float32), rotations=xyzw[:, [3, 0, 1, 2]],
        scales=logscale, densities=density, features_albedo=dc,
        features_specular=np.empty((len(positions), 0), np.float32), n_active_features=0,
        radiance_sph_degree=0, global_z_order=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = args.output.with_suffix(".nurec")
    payload.write_bytes(gzip.compress(msgpack.packb(template), compresslevel=1))
    # Schema and field attributes follow NVIDIA's nurec/serializer.py. Positions
    # already use the shared world frame, so no extra axis flip is applied.
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdVol
    stage = Usd.Stage.CreateNew(str(args.output.resolve()))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    volume = UsdVol.Volume.Define(stage, "/World/RoomGaussians")
    prim = volume.GetPrim()
    prim.CreateAttribute("omni:nurec:isNuRecVolume", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("omni:nurec:useProxyTransform", Sdf.ValueTypeNames.Bool).Set(False)
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    volume.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    for key, value in [("offset", [0., 0., 0.]), ("crop:minBounds", lo), ("crop:maxBounds", hi)]:
        prim.CreateAttribute("omni:nurec:"+key, Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*map(float, value)))
    for field, kind in [("density", "float"), ("emissiveColor", "float3")]:
        field_path = str(prim.GetPath())+"/"+field
        field_prim = stage.DefinePrim(field_path, "OmniNuRecFieldAsset")
        volume.CreateFieldRelationship(field, field_path)
        field_prim.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set("./"+payload.name)
        for key, value in [("fieldName", field), ("fieldDataType", kind), ("fieldRole", field)]:
            field_prim.CreateAttribute(key, Sdf.ValueTypeNames.Token).Set(value)
        if field == "emissiveColor":
            for channel, value in zip("RGB", [[1,0,0,0], [0,1,0,0], [0,0,1,0]]):
                field_prim.CreateAttribute("omni:nurec:ccm"+channel, Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*value))
    prim.CreateRelationship("proxy")
    stage.GetRootLayer().Save()
    report = dict(source=str(args.input.relative_to(ROOT)), source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  source_header=header, renderer_format="NuRec Volume USD with Gaussian payload", original_splats_unchanged=True,
                  native_to_sim=transform, spherical_harmonics_preserved=True, opacity_endpoint_clamp=1e-6,
                  exporter_repository="https://github.com/nv-tlabs/3dgrut",
                  exporter_commit=(json.loads((vendor/"SOURCE.json").read_text())["commit"] if (vendor/"SOURCE.json").exists()
                                   else subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vendor, text=True).strip()),
                  output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  gaussian_payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(), precision="float16 NuRec tensors")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
