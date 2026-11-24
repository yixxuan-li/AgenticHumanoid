"""Validate Gaussian source identity, native stages, render records, and exports."""
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import msgpack
import numpy as np
from PIL import Image
from pxr import Sdf, Usd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/figures_v6"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    provenance = json.loads((OUT/"provenance.json").read_text())
    converted = provenance["conversion"]
    source = ROOT/converted["source"]
    assert sha(source) == converted["source_sha256"] == "578508d8e7236c4e81fa44602041f9968ec9879e752c20da619b3af6073f153d"
    assert sha(OUT/"scene/room_nurec.usda") == converted["output_sha256"]
    assert sha(OUT/"scene/room_nurec.nurec") == converted["gaussian_payload_sha256"]
    payload = msgpack.unpackb(gzip.decompress((OUT/"scene/room_nurec.nurec").read_bytes()), raw=False)
    data = payload["nre_data"]["state_dict"]
    for key, columns in [("positions", 3), ("rotations", 4), ("scales", 3), ("densities", 1), ("features_albedo", 3)]:
        values = np.frombuffer(data[".gaussians_nodes.gaussians."+key], np.float16)
        assert values.size == 2000000*columns
        assert np.isfinite(values).all(), key
    assert data[".gaussians_nodes.gaussians.features_specular"] == b""
    renders = provenance["views"]
    assert len(renders) == 10
    for name, report in renders.items():
        assert report["renderer"] == "Isaac Sim 5.1 RTX / NuRec"
        assert report["room_mesh_appearance"] is False
        assert report["policy_executed"] is False
        assert report["rgb_std"] > 10
        with Image.open(OUT/"assets"/(name+".png")) as image:
            assert list(image.size) == report["resolution"]
        stage = Usd.Stage.Open(str(OUT/"scene"/(name+"_isaac.usda")))
        volume = stage.GetPrimAtPath("/World/Reconstruction/RoomGaussians")
        assert volume.GetAttribute("omni:nurec:isNuRecVolume").Get() is True
        field = stage.GetPrimAtPath("/World/Reconstruction/RoomGaussians/density")
        asset = field.GetAttribute("filePath").Get()
        assert asset.resolvedPath and Path(asset.resolvedPath).is_file()
        if report["foreground"]:
            assert stage.GetPrimAtPath("/World/Foreground/G0") or any(
                str(p.GetPath()).startswith("/World/Foreground/G") for p in stage.Traverse())
    for name, digest in provenance["assets"].items():
        assert sha(OUT/"assets"/name) == digest
    for stem, size in [("teaser_v6", (4200,1500)), ("system_v6", (4200,2310))]:
        assert Image.open(OUT/(stem+".png")).size == size
        assert (OUT/(stem+".pdf")).stat().st_size > 10000
        svg = ET.parse(OUT/(stem+".svg"))
        assert svg.findall(".//{http://www.w3.org/2000/svg}text")
    result = dict(status="passed", gaussian_count=2000000, sh_degree=0, native_render_views=10,
                  original_spz_hash="unchanged", gaussian_payload_arrays="finite and complete",
                  native_usd_references="resolved", raster_hashes="passed", png_pdf_svg="passed",
                  scope="Rendering and artifact checks only; no physics, control, or skill success validation")
    (OUT/"validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
