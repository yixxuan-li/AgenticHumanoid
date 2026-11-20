# Video-to-simulator workflow

The self-contained toolbox is at `../assets/toolbox/`. Copy it into a working directory; run commands there. Its generic components extract frames, run pinned VGGT camera/depth inference, align from configured source regions, export one manifest to MJCF/USD, inspect a scene, and rescale it consistently. It contains no office layout, private source frames, robot weights, or assumed user measurements.

Use Python 3.12 with NumPy, SciPy, Pillow, OpenCV, torch, MuJoCo, usd-core, imageio, imageio-ffmpeg, and matplotlib. VGGT requires a compatible CUDA GPU. Isaac Lab needs its separately installed supported Python/runtime. Do not copy this project's local `.venv` paths to another machine.

## Geometry

```bash
python scripts/inspect_video.py /path/to/video.mp4
python scripts/bootstrap_sources.py --reconstruction-only
mkdir -p checkpoints
python scripts/download_ranges.py \
  https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt \
  checkpoints/vggt_model_ranges.pt --workers 8 \
  --sha256 d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0
python scripts/run_vggt.py --stride 1 --max-frames 34 --output outputs/geometry
python scripts/process_geometry.py --input outputs/geometry/predictions.npz \
  --alignment-config alignment.json --report outputs/geometry/audit.json
```

Choose frame sampling to retain camera overlap and useful translation. The commands' frame limit is an example; increase it for appropriate hardware/coverage. Inspect extracted upright pixels and actual image dimensions. The official pad loader uses white padding; the saved source footprint mask excludes it. Preserve genuine black surfaces and windows.

The model loader checks the official checkpoint digest and loads camera/depth weights strictly. Files are large; use resumable transfers. Sources are pinned and licenses remain applicable. Do not silently substitute synthetic depth on download/inference failure.

`alignment.json` uses original upright image pixel coordinates. Sample actual floor and the same horizontal reference surface in several frames, excluding occluders. Example structure (coordinates must be chosen for the new video):

```json
{
  "floor_samples": [
    {"frame": "frame_0001.jpg", "bbox": [10, 450, 150, 550]}
  ],
  "anchor": {
    "description": "height of the identified table above the fitted floor",
    "height_m": 0.74,
    "measured": false,
    "samples": [
      {"frame": "frame_0002.jpg", "bbox": [40, 330, 240, 380]}
    ]
  },
  "forward_frame": "frame_0002.jpg"
}
```

The example height is an assumption. Set `measured: true` only for an actual supplied measurement of the sampled physical surface. This establishes one scale anchor, not ground truth for every dimension. Generic alignment places the first camera's floor projection at XY origin and the chosen camera's horizontal forward direction along +Y. Rebase the semantic manifest consistently if desired.

The bundled processor requires an explicit alignment configuration for every new video. `--sfm` may compare independent COLMAP camera orientations for the exact same source video. A low reprojection error or good agreement between learned depths is not independent metric/depth truth. Inspect plane inliers/residuals, cross-view depth consistency, pose agreement if available, source views, and translated novel views. Retain holes and uncertainty in glass/occluded areas.

## Optional surface fusion

Compare fused and original surfaces when per-view overlaps flicker. With Open3D installed:

```bash
python scripts/fuse_geometry.py --geometry outputs/geometry --output outputs/fusion
python scripts/texture_fused_geometry.py --geometry outputs/geometry --fusion outputs/fusion
```

This generates a standalone `photo_visual.xml` / `photo_visual.usda` and per-view textured meshes. `--bounds XMIN YMIN ZMIN XMAX YMAX ZMAX` optionally crops in the aligned geometry frame; choose bounds for the actual video. Use a separate fusion directory. `--overwrite` replaces generated fusion mesh output. Fusion can erase thin chair panels or worsen a local view. Retain original per-object skins when they compare better, and inspect independent video frames as review views. Neither fusion nor low image error certifies physical scale.

## Interactive bodies

Author a manifest from the source-grounded inventory using [manifest.md](manifest.md). The supplied exporter accepts box/cylinder collision proxies, textured/rounded visual surfaces, free bodies, and scene-anchored Z hinges. Extend it for mechanisms the source actually supports.

Photo mesh outputs are visual hypotheses with collision disabled. To make a movable body, assign source surface triangles to that object, transform vertices by the inverse initial body pose, attach them below that body's simulator transform, and remove them from the static surface set. Class colors can help association but require an object-moving review; loose distance gates can attach tabletop/floor pixels to small props. Keep primitive fallbacks for unseen faces explicitly estimated. Use `inertia="shell"` for open/planar, mass-zero MuJoCo photo meshes; otherwise valid panels can be rejected for zero enclosed volume. Planar atlas textures can remove coplanar multi-view flicker; depth-test source pixels and disclose inpainted occlusions.

Export in a small Python driver:

```python
from pathlib import Path
import json
from scene_reconstruction.export import export_all
manifest = json.loads(Path('manifest.json').read_text())
export_all(manifest, Path('outputs/scene'))
```

Texture names used by parts need corresponding `textures/NAME.png` files in the output. The exporter writes explicit per-face UVs, separate hidden box collisions, meter/kg/Z-up USD, and named free/hinge joints. USD hinges connect to collider-free kinematic anchors whose pose matches the initial body pose. This makes the closed angle zero and preserves the hinge when the referenced room is translated or rotated. Anchor mass/inertia are bookkeeping values, not physical room objects. Avoid intersecting furniture caused by guessed hidden supports. Record any correction within the stated fitting uncertainty.

## Validation and use

```bash
python scripts/validate_export.py --scene outputs/scene
python scripts/audit_assets.py --scene outputs/scene --report outputs/scene/dependencies.json
python scripts/view_scene.py --scene outputs/scene/scene.xml
```

With an installed Isaac Lab interpreter, run `scripts/validate_isaac.py --scene outputs/scene/scene.usda --headless --fast-exit` to check all dynamic bodies, masses, and initial hinge frames in actual PhysX. Run `scripts/validate_scene_placement.py --scene outputs/scene/scene.usda --headless --fast-exit` to verify three translated/rotated room copies and their hinge frames. The generic validation checks real settling/contacts and USD structure. Add meaningful support/drop, friction, hinge travel, navigation, and task tests for the affordances actually present. Open USD in Isaac Lab, deactivate duplicate composed PhysicsScenes, and inspect PhysX rigid-body state, not stale USD transforms. Record the runtime version. An unavailable Isaac runtime means validation is pending, not passed.

Native viewer: P pause, R reset, O/C cabinet hinge if present, 1–4 camera presets, brackets source-camera cycling, V cutaway for recognized review bodies, F collision view, and standard MuJoCo mouse perturbations. A desktop display is required. If the process crashes during interpreter graphics teardown after the window closes, use `--fast-exit` to flush output and exit directly.

Uniform calibration of an already exported scene:

```bash
python scripts/rescale_scene.py --scene outputs/scene --output outputs/calibrated \
  --calibration anchor.json
```

`anchor.json` contains `description`, `reconstructed_length_m`, and `measured_length_m` for the same physical segment. `--scale-factor 1.05` instead creates an explicitly hypothetical sensitivity variant. The output must be a new directory. Robot body radius, stance heights, hand offsets, and gains remain in the robot's published metric units; rescaling room geometry does not resize the robot. Meshes, body-local parts, cameras, support geometry, and planning coordinates scale together; assumed masses follow volume. Re-run simulator and robot tests. Keep previous reports bound to their old scene fingerprints.

For WBC integration, obtain pinned official sources/checkpoints and motor configuration. Use named joint/actuator indexing so furniture and added fingers cannot shift policy observations. Match frame conventions, quaternion order, future targets, history, gains, armatures, limits, and rates. Start with standing/reference tracking and clear-floor walking, then collision-aware routes, then separate contact grasp and full navigation-to-grasp tests. Report failures and state access honestly; map-based planning is not vision-language exploration.
