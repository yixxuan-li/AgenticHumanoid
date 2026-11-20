# Portable workflow

Run from the new project root after copying `assets/toolbox/` there. Create an environment and install its `requirements.txt`. Isaac Lab uses its own compatible Python environment, not `usd-core` from the reconstruction environment. Install/sign in to the official World Labs CLI using [the current quickstart](https://docs.atlas.worldlabs.ai/quickstart.md).

Credentials: prefer saved CLI login or `WLT_API_KEY`. The runner also accepts an authorized first-line key file through `--key-file`; its default is the ignored project-root `worldlabs` file. Never echo the key. Raw requests/results/download metadata remain private under the workspace; do not commit them because operation responses can contain signed URLs.

## Capture and poses

```bash
python scripts/inspect_video.py capture.mp4 --output outputs/inspection --fps 2
python scripts/select_worldlabs_frames.py --frames outputs/inspection \
  --workspace outputs/worldlabs --count 12 --sample-fps 2
python scripts/worldlabs_scene.py pose --workspace outputs/worldlabs \
  --resolution 720 1280
python scripts/worldlabs_scene.py status --stage pose --workspace outputs/worldlabs
# Download only after status reports successful completion.
python scripts/worldlabs_scene.py download --stage pose --workspace outputs/worldlabs
python scripts/inspect_worldlabs_scene.py --workspace outputs/worldlabs
```

The resolution arguments are WIDTH HEIGHT; choose a currently supported grid with the capture's aspect. The frame selector chooses the sharpest view in each time interval, not guaranteed geometric coverage. Review the contact sheet, overlap and sharpness before upload. Very featureless/low-parallax videos can remain geometrically uncertain even when Atlas produces plausible images.

The runner saves deterministic submission keys and operation IDs. `status` is a single poll; schedule additional polls as needed. `download` resumes completed stages without submitting a task. Use fresh stage names for intentionally changed requests and fresh workspaces for different captures. An existing stage with a different request is rejected. CLI transport errors have a bounded retry; failed operations are surfaced for inspection.

## Completion, splats and mesh

Read `atlas.md` and current [video documentation](https://docs.atlas.worldlabs.ai/using-video-footage.md). Choose novel cameras from observed free space; submit `atlasGenerate` with returned depth, review small rounds, and record retained/rejected frames. Generation is optional if observed coverage suffices. Build request JSONs from the current documented schema and the paired returned camera/RGB/depth assets. Every final posed frame needs a mask or none does; use white keep masks on unaffected frames when masking one image.

```bash
python scripts/worldlabs_scene.py submit --stage gen_round1 \
  --task atlasGenerate --request outputs/worldlabs/gen_round1_input.json
# Poll, download, inspect and curate before constructing splats_input.json.
python scripts/worldlabs_scene.py submit --stage splats \
  --task posedImages2Splats --request outputs/worldlabs/splats_input.json
# Poll and download stage splats after successful refinement.
python scripts/prepare_worldlabs_mesh.py --workspace outputs/worldlabs
python scripts/worldlabs_scene.py submit --stage mesh \
  --task splats2Mesh --request outputs/worldlabs/mesh_input.json
# Poll and download stage mesh before export.
```

Each command accepts `--workspace`; pass it consistently when not using `outputs/worldlabs`. `prepare_worldlabs_mesh.py` selects the refined asset closest to two million splats and crops to captured RGBD percentiles plus a configurable native-unit margin. Inspect these bounds before submission: generated completion outside that crop may be lost. Default 300,000 faces / 1,000 upright boxes / 4096 image texture is a starting point, not a quality guarantee. Preserve prior extraction results when adjusting settings for an observed defect.

## Calibration and export

Select several well-spread native XYZ floor points and two native anchor endpoints corresponding to a known length. `calibrate_worldlabs_scene.py --config points.json --output outputs/worldlabs/calibration.json` fits the floor with SVD and computes metres per native unit from the anchor. The config schema is:

```text
floor_points: array of at least three non-collinear native XYZ points
anchor.points: two distinct native XYZ endpoints
anchor.meters: positive physical distance
anchor.measured: true for a real measurement, false for an assumed dimension
anchor.description: the measured object/distance and provenance
```

Inspect floor residuals and alignment; exclude furniture points rather than fitting a floor through them. The calibration output has `scale`, `floor_y`, and `floor_rotation`: rotate native Y-up points first, subtract `[0,floor_y,0]` in that frame, then scale and convert to simulator Z-up. The exporter additionally shifts XY into positive room coordinates and records the full transform in the manifest. One anchor cannot establish global metric accuracy. `rescale_scene.py` can update an exported room later; rerun validation and planning afterward.

```bash
python scripts/export_worldlabs_scene.py --workspace outputs/worldlabs \
  --mesh outputs/worldlabs/mesh/mesh.glb \
  --colliders outputs/worldlabs/mesh/colliderBoxes.glb \
  --calibration outputs/worldlabs/calibration.json --output outputs/worldlabs_scene
python scripts/render_worldlabs_scene.py --workspace outputs/worldlabs \
  --scene outputs/worldlabs_scene --video
python scripts/validate_worldlabs_mujoco.py --scene outputs/worldlabs_scene
python scripts/validate_export.py --scene outputs/worldlabs_scene
python scripts/audit_assets.py --scene outputs/worldlabs_scene
```

The exporter expects an image-textured GLB and upright collider-box GLB from the same native frame. It creates fixed room boxes, an inferred rectangular floor, and an added 5 cm / 80 g cube. Review the floor footprint and the 10 cm low-collider consolidation threshold, especially for stairs, raised floors or incomplete room boundaries. It reserves candidate navigation routes before placing the cube. If there is no free corridor or off-route cube placement, export stops for review instead of claiming navigability. Baking/unlit shading preserves source appearance; this is not relightable material recovery.

For simulator runtime checks, launch `scripts/validate_worldlabs_isaac.py --scene outputs/worldlabs_scene/scene.usda --headless` in an Isaac Lab installation. Use its normal AppLauncher environment. The workspace's `scripts/run_isaac.sh` is a local environment wrapper and is not needed by the portable toolbox. Native PhysX validation checks cube support, drop and mass; `validate_export.py` only checks MuJoCo dynamics and USD structure.

For held-out appearance review, reserve actual video frames before reconstruction. `prepare_worldlabs_review_geometry.py --workspace ... --scene ... --output ...` prepares Atlas depth for review-camera PnP. Then use `validate_heldout_views.py --video capture.mp4 --scene ... --geometry ... --camera source_000 --times ...`. Exclude review frames from all model inputs and texture creation. SSIM describes appearance at fitted views; it is not a metric geometry score.

## Native splat viewer

Install compatible renderer packages in a local directory (tested Spark 2.4.0 and Three.js 0.187.0). Keep their licences with the bundle. The viewer builder needs the downloaded `splats/` files and `mesh_selection.json`; an optional `curation.json` has `retained_generated: [{stage, index}, ...]` referencing generated operation records. Without curation metadata it includes only real cameras. Optional `heldout_cameras.json` adds review cameras.

```bash
npm install --prefix outputs/viewer_runtime @sparkjsdev/spark@2.4.0 three@0.187.0
python scripts/prepare_worldlabs_viewer.py --workspace outputs/worldlabs \
  --node-modules outputs/viewer_runtime/node_modules --output outputs/worldlabs_viewer
python3 -m http.server 8870 --bind 127.0.0.1 --directory outputs/worldlabs_viewer
```

Review actual rendered source and novel views at localhost:8870. The viewer uses local dependencies and downloaded assets, with no API credentials. WASD moves, Q/E changes height, dragging orbits, wheel zooms, and the dropdown selects cameras. It provides visual exploration; physics lives in MJCF/USD.

Deliver the selected scene, viewer, provenance, assumptions, and relevant test reports. Keep generated assets and ZIPs ignored; share a local bundle separately when requested. Do not infer manipulation success or real-world transfer from a navigation demonstration.
