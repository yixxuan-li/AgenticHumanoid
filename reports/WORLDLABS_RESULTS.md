# World Labs Atlas reconstruction

The native Atlas splat and simulator exports are ready for review. **Scale is provisional, and original furniture is static.**

- [Open local Spark viewer](http://127.0.0.1:8870/) or serve `outputs/worldlabs_viewer/` locally.
- [Native two-million-splat SPZ](../outputs/worldlabs_viewer/room.spz).
- [MuJoCo MJCF](../outputs/worldlabs_scene/scene.xml), [USD](../outputs/worldlabs_scene/scene.usda), [manifest](../outputs/worldlabs_scene/manifest.json).
- [Selected textured GLB](../outputs/worldlabs/constrained_mesh.glb).
- [Two-engine navigation video](../outputs/worldlabs_scene/navigation_both_engines.mp4).
- [Actual held-out video / native SPZ / final mesh comparison](../outputs/worldlabs_scene/appearance_comparison.jpg).
- [Mesh source comparisons](../outputs/worldlabs_scene/source_comparison.jpg), [novel mesh views](../outputs/worldlabs_scene/novel_views.jpg), [native RTX view](../outputs/worldlabs_scene/isaaclab_source_006.png).
- [Portable local bundle](../deliverables/worldlabs_scene_bundle.zip), [checksums and relocation check](worldlabs_delivery_integrity.json).

## Construction

Used the official [Atlas quickstart](https://docs.atlas.worldlabs.ai/quickstart.md), [atlas-splats skill](https://github.com/worldlabsai/skills/tree/main/skills/atlas-splats), and [worldlabs-cli skill](https://github.com/worldlabsai/skills/tree/main/skills/worldlabs-cli). CLI 0.6.0 and those skills are installed. The API key was read from the authorized local file into a child-process environment; it is ignored by Git and absent from viewer assets and delivery archives.

1. Selected 12 sharp, overlapping real frames across the complete 17-second recording. Uploaded them as private project assets.
2. Ran `images2PosedRGBD` jointly at 720 × 1280, with accepted pose refinement. Returned RGB, confidence, depth and cameras remain paired.
3. Generated two rounds of three novel RGBD views from camera centers supported as free space by the observed depth. Kept three generated views, rejected inconsistent chair layouts, and masked an invented computer cart. All 15 final frames carry aligned masks, with full-white masks for unaffected images.
4. Ran one successful refined `posedImages2Splats` reconstruction, returning 2,000,000 splats. The operation is `9bd2eabd410244e180e221fb5094f296`. Source and novel views were rendered in Spark on the RTX 5090.
5. Extracted a textured mesh and upright collider boxes with `splats2Mesh`. A second stereo/trimmed extraction was reviewed and retained separately: it opened additional holes in walls and glass. The selected simulator scene uses the more complete base extraction plus documented visual constraints.
6. Fitted the table outline to decoded refined-splat points, rectified curated input RGB onto its assumed 0.74 m support plane, and removed unsupported gray bridge surfaces beneath it. The tabletop atlas has 76.4% projected coverage and 23.6% inpainted occlusion coverage. Low floor vertices were constrained to the fitted plane. These are authored priors, not recovered CAD.
7. Converted the Y-up Atlas world to Z-up, applying the floor rotation, provisional scale and translation to geometry, cameras and collider boxes together. The room has 890 fixed fitted boxes, an inferred continuous floor, and one added free benchmark cube. Low fitted floor patches are consolidated into the flat support slab. Photo appearance is baked/unlit.

The apparent envelope is approximately 6.12 × 6.87 × 3.11 **provisional metres**. The table-height anchor is an assumption. Floor ROI fitting retained 68.4% of sampled points within a 0.025-native-unit fitting tolerance. Neither Atlas nor this residual proves physical measurement accuracy.

## Executed checks

| Check | Result | Evidence |
|---|---|---|
| MuJoCo support, drop, friction, mass and penetrations | 10/10 pass | [Report](worldlabs_mujoco_validation.json) |
| Native PhysX support, drop, mass and finite states | 5/5 pass | [Report](worldlabs_isaac_validation.json) |
| Generic exporter and dependency checks | Pass | [Exporter](worldlabs_export_validation.json), [dependencies](worldlabs_asset_dependencies.json) |
| MuJoCo ScaleBFM exploration | Pass; 3.8 cm final goal error | [Report](worldlabs_final_explore_scalebfm_explore_mode0.json) |
| PhysX ScaleBFM exploration | Pass; 7.1 cm final goal error | [Report](worldlabs_native_explore_scalebfm_isaac_explore.json) |
| Local native-SPZ viewer | Load, view switch and WASD movement pass | [Report](worldlabs_viewer_validation.json) |

Both navigation runs follow the same 3.384 m A* route from (5.12, 2.92) to (2.96, 4.80), with 0.32 m planner clearance. Each runs for 38 simulated seconds without falling, root teleportation or reference forcing. MuJoCo also audits furniture contacts above 1 N; the native PhysX navigation trial does not provide that contact audit. This is a map/state-based agent and proprioceptive WBC, not learned visual navigation.

Six additional actual video frames, excluded from input images and reconstruction textures, provide descriptive appearance review. Camera poses are fitted only for review using SIFT and learned-depth PnP. Mean grayscale SSIM is **0.821 for the native splat** and **0.717 for the final mesh**. These are interpolated viewpoints along the capture, not unseen backsides or a metric-depth benchmark. [Splat measurements](worldlabs_splat_appearance_validation.json), [mesh measurements](worldlabs_appearance_validation.json).

The native splat is the better visual representation. The simulator mesh has visible thin-chair, glass, blur and occlusion defects. The unseen entrance door, air conditioner and other uncaptured surfaces are model hypotheses. Furniture is fixed in this export; Atlas did not recover free joints, cabinet mechanisms, real masses, friction or relightable materials. The added 5 cm / 80 g cube is absent from the video. Tests validate its assigned simulation behavior. No grasp success is claimed for this static Atlas room.

## Reusable skill

The portable [worldlabs-scene-reconstruction skill](../skills/worldlabs-scene-reconstruction/SKILL.md) packages configurable workspaces, cached API stages, measured-anchor calibration, export and checks. [Offline portability checks](worldlabs_skill_validation.json) passed using a separate project and synthetic geometry, without paid API calls. The retired reconstruction workflow has been removed.

## Open and reproduce

```bash
# Native splat viewer; no API key needed after download.
python3 -m http.server 8870 --bind 127.0.0.1 --directory outputs/worldlabs_viewer

# Native interactive MuJoCo scene.
.venv/bin/python scripts/view_scene.py --scene outputs/worldlabs_scene/scene.xml --fast-exit

# Current Atlas job status; resumes without submitting duplicate work.
.venv/bin/python scripts/worldlabs_scene.py status --stage splats

# The actual map/state navigation test.
.venv/bin/python scripts/run_skill_agent.py --scene outputs/worldlabs_scene \
  --task navigate --run-name atlas_explore --device cuda --video

# Same route in native PhysX, using the saved scene-bound plan.
scripts/run_isaac.sh scripts/run_scalebfm_isaac.py --task explore \
  --scene outputs/worldlabs_scene/scene.usda --plan outputs/worldlabs/final_navigation_plan.json \
  --seconds 38 --run-name atlas_native --headless --fast-exit --device cuda:0
```

Viewer controls: select a source/novel camera, drag to orbit, wheel to zoom, WASD to move, Q/E down/up. The browser viewer is visual-only; collision/dynamics run in MuJoCo and Isaac Lab. The viewer folder is self-contained and contains neither the API key nor signed download URLs. Node rendering tools were tested with Spark 2.4.0, Three.js 0.187.0 and Playwright; the simulator project adds `trimesh` for GLB conversion.

The selected raw requests, resumable operation records, calibration and intermediate variants are private local data under `outputs/worldlabs/`. They are not part of the public source repository. The `worldlabs_scene.py` helper uses the CLI's saved login, `WLT_API_KEY`, or the authorized local first-line credential file. Use a fresh output workspace and source-specific frame/ROI/camera configuration for another recording; this office's fitting constants are not universal. See the reusable skill's [Atlas integration reference](../skills/worldlabs-scene-reconstruction/references/atlas.md).

Earlier extraction artifacts and the blocked planning attempt are preserved. A small benchmark cube initially closed a narrow conservative clearance corridor; the selected scene reserves the routes before placing that prop. The original, stereo/trimmed and selected constrained meshes remain separate for review. Generated assets and archives remain Git-ignored, as requested.
