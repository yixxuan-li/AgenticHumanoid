# Video reconstruction for MuJoCo and Isaac Lab

The delivered scene is **`outputs/final_scene/`**. It combines geometry recovered
from `data/scene/scene.mp4`, captured source appearance, fitted collision bodies,
12 free objects, and an inferred cabinet hinge. Released ScaleBFM weights drive
navigation followed by a physical Dex3 grasp in both MuJoCo and Isaac Lab/PhysX.

**Metric calibration is pending.** The video supplies no verified length. The
scene assumes a **0.74 m table height**; the resulting approximate envelope is
**6.17 × 5.66 × 2.74 m**, including an inferred entry landing. These are provisional
model dimensions, not measurements. Mass, friction, hidden geometry, and the
cabinet mechanism are estimates. See [the measured results](reports/RESULTS.md).

## Review and interact

Git excludes generated delivery archives and individual assets larger than
10 MiB. The full scene USD and converted robot base USD remain in the local
workspace, along with the ZIP bundles. A fresh clone needs the reconstruction
and robot-import commands below to regenerate these files before running Isaac
Lab. Some links in the original results report refer to these local artifacts.
The original validation/provenance reports describe the complete local bundle.

- [MuJoCo environment](outputs/final_scene/scene.xml) and [matching USD](outputs/final_scene/scene.usda).
- [Editable manifest](outputs/final_scene/manifest.json): inventory, poses, dimensions, assumptions, and benchmark profile.
- [Source/render comparison](outputs/final_scene/photo_comparison.png) and [independent review frames](outputs/final_scene/heldout_comparison.jpg).
- [Side-by-side two-engine demo](outputs/final_scene/navigation_pick_both_engines.mp4).
- [MuJoCo navigation-to-pick video](outputs/controller/pick/final_pick/dex3_pick_trial.mp4).
- [PhysX navigation-to-pick video](outputs/controller/isaac_pick/final_sequence/pick.mp4).
- [MuJoCo exploration video](outputs/controller/final_explore/explore_mode0.mp4).
- [Force-driven furniture interaction](outputs/final_scene/interaction.mp4).
- [Trajectory/height plots](outputs/final_scene/benchmark_results.png).

The scene contains one central folding table, four green chairs, one gray chair,
two black office chairs, perimeter workstations, a wall cupboard, a vertical
storage stack, and movable props. The red `test_block` is an **added benchmark
object**, absent from the source video. The grasp tests target it. The small
observed black table object remains movable but has no validated grasp profile.
Papers, cables, cloth clutter, and some small details remain visual-only.

Open the native MuJoCo viewer from the project root:

```bash
.venv/bin/python scripts/view_scene.py --scene outputs/final_scene/scene.xml --fast-exit
```

Controls: **P** pause, **R** reset, **O/C** cabinet open/close, **1–4** camera
presets, **[ / ]** recorded-camera views, **V** cutaway, **F** collision view,
plus MuJoCo mouse navigation/physical perturbations. A desktop display is needed.
`--collision` opens with collision proxies visible. The photos on movable objects
follow their rigid-body transforms; their captured shadows and lighting remain
baked into the images. `--fast-exit` avoids an interpreter graphics-shutdown
crash observed after the native window closed on this runtime.

For a fresh machine, use Python 3.12 and `requirements.txt`. The local `.venv`
currently shares installed scientific packages with a neighboring environment;
it is not a distributable environment. The exported scene itself uses relative
asset paths. Tested runtimes are MuJoCo 3.11.0 and Isaac Sim 5.1 / Isaac Lab 0.47.2.

## Run the controller and simulator checks

The agent uses the known scene map and simulator state, A* approach planning,
ScaleBFM body references, task-space feedback, and separate Dex3 finger control.
It is a **map-based scripted skill agent**, with no learned visual perception.
Neither robot root nor object pose is forced during an episode. No object weld
or external object force is used in the grasp trials.

```bash
.venv/bin/python scripts/validate_scene.py \
  --scene outputs/final_scene --report reports/final_scene_physics.json
.venv/bin/python scripts/run_skill_agent.py --scene outputs/final_scene \
  --task navigate --run-name my_explore --device cuda --video
.venv/bin/python scripts/run_skill_agent.py --scene outputs/final_scene \
  --task pick --run-name my_pick --device cuda --video
```

The pick router checks a target's affordance, a tested hand profile, its support
pose, and a collision-free approach. A declared affordance alone is insufficient.
`--target pick_box` currently rejects the unvalidated observed object; this is
not counted as a grasp failure or success. Use `--dry-run` to inspect a plan.

For Isaac Lab, `scripts/run_isaac.sh` selects the installed runtime and its needed
`libstdc++` preload. Set `ISAAC_PYTHON` / `ISAAC_LIBSTDCXX` on another installation.
`--fast-exit` flushes artifacts and avoids a Kit shutdown hang observed here.

```bash
scripts/run_isaac.sh scripts/isaaclab_smoke.py --headless --fast-exit \
  --scene outputs/final_scene/scene.usda --report reports/final_isaaclab_validation.json

.venv/bin/python scripts/run_skill_agent.py --scene outputs/final_scene \
  --task pick --run-name final_plan --dry-run
scripts/run_isaac.sh scripts/run_pick_isaac.py --headless --fast-exit \
  --scene outputs/final_scene/scene.usda --navigate-first \
  --plan reports/skill_agent_final_plan.json --run-name my_pick
```

Add `--enable_cameras --video` to the PhysX grasp command to record it. The
converted G1/Dex3 USD is at `outputs/robots/g1_dex3_v4/`; on a fresh setup, import it
with `scripts/import_dex3_isaac.py --output outputs/robots/g1_dex3_v4 --headless
--fast-exit` through the same Isaac wrapper. The importer preserves source
inertias, removes empty sensor/extra articulation artifacts, and sets small
contact offsets. [Robot transfer checks](reports/robot_transfer_audit.json)
compare source masses, rest transforms, all 43 joint ranges, and signed axes.

For a source-camera RTX image, add `--enable_cameras --screenshot --camera
source_016` to the scene smoke command. [The recorded view](outputs/final_scene/isaaclab_source_016.png)
uses the matching camera pose, roll, and vertical field of view.

## Reproduce this recording's reconstruction

Fetch pinned official sources and checkpoints when absent. Both model loaders
verify their checkpoint hashes; the BFM adapter also pins deployment metadata.
Weights/vendor sources are separate from the reusable skill and code bundle.

```bash
.venv/bin/python scripts/bootstrap_sources.py --isaac
.venv/bin/python scripts/download_ranges.py \
  https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt \
  checkpoints/vggt_model_ranges.pt --workers 8 \
  --sha256 d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0
.venv/bin/python scripts/download_ranges.py \
  https://huggingface.co/WeishuaiZeng/ScaleBFM/resolve/main/checkpoint/humanoid_transformer_m/model_22200.pt \
  checkpoints/scalebfm_m.pt --workers 8 --chunk-mb 4 \
  --sha256 88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3
curl -fL https://huggingface.co/WeishuaiZeng/ScaleBFM/resolve/main/compiled_checkpoint/humanoid_transformer_m/linux/model_22200_tensorrt_metadata.json \
  -o checkpoints/scalebfm_m_metadata.json
```

Expected BFM metadata SHA-256:
`3024f18136314fade2676e22d6eafb1a8b2befa388da55b0885a53fcb1368d16`.
Existing hash-verified files are reused without a network request.

```bash
.venv/bin/python scripts/inspect_video.py data/scene/scene.mp4
.venv/bin/python scripts/run_vggt.py --stride 1 --max-frames 34 --output outputs/geometry_full
.venv/bin/python scripts/process_geometry.py --input outputs/geometry_full/predictions.npz \
  --report reports/dense_geometry_full.json
.venv/bin/python scripts/fit_room_planes.py
.venv/bin/python scripts/fit_scene_features.py --geometry outputs/geometry_full \
  --room-envelope reports/room_plane_fit.json
.venv/bin/python scripts/fuse_geometry.py --geometry outputs/geometry_full --output outputs/fusion
.venv/bin/python scripts/texture_fused_geometry.py --geometry outputs/geometry_full \
  --fusion outputs/fusion --bounds -.35 -.75 -.06 6.35 5.55 3.0 --overwrite
.venv/bin/python scripts/build_dense_scene.py --geometry outputs/fusion \
  --dynamic-geometry outputs/geometry_full --output outputs/final_scene --hide-front-wall-visual
```

Use a fresh output directory or the explicit overwrite option when retexturing
existing fusion output. These object ROIs, room bounds, colors, and table-height
assumptions describe **this recording**. For another video, use the reusable skill,
configure its own floor/anchor regions, review object identities across the pan,
and fit a new semantic manifest.

`run_colmap.py` optionally provides an independent sparse reconstruction.
`process_geometry.py --sfm outputs/colmap/sparse/0` compares camera orientations
for that exact video. Good reprojection or learned-depth agreement does not
establish ground-truth depth or metric scale. Original and fused surfaces remain
in `outputs/geometry_full/` and `outputs/fusion/`; thin movable-object surfaces
use the original captures where fusion loses detail.

## Calibrate and reuse

Copy `data/scene/calibration.example.json`, replace its null value with a real
measurement of the corresponding physical segment, then run:

```bash
.venv/bin/python scripts/rescale_scene.py --scene outputs/final_scene \
  --output outputs/calibrated_scene --calibration my_measurement.json
```

The output must be a new directory. Geometry, photographic meshes, cameras,
planning coordinates, and support heights rescale together; assumed masses
follow volume. The robot's radius, stance height, hand geometry, and gains stay
at their published metric size. Re-run physics and controller tests. One anchor
sets global scale; it does not verify each inferred furniture dimension.

The [±5% sensitivity run](reports/scale_sensitivity.json) is hypothetical, not a
confidence interval: both variants pass physics checks, the smaller variant's
pick approach is rejected at 0.32 m planner clearance, and the larger variant
completes a pick. This makes a measured anchor important before relying on fit
or reachability.

The reusable [reconstruct-video-scene skill](skills/reconstruct-video-scene/SKILL.md)
contains a portable geometry/export toolbox and explicit scale, appearance,
physics, ownership, and controller acceptance gates. It excludes private source
frames, this office's layout, and model weights. Its validation covers actual
source-frame ROI alignment/fusion and independent export/settling checks; it
does not certify arbitrary future videos.

Remaining appearance limits include low source resolution, reflective-window
holes, thin/occluded surfaces, approximate backsides, and baked shadows. The
entry closure and cabinet hinge are inferred. Additional translated views and
measurements would improve fidelity. Earlier primitive prototypes and failed
trials remain for audit; `outputs/scene/` is **not** the delivered reconstruction.

Upstream sources: [VGGT](https://github.com/facebookresearch/vggt) and
[ScaleBFM](https://github.com/zengweishuai/ScaleBFM), pinned in `vendor/` provenance
files. Their code, models, and robot-asset licenses remain applicable.
