# Reconstruction and controller results
Updated: 2026-11-20T02:45:54+08:00

**The interactive scene and controller demonstrations run in both simulators. Absolute scale is not verified.**

The delivered assets are `outputs/final_scene/`, identity `a33e42e3cba84ebb9b5f1d0feea63f1931752413a8fb83f530d9ede0743edd59`. The model assumes a 0.74 m table height and has an approximate 6.17 × 5.66 × 2.74 m envelope. One real dimension is still needed for global calibration; it will not remove all geometric uncertainty.

| Executed check | Result | Evidence |
|---|---|---|
| MuJoCo scene dynamics + USD structure | 13/13 passed | [Report](final_scene_physics.json) |
| PhysX all-body settling, mass, support and hinge torque | 8/8 passed | [Report](final_isaaclab_validation.json) |
| Map-based MuJoCo exploration | Passed; goal error 2.7 cm | [Report](final_explore_scalebfm_explore_mode0.json) |
| MuJoCo continuous navigation → pick | Passed; lifted hand contact 4.48 s | [Report](dex3_pick_final_pick.json) |
| PhysX continuous navigation → pick | Passed; lifted hand contact 4.30 s | [Report](isaac_dex3_pick_final_sequence.json) |
| G1/Dex3 import comparison | Passed | [Mass, rest transforms, 43 joint ranges/axes](robot_transfer_audit.json) |
| Asset dependency locality | Passed | [Report](asset_dependency_audit.json) |

## Watch and open

- [MuJoCo navigation and pick](../outputs/controller/pick/final_pick/dex3_pick_trial.mp4)
- [Isaac Lab / PhysX navigation and pick](../outputs/controller/isaac_pick/final_sequence/pick.mp4)
- [Exploration](../outputs/controller/final_explore/explore_mode0.mp4)
- [Force-driven chair/cabinet interaction](../outputs/final_scene/interaction.mp4)
- [Measured trajectories and object lift](../outputs/final_scene/benchmark_results.png)
- [Source and novel-view comparison](../outputs/final_scene/photo_comparison.png)
- [Real held-out-frame review](../outputs/final_scene/heldout_comparison.jpg)
- [Native RTX source camera](../outputs/final_scene/isaaclab_source_016.png)
- [MuJoCo scene](../outputs/final_scene/scene.xml), [USD scene](../outputs/final_scene/scene.usda), [manifest](../outputs/final_scene/manifest.json)
- [Commands and reconstruction procedure](../README.md)

## What the tests establish

The modeled table supports a freely dropped block with 0.97 mm settling error. The friction probe moves 0.079 m toward the table interior and stops while supported. MuJoCo finds no initial penetration over 5 mm. PhysX checks all 13 movable bodies, matches the assigned masses, and opens the cabinet to -1.600 rad under a real −3 Nm torque. This validates simulator behavior of the model, not true room dimensions or real material parameters.

Both pick trials start at the exploration origin, follow a scene-matched A* route, settle, reach, close the fingers, lift, and hold. Final block heights are 0.974 m (MuJoCo) and 0.972 m (PhysX), about 20 cm above their initial positions. Neither trial welds the object or forces its pose, and the robot root is not teleported during execution. The released BFM controls the 29 body joints at 50 Hz with 200 Hz physics; separate bounded hand control drives 14 Dex3 joints.

The agent uses the known collision map, object/root/wrist simulator state, timed task references, and integral task-space corrections. It is not a trained visual or language-model exploration system. The red block is an added 5 cm / 80 g benchmark prop. Successful picks do not establish that the observed black object, arbitrary objects, or an unknown real room can be grasped. MuJoCo audits navigation furniture contacts above 1 N; PhysX grasp evidence uses filtered hand–object contacts. Grasping may contact the tabletop with the fingertips, as recorded in the contact log.

## Geometry and appearance evidence

VGGT used 34 upright video frames, with exact source-footprint masks excluding artificial padding. The fitted floor's median residual is 4.92 provisional mm. Independent sparse camera orientation comparison and learned-depth cross-view agreement are in [the dense audit](dense_geometry_full.json); these are consistency evidence, not ground-truth depth or scale.

The final review uses 6 additional real video frames excluded from model input and reconstruction textures. Their camera poses are estimated only for rendering with SIFT/3D PnP. Gray-image SSIM ranges from 0.557 to 0.773 (mean 0.652); this is descriptive image agreement, not an acceptance certificate. The review frames are intermediate viewpoints along the recorded path, not unseen backsides. [Per-view pose/errors and primitive visibility](final_appearance_validation.json).

Static room surfaces use TSDF fusion with source-view textures; movable-object skins retain original captures where fusion loses thin details. The central tabletop uses a depth-tested planar atlas. Floor/table/chair-panel constraints are recorded in the ownership/texture provenance. Photo meshes never collide. Fitted boxes/cylinders supply contact; inferred fallback visuals fill some unseen regions. Reflection holes, occlusion boundaries, texture seams, missing backsides, and baked lighting/shadows remain visible. Wall slopes/recesses and the entry closure are approximate; some room-plane fits have low inlier fractions.

## Calibration and reusable workflow

The ±5% runs are hypothetical scenario tests, not a statistical confidence interval. Both pass the physical scene checks. The −5% case rejects the declared grasp approach at 0.32 m clearance; it does not execute a pick. The +5% case completes a contact pick with the metric robot unchanged. [Sensitivity evidence](scale_sensitivity.json). This is why the 0.74 m assumption cannot be treated as a verified scale anchor.

Use an actual dimension of the corresponding segment with `scripts/rescale_scene.py`. Meshes, cameras, proxy geometry, support heights and planning positions rescale together; masses follow the explicit volume assumption. Robot dimensions, stance heights, hand offsets and gains remain fixed. Re-run the checks for the new scene identity.

The [reusable reconstruction skill](../skills/reconstruct-video-scene/SKILL.md) packages a generic, configurable geometry/export toolbox and acceptance gates. It includes neither private room frames nor this office layout. Its checks include a synthetic exporter/settling contract, actual source-derived scene settling, and a four-frame ROI-alignment/fusion run. It is a reusable workflow, not a guarantee that arbitrary videos recover correct metric geometry.

## Preserved limitations and failed attempts

Earlier reach probes and navigation-to-grasp attempts failed or pushed the block off the table; their reports remain under `reports/` with trial-specific names. The initial Isaac motor mismatch, importer/path failures, and corrected USD hinge-frame issue are retained. Earlier primitive scenes and reports refer to different identities and are not evidence for the delivered scene. The current selected reports above must match its exact fingerprint.

Outstanding input: a real metric correspondence. Outstanding review: whether the visible reconstruction quality meets the intended camera-observation benchmark. Mass/friction and hidden mechanisms would need additional measurements for a high-fidelity physical twin.

## Placement and delivery checks

PhysX checks three independent scene references at the origin, translated 8 m, and translated/rotated 90 degrees: passed. All 39 movable-body placements and three closed hinge frames are checked. Cabinet hinges use collider-free kinematic anchors within the scene, rather than absolute world anchors. This establishes basic reference placement, not large-batch training performance. [Report](final_placement_validation.json).

The skill is supplied as a self-contained folder and ZIP; it is not globally installed. Its generic exporter was also tested on a separate synthetic floor, cube and hinged panel. [Reusable skill checks](skill_reuse_validation.json).

[Portable scene ZIP](../deliverables/scene_bundle.zip), [reusable skill ZIP](../deliverables/reconstruct-video-scene.zip), and [project bundle](../deliverables/reconstruction_project.zip). The project includes the converted robot, source video, code, selected evidence and demos; upstream source/weights are fetched with the pinned bootstrap commands. [Delivery checksums](delivery_integrity.json).

[Side-by-side recorded demo](../outputs/final_scene/navigation_pick_both_engines.mp4).

The native MuJoCo window opens and simulates. On this runtime, interpreter graphics teardown crashed after the window closed; `--fast-exit` flushes output and skips that teardown. The workaround exited successfully. [Startup/shutdown evidence](native_viewer_startup.json). Keyboard and mouse interaction were not automated by this smoke check.

## Additional body-controller checks

| Trial | Root tracking RMSE | Result |
|---|---:|---|
| MuJoCo straight walking | 0.066 m | Passed |
| MuJoCo squat tracking | 0.060 m | Passed |
| PhysX straight walking | 0.070 m | Passed |
| PhysX squat tracking | 0.047 m | Passed |
