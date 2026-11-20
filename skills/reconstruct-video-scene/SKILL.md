---
name: reconstruct-video-scene
description: Reconstruct a supplied room video into interactive MuJoCo and Isaac Lab scene assets, with source appearance, explicit metric calibration, and physics validation. Use for video-to-simulator reconstruction and its reusable pipeline, not ordinary video editing.
---

Produce a source-grounded scene whose visible objects and collision bodies agree. Read [the workflow](references/workflow.md) for the runnable toolbox and [the manifest contract](references/manifest.md) when exporting semantic objects.

Treat monocular scale, unobserved geometry, mass, friction, and joints as separate uncertainties. A conventional table height supplies an assumption, not a measurement. Record the anchor and its provenance. Keep the robot at its published metric size. An unknown metric anchor does not prevent a provisional reconstruction, but the result must retain that status.

Review the video across the full camera pan before counting objects. Track repeated appearances across frames. Fit floor/support planes and furniture poses to recovered geometry, then verify each object against source frames. The example office inventory and pixel regions are specific to that recording.

Use photo surfaces for appearance and simple fitted solids for contact. Attach movable-object photographs in body-local coordinates; remove their copies from the static background. Check an object-moving render for leftover ghosts, attached floor patches, and missing backsides. Preserve original geometry before applying support-plane or panel constraints. Baked source lighting does not demonstrate recovered material properties.

Validate actual MuJoCo dynamics and USD loading in Isaac Lab when that runtime is available. Record executed tests separately from planned paths and schema checks. Bind reports to the exact scene/asset hashes. Re-run after geometry, scale, or motor changes. Preserve failures and distinguish added benchmark props from observed objects.

For controller experiments, use the requested WBC's published joint order, motor model, observation history, and control rate. Report navigation, reach, contact grasp, and complete skill sequences separately. A scripted or map-based planner must be identified as such. A lifted object needs sustained physical hand contact; a welded object or a successful reach is insufficient evidence.

Deliver the interactive scene, matching simulator exports, source/render comparisons, physical assumptions, reproducible commands, and test evidence. State remaining scale and coverage limitations plainly. The toolbox supports reconstruction; new scenes still require source-specific semantic fitting and visual review.
