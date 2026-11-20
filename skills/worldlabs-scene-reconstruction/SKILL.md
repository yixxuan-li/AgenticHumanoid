---
name: worldlabs-scene-reconstruction
description: Reconstruct a recorded environment with World Labs Atlas and export calibrated static scenes to MuJoCo and Isaac Lab, with a native splat viewer and physics validation. Use for video or image captures intended for robot simulation.
---

Use Atlas to reconstruct appearance and estimated geometry, then build a separate simulator contact representation. Deliver the native SPZ, a local viewer, matching MJCF/USD, calibration provenance, and validation reports appropriate to the requested task.

Read [the workflow](references/workflow.md) for commands and portable tooling. Read [Atlas integration](references/atlas.md) before submitting jobs or choosing novel views; it covers resumability, mask consistency, coordinates, and visual review. Read [the manifest](references/manifest.md) when authoring movable bodies or mechanisms beyond the static Atlas output.

Copy `assets/toolbox/` into a new project directory. Its scripts are self-contained and use the project working directory plus explicit `--workspace` paths; no source recording, credentials, weights, or generated assets are bundled. The official `worldlabs` CLI is required for API work; obtain setup and request schemas from current Atlas documentation. Use installed official Atlas skills when available, without making them a prerequisite for downloaded-scene export.

Keep these distinctions in the result:

- Generated views complete missing areas and can contradict observed furniture. Review and curate them before reconstruction; record which surfaces are inferred.
- Atlas units do not establish metres. Use a measured anchor when supplied, otherwise state the scale assumption. Apply one consistent transform to geometry, cameras and collision boxes; keep robot dimensions fixed.
- Splats provide appearance. Exported textured meshes provide simulator visuals, with independent fitted collision boxes. Inspect both because mesh extraction can introduce bridges and holes absent from the splats.
- Furniture in this exporter is static. The added validation cube is an authored prop. Movable objects require segmentation, exclusive visual ownership, masses, support contacts, and joints where applicable; affordance labels alone do not validate picking.
- Support/drop/friction tests validate assigned parameters. They do not prove real-world accuracy. Bind reports to geometry and texture hashes; native PhysX tests are required before claiming Isaac Lab dynamics work.

Persist operation IDs and requests, resume existing jobs, and reuse completed downloads. Do not launch another paid reconstruction merely to package or review existing assets. Keep API keys and raw responses with signed download URLs outside Git and delivery archives.

The repository's `constrain_worldlabs_mesh.py` and `extract_worldlabs_splats.mjs` are optional office-specific corrections, deliberately excluded from the portable toolbox. Derive any such constraints from the new capture rather than reusing this office's coordinates or 0.74 m height assumption. Navigation WBC, symbolic skill discovery, and real-robot transfer are separate downstream tasks.
