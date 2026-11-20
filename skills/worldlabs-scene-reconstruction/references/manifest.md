# Semantic manifest contract

All lengths use meters, mass kilograms, angles radians; Z is up. This contract describes the supplied exporter, not every simulator feature.

Required root fields: `dimensions: [width, depth, height]`, `scale_status`, and `bodies`. Optional `desk_height` is a documented support height, or null when no desk-height prior is used. Recommended: `metric_anchor`, `uncertainty`, `navigation: {start, goals, radius}`, and source provenance. `global_scale` defaults to 1. Optional `cameras` is a list of `[name, position_xyz, target_xyz]` triples.

Every body has:

- `name`: unique identifier usable in both simulators.
- `pos: [x,y,z]`, `yaw`: initial world pose.
- `kind`: `fixed`, `free`, or `hinge`.
- `mass`: zero for fixed bodies, positive for dynamic bodies.
- `evidence`: source frames/features or explicit inference/benchmark status.
- `parts`: nonempty list of primitives.

A part has `name`, `shape` (`box` or `cylinder`), `pos`, `rotation: [rx,ry,rz]`, `size`, `material`, `texture`, and `collision`. Coordinates are body-local. Box sizes are full XYZ dimensions. Cylinder size is `[radius, full_height]`, with axis along local Z before rotation. `texture` is null or the basename of an output PNG. `material` names are defined in the primitive library's `COLORS`; extend them when appropriate. `visible: false` keeps a collider while suppressing that part's visual. `visual_style: rounded` gives a rounded box visual with a box collider.

Use helpers `box(...)` and `cylinder(...)` from `scene_reconstruction.layout` to supply the part defaults. That bundled module contains primitives and colors only.

A hinge body additionally provides:

```json
{"joint": {"axis": [0,0,1], "range": [-1.6,0], "damping": 1.2, "friction": 0.2}}
```

The supplied USD hinge is attached around Z at body origin to a collider-free kinematic anchor within the room. Its initial pose matches the body, making the closed angle zero while preserving referenced-room placement. Adapt the exporter for another axis, a parent articulation, slider, or multiple links. Define collision solids that allow the requested mechanism to travel; passing a joint range check alone does not establish that it can open.

Per-part masses are allocated by primitive volume and sum to the body's assumed mass. Overlapping decorative primitives are not a measured density model. Use explicit inertial properties for a task that requires them. `affordances` may list intended tasks but does not itself prove the task works.

Keep observed objects, inferred backsides/supports, and added benchmark props identifiable in evidence. Record a source-to-world transform and any support-plane constraints. For photographic moving meshes, use the initial body pose inverse; do not leave their surfaces at world coordinates after attaching them.
