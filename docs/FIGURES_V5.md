# Native MuJoCo figures, version 5

This revision replaces the generated scene and robot artwork in version 4 with
native MuJoCo renders. It extends the existing project; versions 3 and 4 remain
available. All figure labels, captions, code, and documentation are in English.

## Deliverables

- `outputs/figures_v5/teaser_v5.{png,pdf,svg}`: a scene-led teaser with four
  temporal exposures, two onboard camera insets, and six supporting skill views.
- `outputs/figures_v5/system_v5.{png,pdf,svg}`: the merged reconstruction,
  execution, program evolution, and symbolic knowledge architecture.
- `outputs/figures_v5/render_provenance.json`: engine version, robot source,
  URDF and mesh hashes, reconstruction fingerprint, and authored pose records.
- `outputs/figures_v5/provenance.json`: figure provenance and layout reference.
- `outputs/figures_v5/scene/`: composed MJCF models, pose arrays, and projected
  label anchors.
- `outputs/figures_v5/validation.json`: checks performed on the final artifacts.
- `deliverables/figures_v5.zip`: figures, code, English notes, source room,
  official robot assets, and the metadata used for the initial standing pose.

The teaser is 4200 × 1500 pixels. The system figure is 4200 × 2310 pixels.
SVG/PDF typography, arrows, labels, and diagram boxes remain editable. The native
simulator views are embedded raster images.

## Scene and robot

The room is the existing real-office reconstruction in
`outputs/worldlabs_scene/scene.xml`, with its original Atlas texture meshes.
The source reconstruction is read only. Its fingerprint before and after figure
rendering is checked for equality:

```
d991e9181b6d9d428171375e2ab7ecdafa795f6f0cabbbd90eeb2581873ee2fa
```

Robot source: [Unitree Robotics unitree_ros](https://github.com/unitreerobotics/unitree_ros),
commit `157ff950351e5f535e03c5d85a16b7d594b3dbcc`, file
`robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf`.
The pinned URDF, its 49 referenced STL meshes, upstream source record, and BSD
license are under `vendor/unitree_g1/` and included in the archive.

MuJoCo imports this URDF directly through `MjSpec.from_file`. The renderer adds
a floating root, palm sites for pose authoring, and an illustrative torso-mounted
camera. The model retains the original link geometry and all 29 body and 14
finger joints. Duplicate collision shapes are hidden from the figure, while
the URDF visual meshes are rendered. MJCF files are saved after composition.

An orange cube, a target region, and an articulated drawer are authored task
props added to the composed scene. The drawer stands beside the room's door
and whiteboard; its articulation was not recovered from the capture. Room
scale and geometry retain the existing reconstruction's provisional calibration.

## What the images establish

All teaser views are actual MuJoCo renders. The system diagram also includes
an actual input-video frame and a native render of the original room. No image
generation service is used for version 5.

The four hero exposures show pick, turn, carry, and place configurations. The
gallery shows posture adjustment, reach, push, drawer opening, placement, and
drawer closing. These are authored poses produced by joint-bounded kinematic
optimization, with direct root and prop placement for illustration. Earlier
exposures are made translucent inside the simulator. The dashed route is an
authored narrative guide. It is not a collision-validated navigation result.

These images do not establish dynamics, stable grasps, learned skills, contact
success, or a successful ScaleBFM rollout. No policy is executed by the figure
renderer. The camera extrinsics are illustrative rather than hardware
calibration. The teaser itself says **MuJoCo · authored poses**; the detailed
distinctions belong in the caption rather than adding many words to the image.

The system diagram retains the existing implemented-component versus research
extension distinction. Figure authoring does not change the research system's
controller, scoring, skill library, API integration, or experiment results.

## Design

The composition follows the requested design principles of
[HITCH, Figure 1](https://arxiv.org/pdf/2611.08039): a large multi-exposure scene,
small onboard insets, compact supporting views, and short action labels.
No artwork, data, or benchmark percentages from that paper are used.

## Reproduce

From the project root, using the existing environment:

```bash
.venv-skill-lab/bin/python scripts/render_research_teaser_v5.py
.venv-skill-lab/bin/python scripts/draw_research_figures_v5.py
.venv-skill-lab/bin/python scripts/validate_research_figures_v5.py
```

The archive preserves repository-relative paths. Extract it into an empty
directory and run from that directory. To create a separate environment:

```bash
python3.12 -m venv .venv-figures
.venv-figures/bin/pip install -r requirements-figures.txt
.venv-figures/bin/python scripts/render_research_teaser_v5.py
.venv-figures/bin/python scripts/draw_research_figures_v5.py
.venv-figures/bin/python scripts/validate_research_figures_v5.py
```

The recorded platform uses Python 3.12.15, MuJoCo 3.11.0, and headless EGL.
A working EGL/OpenGL driver is required. The figure scripts do not need GPT API
credentials, a ScaleBFM checkpoint, or a connection to a real robot. They read
`checkpoints/scalebfm_m_metadata.json` only to initialize the body joint pose.

Both rendering and composition accept `--output PATH`. The compositor requires
the captured input frame at `PATH/assets/captured_13s.png`; it is already in the
archive. In the full working repository, the version 3 captured frame is used
as a fallback when creating a new output directory.

For individual pose replay, load `scene/composed_1.xml` and assign the matching
`*_pose.npz` array to `data.qpos`, then call `mujoco.mj_forward`. For the hero,
load `composed_4.xml`, assign `hero_poses.npz` arrays to `data.qpos`,
`model.geom_rgba`, and `model.geom_group`. Use the fixed `hero` camera. Render
geometry groups 1 and 2; exclude groups 0 and 3. The scripts provide the full
camera, lighting, composition, and visibility settings.

## Captions

**Teaser.** AgenticHumanoid explores editable skill programs and evolving
symbolic knowledge in a reconstructed real-world office. Native MuJoCo renders
use the official Unitree G1 URDF with articulated hands. Four temporal exposures
and supporting views illustrate pick, turn, carry, place, posture adjustment,
reach, push, and drawer interaction. Poses, route, camera mounting, and task props
are authored for visualization; they are not successful controller rollouts or
evidence of acquired contact skills. The room is the existing static
reconstruction; drawer articulation is an added task prop.

**System.** Captured-room reconstruction and a versioned robot model ground a
local observation, belief, skill execution, reference generation, and frozen
ScaleBFM control loop. Episode-level GPT proposals revise skill programs and
symbolic knowledge through validation and version selection. Independent
evaluation, evidence records, rollback, and frozen testing separate candidate
changes from accepted capability. Dashed components denote extensions not yet
validated end to end.
