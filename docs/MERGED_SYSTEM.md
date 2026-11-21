# Merged AgenticHumanoid system

The previous reconstruction version and the skill exploration version form one
project. `scene_reconstruction/` remains the scene/export and shared policy layer;
`humanoid_skill_lab/` supplies program execution and learning. The canonical
command is `agentic-humanoid`. The earlier `humanoid-lab` alias and existing
scripts continue to work.

## What changed between versions

| Capability | Previous version | Skill exploration addition | Merged behavior |
|---|---|---|---|
| Environment | Atlas capture reconstruction, native splats, calibrated MJCF/USD export | Explicit tabletop benchmark | Retains both; online controller probes can use the original Atlas room. |
| Control | Frozen ScaleBFM in MuJoCo and Isaac, map-based navigation | Official ScaleBridge buffers/loop with online references | Shared released weights, metadata, robot assets and eager loader; task-specific adapters remain. |
| Agent | Map/manifest router and collision-aware navigation | Schema-checked stage programs, RGB-D, GPT revision loop | One command surface for navigation, scene inspection and program experiments. |
| Knowledge | Hand-declared execution checks | Candidate rules, evidence, versions, grounded symbolic search | Research knowledge is retained alongside existing scene provenance; unvalidated rules do not authorize room manipulation. |
| Evaluation | Scene/export/physics checks and navigation reports | Independent task score, paired selection, frozen tests | Both record types are preserved and identified by their scene and controller provenance. |
| Delivery | Reconstruction sources and reusable reconstruction skill | Skill prototype sources and selected evidence | One merged archive contains both source workflows and regression records. |

The earlier delivery added the skill package alongside the reconstruction tools,
but its runner was hardcoded to the tabletop fixture. This merge connects the
runner to existing exported scenes, adds a common CLI and packages both versions.

## One entry point

Run from the repository root after fetching pinned ScaleBFM assets as described
in the root README. No reconstruction API call is needed to use an existing room.

```bash
UV_PROJECT_ENVIRONMENT=.venv-skill-lab uv sync --frozen --extra test --extra plots
.venv-skill-lab/bin/agentic-humanoid doctor

# Previous scene and navigation capabilities.
.venv-skill-lab/bin/agentic-humanoid scene --scene outputs/worldlabs_scene
.venv-skill-lab/bin/agentic-humanoid view --scene outputs/worldlabs_scene
.venv-skill-lab/bin/agentic-humanoid navigate --scene outputs/worldlabs_scene \
  --run-name atlas_navigation --device cuda --video

# New online stage execution in that same scene.
.venv-skill-lab/bin/agentic-humanoid probe --task reach \
  --scene outputs/worldlabs_scene --output outputs/merged_system/my_reach
.venv-skill-lab/bin/agentic-humanoid probe --task height \
  --scene outputs/worldlabs_scene --output outputs/merged_system/my_height

# Task learning on the independently defined contact benchmark.
.venv-skill-lab/bin/agentic-humanoid run \
  --skill humanoid_skill_lab/configs/push_calibrated.json \
  --seconds 22 --output outputs/skill_lab/my_push
.venv-skill-lab/bin/agentic-humanoid explore --bashrc \
  --output outputs/skill_lab/my_exploration
```

For reconstruction and MJCF/USD export, add the locked `reconstruction` optional
dependency group. The existing `scripts/worldlabs_scene.py`, export tools,
reusable reconstruction skill and native splat viewer are retained. Isaac Lab
continues to use its own compatible environment and the original
`scripts/run_isaac.sh` entry point. This merge does not rerun paid reconstruction
jobs or claim new Isaac validation.

## Scene and frame contract

Imported scenes use the previous `manifest.json`, `scene.xml`, `scene.usda` and
`scene_reconstruction.provenance.scene_fingerprint` contract. The original
navigation start initializes G1. Source scene files are never rewritten, and the
fingerprint is checked again after a probe. Provisional scale and scene limitations
are carried into the run manifest.

Programs may set `frame: episode` for absolute wrist/root position commands.
That frame translates local meter coordinates by the initial root XY position;
its axes remain parallel to world axes and floor height remains world zero.
`frame: world` remains the default for existing programs. Object-relative offsets
remain world-axis offsets. Reach probe version 2 declares the episode frame, so
the same small motion runs at both the tabletop origin and the Atlas spawn point.

Imported controller probes use a separate control evaluator and declare no task
objects. They record real onboard RGB-D, joint/body trajectories, references and
contacts, but do not apply the tabletop color detector to arbitrary room colors.
Manipulation scores remain null. A push exploration request on an imported room
without declared task bindings/evaluator is rejected rather than inserting a
hidden benchmark object or treating static furniture as articulated.

Navigation retains the previous tested map/router and controller loop. Online
probes use the new ScaleBridge environment loop. Both use the same maintained
`scene_reconstruction.scalebfm.ScaleBFM` loader, frozen checkpoint and metadata.
Keeping the two task adapters preserves the prior navigation behavior while
exposing them as components of the same application.

## Evidence and limits

Merge checks are saved under `outputs/merged_system/`, with summary and hashes
in `reports/merged_system_validation.json`. They include the existing navigation
workflow, online reach/height probes in the original Atlas scene and a regression
of the calibrated tabletop push. No additional GPT calls are needed for this merge.

| Merge regression | Actual outcome |
|---|---|
| Tests | 36 passed. |
| Previous room navigation | Passed; 2.83 m root displacement, 3.82 cm goal error. |
| Online reach in that room | Passed; 25.78 cm hand displacement, 3.46 cm final wrist error. |
| Online height program in that room | Passed; 6.78 cm actual vertical range for an 8 cm requested range. |
| Calibrated tabletop push | Passed again; 5.68 mm final cube-center error. |
| Scene preservation | Original reconstruction fingerprint unchanged. |

Rerun the record checks with
`.venv-skill-lab/bin/python scripts/validate_merged_system.py`.

Historical skill results remain under `outputs/skill_lab/`; their recorded source
hashes identify the earlier implementation. The original prototype archive is
retained. The historical artifact audit verifies consistency within that recorded
release; `--require-current-source` additionally checks against the working tree
and is expected to fail after a source revision.

The merged project does not establish reliable onboard push learning, successful
GPT promotion, validated multi-skill symbolic transfer, drawer/grasp capability
or real-robot deployment. The original static reconstruction and provisional
scale assumptions remain explicit.

Build the merged delivery with:

```bash
.venv-skill-lab/bin/python scripts/package_agentic_humanoid.py
```

The result is `deliverables/agentic_humanoid_merged.tar.gz`, with a file-hash
manifest. It includes previous tracked reconstruction code, the reusable skill,
new skill code/docs, selected earlier evidence and merge regression records.
Checkpoints, environment directories, credentials and downloaded scene assets
remain excluded and are restored through the documented setup workflow.
