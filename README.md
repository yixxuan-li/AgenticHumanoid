# AgenticHumanoid

One system combines **World Labs Atlas scene reconstruction**, shared frozen
**ScaleBFM control**, and **editable humanoid skill exploration**. The previous
reconstruction, splat viewer, MuJoCo/Isaac exports and navigation workflows are
retained. The skill layer adds online ScaleBridge references, onboard RGB-D,
GPT proposals, symbolic knowledge and independent validation.

See [the merged system and version comparison](docs/MERGED_SYSTEM.md),
[the skill guide](docs/SKILL_LAB.md), and
[the development record](docs/DEVELOPMENT_RECORD.md). `agentic-humanoid` is the
combined command; `humanoid-lab` remains a compatible alias.

```bash
UV_PROJECT_ENVIRONMENT=.venv-skill-lab uv sync --frozen --extra test
.venv-skill-lab/bin/agentic-humanoid doctor
.venv-skill-lab/bin/agentic-humanoid scene --scene outputs/worldlabs_scene
.venv-skill-lab/bin/agentic-humanoid navigate --scene outputs/worldlabs_scene \
  --run-name my_atlas_navigation --device cuda --video
.venv-skill-lab/bin/agentic-humanoid probe --task reach \
  --scene outputs/worldlabs_scene --output outputs/merged_system/my_atlas_reach
```

Online controller probes now run directly in the previous reconstructed room,
with its original spawn point and scene fingerprint. Push learning still uses
the explicitly modeled tabletop benchmark; reconstructed static furniture has
no inferred manipulation affordances. For reconstruction/export dependencies,
install the optional locked group with
`UV_PROJECT_ENVIRONMENT=.venv-skill-lab uv sync --frozen --extra test --extra reconstruction`.
Isaac Lab retains its separate runtime installation. The commands below remain
available for reconstruction and engine-specific operations.

Video-to-scene reconstruction with **World Labs Atlas**, followed by calibrated
MJCF/USD export and ScaleBFM whole-body control in MuJoCo and Isaac Lab.
The World Labs workflow replaces the earlier VGGT reconstruction.

The current local scene is `outputs/worldlabs_scene/`, and the native Gaussian
splat viewer is `outputs/worldlabs_viewer/`. Generated assets, model weights,
vendor sources, API responses, credentials and delivery archives are ignored by
Git. A fresh checkout contains source, the input video, the reusable skill and
sanitized validation reports; reconstruct or restore the local bundle to obtain
scene assets.

See [the reconstruction results](reports/WORLDLABS_RESULTS.md) for the actual
pipeline, metrics and limitations. Absolute scale currently assumes a **0.74 m
table height**, pending a real measurement. Original furniture is static;
one added free cube supports physics checks. Navigation succeeded in both
engines. Picking in this Atlas room, symbolic skill discovery and real-robot
transfer have not been validated.

## Reusable reconstruction skill

Use [worldlabs-scene-reconstruction](skills/worldlabs-scene-reconstruction/SKILL.md)
for another capture. It includes portable tools for frame selection, resumable
API stages, floor/scale calibration, simulator export, local splat viewing and
physics checks. Follow [its workflow](skills/worldlabs-scene-reconstruction/references/workflow.md)
for installation and commands. Copy the skill folder to `~/.codex/skills/` to
install, or use the locally installed skill as `$worldlabs-scene-reconstruction`.
The official `worldlabs` CLI handles Atlas jobs. The API does not provide the
simulator physical parameters, articulation or downstream robot control.

Build a standalone skill archive after editing maintained tools:

```bash
python scripts/package_worldlabs_skill.py
# Offline portability and simulator checks; no paid API calls.
.venv/bin/python scripts/verify_worldlabs_skill.py
```

The archive is `deliverables/worldlabs-scene-reconstruction.zip`. Office-specific
mesh corrections remain separate from the generic toolbox.

## Open the local scene

```bash
python3 -m http.server 8870 --bind 127.0.0.1 --directory outputs/worldlabs_viewer
.venv/bin/python scripts/view_scene.py --scene outputs/worldlabs_scene/scene.xml --fast-exit
.venv/bin/python scripts/worldlabs_scene.py status --stage splats
.venv/bin/python scripts/validate_worldlabs_mujoco.py
scripts/run_isaac.sh scripts/validate_worldlabs_isaac.py \
  --scene outputs/worldlabs_scene/scene.usda --headless --fast-exit
```

The splat viewer needs no API key after download. For API work, use CLI login,
`WLT_API_KEY`, or an authorized ignored first-line `worldlabs` credential file.
Raw operation responses can include private signed URLs; keep them local.

## ScaleBFM control

The shared controller tools use the released medium checkpoint from
[ScaleBFM](https://github.com/zengweishuai/ScaleBFM), with pinned source revision,
checkpoint and deployment metadata. The adapter supplies task mode, three
history states and six future reference states at 50 Hz; physics/PD run at
200 Hz. Supported adapter tasks are `squat`, `navigate`, `explore` and `reach` in
MuJoCo; Isaac Lab supports `squat`, `navigate` and `explore`. Exploration uses a
known collision map and simulator state. Raw upstream mode indices are model
conditioning labels, not independently validated skills.

Set up reconstruction dependencies with `pip install -r requirements.txt` in a
Python environment. Isaac Lab requires its own compatible installation;
`scripts/run_isaac.sh` accepts `ISAAC_PYTHON` and `ISAAC_LIBSTDCXX` for local paths.
Fetch the pinned WBC source and weights separately:

```bash
python scripts/bootstrap_sources.py --isaac
python scripts/download_ranges.py \
  https://huggingface.co/WeishuaiZeng/ScaleBFM/resolve/main/checkpoint/humanoid_transformer_m/model_22200.pt \
  checkpoints/scalebfm_m.pt --workers 8 --chunk-mb 4 \
  --sha256 88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3
curl -fL https://huggingface.co/WeishuaiZeng/ScaleBFM/resolve/main/compiled_checkpoint/humanoid_transformer_m/linux/model_22200_tensorrt_metadata.json \
  -o checkpoints/scalebfm_m_metadata.json
```

Expected metadata SHA-256:
`3024f18136314fade2676e22d6eafb1a8b2befa388da55b0885a53fcb1368d16`.
Robot USD layers can be regenerated with `scripts/import_dex3_isaac.py` when
needed; inspect that script's options for the local Isaac runtime.

```bash
.venv/bin/python scripts/run_skill_agent.py --task navigate \
  --scene outputs/worldlabs_scene --run-name atlas_explore --device cuda --video
scripts/run_isaac.sh scripts/run_scalebfm_isaac.py --task explore \
  --scene outputs/worldlabs_scene/scene.usda \
  --plan outputs/worldlabs/final_navigation_plan.json --seconds 38 \
  --run-name atlas_native --headless --fast-exit --device cuda:0
```

The saved native-engine route is scene-bound and included in the local delivery
bundle. MuJoCo exploration also writes its route into its decision report.
Generic Dex3 grasp tools are retained for future configured targets; the skill
router rejects picking without a declared affordance and a validated controller
profile. Static Atlas furniture is not automatically a manipulation target.

The [version 5 figure notes](docs/FIGURES_V5.md) document the native MuJoCo
teaser and system diagram, using the reconstructed office and official G1 URDF.
The figure poses are authored illustrations; they do not change the experimental
results or controller validation status.

The [version 6 figure notes](docs/FIGURES_V6.md) describe native Isaac Sim
Gaussian rendering of the same office, including source splat conversion,
URDF-derived visual poses, and the approximate tabletop occlusion proxy.

See [onboard exploration](docs/ONBOARD_EXPERIMENT.md) for the bounded RGB-D
skill experiment and the source cleanup record.
