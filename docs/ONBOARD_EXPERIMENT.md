# Onboard RGB-D skill exploration

The maintained sensor experiment is a physical MuJoCo tabletop push executed by
the frozen ScaleBFM policy through the ScaleBridge adapter. The skill executor
uses torso-mounted RGB, metric depth, camera calibration, and proprioception.
Simulator object poses are excluded from onboard perception and retained only
in the independent evaluator. Detection assumes known cube geometry and colored
object/target markers; it is controlled RGB-D perception, not general vision.

The Gaussian Isaac Sim figure stages are static visual snapshots. They are not
the contact experiment or evidence of controller execution. The manipulation
benchmark uses the existing controller's fixed-hand G1 asset.

`humanoid_skill_lab/configs/experiment_onboard.json` fixes the data splits before
execution: one development layout, three paired validation layouts, and two
frozen test layouts. All coordinates differ from the previous experiment's
held-out configurations. Limits are three GPT calls, 150,000 reported tokens,
24 exploration episodes, and 22 simulated seconds per episode. API credentials
are loaded from `.bashrc` only by the isolated LLM worker.

```bash
.venv-skill-lab/bin/humanoid-lab explore --bashrc \
  --config humanoid_skill_lab/configs/experiment_onboard.json \
  --skill humanoid_skill_lab/configs/push_calibrated.json \
  --output outputs/skill_lab/onboard_20261124

.venv-skill-lab/bin/humanoid-lab evaluate \
  --config humanoid_skill_lab/configs/experiment_onboard.json \
  --registry outputs/skill_lab/onboard_20261124/registry \
  --output outputs/skill_lab/onboard_20261124_frozen_test
```

Each episode saves its actual program, RGB-D frames, onboard video, observations,
references, states, contacts, independent score, and source/checkpoint hashes.
The content-addressed registry saves program/knowledge pairs and accepts a new
version only after strict paired validation improvement without safety failure
or regression. A saved candidate is not an acquired or validated skill. Frozen
test results must not be fed into further tuning.

The first run completed 18 physical episodes and three live GPT calls without
promoting a new program. `experiment_onboard_occlusion.json` defines a separate
two-proposal follow-up, using only development feedback from that run. It tests
camera-initialized approach followed by phase-relative motion while the hand
occludes the targets. Its maximum is 14 exploration episodes; the same frozen
test layouts remain unused until final selection. Phase-relative motion is a
bounded stroke within this task domain, not continuous visual servoing or broad
goal generalization.

Use the follow-up configuration with `explore` and a fresh output directory.
Save the selected program and separately labeled rejected candidates with:

```bash
.venv-skill-lab/bin/python scripts/save_explored_skill.py \
  --run outputs/skill_lab/onboard_20261124_occlusion \
  --test outputs/skill_lab/onboard_20261124_frozen_test \
  --output skill_library/push_camera_20261124
```

The `--test` argument is optional and must identify a frozen evaluation of the
same selected snapshot. Exporting an initial baseline or a rejected proposal
does not label it as validated.

## Executed result — November 24, 2026

The two runs completed 31 exploration episodes and five live GPT proposals.
Two further episodes evaluated the frozen selected program, for **33 actual
MuJoCo physics episodes** in total. All recorded agent object poses came from
the camera estimator; no oracle object poses entered the skill execution path.
The reported API usage was 56,781 tokens.

Version 2 of `push_to_region_calibrated` was selected. GPT changed the push
endpoint by 3 cm in +x: the target-relative
wrist offset changed from `[-0.085, -0.015, 0.055]` to
`[-0.055, -0.015, 0.055]`. The stage structure stayed the same, so this is
**parameter refinement**, not acquisition of a new program structure.

| Check | Result |
|---|---|
| Baseline paired validation | 0/3 successes |
| Selected version 2 paired validation | 1/3 successes |
| Frozen held-out layouts | 1/2 successes |
| Symbolic preconditions/effects | Candidate evidence only; not validated |

The successful episodes placed the full cube footprint inside the target,
cleared the hand, and held the outcome. Remaining layouts failed because fresh
visual poses became unavailable. These small samples do not establish broad
reliability, unseen task composition, or real-world transfer. No tuning followed
the frozen test results.

The saved reusable pair is in
[`skill_library/push_camera_20261124`](../skill_library/push_camera_20261124/README.md).
Its immutable snapshot is
`97e63e484bb7fe17345fd910883eccf3f8238b686ec09c378559afe8c9b45fc0`.
The replay configuration uses the successful development layout; it does not
silently switch to the CLI's default layout.

The compact per-episode result table is
[`reports/onboard_20261124_results.json`](../reports/onboard_20261124_results.json).
Local RGB-D frames, trajectories, onboard videos, and evaluation records remain
under `outputs/skill_lab/`. The local delivery
`deliverables/onboard_skill_20261124.zip` contains all episode scores and full
records for the selected development success, selected validation success, and
both frozen tests. Bulk recordings are excluded from Git.

## Source cleanup

The obsolete version 1–4 figure drafts, their prompts, and version 3–4 figure
notes were archived before removal from the active source tree at
`outputs/code_archive/legacy_figure_sources_20261124.zip`. The archive contains
ten files and is intentionally excluded from Git. Previously rendered figures
remain under their original output directories. The native MuJoCo and Gaussian
Isaac renderers remain maintained; their repeated typography and system diagram
code now share `scripts/figure_layout.py`.

Large reconstructed scenes, splats, robot downloads, checkpoints, videos, and
delivery archives remain ignored. The remote repository contains code and
English documentation, not API credentials or generated bulk assets.
