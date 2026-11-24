# Development record

The subsequent [merged system update](MERGED_SYSTEM.md) connects this skill layer
to the previous Atlas scene/navigation workflow and provides the common
`agentic-humanoid` command. Its separate report is
`reports/merged_system_validation.json`. The experiments below retain their
original source hashes and results.

This record describes the implemented prototype and actual local experiments on
2026-11-21. All new source, configuration descriptions and documentation are in
English. The work preserves the pre-existing Atlas reconstruction pipeline and
the user's untracked teaser scripts.

**Current result:** the official ScaleBridge execution interface is connected to
online skill references and the released frozen ScaleBFM weights. Standing,
reaching, body-height adjustment and reference replay have run in MuJoCo. A
manually calibrated push program has achieved the independent success criterion.
Live GPT proposals have been executed and independently validated, but none has
qualified for promotion. Validated symbolic task composition is not established.

## Environment and provenance

| Item | Observed configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GB |
| Driver | 580.76.05 |
| Python | 3.12.15 |
| PyTorch execution | 2.11.0+cu128, eager upstream export wrapper |
| MuJoCo | 3.11.0 |
| Upstream commit | `abd6f17c02fe0baabc14709feb8d9ea4959aa621` |
| Checkpoint SHA-256 | `88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3` |
| Metadata SHA-256 | `3024f18136314fade2676e22d6eafb1a8b2befa388da55b0885a53fcb1368d16` |
| Controller mode | Fixed mode 2, pelvis and both wrists; replay uses mode 7 |
| Physics / policy / camera | 200 / 50 / 10 Hz in simulated time |
| Robot hands | Fixed rubber hands, no finger actuators |

The upstream source tree was checked against GitHub's pinned tree. Every runtime
import now checks 78 required source/robot/motion/mesh files against pinned Git
blob hashes. The source pin, checkpoint digest and metadata digest are enforced;
this is not an adapter with unverified placeholder weights.

No official precompiled TensorRT binary was run. The adapter executes the
upstream PyTorch network/export wrapper and preserves `BFMAgent.get_action`,
`BaseEnv` state/action history and the official frame-transform functions. Its
headless simulator adds named joint indexing and metadata torque clipping.

The existing development `.venv` inherits Isaac Lab packages. The isolated
`.venv-skill-lab` was installed against `uv.lock`; installation records are
under `outputs/skill_lab/`. To avoid downloading already installed multi-hundred
megabyte CUDA libraries again, three matching NVIDIA distributions were copied
into the isolated environment after verifying their installed `RECORD` hashes.
That reuse is documented in `local_dependency_reuse.json`; it is not described
as a fresh download of every wheel. The reference installation command remains
`UV_PROJECT_ENVIRONMENT=.venv-skill-lab uv sync --frozen --extra test`.
The isolated environment passed all **32 tests**, including the real-checkpoint
loop and image-only RGB-D localization checks. It reproduced the 3.46 cm reach
error and the successful calibrated push's 5.68 mm final object error. There are
14 upstream TorchScript deprecation warnings; no test failures.

## Implemented code

| Area | Delivered behavior |
|---|---|
| Integration | Official agent/environment loop, eager policy compatibility loader, source/weight pins, headless MuJoCo, online future references. |
| Programs | Strict JSON stage schema, transparent motion primitives, finite timeouts, branches, bounded retries, position/orientation interpolation. |
| Perception | Timestamped RGB-D/proprioception/calibration; labeled oracle poses; controlled known-color/geometry image estimator; no-depth and unknown-state handling. |
| Evaluation | Fixed truth-based goal/footprint, post-exit hold, clearance, fall, contact and timeout checks; separate controller probe scores. |
| GPT | Existing `GPT_API_KEY` compatibility, configurable model/endpoint, isolated worker, multimodal development context, bounded calls/tokens/episodes, sanitized failure metadata, saved responses. |
| Experiments | Separate development/validation/test layouts, paired selection, retained rejections, frozen tests, four ablation paths, response replay. |
| Symbolic layer | Three-valued predicates, exact version binding, candidate/validated separation, add/delete effects, evidence records, contrastive-validation gate, dependency graph, grounded conjunctive BFS and occupancy conflicts. |
| Version control | Content-addressed program/knowledge pairs, atomic current pointer, integrity checks and rollback. |
| Records | Program, knowledge, manifests, source hashes, images/depth, video, references, actual body/joint states, contacts, scores and model usage. |
| Packaging | Installable Python project, hashed dependency lock, environment/replay/probe/run/explore/evaluate/rollback/report commands. |

Entry points and usage are in [SKILL_LAB.md](SKILL_LAB.md). Hardware placeholders
and operator requirements are in [REAL_ROBOT_PREPARATION.md](REAL_ROBOT_PREPARATION.md).

## Actual controller results

| Trial | Actual result | Artifact directory |
|---|---|---|
| Upstream squat reference replay, 8 s | Completed without fall; mean root error 5.88 cm, mean right-wrist error 8.85 cm. | `outputs/skill_lab/replay_004/` |
| Standing | Completed; final root error 3.09 cm, no forbidden contact. | `outputs/skill_lab/probe_stand_001/` |
| Online wrist reach | Hand displacement 25.76 cm; final wrist error 3.46 cm; no forbidden contact. | `outputs/skill_lab/probe_reach_001/` |
| Online body-height adjustment | 8 cm requested range produced 6.83 cm actual vertical range; final root error 3.06 cm. | `outputs/skill_lab/probe_height_001/` |

These results measure physical simulated motion, rather than tensor-shape checks.
The replay uses the upstream motion's world origin; the tabletop is not its task.
Squat scoring permits legitimate low squat heights and is separate from the
manipulation fall-height threshold.

## Auditable GPT failure/revision/validation cycle

The initial push program kept the initialized downward wrist orientation and
commanded the wrist too far forward. It contacted the table and failed the goal.
The API settings in `.bashrc` were accessed only in an isolated worker. A local
proxy setting initially prevented HTTPX initialization; the LLM configuration
now disables inherited proxy settings for this machine. No credentials were
printed or saved.

One live proposal was rejected because its rule used an array for `evidence`
instead of the required object. Subsequent valid proposals changed stages,
body reference, wrist orientation and approach geometry and were executed.

The most useful complete example is:

| Step | Artifact | Outcome |
|---|---|---|
| Initial development failure | `exploration_003/round_00_baseline_development_00/` | Failed, with table-contact violations. |
| Actual model proposal | `exploration_003/proposal_00/response.json` | Explicit wrist orientation and a changed root/approach sequence. |
| Revised physical development trial | `exploration_003/round_00_candidate_development_00/` | No forbidden contact; push still failed. |
| Independent validation | `exploration_003/round_00_candidate_validation_00/` through `_02/` | 0/3 successes. |
| Selection decision | `exploration_003/proposal_00/decision.json` | Rejected: no improvement in successful episodes. |
| Symbolic evidence | `exploration_003/proposal_00/knowledge_after_validation.json` | Proposed rule remains a candidate, with failure evidence. |
| Frozen current pair | `exploration_003/registry/` | Initial program retained. |

A second valid proposal in this experiment also scored 0/3. Earlier valid
proposals are retained in `exploration_002/`. Reduced tracking/contact error
was not treated as task success. No fake model response or substitute controller
was used to manufacture a successful improvement.

Five real API requests have reported usage totaling **50,716 tokens**, including
one diagnostic proposal that was not a selection experiment. Three earlier
worker-initialization failures made no recorded API request. Replay ablations
made no additional API calls. Usage, model, latency and attempts are saved in
individual `call.json` files; credential values are absent.

`final_audit_cycle_v2/` repeats the entire proposal-validation-selection path in the
isolated environment with the final implementation, replaying that original GPT
response instead of making another live request. Its `replay.json` preserves
the response hash. It is an offline response replay with actual physics, not a
new live GPT run.

The final simulator refreshes body/camera kinematics after each integrated policy
step. Earlier transform logs can lag joint state by one 5 ms physics step because
of MuJoCo's integration order; those records retain their original source hashes.
The final audit and `isolated_calibrated_push_final/` use synchronized transforms.
An independent artifact audit passed for these nine episodes, checking required
files, source identity, finite arrays, 50 Hz timing, quaternion normalization,
recorded world/root transforms, image/depth references and the rejection decision.
Its report is `reports/skill_lab_integrity.json`; rerun it with
`.venv-skill-lab/bin/python scripts/audit_skill_lab_delivery.py`.

## Developer calibration, separated from GPT learning

Robot asset inspection showed that the rubber fingers extend about 17 cm beyond
`right_wrist_yaw_link`. Wrist position alone is therefore an inadequate contact
reference. Retaining the initial downward pitch can make the fingers strike the
table even when the wrist target is above it.

Developer trials varied approach height, explicit orientation, wrist/tool
position offsets, root position and withdrawal structure. They are named
`geometry_diagnostic_*` and `root_diagnostic_*`, and are not GPT discoveries.
The failed trials remain on disk. No object state was changed during execution.

`geometry_diagnostic_007` succeeded under the fixed evaluator: final cube-center
error **5.68 mm**, object displacement **15.55 cm**, no fall, no forbidden contact,
hand clear, and the full rotated cube footprint maintained in the target after
program exit. The retained developer template is
`humanoid_skill_lab/configs/push_calibrated.json`, with its own skill name.
Independent layout validation and a separate onboard-vision trial are recorded
under `calibrated_validation/` and `calibrated_onboard_001/`. A single successful
development layout does not establish reliable visual manipulation.

The calibrated program passed **1/3 independent oracle layouts**, with no
forbidden contact in any of those three. Its separate development layout in
that protocol failed the full-footprint condition. Both recorded camera trials
failed; the final controlled RGB-D estimator trial, `calibrated_onboard_002`,
finished 3.48 cm from target with no forbidden contact. Improved top-face/marker
geometry fitting localizes the initialized cube within 1 cm in an integration
test, but that does not establish reliable contact execution.

## Ablations and frozen evaluation

The four small engineering ablations replayed the same actual saved proposal
from `exploration_003/proposal_00/response.json` and ran actual policy/physics
rollouts. They isolate update pathways without model-sampling differences.

| Arm | Executed episodes | Candidate validation successes | Selected version |
|---|---:|---:|---:|
| Fixed | 4 | No candidate; initial validation 0/3 | 1 |
| Skill only | 8 | 0/3 | 1 |
| Symbolic only | 8 | 0/3 | 1 |
| Joint | 8 | 0/3 | 1 |

This is not evidence for an improvement from joint learning: no arm acquired a
validated push rule or promoted a program. The symbolic-only program is unchanged,
so it retains the baseline's physical failures.

`frozen_test_001` evaluated the frozen retained baseline on two held-out push
layouts and scored **0/2**. No test images or per-episode feedback entered a GPT
context, and the baseline snapshot was not changed by that command. These are
held-out layouts, not unseen multi-skill compositions. No generalization result
is claimed.

Machine-readable results and episode hashes are in
`reports/skill_lab_results.json` and `reports/skill_lab_episodes.csv`.
`reports/skill_lab_tracking.png` plots actual development wrist error and contact
samples. `deliverables/humanoid_skill_lab_prototype.tar.gz` contains maintained
source/docs and selected complete trial records; its `BUNDLE_MANIFEST.json`
lists file hashes. Weights, credentials, vendor downloads and environments are
excluded. The complete local run collection remains in `outputs/skill_lab/`.

## Tests and known exclusions

The software suite covers invalid/bounded programs, finite retries, unknown and
stale predicates, fresh poses, full-root transforms against independent SciPy
calculations, future frame timing, preservation of unmodified references,
action history, missing credentials, rule/schema trust boundaries, all-condition
planning, bindings, occupancy conflicts, effect deletion, version invalidation,
contrastive/coverage gates, paired promotion, rollback/tamper checks and split
isolation. One marked integration test executes the real checkpoint through
several official-loop physics steps. Evaluator unit fixtures directly initialize
states; those are software tests and are not reported as rollout successes.

Early debugging artifacts are excluded from acceptance:

- `replay_001`: an evaluator indexing error stopped the run.
- `replay_002` and `replay_003`: a generic standing-height threshold incorrectly
  identified legitimate deep squats as falls.
- `push_initial_001`: unnamed robot geom bodies were missing in the contact log,
  causing false forbidden-contact counts.
- `exploration_001`: LLM worker startup failed due to the shell's proxy setting;
  no successful API call or GPT rollout occurred in that experiment.

Later records preserve exact source hashes so that evaluator changes can be
identified. `skill_lab_results.json` flags excluded episodes explicitly.

## Acceptance against the requested milestones

| Milestone | Status and remaining work |
|---|---|
| M0 environment | Source, checkpoint, metadata, GPU and dependencies inspected; isolated locked environment installed and tested. |
| M1 official baseline | Adapted official-loop reference replay executed with real released weights; official precompiled TensorRT deployment untested. |
| M2 online reference | Standing, wrist reach and height changes physically measured. Root/hand tracking still has task-dependent bias. |
| M3 contact task | Developer-calibrated oracle push succeeds on one layout and 1/3 independent layouts; onboard push success remains unverified. |
| M4 GPT exploration | Live failure/proposal/execution/validation/rejection loop executed. No accepted improvement. |
| M5 symbolic planning | Implemented and software-tested; candidate evidence updated. No empirically validated rule set or physical unseen composition. |
| M6 evaluation | Four small physical ablations, a frozen two-layout test, machine-readable results and a tracking/contact plot. Not a powered research study. |
| M7 hardware preparation | Disabled configuration and operator documentation. No physical camera calibration, hand profile or real G1 execution. |

## Next work

1. Reduce wrist position/orientation bias and cube yaw variation with local,
   measured feedback primitives; validate the calibrated push over a wider
   predeclared development/validation domain.
2. Improve the controlled RGB-D estimator's surface-to-center and occlusion bias;
   keep oracle and camera results separate.
3. Execute matched contrastive experiments and measure rule coverage before
   promoting preconditions/effects. The current loop records suggestions but does
   not automatically execute such interventions.
4. Integrate grounded plan bindings with multi-object perception and execution;
   validate new task compositions physically before claiming transfer.
5. Confirm a finger-equipped asset/controller profile, then add articulated drawer
   manipulation and grasp/place/close experiments using transparent primitives.
6. Obtain real G1 camera/hand/localization configuration and physical calibration.
   Hardware execution remains an operator action after simulation acceptance.

## Native figure revision — November 23, 2026

Version 5 replaces the generated teaser artwork with actual MuJoCo renders of
the existing office reconstruction and a directly imported official Unitree G1
URDF. The robot source is pinned to Unitree's `unitree_ros` commit
`157ff950351e5f535e03c5d85a16b7d594b3dbcc`; the referenced 49 STL meshes and
upstream license are retained with the local delivery.

The deliverable includes four temporal robot exposures, two torso-camera views,
six skill close-ups, and a revised system diagram, with PNG/PDF/SVG exports.
The existing source room is unchanged. Cube/target/drawer props, inverse
kinematics poses, and the narrative route are authored for figure production.
The rendering scripts do not execute ScaleBFM, modify the skill library, or
establish successful contact behavior. Prior experimental results and milestone
status above remain unchanged.

Source and asset hashes, saved MJCF models, pose arrays, reproducible rendering
commands, validation checks, and English captions are documented in
[FIGURES_V5.md](FIGURES_V5.md). Versions 3 and 4 are preserved.

## Gaussian figure revision — November 24, 2026

Version 6 renders the original two-million-Gaussian Atlas room natively in
Isaac Sim 5.1 RTX / NuRec. The SPZ was converted to a NuRec Volume and Gaussian
payload using NVIDIA's pinned 3DGRUT template functions. The original splat file
and source reconstruction remain unchanged; no reconstruction service was called.

Ten native views cover the hero sequence, six skill close-ups, two onboard
insets, and a room-only view. G1 visual triangles come from the existing official
URDF-derived authored poses. One approximate tabletop matte proxy supplies an
occlusion boundary. Full mesh matte proxies were rejected because their geometry
mismatch created artifacts. This changes figure appearance, not physics,
controller validation, symbolic evidence, or measured task outcomes.

The teaser/system PNG, PDF, and SVG exports, native stages, rendering records,
validation, and English reproduction notes are documented in
[FIGURES_V6.md](FIGURES_V6.md).

## Onboard skill experiment — November 24, 2026

Cleaned and pushed the maintained code, archived ten obsolete figure drafts,
and consolidated shared figure layout code. The 36 software tests and both
native figure artifact checks passed.

Executed two bounded RGB-D exploration runs and one frozen evaluation in the
MuJoCo tabletop benchmark: 33 physical episodes, five live GPT calls, and
56,781 reported tokens. Version 2 of the push program improved paired validation
from 0/3 to 1/3 successes and passed 1/2 frozen held-out layouts. Its accepted
change is a 3 cm forward push-reference adjustment; this is parameter refinement.
Remaining failures include loss of fresh camera poses during occlusion. The
saved symbolic rule remains a candidate, and no task-composition, Gaussian-stage
contact, finger-grasp, or hardware success is established.

The reusable program/knowledge pair, rejected proposals, replay configuration,
and evidence digests are saved under `skill_library/push_camera_20261124/`.
See [ONBOARD_EXPERIMENT.md](ONBOARD_EXPERIMENT.md) and
`reports/onboard_20261124_results.json` for the precise scope and results.

## Added scene objects and position sweep — November 24, 2026

Added explicit workcell/task bindings for the existing reconstructed scene,
including a free red cube, target region, and two free colored distractors.
Added tables in two clear room aisles without changing the captured room. An
explicit episode frame handles translation/yaw of the frozen skill, and a
camera-only calibrated workspace filter separates task markers from room clutter.
The evaluator now supports declared support heights and rotated target regions,
and checks distractor disturbance. The original unbound-scene rejection remains.

All 41 software checks passed. The predeclared 15-case sweep executed 12 physical
episodes and succeeded in four: the nominal position in both open controls and
both office aisle placements. Shifted positions remain unreliable. Three
original-table cases were blocked before execution by 16–29 mm object/collider
penetration; no higher-table policy result is claimed. Source reconstruction
identity is unchanged. No GPT calls, skill tuning, or knowledge promotion were
performed during this evaluation. Details are in
[SCENE_POSITION_TEST.md](SCENE_POSITION_TEST.md).
