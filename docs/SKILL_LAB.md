# Humanoid Skill Lab

This is a research prototype for editable G1 skill programs executed through
the pinned ScaleBridge environment and a frozen ScaleBFM policy in MuJoCo.
The original Atlas reconstruction tools are integrated into the same project.
Online controller probes accept their exported scene with `--scene`; push
experiments use an independent tabletop benchmark. Neither path submits an Atlas
job or requires Isaac Lab. See [MERGED_SYSTEM.md](MERGED_SYSTEM.md) for the common
entry points and physical merge checks.

Read [DEVELOPMENT_RECORD.md](DEVELOPMENT_RECORD.md) for measured results,
excluded debugging runs, and the current acceptance status. Software tests and
symbolic examples do not establish robot manipulation capability.

## Installation

Run commands from the repository root. Python 3.12, Linux, NVIDIA EGL and the
CUDA 12.8 PyTorch wheels are the reference deployment. The lock is in `uv.lock`;
it includes artifact hashes and uses the official PyTorch CUDA 12.8 index.

```bash
UV_PROJECT_ENVIRONMENT=.venv-skill-lab uv sync --frozen --extra test
python scripts/bootstrap_sources.py
python scripts/download_ranges.py \
  https://huggingface.co/WeishuaiZeng/ScaleBFM/resolve/main/checkpoint/humanoid_transformer_m/model_22200.pt \
  checkpoints/scalebfm_m.pt --workers 8 --chunk-mb 4 \
  --sha256 88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3
curl -fL \
  https://huggingface.co/WeishuaiZeng/ScaleBFM/resolve/main/compiled_checkpoint/humanoid_transformer_m/linux/model_22200_tensorrt_metadata.json \
  -o checkpoints/scalebfm_m_metadata.json
.venv-skill-lab/bin/humanoid-lab doctor
.venv-skill-lab/bin/python -m pytest -q
```

The official medium checkpoint and metadata digests are checked before policy
use. Seventy-eight required upstream source, robot, mesh and motion files are
checked against Git blob hashes from commit
`abd6f17c02fe0baabc14709feb8d9ea4959aa621`. Changed or missing files stop execution.
The loader never downloads or runs an unpinned controller automatically.

The existing `.venv` is a development environment that inherits Isaac Lab
packages. Its exact observed versions are recorded separately from the clean
locked environment. An editable installation is intended; keep vendor sources
and weights in this repository. `pip install -e '.[test]'` is also supported,
but use the lock command above for the recorded dependency resolution.

## Commands

Choose a new output directory for every run. Existing runs are not overwritten.

```bash
# Reference-motion replay through the adapted official loop.
.venv-skill-lab/bin/humanoid-lab baseline --task replay \
  --output outputs/skill_lab/my_replay --seconds 8

# Continuous program-generated controller references and measured error.
.venv-skill-lab/bin/humanoid-lab probe --task stand \
  --output outputs/skill_lab/my_stand
.venv-skill-lab/bin/humanoid-lab probe --task reach \
  --output outputs/skill_lab/my_reach
.venv-skill-lab/bin/humanoid-lab probe --task height \
  --output outputs/skill_lab/my_height

# A physical push trial, first with explicitly labeled simulator poses.
.venv-skill-lab/bin/humanoid-lab run \
  --skill humanoid_skill_lab/configs/push.json \
  --observation-mode oracle_debug --output outputs/skill_lab/my_push

# Controlled RGB-D perception; object poses are estimated from images/depth.
.venv-skill-lab/bin/humanoid-lab run \
  --observation-mode onboard_vision --output outputs/skill_lab/my_vision_push

# GPT proposal -> schema validation -> development -> independent validation.
.venv-skill-lab/bin/humanoid-lab explore --bashrc \
  --output outputs/skill_lab/my_exploration

# Freeze the current program/knowledge pair and evaluate held-out layouts.
.venv-skill-lab/bin/humanoid-lab evaluate \
  --registry outputs/skill_lab/my_exploration/registry \
  --output outputs/skill_lab/my_frozen_test

# Select a previously committed pair, after checking its content hash.
.venv-skill-lab/bin/humanoid-lab rollback \
  --registry outputs/skill_lab/my_exploration/registry --snapshot HASH

# Aggregate saved results; plots require the optional plots dependency group.
UV_PROJECT_ENVIRONMENT=.venv-skill-lab uv sync --frozen --extra test --extra plots
.venv-skill-lab/bin/humanoid-lab report --plot

# Maintained source, documentation and selected raw evidence, without weights.
python scripts/package_skill_lab.py
```

`--device cpu` is available for functional checks. `--no-video` disables video
encoding but retains onboard images, depth and observations. No command launches
a real robot. A failed task normally exits successfully as a completed
experiment and records `success: false`; a dependency or execution error has a
nonzero process exit status.

## GPT configuration and boundaries

The short-lived LLM worker reads `OPENAI_API_KEY`, falling back to the user's
existing `GPT_API_KEY`. `OPENAI_MODEL` overrides the explicit model selection
in `configs/llm.json`; `OPENAI_BASE_URL` optionally overrides the endpoint.
The client has no hardcoded model fallback. Without credentials or a configured
model it errors instead of returning a synthetic proposal.

With `--bashrc`, only the worker uses an interactive shell to load `.bashrc`.
The supervisor and simulator remove inherited API-key/token variables. Shell
stdout/stderr is captured and discarded because arbitrary startup scripts may
print private data. SDK errors are reduced to error type and HTTP status;
authentication headers, keys and environment dumps are never saved. The local
LLM configuration sets `trust_env: false` because this machine's shell exports
an HTTPX-incompatible `socks://` proxy URL. Change that setting for a deployment
that requires an explicitly configured working proxy.

GPT receives development images, timestamped agent observations, reference and
actual hand motion, task scores, prior development attempts and remaining
budget. Validation returns aggregate selection feedback. Frozen test results
are produced by a separate command with no proposal call path.

`experiment.json` bounds rounds, worker calls, total reported tokens and physical
episodes. A conservative prompt allowance plus the maximum response length is
reserved before a call. Unknown API usage is recorded as unknown and blocks
further live proposals. These are research execution budgets, not a monetary
billing cap. Rejected JSON and failed physics episodes are retained.

For offline response replay, add `proposal_replay: ["path/to/response.json"]`
to an experiment configuration and set `rounds` accordingly. The response is
validated against the starting program; physical episodes still run. Replay
records the original response's SHA-256 and does not claim a new API call or
charge the original usage twice. Replay is useful for paired ablations.

## Control integration

The local adapter subclasses the official `BaseEnv`, headless
`MujocoSimulator`, and `BFMAgent`. It executes:

```python
obs_dict = env.reset()
target_joint_positions, action = agent.get_action(obs_dict)
obs_dict = env.step(target_joint_positions, action)
```

The released `.pt` checkpoint is loaded into the upstream network and export
wrapper. This is an **eager PyTorch backend**, not an official precompiled
TensorRT artifact. To import `BFMAgent` without TensorRT, the compatibility loader
removes exactly its unconditional `import torch_tensorrt` AST node; it retains
the upstream class and `get_action` body. No fake TensorRT module is installed.
The upstream deployment code in `vendor/` is not edited.

The adapted simulator replaces the GUI and positional-index assumptions, uses
name-resolved joints/actuators, and applies metadata torque limits. This clipping
differs from upstream's unclipped PD assignment and is recorded here. Robot and
object configuration/velocity assignment is confined to initialization. After
initialization, only the frozen policy's joint targets and PD torques move them.

Policy rate is 50 Hz; MuJoCo/PD rate is 200 Hz. State/action histories have three
entries; reset broadcasts the calibrated initial state and zeros action history.
Future offsets `[0,1,2,3,4,5]` are frame counts: 0 through 100 ms at 50 Hz. They
are passed unchanged to the official wrapper, which handles feature timing.
The provider interpolates positions with smoothstep and orientations with SLERP.
It retains calibrated references for unmodified bodies instead of filling zeros.

World references have shape `(1,6,14,3)` and `(1,6,14,4)`, with metadata body order
and `wxyz` rotations. Official observation functions subtract the current root
position and apply the inverse **full root rotation**, including roll/pitch.
Reference forcing is disabled: simulator root position supplies localization.
Derived body/camera transforms are refreshed after integration so their
timestamps match the current joint/root state.
Mode 2 conditions the pelvis and both wrists; it stays fixed during an episode.
The current G1 asset has fixed rubber hands and no finger actuator interface.

## Program interface

`skills/schema.py` rejects unknown fields/primitives, nonfinite values, dangling
transitions and invalid rotations. There are at most 16 stages and 32 transitions.
Each stage has a finite timeout; the global transition budget bounds retries and
cycles. Worst-case programmed duration is at most 240 seconds; the independent
episode deadline may be shorter. Motion reference peak speed is limited to
0.5 m/s. This is a conservative software exploration envelope, not a hardware
safety certification.

| Primitive | Implementation and limits |
|---|---|
| `wait`, `observe` | Retain current targets; perception continues at 10 Hz. |
| `move_hand_relative` | Latch a fresh object/target pose on stage entry; add a world-axis offset; interpolate wrist position and optional world orientation. |
| `move_hand_world` | Interpolate wrist toward an explicit world position and optional orientation. |
| `retract_hand` | Add an offset to the current reference, not an assumed actual hand pose. |
| `set_body_height` | Change pelvis height within 0.60–0.85 m. |
| `move_root_world` | Change pelvis reference within the declared bounded local workspace. |

These are geometric tools; none contains a complete push, grasp or drawer
procedure. Conditions are three-valued. Unknown never satisfies a branch.
Missing/stale object poses cause explicit failure. A completed program does not
imply successful contact or task completion.

## Perception and evaluation

The simulation-only camera is attached to `torso_link`, offset `(0.10,0,0.24)` m,
with a 50-degree downward angle, 75-degree vertical FOV and 320×240 RGB-D output.
This mounting is an experimental design, **not physical G1 calibration**.
Optical coordinates are right/down/forward. Camera extrinsics use ideal simulated
proprioceptive root localization and known robot kinematics.

`oracle_debug` labels object/target poses as `oracle`. `onboard_vision` estimates
them using known red/green colors, known upright cube geometry and metric depth.
It cannot receive oracle object poses; the estimator raises if they are present.
The estimator fits a visible top rectangle for cube center/yaw, assuming an
upright cube, and fits the flat target's edge bounds. Roll/pitch are not estimated.
Color confidence is a pixel-count heuristic, not a
calibrated probability. Without depth it does not invent metric object poses.
Hand occupancy remains unknown because no tactile/gripper detector is present.

The independent evaluator reads MuJoCo truth. Its success rule requires the
entire cube's rotated footprint inside the target, supported height, wrist
clearance of at least 10 cm, one second of maintained result after program exit,
no fall, no timeout and no forbidden robot/environment contact. Foot-floor and
hand-cube contacts are allowed; table contacts are forbidden. Cube initialization
settling alone never counts as a push success. The evaluator is outside the
candidate's editable interface. Controller probes have separately named scores
and never claim manipulation success.

## Symbolic knowledge and planning

Rules bind to an exact skill version. Candidate rules and invalidated old rules
cannot be planned with. The finite grounded BFS checks all preconditions jointly,
all object bindings, accepted program versions and explicit add/delete effects.
It rejects simultaneous holding/free-hand states. Unknown preconditions block
application. Graph edges are explanatory; an enabling edge is insufficient to
establish an executable plan. Reachability is a separate geometric screen.

Supporting validation episodes alone do not establish necessary preconditions.
The rule validator requires matched contrastive experiments for each proposed
precondition and a minimum declared domain coverage. The current experiment
loop records proposed contrastive experiments; it does not automatically execute
them or upgrade candidate rules to validated status. Only empirically validated
rules should be supplied to real task composition. Synthetic planner tests are
explicit software checks.

Promotion compares paired validation configurations, requires more successful
episodes, permits no success regression and rejects candidate safety/timeout
failures. A content-addressed snapshot commits a program and its corresponding
knowledge together. The current pointer is replaced atomically. Rejected rules
retain their supporting/failing evidence alongside the proposal.

## Records and experiment splits

Every successfully initialized episode writes:

| Artifact | Contents |
|---|---|
| `manifest.json` | Source/checkpoint/metadata hashes, seed, split, scene, camera, backend and rates. |
| `skill.json`, `knowledge.json` | Versions actually used for execution. |
| `observations.jsonl`, `frames/` | Agent-visible images, depth, poses, timestamps, calibration and source labels. |
| `references.npz` | Timestamped world/root future references and quaternions. |
| `robot_states.npz` | Joint/root states, actual body positions, policy actions, joint targets and tracking errors. |
| `events.jsonl` | Stage transitions and 200 Hz contact samples, sorted by time. |
| `onboard.mp4`, `overview.mp4` | Onboard and external review videos, when enabled. |
| `evaluation_trace.json`, `metrics.json` | Independent truth-based result and detailed scoring trace. |

Proposal directories retain raw model JSON, sanitized call usage/latency/retry
metadata, candidate validation decisions and post-validation rule evidence.
Run artifacts and credentials are Git-ignored. Sanitized reports are maintained
under `reports/`. Each manifest hashes local source files to distinguish code
changes during development even when the Git commit is unchanged.

Development layouts feed proposals. Validation layouts select versions, returning
only aggregates to GPT. Frozen test layouts are consumed by `evaluate` and are
never included in exploration contexts. Split checks reject identical physical
configurations even when their seeds differ. Current test layouts exercise
held-out pushes; they are **not unseen multi-skill task compositions**.

`configs/ablation_fixed.json`, `ablation_skill_only.json`,
`ablation_symbolic_only.json`, and `ablation_joint.json` provide the four requested
controls. Identical response replay can isolate the update pathways. An unchanged
physical program in the symbolic-only arm is expected to have the same physical
push score; composition benefits require validated rules and further tasks.
