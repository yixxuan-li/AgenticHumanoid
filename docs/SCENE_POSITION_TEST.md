# Objects and position tests in the reconstructed room

The saved camera-guided push program is frozen for this robustness test. New
task bodies are composed into a copy of the existing MuJoCo office scene;
the source reconstruction, its furniture, and its visual assets are unchanged.
The scene composition also supports an open workcell for comparison.

Each case adds one movable red 8 cm cube, a green square target, and two movable
blue/yellow distractors. Some office cases add a small 0.575 m table in the aisle;
the existing-table cases use the reconstructed central table at the previously
assumed 0.74 m height. Props, added tables, masses, and friction are authored
simulation settings, not recovered physical properties or furniture articulation.

`humanoid_skill_lab/configs/scene_positions.json` predeclares five workcells and
three positions per workcell, for 15 cases:

| Workcell | Room | Support | Robot heading |
|---|---|---|---|
| `open` | Open comparison workcell | Added table, 0.575 m | 0 degrees |
| `open_yaw90` | Open comparison workcell | Added table, 0.575 m | 90 degrees |
| `office_east` | Existing captured office | Added aisle table, 0.575 m | 90 degrees |
| `office_west` | Same captured office | Added aisle table, 0.575 m | 90 degrees |
| `office_original_table` | Same captured office | Existing central table, assumed 0.74 m | 0 degrees |

Object centers in each workcell are `[0.455, -0.185]`, `[0.435, -0.155]`, and
`[0.475, -0.215]` metres. The target is 0.13 m forward from each object.
The nominal open case is a sanity check; it is not a new independent hold-out.
Office placements are variants of one captured room, not independent real rooms.

## Execution and scoring

The same released ScaleBFM checkpoint executes the skill through the maintained
ScaleBridge loop. The skill parameters and stage graph are unchanged. An explicit
`episode` frame rotates/translates offsets and wrist orientations with the
declared workcell heading. The original and executed program hashes are saved.
This coordinate port is distinguished from running the original world-frame JSON
at an unrelated origin. No GPT calls, parameter search, or rule promotion occur.

Onboard RGB-D and proprioceptive camera transforms provide object estimates.
Known color/geometry detection is limited to a declared workcell volume, using
static support/frame calibration rather than simulator object poses. This keeps
nearby green chairs and colored room clutter out of the marker detector. The
camera still may fail under occlusion; missing poses remain unknown.

The independent evaluator checks full-footprint containment in target axes,
support height, hand clearance, a one-second held result, falls, timeouts,
forbidden robot/environment contacts, and distractor motion beyond 2 cm. These
criteria are fixed before the sweep. Table height comes from declared task
configuration instead of the old fixed benchmark height.

Initial robot/object penetrations deeper than 8 mm block a case before a rollout.
Blocked cases are reported separately and count as unsuccessful cases, not as
executed physics episodes. No placement is moved after an outcome to improve
the score. The source scene fingerprint must match before and after the suite.

```bash
.venv-skill-lab/bin/python scripts/evaluate_scene_positions.py \
  --output outputs/skill_lab/scene_positions_20261124
```

Per-case directories contain the actual composed scene, skill, onboard RGB-D,
video, trajectories, references, contacts, metrics, and source/checkpoint hashes.
The suite summary separates the scene/workcell groups, executed episodes, blocked
initializations, and errors. These are MuJoCo contact trials; the Gaussian Isaac
Sim figure stages are not used as physical execution evidence.

## Actual results

The frozen sweep completed 12 physical episodes. Four succeeded. Three further
cases were blocked by the initialization audit and were not executed.

| Workcell | Successful / executed | Blocked |
|---|---|---|
| Open comparison | 1/3 | 0 |
| Open, rotated 90 degrees | 1/3 | 0 |
| Captured office, east aisle | 1/3 | 0 |
| Captured office, west aisle | 1/3 | 0 |
| Captured office, original table | Not executed | 3 |

The nominal position succeeded in all four executed workcells. All near/left
positions missed full-footprint containment. All far/right positions failed
because a fresh camera pose was unavailable. No executed episode fell, timed
out, or registered a forbidden contact. Distractor motion stayed below the
declared 2 cm threshold; approximately 5–6 mm came from initial settling.

The original-table object placements intersected collision boxes
`atlas_collider_0123` / `atlas_collider_0729` by approximately 16–29 mm. These
blocked cases do not measure policy performance at the higher surface. The
existing table needs collision-geometry review before such manipulation trials;
the suite did not edit the source furniture or move the blocked objects after
seeing outcomes.

The source reconstruction fingerprint remains
`d991e9181b6d9d428171375e2ab7ecdafa795f6f0cabbbd90eeb2581873ee2fa`.
All available agent object poses were labeled `camera`; the skill parameters
were frozen and no GPT calls or symbolic updates occurred.

See the [case matrix](../reports/scene_positions_20261124.png) and
[complete machine-readable result](../reports/scene_positions_20261124.json).
Local `outputs/skill_lab/scene_positions_20261124/scene_examples.png` combines
actual recorded final physics states rendered from the room overview with
the original onboard RGB observations. It uses no generated robot poses.
