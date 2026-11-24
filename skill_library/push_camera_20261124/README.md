# Saved onboard experiment skill

Status: **validation_selected**. Observation mode: **onboard_vision**.

`skill.json` and `knowledge.json` are the registry-selected pair. Candidate programs and their decisions are retained under `candidates/`. Saving a file does not establish acquired capability or validated symbolic rules. See `manifest.json` and `evidence.json` for selection and physical execution evidence.

Replay in the same bounded simulation benchmark:

```bash
.venv-skill-lab/bin/humanoid-lab run --skill skill_library/push_camera_20261124/skill.json --config skill_library/push_camera_20261124/replay_development.json --observation-mode onboard_vision --output outputs/skill_lab/new_replay
```
