# PegInsertionSide subtask evidence pipeline

Graph-team code for [Specification v0](../../docs/subtask.md). No new model,
training, randomized reset or GCSL implementation. Technical decisions and
measured results: [subtask_segmentation.md](../../docs/subtask_segmentation.md).

Run from repository root, using the existing Python 3.11 environment and datasets:

```bash
.venv/bin/python -m pip install -r graph/subtasks/requirements.txt
# Live CPU replay: original verified s0 + controller reset + unchanged actions.
.venv/bin/python -m graph.subtasks.audit
# Train-only evidence first. Final thresholds live in graph/subtasks/config.toml.
.venv/bin/python -m scripts.subtasks.inspect
.venv/bin/python -m scripts.subtasks.segment
.venv/bin/python -m scripts.subtasks.report
.venv/bin/python -m graph.subtasks.visualize
.venv/bin/python -m graph.subtasks.visualize --render --episodes 0 1 2 13 29 38 54 90
# Annotated expert replay videos (one MP4 per requested episode).
.venv/bin/python -m graph.subtasks.visualize --video \
  --output outputs/subtasks/videos --episodes 0 1 2 13 29 38 54 90 --fps 10
# Use --stride 2 or 4 to save every second/fourth rendered state while replaying
# every action; this reduces file size without changing the simulated trajectory.
SUBTASK_INTEGRATION=1 .venv/bin/python -m unittest graph.subtasks.tests.test_subtasks -v
.venv/bin/python -m unittest discover -s graph/prototype/tests -t . -v
.venv/bin/python -m unittest graph.prototype.v2.tests.test_v2 graph.prototype.v3.tests.test_v3 graph.prototype.v4.tests.test_history -v
.venv/bin/python -m unittest tests.test_maniskill_env -v
```

Audit resumes existing `episode_ID.npz` files after source checksum verification.
Use a **new `--output` directory** for a fresh replay; deleting the source dataset
is never needed. Default source metadata is `outputs/graph/dataset.json`, whose
existing 73/9/9 split is preserved. Audit `--episodes ID ...` allows partial runs;
segmentation requires all manifest episodes. Other CLIs accept `--manifest`,
`--audit`, `--output`, and `--config` as shown by `--help`.

Outputs (ignored by git):

- `outputs/subtasks/audit/`: complete T+1 live GT, contact validity/forces,
  original T actions, exact planner-derived peg-local goals, replay checks.
- `outputs/subtasks/segments/index.json`: five records per source episode,
  boundaries/status, source checksums, original split, prefix restoration refs,
  and goal metadata. `metrics_ID.json` includes each state and success runs.
- `outputs/subtasks/reports/`: train threshold distributions, coverage table,
  full per-episode coverage/length/ambiguity/provenance validation.
- `outputs/subtasks/figures/`: eight timelines, eight geometry contact sheets,
  eight real simulator boundary contact sheets, all-episode duration plot.
- `outputs/subtasks/videos/`: annotated MP4 expert replays and `videos.json`.
  The overlay identifies phase, source state index, gripper command, live contact,
  official success, lateral error, and insertion depth. MP4 rendering uses the
  same original-s0 restoration and exact action replay as the audit.

Read a segment without duplicating the full source trajectory:

```python
from graph.subtasks.dataset import load_segments, segment_arrays
index = load_segments('outputs/subtasks/segments/index.json')
reach = index['episodes'][0]['segments'][0]
sample = segment_arrays(index, reach)
assert len(sample['states']) == len(sample['actions']) + 1
# Actions are exactly a[start_t:end_t]; geometry includes the terminal state.
```

Use the same conditions in a live simulator. Initialize the accessor after
reset/original s0 restoration and reset temporal context per episode:

```python
from graph.subtasks.live import LiveSubtaskAccessor
from graph.subtasks.specs import SubtaskSpec, load_config
from graph.subtasks.conditions import ConditionContext

cfg = load_config()
accessor = LiveSubtaskAccessor().reset(env)
ctx = ConditionContext()
align = SubtaskSpec('align', cfg)
# t=0 is an initialized state, contact caches are unknown.
state = accessor(env, contact_valid=False)
metrics = align.compute_metrics(state)
ctx.update(metrics, 0, cfg, align.align_raw(metrics))
# Following env.step(action), call accessor(..., contact_valid=True), compute
# metrics and update ctx with t=1,2,... exactly once per control timestep.
# ctx.start_t is the independent subtask's starting timestep for timeout checks.
```

`check_start`, `check_success`, `check_failure` return booleans;
`compute_metrics` returns the numeric evidence; `failure_reasons` returns reasons.
Grasp/Align temporal conditions need a context. Geometry boundaries are estimates,
not recovered oracle phase labels. Only Reach is currently marked `bc_eligible`;
other extracted segments remain candidates for independent rollout validation.
