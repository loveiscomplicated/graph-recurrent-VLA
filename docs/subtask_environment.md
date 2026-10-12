# Independent PegInsertionSide environments

Graph-team interface: `graph.subtasks.env.make_subtask_env`. The shared simulator,
other teams, v1–v4 models and original 91-episode segmentation are preserved.
No model is trained by this workflow.

```python
from graph.subtasks.env import make_subtask_env

env = make_subtask_env("reach", control_mode="pd_ee_delta_pose")
try:
    obs, info = env.reset(seed=100)
    while True:
        action = policy(obs)  # native ManiSkill observation, optional live GT
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print(info["subtask_success"], info["subtask_step"])
finally:
    env.close()
```

One `physx_cpu` environment is supported. ManiSkill observations remain batched
(state mode `[1,43]`), and controller actions accept `[7]` or `[1,7]` in
`pd_ee_delta_pose`. The wrapper returns Python boolean termination flags and a
scalar sparse success reward, following the single-environment Gymnasium API.
The underlying task's full insertion flags and registered 100-step limit do not
terminate an independent subtask. `full_task_success` reports the unchanged task
predicate separately. `subtask_failure` indicates a physical failure predicate;
timeout sets `truncated=True`, `failure_reason="timeout"`, and
`subtask_failure=False`. Success or a physical failure sets `terminated=True`.
Stepping after done or an invalid reset raises an error.

## Explicit source selection and reset semantics

```python
env = make_subtask_env("transport", index="outputs/subtasks/segments/index.json")
obs, info = env.reset(options={"source_episode_id": 29})
```

`source_episode_id` identifies `traj_ID` in that index's converted H5. It is local
to a dataset; `source_uid` identifies a legacy or generated source across datasets.
Middle subtasks require an explicit source ID. Reach with a source ID reproduces
the selected expert's start; Reach with `seed` only performs a fresh reset.
Supplying both a source ID and a seed asserts the source's seed: a conflicting
seed is rejected. The optional `options["subtask"]` must match the wrapper's task.
To run a different task, create another wrapper. Fresh Reach can use `index=None`
when source artifacts are unavailable.

All source resets force a reconfiguration using recorded reset kwargs, seed and
environment configuration, then execute every action in `[0,start_t)` in the real
simulator. Observations, object dimensions/poses, state arrays and actual grasp
contacts are checked against the source. Contact and temporal grasp/alignment
evidence are accumulated throughout this uninterrupted prefix. The prefix never
returns subtask termination; its steps do not enter the learner clock. Workspace
exit, acquisition displacement, sustained grasp loss or replay divergence rejects
the reset. A new accessor/context is created per episode. No extra reset occurs
after replay, and the controller is not reset at the subtask boundary.

Two initialization methods are explicit in `initial_state_ref`:

- `original_s0_then_expert_prefix`: legacy 91 episodes retain the verified
  original-s0 restoration and controller reset that repaired the old conversion's
  s0 recording bug.
- `seed_reset_then_expert_prefix`: new recorder trajectories have correct s0.
  A natural seeded reset must match recorded s0 within `2e-6`; its actual state
  and controller are retained. Snapshot injection at this point was tested and
  introduced later contact/velocity divergence on Linux, so it is not used.

The default method remains action prefix replay. No cached middle-state reset,
pose teleport, synthetic grasp or snapshot-based contact claim is implemented.
Every reset must satisfy the existing start condition, not already satisfy the
selected success condition, and have finite live simulator state. Invalid cases
raise `InvalidSubtaskReset`; evaluation records them without replacing samples.

## Budgets and termination evidence

The thresholds in `graph/subtasks/config.toml` are unchanged. Independent budgets
are in `graph/subtasks/environment.toml` and may be overridden with `max_steps`.
They use twice the original train-only P95 segment duration, rounded up to ten:

| Task | Train P95 actions | Independent max steps |
|---|---:|---:|
| Reach | 63 | 130 |
| Grasp | 22 | 50 |
| Transport | 52 | 110 |
| Align | 47 | 100 |
| Insert | 26 | 60 |

Only the previously implemented workspace, displacement and sustained live-grasp
loss predicates are used. Collision force is diagnostic; no severe-collision,
IK-unreachable or insertion-stall threshold is invented. Reset/replay tolerances
are `0.001` for maximum state/observation component error; state error includes
positions, quaternions and velocities in their native units, not only metres.
Actual geometry equality is checked more tightly (`2e-6`).

`info` includes `subtask`, `subtask_success`, `subtask_failure`, `failure_reason`,
`failure_reasons`, `subtask_step`, `metrics`, `full_task_success`, `prefix_steps`,
`start_state`, `initialization`, `reset_valid`, and `time_limit_reached` after steps.
Native info is retained, including native `elapsed_steps`, which includes prefix
steps; use `subtask_step` for policy evaluation.

## Expert and learner evaluation

```bash
.venv/bin/python -m graph.subtasks.evaluate \
  --output outputs/subtasks/independent_new_run
# Short run and reset/end PNGs:
.venv/bin/python -m graph.subtasks.evaluate --episodes 0 1 2 --render \
  --output outputs/subtasks/independent_new_smoke
```

The evaluator resets each subtask separately and executes only its saved segment
actions. SR terminates at the first success; a diagnostic continuation through
the saved segment endpoint checks that its entire BC interval remains aligned
and its endpoint still succeeds. These continuation steps are excluded from
reported completion steps. Earlier successful termination is expected for Reach
and sometimes Align; early failure/timeout is reported separately.

`replay.json` reports attempted sources, valid starts, successes per valid start,
completion steps, prefix and segment state/observation errors, endpoint success
and failure reasons. `eligibility.json` is a checked sidecar: the original index
is never rewritten. Reach, Grasp and Insert can be verified after valid start,
live independent success, successful endpoint, no segment failure and aligned
state/action replay. Transport and Align remain candidates because a continuous
planner motion and refinement are partitioned geometrically, with possible
intermediate success exits. This is a conservative eligibility decision, not an
assertion that these environments cannot execute.

`evaluation_starts.json` stores explicit seed/source/start-index/reset kwargs,
split, index hash and source hash for paired evaluation. Use its unchanged
geometric `segments/index.json`, not a different index or random replacement:

```python
import json
from graph.subtasks.evaluate import evaluate_policy
starts = json.load(open("outputs/subtasks/independent_verified/evaluation_starts.json"))
env = make_subtask_env("reach", index=starts["index"])
results = evaluate_policy(env, policy, [s for s in starts["starts"] if s["split"] == "test"])
env.close()
```

Manifest hashes are checked before policy evaluation. All models can use the
same manifest and simulator software/backend; cross-host equivalence requires
its own replay verification. `initialization="fresh_reset"` and
`"expert_prefix"` distinguish fresh Reach from source-conditioned evaluation.
Results for the executed hosts are in [subtask_data_generation.md](subtask_data_generation.md).
