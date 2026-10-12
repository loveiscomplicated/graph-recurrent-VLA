# Randomized official expert data

Implementation is Graph-team owned under `graph/subtasks/`. Existing raw/converted
91-episode data, segmentation, 73/9/9 split, Visual/Recurrent code and v1–v4
models/results are preserved. New outputs use separate directories. No BC, RL,
GCSL, hindsight, encoder or reward-model training is performed.

## Implementation inventory

All implementation changes are inside the Graph subtask directory or its docs:

| File in `graph/subtasks/` | Purpose |
|---|---|
| `env.py`, `environment.toml` | Prefix-based single-task Gym wrapper and independent budgets |
| `generate.py`, `requirements-generator.txt` | Official recording, conversion, CPU verification, aggregation and pinned Linux runtime |
| `evaluate.py` | Independent segment replay, conservative eligibility and paired policy evaluation |
| `export_bc.py` | Existing Flat MLP-compatible verified geometry/action tensor export |
| `collection_report.py`, `reproducibility.py` | Measured distributions and exact same-seed array comparison |
| `dataset.py` (extended) | Episode-level split/source leakage checks |
| `visualize.py` (extended) | Explicit initialization provenance and new-recording CPU readback |
| `tests/test_environment.py` | Unit and actual CPU integration coverage, including policy entry point |
| `README.md` (extended) | Runnable entry points and documentation links |

Documentation: `docs/subtask_environment.md` and this file. The condition/spec,
segmentation threshold and prototype/model files were not changed. Generated
artifacts remain under ignored `outputs/subtasks/`; preserve/copy those directories
alongside source code when moving the benchmark.

## Installed API audit

The installed ManiSkill version on the Mac and the collection server is **3.0.1**,
with SAPIEN 3.0.3 and Gymnasium 1.3.0. The installed
`mani_skill/examples/motionplanning/panda/solutions/peg_insertion_side.py` was
inspected alongside the [official source](https://github.com/mani-skill/ManiSkill/blob/main/mani_skill/examples/motionplanning/panda/solutions/peg_insertion_side.py),
[recording documentation](https://maniskill.readthedocs.io/en/latest/user_guide/wrappers/record.html),
and [conversion documentation](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html).
The installed code, rather than the moving main branch, determines runtime behavior.

`solve(env, seed=...)` calls `env.reset(seed=...)` internally and asserts
`pd_joint_pos` or `pd_joint_pos_vel`. The planner captures the passed wrapper
before the solver assigns `env=env.unwrapped`; its `follow_path` and gripper
methods call that captured wrapper's `step`. A `RecordEpisode` subclass therefore
records actual reset s0, every native action and T+1 states/observations without
modifying the installed library. Its reset hook also records randomization before
planning. Planner waypoint labels are not captured or claimed as research phases.

Native joint actions are `[T,8]`; BC EE actions are `[T',7]`. The generator calls
the installed official `from_pd_joint_pos` action converter on two actual CPU
environments, recording the converted target environment with `RecordEpisode`.
It uses natural same-seed resets verified against native s0. This avoids the
installed high-level `replay_cpu_sim --use-first-env-state` path's known s1-in-s0
buffer replacement bug. No state-per-timestep playback is used for conversion
or verification. T' may differ from T: seed 10002 had 205 native and 217 converted
actions. All segmentation is performed on the converted, verified trajectory.

## Collection commands and tested Linux environment

The Mac's existing Python environment supports CPU replay but has no mplib wheel.
Installing mplib on macOS ARM failed on its build dependency. Docker's Linux
environment also failed rendering-device initialization, including an attempted
`render_backend="none"`. These failed attempts were preserved as diagnostic logs.
Official collection was then run on the user-provided Vast.ai instance **55012802**.
Simulation is `physx_cpu`, one environment. The RTX 3090 is used for SAPIEN Vulkan
rendering, not CUDA simulation. Collection uses `sapien_cuda` without camera reads;
RGB visualization uses `sapien_cpu` for CPU readback from the Vulkan renderer,
because `get_picture_cuda` cannot run with CPU-only Torch. Both modes were executed.
CUDA simulator/vectorized execution
was not tested. The Mac uses CPU simulation and the MoltenVK renderer.

Use a **separate Linux Python 3.11 environment** for the pinned collection stack:

```bash
uv venv --python 3.11 .venv-collection
uv pip install --python .venv-collection/bin/python torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-collection/bin/python \
  -r graph/subtasks/requirements-generator.txt
# On the tested server:
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

.venv-collection/bin/python -m graph.subtasks.generate \
  --num-episodes 3 --seed-start 10000 --output outputs/subtasks/generated_smoke_new
.venv-collection/bin/python -m graph.subtasks.generate \
  --num-episodes 50 --seed-start 11000 --output outputs/subtasks/generated_50_new
```

The tested stack uses Torch 2.8.0+cpu, NumPy 1.26.4, mplib 0.1.1 and Pinocchio
2.7.0. The initial NumPy 2.3.5/Pinocchio 3.8.0 combination crashed in native
`mplib.ArticulatedModel` construction; Pinocchio also had an unresolved urdfdom
library dependency. The pinned combination executed successfully and `uv pip check`
passed. The Mac replay environment is unchanged (Torch 2.14.1, Pinocchio 4.1.0).
This is an independent collection environment, not a downgrade of shared requirements.

The existing tested instance uses Python at
`/workspace/peg-subtasks-55012802/.venv/bin/python` and repository root
`/Users/jeong-yunseong/Documents/programming/graph-recurrent-VLA` (the same
absolute path as the local copy, so recorded references remain portable).
Replace `.venv-collection/bin/python` in the examples with that absolute Python
path when reusing the current instance. Do not run the collection CLI through
the Mac `.venv`. The runtime report records the dependency versions at
`generated_50/reports/report_runtime.json`.

Python entry point:

```python
from graph.subtasks.generate import generate_expert_data
summary = generate_expert_data(num_episodes=200, seed_start=10000,
                               output_dir="outputs/subtasks/generated_new")
```

`--seeds 10000 10001 ...` accepts a fixed explicit list. Existing/nonempty output
directories are rejected; choose a new directory. Each seed runs in an isolated
worker with a 300-second limit (`--worker-timeout`), so native crashes/timeouts
become failed records with stage and log evidence. Source failures are preserved;
full-task failure alone does not disqualify earlier successful subtask milestones.
Invalid converted replay is excluded from the segment corpus automatically.

## Data and provenance

```text
generated_50/
  raw/seed_SEED/{trajectory.h5,trajectory.json,initial.json}
  raw/{dataset.h5,dataset.json}
  converted/seed_SEED/{trajectory.h5,trajectory.json}
  converted/{dataset.h5,dataset.json}
  audit/episode_ID.npz
  manifests/{seeds.json,dataset.json}
  segments/{index.json,verified_index.json,metrics_ID.json}
  reports/{collection.json,seed_SEED.json,seed_SEED.log,summary.json}
  reports/independent/{replay.json,eligibility.json,evaluation_starts.json}
```

Failed per-seed raw and converted files remain even when not accepted into
aggregate converted data. H5 `traj_ID` and segment `source_episode_id` share a
dataset-local numeric ID; `source_uid=generated:HASH` distinguishes sources from
`legacy91:ID` and is stable for the same seed/configuration. Native environment
kwargs, reset kwargs, seed, versions, dimensions, object poses, robot qpos, actual
actions, states, source success, conversion outcome and replay checks are saved.

`segments/index.json` holds the unchanged geometric segmentation decision.
`verified_index.json` adds independently evaluated BC eligibility and replay
evidence. A segment stores source seed/UID, `[start_t,end_t)`, reset/prefix
reference, success/status, eligibility, geometric provenance and goal metadata.
`segment_arrays(index,segment)` reconstructs exactly `s_i...s_j` and
`a_i...a_(j-1)` plus observations and audit geometry, with T+1/T checks and GT
pose/qpos comparisons. Source H5 and audits have checked SHA256 provenance.

Splits are fixed from the full attempted seed list, before success filtering,
at episode level. For 50 attempts they contain 40/5/5 seeds. All five segments
from a source remain in its split; duplicate seeds and cross-split reset groups
are rejected. The existing split is unchanged. Reproduction runs are diagnostic
copies, **not additional independent data**; do not merge overlapping seed lists
from separate manifests into different splits.

## Randomization and figures

Installed task reconfiguration samples peg half length uniformly in `[.085,.125]`
metres and half width/height in `[.015,.025]`; hole radius is width plus `.003`.
Hole Y/Z offsets are sampled from `±.5*(half_length-half_width)`. Episode reset
samples peg XY in `[-.1,.1] × [-.3,0]`, box XY in `[-.05,.05] × [.2,.4]`, and
peg yaw in `[30°,150°]` and box yaw in `[67.5°,112.5°]`. The task resets arm qpos
around `[0,π/8,0,-5π/8,0,3π/4,-π/4]` with Gaussian noise of standard deviation
`.02` radians; both fingers start at `.04` metres. No custom physical randomization
is added. Across the 50 seeds, peg position/yaw/dimensions, box and hole pose,
hole offsets/radius and arm qpos all varied. Finger initial positions remained
fixed. Observed peg half lengths were `.085114–.124836` m, half widths
`.015306–.024949` m, and hole radii `.018306–.027949` m. See the component-wise
observed bounds in `generated_50/reports/randomization.json`; these sample bounds
are not claims about the full theoretical distribution.

```bash
.venv/bin/python -m graph.subtasks.collection_report outputs/subtasks/generated_50
.venv/bin/python -m graph.subtasks.visualize \
  --index outputs/subtasks/generated_50/segments/index.json \
  --output outputs/subtasks/generated_50/figures --episodes 0
.venv/bin/python -m graph.subtasks.evaluate \
  --index outputs/subtasks/generated_50/segments/index.json \
  --output outputs/subtasks/generated_50/reports/rendered_example --episodes 0 --render
.venv/bin/python -m graph.subtasks.visualize \
  --index outputs/subtasks/generated_50/segments/index.json \
  --output outputs/subtasks/generated_50/videos --episodes 0 --video --stride 2
```

Use the collection Python/environment for replay if changing hosts produces
state/contact divergence. Randomization JSON reports observed min/max/std and
unique rows for positions, orientations, dimensions, offsets and robot qpos.
Figures include observed distributions, detected segment lengths, collection
stages, existing timeline/geometry visuals and representative reset/end PNGs.
Video rendering is limited to representative episodes.

## Flat MLP compatibility and validation

```bash
# Existing Reach dataset: keeps 73/9/9 and 3,931 actions.
.venv/bin/python -m graph.subtasks.export_bc \
  --output outputs/subtasks/bc/reach_legacy91_new.pt
# New dataset: use independent eligibility evidence with geometric index.
.venv-collection/bin/python -m graph.subtasks.export_bc \
  --index outputs/subtasks/generated_50/segments/index.json \
  --eligibility outputs/subtasks/generated_50/reports/independent/eligibility.json \
  --output outputs/subtasks/generated_50/bc/reach.pt
```

The exporter reuses `GTAdapter`, existing surface sampling and `GeometryState`,
and preserves episode splits. It emits the existing `load_dataset` format with
actions, episode IDs, timesteps, geometry and configuration. No feature/model
interface is redesigned. The unchanged Flat MLP normalization and forward path
was executed on both legacy and new-50 Reach exports (`[8,1915] → [8,7]`, finite
output). The new export has 2,084 actions, with 1,663/225/196 action samples from
40/5/5 train/validation/test episodes. Normalization uses only train episodes.
The next task can use `graph.prototype.train --model flat --dataset PATH`; no
training command was run in this work. Tiny smoke manifests may have too few
eligible sources for nonempty validation/test; use a larger corpus for training.

```bash
SUBTASK_INTEGRATION=1 .venv/bin/python -m unittest \
  graph.subtasks.tests.test_subtasks graph.subtasks.tests.test_environment -v
SUBTASK_GENERATION_INDEX=outputs/subtasks/generated_50/segments/index.json \
  .venv-collection/bin/python -m unittest \
  graph.subtasks.tests.test_environment.GeneratedDatasetIntegrationTests -v
.venv-collection/bin/python -m graph.subtasks.reproducibility \
  outputs/subtasks/generated_vast_smoke3 outputs/subtasks/generated_smoke_final \
  --output outputs/subtasks/reproducibility.json
```

Reproducibility compares actual randomization, native/converted observations,
actions and all state arrays, not H5 file bytes or timestamps. Unit tests are
supplemented by actual CPU full replay, independent prefix/reset/step execution,
conversion, geometry bounds, save/load and episode split checks.

## Executed results

Existing 91 episodes, Mac CPU: **455/455 valid resets and independent expert
successes**; all saved segment endpoints also succeed, all segment state and
observation replay errors are zero. Original Reach has 91 verified episodes /
3,931 actions. Grasp and Insert have 91 independently verified episodes each in
the eligibility sidecar; Transport and Align retain 91 candidates each. Average
independent completion steps: Reach 34.01, Grasp 16.81, Transport 33.84, Align
25.41, Insert 19.96. These differ from saved segment lengths because evaluation
terminates on first success.

Final smoke, Linux CPU, seeds 10000–10002: source full success 2/3, native replay
valid 3/3, conversion full success 2/3, converted replay valid 3/3 and full success
2/3. Detection: Reach/Grasp/Transport/Align 3/3, Insert 2/3. Independent starts
and success: 3/3 for the first four tasks and 2/2 valid Insert starts; all errors
zero. BC eligibility is Reach 3, Grasp 3, Insert 2, Transport/Align candidates 3
each. Failed seed 10002 is retained with its successful earlier milestones.

### Main collection: 50 new seeds, Linux CPU

Seeds **11000–11049**, `outputs/subtasks/generated_50`: 50 native trajectories
(7,584 actions) and 50 verified converted trajectories (7,851 actions).
Native full success is **39/50 (78%)**; native replay validity is **50/50**.
Conversion was attempted for **50/50**, with converted full success **35/50
(70%)**; converted replay validity is **50/50**, and replayed full success is
**35/50**. Replay validity means reproducing the saved converted trajectory,
not reproducing the native joint trajectory with identical timesteps/states.
The converter can change motion and lose full-task success. No such source is
silently presented as a successful full insertion demonstration.

| Subtask | Detected / attempted | Valid start / attempted | Expert success / valid start | BC-eligible episodes (actions) | Avg completion steps |
|---|---:|---:|---:|---:|---:|
| Reach | 50/50 | 50/50 | 50/50 | 50 (2,084) | 32.06 |
| Grasp | 50/50 | 50/50 | 50/50 | 50 (841) | 16.82 |
| Transport | 50/50 | 50/50 | 50/50 | 0; 50 candidates | 35.14 |
| Align | 50/50 | 50/50 | 50/50 | 0; 50 candidates | 27.30 |
| Insert | 35/50 | 35/50 | 35/35 | 35 (723) | 20.66 |

All **235 detected segments** have valid independent starts, successful expert
replay, successful saved endpoints, and zero prefix/segment observation/state
error. The other 15 Insert segments are unresolved; they are excluded from
start manifests, not counted as 15 failed valid-start rollouts. No threshold
was adjusted to increase coverage. Transport/Align execute successfully but
remain provisional for BC because expert phase/boundary semantics are unresolved.

Source full failures: seeds 11001, 11006, 11007, 11009, 11017, 11024, 11029,
11031, 11032, 11045, 11047. Additionally, source-successful seeds 11014, 11015,
11016 and 11037 lose full success after conversion. Their raw/converted data,
final official success and earlier valid milestones are retained. These are
observed predicate outcomes, not a claim that a specific unmeasured collision
or planner threshold caused each failure. Per-seed solver logs and geometric
metrics support further diagnosis. Collection workers took 558 seconds total;
the supervised collection including independent evaluation took about 14 minutes.

The smoke and main corpus contain **53 distinct new seeds**, not counting
same-seed reproduction runs: Reach 53 / 2,208 actions, Grasp 53 / 895 actions,
Insert 37 / 770 actions; Transport/Align have 53 candidates each. Legacy sidecar
verification adds Reach 91 / 3,931, Grasp 91 / 1,530 and Insert 91 / 1,816.
These counts are separately host-verified; no cross-runtime merged evaluation
corpus is claimed.

### Tests, reproducibility and host limitation

- Mac: `tests_final.log`, **32 cases, 31 passed and one generated-data skip**,
  including actual CPU reset/prefix/learner action/success/timeout tests. The
  physical-failure flag-routing test uses fault injection after a real CPU step;
  it is not empirical calibration of a collision/failure threshold.
- Existing prototype compatibility: **12 + 34 passed**, recorded in
  `tests_compatibility.log`; shared and v1–v4 tests remain unchanged.
- Linux: generated-dataset integration passed for both the 3-seed and 50-seed
  corpora (`tests_reproduction_linux.log`, `tests_generated50_linux.log`).
  This supplements, rather than replaces, full 235-segment CPU execution.
  The final 50-seed integration also executes `evaluate_policy` on all five
  tasks and checks rejection of mismatched paired-manifest hashes
  (`tests_policy_export_linux.log`). Verified Grasp and Insert exports also
  completed: `bc/grasp.pt` (50 episodes / 841 actions, 40/5/5) and `bc/insert.pt`
  (35 episodes / 723 actions, 30/3/2). Their export replays include the complete
  saved segment and check every state/observation against its source.
- Same-seed reruns reproduce initial randomization, every native/converted
  action, observation and state array **exactly for 3/3 seeds**. Evidence:
  `outputs/subtasks/reproducibility.json`. Diagnostic reruns are not new samples.
- New Linux trajectories are **not all reproducible in the existing Mac stack**.
  The Mac smoke check rejects some Grasp and later prefixes, with maximum
  recorded prefix state error about `.231` in native mixed state units.
  Evidence: `generated_smoke_final/reports/mac_independent/replay.json` and
  `tests_generated_mac_final.log`. Do not relax tolerances or use these failed
  starts. New-data SR evaluation/export must use the tested Linux environment;
  cross-host replay equivalence remains unresolved.

Artifacts include a ready-to-load `generated_50/bc/reach.pt`, paired
`reports/independent/evaluation_starts.json` (235 valid starts), distribution and
length figures, smoke reset/end PNGs in `generated_smoke_final/reports/rendered_linux`,
and verified successful/failed full-expert annotated MP4s in
`generated_smoke_final/videos_verified` (episodes 0 and 2, zero replay error).
Older failed attempts and Mac diagnostic videos are preserved but are not the
verified benchmark. All final artifacts were copied back to the local repository.

Reach is ready for an unchanged Flat MLP BC baseline on the Linux benchmark.
All five environments support independent rollout from verified sources.
Transport/Align BC readiness and cross-runtime replay remain the explicit
unfinished research/validation items. No model was trained, and no Visual,
Recurrent, graph architecture, shared simulator or old segmentation threshold
was changed. The user's Vast.ai instance is left available; its collection
supervisor job completed normally and is not automatically restarted.
