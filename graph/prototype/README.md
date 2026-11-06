# Independent Graph prototype

This is Graph-team code. Shared simulator, Visual and Recurrent code are not
changed. The geometry/embedding types here are provisional prototype APIs.
Python 3.11 is required (`tomllib`). No PyG dependency is needed.

Run from the repository root. The existing `.venv` already has all dependencies
on the checked host. For a new environment install
`graph/prototype/requirements.txt` instead of only the shared requirements.

## Inspect and visualize

```bash
.venv/bin/python -m graph.prototype.inspect_env
.venv/bin/python -m graph.prototype.render_diagnostics
.venv/bin/python -m graph.prototype.visualize_graph --geometry outputs/graph/inspection/geometry_seed0.pt
.venv/bin/python -m unittest discover -s graph/prototype/tests -t . -v
```

Geometry tests and model tests use analytic fixtures and need no simulator,
network or expert download. The actual-dataset test skips explicitly if the
dataset is unavailable. Inspection exercises live `physx_cpu`; CUDA was not run.

## Official demonstration preparation

```bash
.venv/bin/python -m mani_skill.utils.download_demo PegInsertionSide-v1 -o outputs/graph/demos
.venv/bin/python -m graph.prototype.replay_demos --traj-path outputs/graph/demos/PegInsertionSide-v1/motionplanning/trajectory.h5 --count 100 --workers 2
.venv/bin/python -m graph.prototype.prepare_dataset --traj-path outputs/graph/demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 --initial-state-source outputs/graph/demos/PegInsertionSide-v1/motionplanning/trajectory.h5
.venv/bin/python -m graph.prototype.validate_expert --horizon 250
```

`replay_demos` delegates to the installed official utility, preserving its
conversion/recording behavior. It resolves paths before the macOS renderer can
change cwd and uses the shared factory's MoltenVK setup. CPU `--workers 2`
means two processes, each running one environment. Rerunning replay replaces
its generated conversion files. Source H5 is preserved by the official utility.

The original H5 is required for the first-state repair. In installed ManiSkill 3.0.1,
CPU replay with `--use-first-env-state` incorrectly stores original state 1 in
the first state row. Extraction restores original state 0, matches its current
observation, checks all T+1 observations and randomized collision dimensions,
and pairs only state t with action t. Original episodes with the same reset seed
are resolved by matching initial observations. Repeated reset groups remain in
one split.

## Smoke and full BC

```bash
.venv/bin/python -m graph.prototype.train --model flat --epochs 1 --max-batches 8 --output outputs/graph/smoke/flat
.venv/bin/python -m graph.prototype.train --model center --epochs 1 --max-batches 8 --output outputs/graph/smoke/center
.venv/bin/python -m graph.prototype.train --model surface --epochs 1 --max-batches 8 --output outputs/graph/smoke/surface

.venv/bin/python -m graph.prototype.train --model flat --epochs 10 --output outputs/graph/runs/flat
.venv/bin/python -m graph.prototype.train --model center --epochs 10 --output outputs/graph/runs/center
.venv/bin/python -m graph.prototype.train --model surface --epochs 10 --output outputs/graph/runs/surface
```

These commands reproduce the pilot's 10 complete epochs. The config defaults
to 30 epochs for subsequent experiments; `--epochs` explicitly overrides it.
Use `--config PATH.toml` for point counts, graph k, hidden sizes, batch size,
optimizer and seed. Dataset-related config values must match extraction.
`--seed` changes initialization/sample order while retaining the saved split.
All three models should use the same options. Checkpoints are selected using
validation action MSE; training never accesses test samples.

To approximate parameter matching, set `flat_width = 192` in a config copy and
use that same config for all three training commands. This option is supported;
the reported pilot does not perform the extra experiment.

## Evaluation and comparison

```bash
.venv/bin/python -m graph.prototype.evaluate --checkpoint outputs/graph/runs/flat/best.pt --output outputs/graph/evaluation/flat.json --rollout --warmstart --horizon 250 --videos 1
.venv/bin/python -m graph.prototype.evaluate --checkpoint outputs/graph/runs/center/best.pt --output outputs/graph/evaluation/center.json --rollout --warmstart --horizon 250 --videos 1
.venv/bin/python -m graph.prototype.evaluate --checkpoint outputs/graph/runs/surface/best.pt --output outputs/graph/evaluation/surface.json --rollout --warmstart --horizon 250 --videos 1
.venv/bin/python -m graph.prototype.compare --results outputs/graph/evaluation/flat.json outputs/graph/evaluation/center.json outputs/graph/evaluation/surface.json
```

Default evaluation horizon is the registered 100 steps. This pilot explicitly
uses 250 because converted expert episodes last 105–219 steps. All models use
identical test seeds, initial snapshots and horizons. Alignment states are
reconstructed by replaying past expert actions to preserve hidden controller and
contact state; the policy receives current geometry only. `compare` rejects differing data hashes, configs, horizons and initial
geometry hashes. `--rollout-episodes N` can limit simulator evaluation; omit it
for all nine test episodes. Omitting `--rollout --warmstart` performs offline
evaluation only. `--videos 1` records a full-task and an alignment MP4.

A selected success example can be reproduced without replacing the comparison:

```bash
.venv/bin/python -m graph.prototype.evaluate --checkpoint outputs/graph/runs/surface/best.pt --output outputs/graph/success_example/surface.json --warmstart --episode-ids 41 --horizon 250 --videos 1
```

Run latency measurements after training processes finish. To refresh latency
and offline metrics while keeping existing rollout results, repeat each
evaluation command with only `--checkpoint`, `--output` and
`--refresh-latency --latency-repeats 100`.

Artifacts are under the ignored `outputs/graph/` directory: dataset and manifest,
source H5/JSON, geometry PNG/PT, diagnostic camera overlays, demonstration phase
graphs, checkpoints, training histories/loss PNG, evaluation JSON, MP4 videos,
comparison Markdown/JSON and error PNG. Source hashes, split IDs, configs,
versions, seeds and CPU timings are recorded. See
[technical report](../../docs/graph_prototype.md) for measured results and limits.

## Follow-up controller diagnostics

The independent v2 experiment and its commands are in [v2/README.md](v2/README.md). V1 checkpoints, data and results remain preserved.

For a small parameter-matched Flat/Surface comparison with 10 training episodes
and three initialization seeds, see [LOW_DATA.md](LOW_DATA.md):

```bash
WANDB_MODE=disabled .venv/bin/python -m graph.prototype.low_data
```
