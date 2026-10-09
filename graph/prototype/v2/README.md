# Graph prototype v2

Graph-team-owned controller and failure experiments. V1 encoders, data, checkpoints and results are preserved. Read [technical report](../../../docs/graph_prototype_v2.md) and [measured tables](../../../outputs/graph/v2/comparison/results.md).

Run from repository root with the existing `.venv`. No new packages are required. All experiments below use CPU physics and PyTorch CPU. Rendering on macOS uses the repository's MoltenVK setup. CUDA/MPS training is not executed.

## Diagnosis first

```bash
.venv/bin/python -m graph.prototype.v2.diagnostics --audit --output outputs/graph/v2/diagnostics/audit.json
.venv/bin/python -m graph.prototype.v2.diagnostics --original-expert --stages full --output outputs/graph/v2/diagnostics/original_expert.json
.venv/bin/python -m graph.prototype.v2.diagnostics --expert --stages full alignment --output outputs/graph/v2/diagnostics/expert.json
for model in flat center surface; do
  .venv/bin/python -m graph.prototype.v2.diagnostics --checkpoint outputs/graph/runs/$model/best.pt --stages full grasp transport alignment --output outputs/graph/v2/diagnostics/${model}_v0.json
done
.venv/bin/python -m graph.prototype.v2.analyze --output outputs/graph/v2/diagnostics/offline_v0.json
.venv/bin/python -m graph.prototype.v2.analyze --summary --output outputs/graph/v2/diagnostics/summary_v0.json
.venv/bin/python -m graph.prototype.v2.analyze --probes --output outputs/graph/v2/diagnostics/one_step_probes.json
```

`--horizon` defaults to 250 learner steps. Expert prefixes are past recorded EE actions; their observation reconstruction must agree within .001. Original expert mode is `pd_joint_pos` and cannot use converted warm-start indices. `--episode-ids` restricts held-out rollouts.

## Flat ablations and equal-condition training

Training refuses to overwrite an existing v2 checkpoint. Use fresh output names for reruns. Dataset and episode split are exactly v1's. `--config` defaults to this directory's `config.toml`. Epoch budget, seed, configuration, normalizers, dataset SHA and dependency versions are stored in checkpoints.

```bash
MPLCONFIGDIR=outputs/graph/v2/.mplconfig .venv/bin/python -m graph.prototype.train --model flat --epochs 30 --output outputs/graph/v2/training/flat_duration_rerun
.venv/bin/python -m graph.prototype.v2.train --model flat --state proprio --epochs 30 --output outputs/graph/v2/training/flat_proprio_rerun
for model in flat center surface; do
  .venv/bin/python -m graph.prototype.v2.train --model $model --state goal --gripper regression --epochs 30 --output outputs/graph/v2/training/${model}_goal_rerun
done
```

`--state none` exactly initializes the v1 encoder/head and adds no state. `proprio` adds qvel9; `goal` adds qvel9 + 22 derived geometric residual features. Existing qpos, gripper, poses, sizes are retained inside each encoder. `--gripper binary` is implemented and unit-tested but **not used in the measured main comparison**. It applies hard ±1 at inference and component BCE/7 during training. `--max-batches` is only a smoke option, recorded explicitly.

The completed runs use names without `_rerun`. Configuration was selected using Flat validation error, before main rollout results, and written to `outputs/graph/v2/training/selection.json`.

## Evaluation, independent fresh configurations, diagnostics

```bash
for model in flat center surface; do
  .venv/bin/python -m graph.prototype.v2.analyze --checkpoint outputs/graph/v2/training/${model}_goal/best.pt --output outputs/graph/v2/evaluation/${model}_goal.json
  .venv/bin/python -m graph.prototype.v2.diagnostics --checkpoint outputs/graph/v2/training/${model}_goal/best.pt --stages full grasp transport alignment --output outputs/graph/v2/diagnostics/${model}_goal.json
  .venv/bin/python -m graph.prototype.v2.diagnostics --checkpoint outputs/graph/v2/training/${model}_goal/best.pt --stages full --new-seeds $(seq 2000 2029) --output outputs/graph/v2/diagnostics/${model}_fresh.json
done
.venv/bin/python -m graph.prototype.v2.recovery --reports outputs/graph/v2/diagnostics/flat_goal.json outputs/graph/v2/diagnostics/center_goal.json outputs/graph/v2/diagnostics/surface_goal.json --output outputs/graph/v2/diagnostics/distribution_v2.json
.venv/bin/python -m graph.prototype.v2.recovery --oracle --output outputs/graph/v2/diagnostics/oracle_feasibility.json
.venv/bin/python -m graph.prototype.v2.compare
.venv/bin/python -m graph.prototype.v2.figures
.venv/bin/python -m graph.prototype.v2.report
.venv/bin/python -m graph.prototype.v2.visualize_goals --output outputs/graph/v2/figures
.venv/bin/python -m graph.prototype.v2.video --checkpoint outputs/graph/v2/training/flat_goal/best.pt --episode-id 29 --stage alignment --output outputs/graph/v2/videos/flat_alignment_episode29.mp4
```

All JSON traces include pre/post poses, qpos/qvel, gripper state, contact forces, raw/applied action, actual TCP movement, relative hole pose and flags. Offline metrics and warm-start SR are separate from full-task SR. Latency excludes GT extraction. Fresh seeds must be unique and absent from the training dataset; fresh runs only support full-task start.

## Tests

```bash
.venv/bin/python -m graph.prototype.v2.overfit --output outputs/graph/v2/diagnostics/overfit_real.json
.venv/bin/python -m unittest discover -s graph/prototype/tests -t . -v
.venv/bin/python -m unittest tests.test_maniskill_env tests.test_graph_gt_nodes -v
.venv/bin/python -m unittest graph.prototype.v2.tests.test_v2 -v
```

V2 integration tests intentionally require the real saved audit/expert/v0/main diagnosis artifacts and overfit results. Analytic fixture geometry is used only for mathematical/model unit tests, never for reported training or control metrics.

## Embedding and state bypass

```python
# Loaded checkpoint provides train-only normalizers, unchanged v1 encoder.
geometry = session.adapter(env)
qvel = env.unwrapped.agent.robot.get_qvel()
graph = geometry_normalizer.apply("surface", geometry, graph_builder)
embedding = policy.encoder(graph)             # [B, 128]
shared = state_normalizer.apply(geometry, qvel, "goal")  # [B, 31]
action = policy.action_head(torch.cat((embedding, shared), -1))  # [B, 7]
```

This is a feasibility interface inside the Graph team directory, not a final shared Visual/Recurrent contract. Model/simulator API claims are verified against installed ManiSkill 3.0.1.

## Complete fresh-output reproduction of the main comparison

This block trains and evaluates the checkpoints it just created. It uses the existing, validated expert-prefix report and the already-selected common setting; it does not re-select hyperparameters. Choose a new RUN directory if it already contains checkpoints. Old artifacts are preserved.

```bash
RUN=outputs/graph/v2/reproduce
for model in flat center surface; do
  .venv/bin/python -m graph.prototype.v2.train --model $model --state goal --gripper regression --epochs 30 --output $RUN/training/${model}_goal
  .venv/bin/python -m graph.prototype.v2.analyze --checkpoint $RUN/training/${model}_goal/best.pt --output $RUN/evaluation/${model}_goal.json
  .venv/bin/python -m graph.prototype.v2.diagnostics --checkpoint $RUN/training/${model}_goal/best.pt --stages full grasp transport alignment --output $RUN/diagnostics/${model}_goal.json
  .venv/bin/python -m graph.prototype.v2.diagnostics --checkpoint $RUN/training/${model}_goal/best.pt --stages full --new-seeds $(seq 2000 2029) --output $RUN/diagnostics/${model}_fresh.json
done
.venv/bin/python -m graph.prototype.v2.compare --root $RUN
```

The reproduced root has main metrics; its unrun historical ablation rows are N/A. Baseline diagnostics, optional video/plot/report generation and regression commands above target the original measured `outputs/graph/v2` artifacts.
