# Graph team

## Temporal estimation workflows

Install Torch, NumPy, PyYAML and Matplotlib, then run from the repository root.
Set the selected recipe's checkpoint, cache and parent-manifest paths first.
Pose GRU can add checksum-matched raw robot frames; insertion Mamba/GRU requires
those frames by default. Hybrid workflows require the original frozen checkpoints
and the identified validation prediction snapshot.

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/hole_gru/train.yaml
WANDB_MODE=disabled python main.py --config graph/configs/temporal/hole_gru/evaluate.yaml
```

Choose the workflow by its input/output task:

| Workflow | Config under `graph/configs/temporal/` | Guide |
| --- | --- | --- |
| Hole position GRU | `hole_gru/train.yaml`, `hole_gru/evaluate.yaml` | [Hole GRU](../docs/temporal_estimation.md#gru-temporal-object-state-estimator) |
| Peg/Hole pose GRU | `pose_gru/train_hole.yaml`, `train_peg.yaml`, `train_joint.yaml`, `evaluate.yaml` | [Pose GRU](../docs/temporal_estimation.md#joint-peghole-temporal-pose-estimator) |
| Pose/relative/lateral loss ablations | `precision/train_*.yaml`, `precision/evaluate.yaml` | [Insertion objectives](../docs/temporal_estimation.md#gru-insertion-precision-objectives) |
| Insertion Mamba and matched GRU | `insertion/train_mamba.yaml`, `train_gru_matched.yaml`, `evaluate.yaml`, `compare_gru_mamba.yaml` | [Insertion estimation](../docs/temporal_estimation.md#mamba-insertion-precision-estimator) |
| Frozen Mamba + Gate hybrid | `hybrid/evaluate.yaml` | [Hybrid evaluation](../docs/hybrid_estimation.md#frozen-mamba--learned-gate-hybrid) |
| Saved hybrid error analysis | `hybrid/analyze_errors.yaml` | [Error attribution](../docs/hybrid_estimation.md#hybrid-h1-relative-geometry-error-attribution) |

`insertion.py` supports both Mamba and GRU; `hole_gru.py` and `pose_gru.py` name
their separate original estimators. CPU Mamba uses the explicit `torch_reference`
backend; official `mamba_ssm` kernels require a compatible CUDA environment.
The guides retain the measured results, split roles and device limitations.

| Location | Responsibility |
| --- | --- |
| `graph/temporal/` | Models, streaming state and observation features |
| `graph/pose_geometry.py` | Pose and insertion geometry |
| `data_collection/temporal/` | Aligned episodes, windows, splits and frozen feature caching |
| `training/gnn/temporal/` | Config validation, frozen-source preparation, training and losses |
| `evaluation/gnn/temporal/` | Paired evaluation, diagnostics and plots |

Run the related CPU tests, including tiny synthetic training workflows:

```bash
WANDB_MODE=disabled .venv/bin/python -m unittest \
  tests.test_temporal_hole_gru tests.test_temporal_pose_gru \
  tests.test_temporal_insertion tests.test_temporal_precision \
  tests.test_temporal_hybrid tests.test_temporal_hybrid_error_analysis \
  tests.test_temporal_gate -v
```

## Learned temporal update gate

With Torch, NumPy, PyYAML and Matplotlib installed, the original gated MPNN
`best.pt`, its extracted published cache, and an explicit parent-disjoint split
manifest, run from the repository root (full runs require separate approval):

```bash
WANDB_MODE=disabled python main.py --config graph/configs/train_temporal_gate.yaml
WANDB_MODE=disabled python main.py --config graph/configs/evaluate_temporal_gate.yaml
```

Set checkpoint/cache/manifest paths in both configs and the learned checkpoint
path in the evaluation config. The gate trains on GNN-train parents only; the
original GNN validation split is reserved for the final three-method comparison.
See [GNN training](../docs/gnn_training.md#learned-temporal-update-gate) for manifest
preparation, initialization, features, assembly ablation and smoke tests.

## Frozen GNN temporal-memory evaluation

With Torch, NumPy and PyYAML installed, an extracted graph cache and its trained
`best.pt`, run from the repository root:

```bash
python main.py --config graph/configs/evaluate_temporal_memory.yaml
```

Set the checkpoint/cache paths in the YAML. Set `evaluation.visibility_labels`
to `null` if the optional earlier depth-visibility diagnostics are unavailable.
This evaluates the saved validation split without training or generating caches.
See [temporal memory](../docs/gnn_training.md#frozen-gnn-temporal-memory-evaluation)
for observation gates, causal state, metrics and limits.

## Simulator GT points

`graph.utils.gt_nodes.get_gt_node_positions(env)` returns TCP-relative gripper,
fingertip, peg-corner and hole-corner coordinates from simulator GT. See
[`utils/gt_nodes.py`](utils/gt_nodes.py) and
[`examples/visualize_gt_points.py`](../examples/visualize_gt_points.py).

## Observed corners to GNN input

Install `requirements.txt` and `visual/corner_detection/requirements.txt` in a
Python 3.11 environment. From the repository root:

```bash
python examples/build_corner_graph.py --input /path/to/frame.npz \
  --tcp-xyz 0.0 0.0 0.2 --model gated --device cpu
```

Supply the actual observed TCP world position (metres). This is an untrained
forward smoke check. `graph/build_graph/` converts Visual `TensorResult` into a
shared `GraphBatch` consumed by both GNNs, preserving PEG/BOX/HOLE distinctions,
unknown semantics/quality and padding masks. See [graph input](../docs/graph_input.md)
for the contract, confidence policy, edge settings and runnable Python example.

## GNN training

For a downloaded graph cache, extract `graph_cache.tar` into `/workspace/gnn_cache`
and run from the repository root with Torch, NumPy and PyYAML installed:

```bash
python main.py --config graph/configs/train_gated_mpnn_cached.yaml
```

This cache-only factory reads the published split, graph tensors and GT targets.
It checks the inventory and file checksums without raw RGB-D, detection, or cache
regeneration. Change `data.cache_dir` for another extraction location. See
[cache-only training](../docs/gnn_training.md#train-from-a-downloaded-cache).

Install `requirements.txt` and `visual/corner_detection/requirements.txt`, extract
approved episode shards and set `data.root` in the gated config. On the training host:

```bash
python main.py --config graph/configs/train_gated_mpnn.yaml
```

The gated recipe enables `training.compile.enabled: true` and dynamic-shape
compilation on CUDA. CPU/MPS use eager execution. Set the flag to `false` to
disable compilation. The first train/validation calls can spend time compiling.

The common `training.gnn.trainer` handles optimization, validation, tracking and
resume; task implementations own output-specific loss/metrics. Frame detection
and graph caches are generated on first access when `data.use_cache` is true.
See [GNN training](../docs/gnn_training.md) for dataset layout, splits, cache policy,
checkpoint semantics and other GNN extensions. No training or frame caching was
executed locally during this implementation.

`train_gnn.yaml` remains a BasicGNN initialization example (`mode: initialize`);
its node outputs need a suitable task before actual training.

## Existing prototypes

See [the independent surface-geometry prototype](prototype/README.md) and
[the subtask pipeline](subtasks/README.md) for their existing workflows.

## Corner prediction model

The vertex graph model is defined in [gated_mpnn.py](gated_mpnn.py).
Its training objectives live in
[training/gnn/loss.py](../training/gnn/loss.py):
`object21_loss` compares geometry with GT and regularizes endpoint corrections;
`paired_delta_loss` supervises position changes between paired samples.
