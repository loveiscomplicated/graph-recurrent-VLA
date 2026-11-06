# Graph team

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

Install `requirements.txt` and `visual/corner_detection/requirements.txt`, extract
approved episode shards and set `data.root` in the gated config. On the training host:

```bash
python main.py --config graph/configs/train_gated_mpnn.yaml
```

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
