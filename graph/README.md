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

## GNN initialization skeleton

From the repository root, with Python 3.11 and `requirements.txt` installed:

```bash
python main.py --config graph/configs/train_gnn.yaml
```

Select `model.name: basic_gnn` for [BasicGNN](basic_gnn.py), or
`model.name: vertex_edge_gated_mpnn` for [VertexEdgeGatedMPNN](gated_mpnn.py).
The latter has its own runnable example config:

```bash
python main.py --config graph/configs/train_gated_mpnn.yaml
```

Both use the same trainer. This entrypoint initializes the selected model, selects a
Torch device, and opens a shared `ExperimentLogger` run. It does not load data,
train, evaluate, or save checkpoints. Model dimensions in the config are example
values to replace when the team chooses its inputs and objective.

For local logging without W&B:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/train_gnn.yaml
```

See [the initialization guide](../docs/graph_prototype.md#basicgnn-initialization-skeleton)
for config fields, recorded metadata and remaining implementation steps.

## Existing prototypes

See [the independent surface-geometry prototype](prototype/README.md) and
[the subtask pipeline](subtasks/README.md) for their existing workflows.

## Corner prediction model

The vertex graph model is defined in [gated_mpnn.py](gated_mpnn.py).
Its training objectives live in
[training/gnn_loss.py](../training/gnn_loss.py):
`object21_loss` compares geometry with GT and regularizes endpoint corrections;
`paired_delta_loss` supervises position changes between paired samples.
