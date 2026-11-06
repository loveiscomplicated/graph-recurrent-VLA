# Graph team

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
