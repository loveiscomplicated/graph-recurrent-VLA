# Observed-corner graph input

This Graph-team prototype connects the standalone Visual corner detector to GNNs:

```text
RGB-D + camera calibration -> visual.corner_detection.detect_tensors
                          -> graph.build_graph.GraphBuilder (+ observed TCP)
                          -> GraphBatch -> BasicGNN / VertexEdgeGatedMPNN
```

It does not define a simulator-wide or cross-team interface. The detector and
shared simulator remain unchanged. GT is not used for graph construction.

## Run a smoke check

From the repository root, use Python 3.11, `requirements.txt`, and
`visual/corner_detection/requirements.txt` in the active environment:

```bash
python examples/build_corner_graph.py --input /path/to/frame.npz \
  --tcp-xyz 0.0 0.0 0.2 --model gated --device cpu
```

The NPZ has `rgb`, `depth`, `intrinsic_cv`, `extrinsic_cv`, and optional `masks`,
as in [corner detection](corner_detection.md). Supply the actual TCP world
position in metres; the numbers above are placeholders. Use `--model basic`
for BasicGNN, `--use-confidence` to enable the quality input, and `--device mps`
or `cuda` on a supported host. This executes an **untrained** model and prints
shapes, not accuracy or task success. It does not create an experiment or upload.

The gated training command is now implemented; configure data first as described
in [GNN training](gnn_training.md):

```bash
WANDB_MODE=disabled python main.py --config graph/configs/train_gated_mpnn.yaml
```

## Python use and ownership

```python
import torch
from visual.corner_detection import detect_tensors
from graph.build_graph import GraphBuilder, GraphSettings, ConfidenceSettings
from graph.gated_mpnn import VertexEdgeGatedMPNN

result = detect_tensors(rgb, depth, intrinsic_cv, extrinsic_cv, device="cpu")
builder = GraphBuilder(
    settings=GraphSettings(intra_k=None, cross_k=None),
    confidence_settings=ConfidenceSettings(),
)
# Each result is one unbatched camera/frame. Supply synchronized TCP separately.
tcp = torch.tensor([[tcp_x, tcp_y, tcp_z]], dtype=torch.float32)
graph = builder([result], tcp)
model = VertexEdgeGatedMPNN(num_semantic_ids=16)
prediction = model(graph)
```

The builder must run outside `torch.inference_mode()`: it detaches and clones
inference tensors into ordinary tensors that autograd can save for GNN parameter
gradients. Detector gradients are not propagated. Build once for cached inputs,
or build under ordinary execution before running the model under `no_grad()`.
Results move to the TCP tensor's device; `.to(device)` moves a complete graph.
The builder consumes only `xyz`, `object_id` and `metadata`, so importing it does
not import OpenCV or the detector. Raw dataset selection/storage stays in
`data_collection/`; graph transformation stays in `graph/build_graph/`.

## GraphBatch contract

All floating tensors are finite float32, coordinates are in metres, IDs are
int64, and masks are bool. Every field is on the same device.

| Field | Shape | Meaning |
| --- | --- | --- |
| `xyz_world` | `[B,N,3]` | Observed world positions; padding is zero |
| `tcp_xyz_world` | `[B,3]` | Synchronized observed TCP position; world axes |
| `object_type` | `[B,N]` | `PAD=-1`, `PEG=0`, `BOX=1`, `HOLE=2` |
| `node_valid` | `[B,N]` | Actual observed node, independent of confidence |
| `semantic_id` | `[B,N]` | Externally established identity, or `-1` unknown |
| `semantic_valid` | `[B,N]` | Whether semantic identity is known |
| `confidence` | `[B,N]` | Evidence quality `[0,1]`; zero if invalid |
| `confidence_valid` | `[B,N]` | Quality was calculable, independent of node validity |
| `edge_index` | `[2,E]` | Directed sender/receiver indices into flattened `[B*N]` |
| `edge_type` | `[E]` | `PEG=0`, `BOX=1`, `HOLE=2` within type; `CROSS=3` across types |
| `relative_xyz` | `[E,3]` | Sender minus receiver, world-aligned axes |
| `distance` | `[E,1]` | Euclidean length of `relative_xyz` |

`GraphBatch.validate()` checks shapes, dtypes, devices, mask/sentinel conventions,
edge endpoints, relation types, duplicate edges and consistency of geometry.
`validate(num_semantic_ids=...)` additionally checks known IDs against a model's
vocabulary. The field names intentionally replace the old model-local
`graph.gated_mpnn.VertexBatch`; callers must migrate. Gated embeddings now have
three object types and four edge types, so old checkpoints are not directly
compatible. No checkpoint migration is provided by this prototype.

### Node identity and box/hole distinction

By default detector IDs `0/1` become PEG/BOX, unchanged in meaning. `contour_hole`
is evidence about image topology, not an automatic HOLE classifier. No aperture
classification is implemented here. An upstream classifier may provide explicit
`object_types=[labels_per_result, ...]` with PEG/BOX/HOLE values. The builder
validates those labels but cannot verify the classifier's physical correctness.
It does not read GT or infer hidden corners to fill gaps.

The detector does not establish physical corner identity. The default semantic
IDs are all `-1`; ordering never defines identity. A caller with established IDs
can supply `semantic_ids=[ids_per_result, ...]`, using `-1` for unknown entries.

### Edges and batching

The default is a complete directed graph over observed nodes, without self
edges. Intra-type edges carry their object type; different types carry CROSS.
This is a small-corner baseline, not a conclusion that complete graphs are best.

Set `GraphSettings(intra_k=..., cross_k=...)` for nearest senders per receiver:
intra k selects the same type, cross k selects across all other types combined.
Each side uses up to k available nodes; edges need not be reciprocal. Equal-
distance ties use input order, so kNN tie cases are not permutation invariant.
No physical cuboid-edge topology is inferred from unordered detections.

The builder pads to the largest sample, or explicit `pad_to`, and offsets edges
by `sample_index * N`. Empty detections are allowed; an all-empty batch gets one
padding slot per sample. An empty result list is rejected. No edges connect
padding or different samples. Padding never has valid semantics or confidence.

## Confidence policy

`ConfidenceSettings` defaults are provisional and recorded explicitly by callers
when used in an experiment:

| Setting | Default |
| --- | --- |
| `residual_scale_px` | 2.0 |
| `plane_reference` | 3.0 |
| `source_reference` | 3.0 |
| residual / plane / source weights | 0.6 / 0.3 / 0.1 |

For available metadata, compute:

```text
residual score = exp(-(line_residual_px / residual_scale_px)^2)
plane score    = min(supporting_planes / plane_reference, 1)
source score   = min(sources / source_reference, 1)
confidence     = sum(weight * available score) / sum(available weights)
```

Absent/None evidence is excluded from both numerator and denominator. Malformed,
negative or nonfinite values raise an input error. If no positive-weight evidence
is available, return zero with `confidence_valid=False`; calculated zero quality
has `confidence_valid=True`. The detector's default plane/source counts are used
as provided, not treated as additional independent observations. Correlated
candidate merges are why source weight is low.

This is a heuristic quality score, **not a calibrated probability**. No mapping
from score to position-error probability, calibration by detection method, or
held-out accuracy validation has been performed. Defaults need saved-frame
error analysis before being used for a production confidence policy.

## GNN integration

Both models accept `model(graph)` and consume the builder's edge geometry;
neither reconstructs topology or reads detector metadata.

- `VertexEdgeGatedMPNN` has `use_semantic_ids=False` and `use_confidence=False`
  defaults, also supported in trainer model config. When enabled, unknown
  semantics contribute zero features, and unknown confidence gets neutral weight
  1. A detected node is never padding merely because its score is unknown or zero.
  BOX/HOLE have separate embeddings and edge types, but share the existing
  target-side message layers/pooling to predict hole pose/radius. `hole_observed`
  means actual HOLE input, `box_observed` means BOX input, and
  `target_support_observed` means either. Existing `hole_anchor` is a target-support
  centroid, not a measured aperture position; mean-confidence outputs describe
  effective pooling weights (1 when quality use is disabled/unknown).
- `BasicGNN` adapts to its dense representation internally, requiring `in_dim=8`
  and `edge_dim=8`. Node features are TCP-relative xyz (3), type one-hot (3),
  confidence and confidence-valid (2). Both quality channels are zero unless
  `use_confidence=True`. Semantic IDs are available in GraphBatch but this
  baseline does not use them. Edge features are relative xyz (3), distance (1),
  relation one-hot (4). Its explicit `model(x, adj, edge_attr)` call still works
  with caller-selected dimensions. The graph call updates only valid nodes and
  computes messages only for real edges; padding outputs are zero.

Gated node encoding, updates and attention exclude padding. Basic normalization
also excludes padding. Any downstream node-level objective or metric must select
`graph.node_valid`; graph-level empty samples produce finite predictions from
model priors, not observed geometry. No success claims follow from this behavior.
Outputs remain model-specific: BasicGNN returns node tensors; gated returns
Object21 geometry. The [common trainer](gnn_training.md) now connects gated Object21 supervision;
other prediction layouts supply their own tasks.

## Optional features for completed corner graphs

`VertexEdgeGatedMPNN(extra_node_dim=5)` accepts a second argument,
`extra_node_features[B,N,5]`. The default `extra_node_dim=0` retains the existing
call and state-dict layout; opting in changes the first encoder layer and needs
a new checkpoint. Features must be finite and on the graph device.

The Graph team's `graph.temporal.corner_tracking.graph.CompletedGraph` is a
separate experimental contract. It retains `observed`, observation age and XYZ
uncertainty alongside an internal GNN geometry adapter. The adapter's presence
mask means included support, so its counts/centroids are **completed support**,
not physical visibility or raw detector counts. It must not be published as an
observed GraphBatch cache. Raw detector graphs and their API remain unchanged.
Track IDs are not semantic IDs. See [tracked corner completion](corner_tracking.md#tracked-corner-completion-before-gnn).

## Verification

```bash
python -m unittest tests.test_graph_builder tests.test_gated_mpnn \
  tests.test_gnn_trainer tests.test_corner_detection -v
```

On 2026-11-06 KST, macOS/Python 3.11/Torch 2.14.1: 39 tests passed on CPU.
Checks cover edge geometry, complete/kNN construction, masks, offsets, unknown
attributes, box/hole distinction, permutation equivariance, unchanged predictions
and parameter gradients under extra padding, mixed empty batches and finite
backward. A synthetic RGB-D frame runs through the actual CPU detector, builder,
both models and an optimizer step. Its dummy target tests gradient plumbing only.
This does not measure learned performance or use a simulator.

Additional local device checks executed both models' MPS forward/backward on a
padded graph, plus the CLI example on a synthetic frame for both models on CPU
and MPS. All completed with finite outputs/gradients. CUDA is unavailable on this
host and was not executed. No simulator rollout, real-dataset training,
confidence calibration, or online publishing was performed.
