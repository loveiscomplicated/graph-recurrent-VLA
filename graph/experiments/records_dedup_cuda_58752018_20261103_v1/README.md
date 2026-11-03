# Task 10-I — GPU current records and greedy3mm deduplication

Production API is `graph.persistent_geometry.tensor_records`:

- `current_records_tensor(observations, coordinate_frame='ee'|'world')` builds
  a columnar `TensorRecords` with aligned measured features and encoded source IDs.
- `deduplicate_tensor(records, .003)` returns resident representative tensors
  and the existing audit. No per-point arrays are downloaded. One-source current
  records only; do not pass an already deduplicated table.
- `numpy_records(records)` explicitly converts to the existing Python dictionary
  contract after deduplication. Default `CurrentGeometry` calls this immediately
  before the next, still-CPU final128-point FPS and graph construction.

Default `triton_fused` uses this path. `PlanA.represent(..., ('SF',),
tensor_records=True)` returns the GPU table; the flag requires the tensor SF-only
path. Existing calls without the flag retain Python-list output via an explicit
post-dedup adapter. CPU perception and the independent `cuda_resident` prototype
retain their existing paths. Robot filtering/gate are outside this change.

Stable source sorting and gathers use Torch. The Triton greedy kernel processes
the earliest unassigned row as representative, assigning only unassigned points
within the strict3mm Euclidean radius and the same object's adjacent3mm cells.
This preserves nontransitive greedy priority and earliest-hit provenance.
Buffers are O(N); the device loop has no per-representative host synchronization.
Dynamic output compaction and one aggregate input validation can synchronize.
CUDA uses float64, correctly rounded cell division/sqrt, and no FP contraction.
Explicit CPU Torch device supports the same API for tests/CPU callers.

The benchmark pairs four endpoints,20 warmups and100 samples, rotating order:

1. `cpu`: CPU observations/flags ready → existing CPU records/dedup.
2. `gpu`: GPU observations ready → synchronized resident GPU records/dedup.
3. `gpu_to_cpu`: endpoint2 plus the actual post-dedup Python compatibility adapter.
4. `legacy_from_gpu`: old observation readback plus endpoint1, starting from the
   same GPU observations as endpoints2/3.

Preparation of packets/observations/view checks is outside timing. Final FPS,
edges, filtering/gate and unused L/H diagnostics are excluded. Same512 real hand
input,1,105 Peg points,0 Box and an empty cloned base view. Full records/provenance
and actual FK/public Graph in EE/World are checked against the preceding frozen
implementation. This is not a two-independent-real-view or nonempty-Box timing.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/source python benchmark.py --inputs /path/to/inputs \
  --reference-source /path/to/frozen/reference --assets /path/to/fk/assets \
  --schema /path/to/normalizer.json --output /path/to/new/output \
  --warmups 20 --repeats 100
```

Shared ExperimentLogger, W&B disabled, no publishing.
Full [report](../../../docs/visual-graph-integration/10i_records_dedup_cuda_58752018_20261103.md).
