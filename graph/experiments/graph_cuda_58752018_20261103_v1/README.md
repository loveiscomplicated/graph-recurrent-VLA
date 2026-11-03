# 10-I CUDA Graph construction

`CurrentGeometry` defaults to resident final-FPS -> CUDA Graph construction.
`tensor_graph` implements current-only Node21/Edge4 and all three edge types.
`tensor_knn`/`tensor_knn_kernels` implement <=128-row compact median KD queries
without CPU neighbor selection. The executed reference is SciPy1.18.1/Linux;
cutoff ties and leaf permutations were compared explicitly. The old CPU Graph
remains the independent reference and explicit CPU-perception path.

`CurrentGeometry.build` returns CUDA Graph tensors plus `TensorRecords` on the
default path. `numpy_graph` exports the legacy CPU/string-ID view; `numpy_records`
exports record dictionaries. Cache/dump writers opt into these readbacks after
construction. The model's `pack_graphs` path consumes the CUDA Graph directly.

Run `python -m graph.experiments.graph_cuda_58752018_20261103_v1.benchmark` with
`--inputs DIR --output NEW_DIR --assets FK_DIR --schema normalizer.json
--baseline PREVIOUS_SPARSE_DEPTH_OUTPUT`. It asserts Vast58752018 and no other
CUDA process, uses ExperimentLogger in disabled/offline mode, performs20 warmups
and100 alternating measurements, and synchronizes CUDA completion inside timers.
It measures ready CPU records, ready CUDA records/views, and CUDA with fresh view
preparation separately. Upstream filtering/gate/Shape-First/dedup/FPS are excluded.
`check_knn` is the independent 225-case tree permutation and neighbor oracle.

The saved512 case contains real hand Peg observations; base depth/calibration
are a clone with empty base masks, and Box is empty. Synthetic tests cover both
objects, empty/short/full clouds, edge gates, identities, provenance rejection,
GPU packing and the existing encoder. These are geometry tests, not a training
or task-success evaluation. See the
[full report](../../../docs/visual-graph-integration/10i_graph_cuda_58752018_20261103.md).
