# Task 10-I — sparse CPU depth lookup for Graph free-space checks

Production change: `persistent_geometry.graph.free_space_violation` obtains image
shape from `depth_raw_mm` and gathers only projected raw pixels, then applies the
existing float64 mm→m and nonpositive→NaN rule. No dense depth getter or persistent
cache. Camera projection, pixel floor/bounds, strict6mm test and all edge rules stay
unchanged. This is a CPU inefficiency fix, not a GPU Graph port.

The benchmark uses the preceding frozen Graph baseline tensors/packets and exactly
the same128-point CPU input boundary.20 warmups/100 samples, alternating EE/World,
with no primary hooks. Separate5 warmups/20 diagnostics reuse the preceding Trace
and explicitly report zero full-depth getter calls. All Graph tensors/metadata/audit
must match the saved before result exactly. The previous timing is a separate run,
not a newly paired CPU baseline.

Same actual512 hand input, Peg128/Box0, empty cloned base;131 nodes/1,006 edges.
Upstream GPU preparation, CPU return, FK, robot filter/gate and inference are outside
timing. This is not a two-independent-real-view or nonempty-Box latency measurement.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/source python benchmark.py --inputs /path/to/inputs \
  --assets /path/to/fk/assets --schema /path/to/normalizer.json \
  --baseline /path/to/saved/before --output /path/to/new/output \
  --warmups 20 --repeats 100 --diagnostic-warmups 5 --diagnostic-repeats 20
```

Existing Vast58752018 environment, shared ExperimentLogger, W&B disabled,
no publishing. Full [report](../../../docs/visual-graph-integration/10i_graph_depth_lookup_58752018_20261103.md).
