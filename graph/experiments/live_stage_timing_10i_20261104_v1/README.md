# Task10-I main-live stage timing (2026-11-04)

Measurement only. `run.py` executes the original
`graph.plan_a_bc.rollout.run_episode` at two native CUDA512 cameras. Actual
Terra grounding, installed SAM2, default robot preparation/filter, wrist gate,
mask projection, Triton PCA/features, Shape-First, tensor packets/observations,
3mm dedup, final128 FPS, edges and Graph all execute. Only the policy **after
Graph** is replaced with a zero native-motion policy through the original codec;
this is a runtime diagnostic, not a manipulation-success evaluation.

Three causal episodes use the same reset seed and source snapshot:

* `natural`: outer image/sensors/Graph completion timing, with no extra internal
  CUDA barriers. Synchronous grounding and CPU preparation receive host spans.
* `detailed`: CUDA completion barriers at stage boundaries. Nested exclusive
  times prevent double counting. The CPU hull worker remains asynchronous;
  its time overlaps the caller and must not be added to the caller's total.
* `natural_warm`: repeat the natural timing after the same trace has executed,
  to distinguish runtime preparation from ordinary completion latency.

The final run uses frame0 for initialization, tracking frames1..8 for warmup,
and frames9..38 for30 steady samples. Terra exact-RGB caching is retained across passes; real
network calls, retries and cache hits are recorded separately. Missing API
credentials fail rather than presenting DINO timing as Terra timing.
The comparison includes bit-exact fingerprints of all five Graph arrays for
each matched frame, exported after the Graph timer stops.

`profile.py` wraps the current Graph function's original AST statements from
`surface_edges_tensor(...)` through the final Edge4 attribute assignment. This
separates all edge generation from node/metadata construction without keeping
a second Graph implementation. Unknown source boundaries fail explicitly.

Run on the existing Vast58752018 CUDA environment from an immutable source
snapshot with `source_manifest.json`:

```bash
WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m graph.experiments.live_stage_timing_10i_20261104_v1.run \
  --runtime /workspace/task10g_f1_abc_20261031/full \
  --output /absolute/path/to/new-output --frames 39 --warmup 8 --seed 42
```

Supply credentials through the environment or an existing
private dotenv file; never include them in source snapshots or experiment
configuration. The shared `ExperimentLogger` records local identity/artifacts;
no publishing is performed. Results belong in `docs/visual-graph-integration/`.

The final controlled run uses a new, experiment-local `TRITON_CACHE_DIR` for
its first pass, leaving existing users' caches untouched. Host spans record
actual Triton specialization/cache-load calls and FPS CUDA Graph creation;
they are diagnostic subspans and are not added to stage times a second time.
