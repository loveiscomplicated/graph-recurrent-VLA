# Task10-I real SAM2 / robot preparation overlap

Runs the installed SAM2.1 Hiera Small checkpoint on saved 512 RGB-D from both
cameras. It never calls a new grounding/model API or reads object GT. Fixed
boxes come from the already frozen RGB-derived actual predictions. Robot FK is
verified and computed before the timer, from the existing Robot76/schema/assets.

Three implementations share the same loaded model, input, prompts, filter
settings and CUDA clean-mask output contract:

- `legacy_serial`: real SAM2, then direct `CudaRobotFilter.filter_cameras` on the
  caller thread. The previous direct path has no snapshot/worker overhead.
- `serial`: real SAM2, then the new snapshot/worker preparation and filter. This
  isolates ordering against the next variant but is not the old baseline.
- `parallel`: submit current-frame preparation, run real SAM2, then join/filter.

There is no extra GPU synchronization between model and filter/preparation.
Primary wall timing synchronizes CUDA only at the start and end. CPU input
loading, separate FK, model loading and verification are excluded. Snapshot and
dispatch overhead are included when the implementation uses them. No Gate,
Shape-First, Graph, controller/action or simulator is timed.

`static` uses `10i_s1_0921_t080`: both real image encodings and three prompts
(base Peg/Box, wrist Box; missing wrist Peg remains missing). Warmup3 per
variant, then15 samples each with rotated/reversed execution order. All masks
are checked against the same reference outside the timer. Three diagnostic
traces run separately with CPU timestamps; their cold-worker/ray-cache costs
are not the primary timings.

`video` uses all21 frames of `10i_s1_0531`, both cameras, with real
`OnlineVisionAdapter` JPEG preprocessing and real forward SAM2 video memory.
Only initialization detection is replaced with the recorded t0 boxes. Each
variant has independent append-only states and sees `num_frames=t+1`, not the
future sequence length. t0 initialization is separate; t1..5 are warm progression;
t6..20 are15 paired steady frames, not15 identical-input repetitions. Report
these distinct inputs/methods separately. No p95/worst-case/realtime guarantee.

All serial/parallel raw and clean masks are compared bitwise, including every
video frame. A separate real `CurrentGeometry` default-constructor smoke runs
its SAM2 callback, CPU preparation and GPU filter through the clean-mask handoff;
its following Gate/Graph method is intercepted because it is outside this scope.

Existing environment, weights and eager predictor code are reused. FP32 weights,
BF16 autocast, TF32 on, Torch threads4 and BLAS threads1. SAM2's optional `_C`
extension is absent in this installation and its own video postprocessing
fallback warning is retained; no extension installation is attempted.

```bash
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 WANDB_MODE=disabled \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m \
  graph.experiments.robot_sam_overlap_live_10i_20261103_v1.run \
  --output /unique/new/output --mode both --repeats 15
```

Uses the existing `ExperimentLogger`, source/input/checkpoint hashes and a new
output root. No production implementation or paused Task10-I run is modified.
The historical filter-only7.16ms is a different boundary and cannot be added to
or subtracted from these combined measurements.
