# Task 10-I robot filter optimization (2026-11-03)

Independent, opt-in computation optimization of the frozen
`graph_latency_10i_20261103_v2.Candidate(9)`. The prior sources, saved actual SAM2
masks, simulator, shared Graph files and suspended processes are read-only inputs.
No inference APIs, model execution, training, rollout, package installation,
instance creation or publishing are part of this experiment.

`protocol.json` freezes the baseline, contracts and selection rules. The old 26
outcome-selected scenarios (18 unique Parents, overlapping across strata) are
all regressed at 128/512/768/1024. `timed.json` contains all original 52 high-size
conditions plus two minimum-SHA scenes per stratum at 128/512. `extra.json` uses
availability and a predeclared hash rule to choose additional Parents; only two
eligible Parents existed. Repeated frames and repeated timings are not independent
population samples. `full_masks.json` deliberately stresses a full-image union.

## Private contract

`SparseGraph` scopes process-local patches around `CurrentGeometry.build`; do not
run concurrent builders in a process. `PlanA.prepare` still performs the existing
sensor/mask validation, overlap removal, wrist gate and C0/C1/C2/C3 construction.
Its private `filters` entries are `SparseDecision`, valid **only** at the exact raw
Peg/Box mask union. They expose no fabricated dense candidate/depth/owner array.
The union is checked again before applying decisions. Full-screen diagnostic APIs
are preserved outside this scope. ROI diagnostics explicitly carry their pixel IDs.
All changes are restored even when a build raises.

At each required pixel all projected link intersections remain eligible. Every
convex collision shape, nearest CV-z depth, 2.5mm tolerance, 6mm uncertainty band,
0.5mm strict interior margin, link ordering and depth ties are preserved. Float64
and `enable_fp_fusion=False` remain. No threshold, object-mask overlap, gate,
Shape-First, 3mm deduplication, budget, observation ID/provenance or Graph policy
is changed. This work preserves existing mistakes; it does not improve or score
real-world filtering accuracy.

| Candidate | Additional work |
|---|---|
| baseline | Frozen Stage 9: grouped at <=128², old per-link GPU above |
| roi | Clip only mask-union ∩ existing projected bounds; sparse decisions/readback |
| hulls | roi plus world hull reuse across the two cameras in one build |
| packed | hulls plus compact rays/depth, batched plane numerator and row transfers |
| rays | packed plus content-keyed O(H+W) calibration-vector cache |
| atomic | packed plus clipping grouped by plane count, exact depth-min/interior-OR reduction |
| strict | Preserve full-ray GEMM and per-link numerator arithmetic; sparse atomic clipping above 128²; Stage 9 fallback at small sizes |
| cpu | Same Stage 9 Graph pipeline with the existing CPU robot classifier |

Hull reuse itself already existed in the earlier grouped candidate. It is not a
new discovery here. No world planes or vertices are reused across builds. A local
hull/plane transform shortcut is intentionally not implemented: world vertex
rounding and Qhull plane decisions would need a separate equivalence proof.
`RayCache` has an eight-entry LRU keyed by full float64 intrinsic contents and H/W;
it caches only calibration vectors. Current depth, robot FK, origin and wrist
extrinsics are renewed every build. The cache key, construction and eviction costs
are inside timing. No output is memoized for repeated benchmark inputs.

Atomic reduction uses exact float64 minima and integer OR. Diagnostic ownership
is resolved only after all depth updates, choosing the minimum original link ID
among exactly equal finite entries. Production masks do not consume ownership;
diagnostics have a separate scratch array and second pass. This differs from the
old full-screen grouped ENTRY/UPPER/reduce layout; it is not a claim that old
grouping became faster at high resolution.

## Reproduction

Use the existing remote environment and baseline paths in `remote.sh`. Copy this
folder to a new code snapshot under the experiment root before running. Outputs
must be new directories. For example:

```bash
bash /workspace/task10i_robot_filter_optimization_20261103_v1/code_v2/remote.sh \
  --cases /workspace/task10i_robot_filter_optimization_20261103_v1/code_v2/timed.json \
  --output /workspace/task10i_robot_filter_optimization_20261103_v1/new_unique_run \
  --variants baseline strict --threads 4 --warmups 3 --repeats 15
```

The report records the actually selected final variant. Validation can use
`--cases .../regression.json --repeats 0 --pixels`. Execute tests with
`TASK10I_FILTER_ENTRY=test_filter.py bash .../remote.sh -v`. The same entry override
selects `filter_benchmark.py` for two-camera filter-to-clean-mask timing, or
`diagnostic.py` for synchronized spans and Torch operator traces. These diagnostics
are separate from the final Graph latency measurements.

Main boundary: CPU sensor arrays, masks and precomputed current robot surfaces to
CPU final Graph, all records and gate, including internal transfers. File reads,
decode/join, FK, models, action inference and GT scoring are excluded. Each
variant has >=3 warmups and >=15 measured repetitions, alternating order. JIT and
first invocation are recorded separately. GC remains enabled. Full record fields
and all prepare conditions are checked in the untimed capture; each measured
Graph checks discrete/position/non-PCA values, source provenance, gate and the
existing PCA-only `atol=2e-5, rtol=0` contract. ROI booleans and owners are exact;
predicted depth uses the existing filter-test `atol=1e-12, rtol=0` contract.

`ExperimentLogger` owns run IDs and local sidecars. W&B is disabled and no publisher
is called. An 80GiB data limit and 180s per-build alarm apply only to new processes.
See the full report in `docs/visual-graph-integration/` for measurements and limits.

The initial `packed`, `rays`, and `atomic` candidates are REJECTED: saved real
input 0044/t80 at 1024 changed 29 Main owner IDs despite identical clean masks
and Graphs. Their last-bit depth changes are not an acceptable owner contract.
`strict` retains baseline ray and per-link numerator arithmetic to preserve these
ties. Its row enumeration scans the sorted union only over each projected y span.
