# Task 10-I Graph latency continuation, 2026-11-03

Independent, opt-in continuation of `resolution_perception_graph_20261103_v1`.
Shared simulator, PlanA, existing Graph source, frozen masks, and paused experiment
are untouched. The old `fast_v5_tree` source is a read-only baseline dependency.
No inference API, SAM2 execution, training, rollout, package installation, or upload.

## Plan and measurement contract

1. Review current source and reproduce the CPU-input -> CPU-final-Graph boundary.
2. Compare deduplication, voxel representatives and final point budget separately.
3. Compare grouped GPU robot clipping, dense features/endpoint evidence and FPS.
4. Remove SF-unconsumed Hybrid work and repeated packet copying; compare small
   final-graph gates on CPU and GPU. Keep the implementation that measurements support.
5. Freeze the candidate, run paired repetitions, diverse frozen actual inputs,
   CPU-thread sensitivity, contracts, memory inspection, and a final diagnostic trace.
6. Write the full report under `docs/visual-graph-integration/` and stop. Never
   restart the original Task 10-I processes.

The timer includes depth conversion, two-camera robot filtering, gate, Shape-First
selection, 3mm stable representative deduplication, 128/object final budget,
Graph21/Edge4 construction/validation and internal CPU/GPU transfers. It excludes
file reads/decoding, input loading/join, separate FK, probe, other modules and
action inference. Inputs are CPU arrays and robot surfaces; outputs are CPU Graph,
records and gate. There is no GPU-resident end-to-end result in this experiment.

## Candidates

| Stage | Cumulative change |
|---|---|
| 0 | Frozen `fast_v5_tree` |
| 1 | Single-threaded C++ 3mm greedy dedup, strict NumPy boundary-pair checks |
| 2 | Stable native voxel representative loops and stable-ID final FPS |
| 3 | GPU robot link clipping grouped by plane count, hull reuse within one build |
| 4 | Existing GPU PCA5 and fused dense features + private endpoint maps |
| 5 | Single-block FPS for <=8192 candidates; source-sorted feature join |
| 6 | SF-only represent: skip unrequested L/H work, use privately owned records;
 selection packet skips a redundant receiver copy/digest, retains source validation |
| 7 | Exact cutoff-inclusive partial top-k Peg/Box edges |
| 8 | CPU-batched small final-graph surface gates, retaining KD-tree neighbor order |
| 9 | Use grouped robot clipping only at <=128² pixels; keep previous GPU clipping above |

Stage 6 has a **private SF-only audit return** containing `SF`, rather than the
unrequested `L/H` audit pools. `CurrentGeometry.build` discards those audits. It
does not change the public shared API or support using this candidate as a
general L/H evaluator. Source packet fields and final Graph/records/gate semantics
are checked separately. Generic native dedup still deep-copies caller-owned
records; only fresh private SF records may transfer ownership without another copy.

Endpoint maps implement the current SF consumer's `candidates()` needs; they are
not a general replacement for `PairEvidence.pixels` or its full edge-list API.
Orientation decisions within 0.001 degrees of the threshold use the reference
pair operator. The GPU feature operator adds ordinary floating-point differences;
the existing PCA5 Graph tolerance remains absolute `2e-5`, relative `0`.
Positions, validity/discrete fields, graph edges, selected and merged source IDs,
and gate dictionaries are compared exactly. Bitwise floating-point equality is
not claimed.

`native.cpp` compiles with an already available C++ compiler into a hash-named
temporary library (`-O3 -ffp-contract=off -fno-fast-math`, no OpenMP). No install
or host-specific instruction flag is used. Small FPS uses float64 and deterministic
first-row ties; larger inputs retain the previous multi-block GPU implementation.
Context patches are restored on exit, including exceptions. The adapter is
process-local and must not run concurrent builders in the same process.

## Reproduction

On the existing remote host, use its existing Python and frozen sources:

```bash
cd /workspace/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/source
export TASK10I_REPO="$PWD"
export TASK10I_ROOT=/workspace/task10i_resolution_perception_graph_20261103_v1/repaired_graph_v2
export TASK10I_REFERENCE_CODE=/workspace/task10i_graph_latency_20261103_v2/reference_code
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/workspace/task10i_stage4a2_qwen35_verifier_20261102_v1/env/bin/python \
  /workspace/task10i_graph_latency_20261103_v2/code/benchmark.py \
  --output /workspace/task10i_graph_latency_20261103_v2/new_unique_run \
  --stages 0 9 --sizes 128 512 768 1024 --warmups 3 --repeats 15 --threads 4
```

Outputs must be new directories. A per-build 180s alarm and 80 GiB data limit
apply only to this new process. Warmups include JIT/allocator setup and are
excluded; stage order alternates across measured rounds. Comparisons, hashes,
Graph saves, logger writes, source/input hashes and memory reads are outside the
timer. Python GC remains enabled. `ExperimentLogger` creates each run identity;
W&B transport is disabled, and no artifact publisher is called.

Run CPU tests locally; CUDA tests skip on the Mac:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  graph.experiments.graph_latency_10i_20261103_v2.test_optimization -v
```

On the remote host add the repo and this folder to `PYTHONPATH`, retain
`TASK10I_REFERENCE_CODE`, and run `code/test_optimization.py -v`. The final report
records measured results, accepted/rejected paths, artifacts and unverified scope.
