# Task10-I: first CUDA wrist-gate comparison

Reuses `graph.persistent_geometry.tensor_gate.wrist_gate_tensor` and its existing
Triton radius-witness kernel. This experiment does not add a second gate or alter
the gate thresholds, point count, masks, production defaults, or live runtime.

Only the nonempty frozen 512 case `10i_s1_0531_track_t011` is benchmarked: 29,775
valid clean wrist Peg points. The input NPZ must retain the original CPU
benchmark's SHA-256. The current CPU gate must first reproduce the saved
reference mask hash and complete trace exactly.

The existing CUDA implementation computes only selected XYZ, represents radius
connectivity with cells and exact inter-cell radius witnesses, and checks local
support and cross-camera consistency on CUDA. It therefore changes both the
device and the computational method; any measured speedup is not a device-only
comparison. No points are downsampled or capped for the gate.

Raw/clean masks and depth are resident on CUDA before its timer. Each gate call
still includes host calibration/identity checks and small uploads, scalar
synchronization, complete gate math and mask/trace construction. Initial input
uploads and validation readback are excluded, as are SAM2, robot filtering and
Graph. The CPU timer retains the original full `core.wrist_gate` boundary.
These are synchronized wall times, not CUDA-event kernel-only timings.

One untimed reference/compile call and three warmups precede 15 alternating-order
CPU/CUDA pairs. Every output mask is compared exactly. Every trace field except
the floating point extent is compared exactly; extent allows absolute 1e-12 m
roundoff. The actual extent difference and whether the entire trace is exactly
equal are retained. Repeated traces must equal their own reference exactly.
The initial GPU call also forbids the CPU components and world-point methods.

Use a frozen source snapshot in an existing CUDA environment, from its root:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled \
python -m unittest graph.experiments.wrist_gate_cuda_10i_20261104_v1.test_contracts -v
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled \
python -m graph.experiments.wrist_gate_cuda_10i_20261104_v1.benchmark \
  --input /path/to/original/cpu/remote --output /unique/new/output
```

Uses the shared `ExperimentLogger`, with local config, raw repetitions, output
masks, source/input hashes and sidecars. No model call, simulator, training,
publishing or package installation is needed. A single saved frame does not
establish equivalence across all scenes or gate thresholds.
