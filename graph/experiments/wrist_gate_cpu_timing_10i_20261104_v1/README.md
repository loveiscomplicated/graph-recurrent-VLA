# Task10-I: original CPU wrist-gate timing

Calls the unchanged `graph.plan_a_geometry.core.wrist_gate` directly. It does not
use the new GPU gate or the alternative connectivity prototype. Three warmups
follow an untimed reference call, then fifteen full-function wall timings per
saved 512 case. Full accepted masks and trace dictionaries must remain exactly
equal after every call; verification happens outside the timer.

Cases are the earlier image scene `10i_s1_0921_t080` (empty wrist Peg) and
tracking `10i_s1_0531_track_t011` (nonempty wrist Peg). Original frozen r0 actual
SAM2 masks are loaded from disk. They are not new SAM2 predictions and are not
assumed identical to the later online-adapter experiment masks. Both cameras'
original calibration/depth and exact old CPU-filtered clean masks are supplied.
The gate runs once for the wrist Peg per call and uses the other camera for
consistency checks.

Input I/O, separate FK, robot filtering, SAM2/detection, transfers and all later
Graph/policy/simulator work are excluded. The original full-frame world-point
creation, SciPy radius components/local-neighbor queries, cross-camera checks,
trace and accepted-mask creation are included. No CPU geometry cache is added.

Run with the frozen source from the earlier real-SAM2 measurement on the existing
Linux host and its installed environment, in a new output directory:

```bash
PYTHONPATH=/workspace/task10i_real_sam_overlap_20261103_v1/code \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WANDB_MODE=disabled \
python run.py --output /unique/new/output --repeats 15 --warmups 3
```

The runner uses the common `ExperimentLogger`, retains input/source hashes and
packed gate inputs, and makes no model/API call or runtime implementation change.
