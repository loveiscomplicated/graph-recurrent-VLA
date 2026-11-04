# Task10-I robot-filter live completion validation

Validates the four integration fixes with the common `ExperimentLogger` and the
existing remote CUDA environment. The runner executes related contract suites,
then two native 512 two-camera filter handoffs and two frozen-input correction
handoffs. Final Graph construction is intercepted; no model/API, simulator,
training, publishing or paused-run resume occurs.

Correction cases compare all consumed boolean robot decisions, raw/clean masks
and gate decisions to the original NumPy classifier, including a proposal outside
the original mask. They require zero CPU classifier calls, one shared preparation,
one full-frame CUDA classification and two fused depth conversions per frame.
Auxiliary CPU/GPU owner differences are retained, not hidden by claiming full
diagnostic byte identity. The original CUDA filter's owner/tie contracts remain
in the unit suite. This runner does not produce a new latency measurement.

```bash
PYTHONPATH=$PWD OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m graph.experiments.robot_filter_live_completion_10i_20261104_v1.run \
  --output /unique/new/output
```

The initial run's overly broad NumPy owner-equality assertion failed and its log
is retained. The final run checks correction's actual fields strictly and reports
auxiliary numerical differences separately. Source reconstruction uses the
recorded base, source archive, test overlay and v2 fix overlay (or the canonical
executed-source archive). Use a fresh output root for any new execution.
