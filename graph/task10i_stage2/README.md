# Task10-I Stage2 initial-state selection

CPU-only, task-owned dataset construction. No simulator, expert, vision/model
inference, training, publishing or Stage3 launch. Original inputs are read-only.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest graph.task10i_stage2.test_contracts -v
```

On designated instance57987772, freeze this package, the existing mirror verifier
and shared ExperimentLogger/artifact modules under a new isolated source root.
The authoritative input paths are in `load_and_verify.py`.

```sh
cd /workspace/task10i_stage2_NEW/source
tmux new-session -d -s task10i-stage2 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python -B -u -m graph.task10i_stage2.run_pipeline /workspace/task10i_stage2_NEW/run > /workspace/task10i_stage2_NEW/pipeline.log 2>&1'
```

The pipeline owns every stage through report/hash verification. Atomic progress
and per-stage PASS records preserve completed work. Resume the same command only
after the prior process is terminal: a lock rejects concurrent execution. Resume
revalidates deterministic stages and requires frozen source/config/results to be
unchanged; invariant failures are not retried or relaxed. A completed root only
reads back its frozen selection and artifact index and returns. New source needs
a new root, with earlier failures preserved; do not select favorable outcomes.

Only existing33 + training220 fit the scale. Greedy only accepts those provenance
IDs, and holdout evaluation requires a valid frozen selection identity. The
distance/weights/floors/RNG/diagnostics are in `core.PROTOCOL`, frozen before
selection. R19 remains eligible for D19. Greedy continuation and independent
random queues provide deterministic reserves, without performing replacements.

Executed result: `/workspace/task10i_stage2_20261031_v2/run`, UUID
`34d10bf652b344a38d54b7672dbfe0c0`. The original root remains BLOCKED after a
distribution-output duplicate-key error. All frozen selections/scales/order/CSVs
match the corrected run byte for byte. Do not rerun to choose a favorable seed.

The separate `audit_completion.py` checks every immutable artifact, reconstructs
340 descriptors, uses independent SciPy SO(3)/RMS distance and greedy objective,
recomputes all coverage and 200 Monte Carlo scores, and audits reserve/branch
budgets. With `--raw-inputs` it also reopens all 300 candidates and 40 historical
parent sources and checks the entire canonical tree. On the original runtime a
separate-process exact repeat checks the full176 greedy trace. On other platforms
late floating-point reserve ties may differ; the frozen queue remains authoritative.

```sh
.venv/bin/python -B -m graph.task10i_stage2.audit_completion \
  outputs/task10i/stage2/run outputs/task10i/stage2/local_completion_audit.json
.venv/bin/python -B -m graph.task10i_stage2.deliver
```

`deliver.py` requires the collected remote and local audit receipts and failed-run
selection proofs. It renders the full Markdown as proper static HTML, resolves
all local report/gallery references, and preserves the original remote artifacts.
It records source hashes and link/byte verification in `delivery_verification.json`.
The full report is saved under `docs/visual-graph-integration/`.

Post-run hardening preserves already indexed ExperimentLogger sidecars when a
new resume UUID attaches existing results; the synthetic interrupted-finalization
identity test covers this. The executed frozen source remains separately retained
in the remote execution snapshot. Current verification/delivery helpers record
their own source hashes. The Stage3 receipt is a collection plan; no Stage3 job is
started by any entrypoint here.
