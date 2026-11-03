# Terra Fast API latency (Task 10-I Stage 4A-2)

User-requested timing extension on the existing 60 native512 images. Reuses the
existing Terra request builder, one-attempt HTTP adapter, strict parser, and
common ExperimentLogger. The only request change is `service_tier="fast"`.
Sequential, at most 60 requests, no warm-up, retry, replacement or Standard rerun.
Successful responses must report `fast` or its documented alias `priority`;
otherwise stop and record the actual tier. Previous artifacts remain immutable.

Run offline guards with `.venv/bin/python graph/experiments/api_terra_fast_latency_20261103_v1/test_contracts.py`.
Stages: `run.py prepare`, `run.py infer`, `run.py summarize`, in that order.
Preparation freezes the protocol, input hashes, source, previous outputs and cost
accounting. A durable RESERVED job is never sent again; concurrent infer processes
are blocked with a file lock. Caffeinate is scoped to the inference process.

Fast uncached prices frozen on 2026-11-03: $2 input / $10 output per million tokens.
Each request reserves twice the original Terra bound; cumulative accounting must
remain under $20. Usage estimates and unknown reservations are not billed costs.
No SAM2, GT evaluation, GPU, simulator, tracking, training, graph/controller,
production changes, commits or external publishing. PAUSED_SCOPE_REVIEW remains.
The Standard comparison is historical and cannot isolate the causal tier effect.
Final report: `docs/visual-graph-integration/10i_stage4a2_api_terra_fast_latency_20261103.md`.
