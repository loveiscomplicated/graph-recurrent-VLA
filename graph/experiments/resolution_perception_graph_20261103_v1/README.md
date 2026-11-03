# Task 10-I: native resolution, recognition, geometry and latency

Independent evaluation of native 128/512/768/1024 RGB-D. Existing production and
shared simulator APIs stay unchanged. No training, policy rollout or publishing.

The frozen protocol names 15 known train Parents at t0/t80 (0134 uses t78), the
already-viewed seed98202 t0, and four previous timing Parents at t0..20. Static
quality has62 camera views/size; causal tracking has8 sequences/size. Terra low
uses three fresh attempts/condition:744 static plus96 tracking-initialization
requests, maximum840 and a separate new20 USD API ceiling. Old results are never
mixed into the new comparison. The old same-scene Terra boxes are smoke inputs
only.

`capture.py prepare` freezes identities and dependencies; `capture.py smoke` and
`sam_worker.py smoke` exercise all sizes. `capture.py capture` restores physical
states without stepping and produces920 native RGB-D camera observations. All
API inputs are captured first, so the remaining tracking capture can finish
while local HTTP calls run. Intrinsics, extrinsics, state and signed-mm depth
are verified. Seed98202's initial physical state is checked against its saved
snapshot. The experiment uses CPU PhysX and CPU SAPIEN readback consistently;
this renderer differs from the previous seed98202 CUDA-rendered preview.

Only `api.py` reads the local API credential. `run.py fetch-inputs` transfers RGB,
sanitized image metadata and the protocol, with no GT arrays/actor IDs or keys.
Requests are sequential and cross resolution order. A locked, fsync'd RESERVED
job is never sent twice, even after a timeout or interruption. No SDK retry,
coordinate repair, schema forcing, prompt tuning or failed-image replacement.
The reservation uses a deliberately generous full-context upper bound; Usage
estimates ignore cache discounts and are not invoices.

After API freeze, `run.py transfer-api`, `run.py sam --arm actual`, and
`run.py graphs --arm actual` freeze real masks/Graph tensors/estimates. Only
then `run.py oracles` creates1024-normalized GT boxes, GT-box SAM2 predictions
and GT-mask Graph controls. Graph remains CurrentGeometry C3/SF, Graph21/Edge4,
PCA5,3mm real-measurement dedup and128 points per object. Wrist Box stays disabled.
The geometric estimator functions are exact source exports of the previous
frozen probe; no fit, new feature or estimator sweep is introduced.

Graph quality uses four CPU processes with one thread each. Its concurrent
wall times are diagnostics, not fair latency. Native Graph latency runs alone
with four Torch threads, one pass per frozen r0 frame, excluding a per-size
warmup. Every successful native r0 is checked against its instrumented Graph.
Quality execution has a180×worker-count second deadline; isolated latency has
180 seconds. A24GiB data-allocation limit bounds each quality process. After all
four workers exit, memory failures alone are replayed singly with an80GiB limit;
the original24GiB receipts remain unchanged. Official isolated latency also uses
80GiB. OOM, timeout and failure are preserved; resolution/algorithm never changes.
Quality-only read-only memoization avoids recalculating identical sensor depth
and inverse extrinsics. Official native timing disables it. Exact five-tensor
comparisons verify both paths, including every successful native r0 condition.

`timing.py static` measures eight balanced rounds using frozen r0 prompts;
`timing.py tracking` measures four, with three image warmups or one full video
warmup per size. Full CPU video preload/first encoding, prompts, t0, t1..20,
CPU mask output and GPU allocation are separate. Temporal horizon is t+1;
preloading is not evidence of an online streaming implementation. JPEG is fixed
at quality100/subsampling0. Pure masks are checked against quality outputs.

`evaluate.py` scores frozen predictions after all three Graph arms finish. Its
common visible1024 reference is not a full or hidden surface. Peg uses both
views, Box uses base only; all-camera support is separate.4mm coverage and
normalized boundary tolerance are fixed. The normal visibility stratum is1024
actor pixels at1024, equivalent to16 pixels at128; tiny and absent observations
are separate. N/A estimates are never zeros. Topology component counts are
diagnostics, not GT edge accuracy. Only final capped Graphs run the fixed pose
probe; dense stages receive surface/support diagnostics.

Raw masks, stage arrays, graphs and logs live under the experiment output root.
The final full Korean report belongs in `docs/visual-graph-integration/`, with
paired Parent-cluster10,000-bootstrap intervals; seed98202 is a separate case.
All stages reuse the shared ExperimentLogger with W&B disabled. Sources,
checkpoint, raw inputs and protocol revisions are preserved. No environment
installation or credential transfer is required.

The initial Graph attempt was rejected by the existing sensor-only guard because
the adapter accidentally placed two timing fields in its input dictionary. The
guard was retained; diagnostics now occupy a separate fifth tuple member. The
failed attempt and its controls remain in the original output root. The corrected
attempt reuses byte-identical native inputs, API responses and actual SAM masks,
then rebuilds all Graphs and fresh controls in `repaired_graph_v2/`. Set
`TASK10I_ROOT` to that subdirectory for evaluation/reporting. The existing server
root is `/workspace/task10i_resolution_perception_graph_20261103_v1/repaired_graph_v2`.
`engineering_recovery.json`, `reused_inputs.json`, `protocol_history/` and
`source_versions/` record the boundary and source changes. No extra API calls
are made during recovery. Each resumed stage receives a new ExperimentLogger ID.

Detailed JPEG profiling observes the original file open/read/seek/close calls
and CPU preload separately; it does not change image bytes. Native Graph memory
uses Linux's per-process high-water RSS reset before each build. This includes
the loaded process, graph copy/hash and unchanged pose probe; it is not a pure
Graph allocator measurement and excludes GPU memory. Timings and memory failures
remain separate from successful conditional latency distributions.

Offline verification:

```bash
.venv/bin/python graph/experiments/resolution_perception_graph_20261103_v1/test_contracts.py
```

### Separate wrist connectivity optimization probe (not deployed)

`radius_components.py` is an opt-in connectivity prototype. It retains every
input point and the original radius, groups guaranteed-connected small spatial
cells, and joins cells using scalar exact KD-tree radius counts. It never
materializes the full radius-edge list. Labels retain first-source order so tied
largest components select the same pixels. No PlanA source, production interface,
paused server code, protocol, or experiment output is replaced by this probe.

`test_radius_components.py` compares original labels and complete gate outputs.
The local test command also discovers the 16 imported existing PlanA contracts:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest graph.experiments.resolution_perception_graph_20261103_v1.test_radius_components -v
```

The [10-I connectivity probe report](../../../docs/visual-graph-integration/10i_wrist_gate_connectivity_optimization_20261103.md)
records Mac CPU measurements, saved-pixel comparisons and remaining validation.
These measurements are not the full resolution experiment or server latency.
