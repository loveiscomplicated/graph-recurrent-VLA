# Task 10-F1 — GT-supervised graph pretraining

The executed result is documented in
[`10f1_gt_supervised_gnn_pretraining_final.md`](../../docs/visual-graph-integration/10f1_gt_supervised_gnn_pretraining_final.md).
Both fixed12k runs and frozen held-out evaluation completed on57987772. The
selected GNN step is5000, checkpoint SHA256
`781f9f67ba317656d2e6718c2c601026883482633c7d17a4a0cf44f7548690c6`.
Absolute errors improved against the evaluated controls, while accurate5mm
geometry response and strict predicted-state policy readiness remain unresolved.
No GRU/action/BC training was run. The current-session continuation superseded
the historical callback watcher; actual callback invocations remained zero.

Robot-only evaluation pads the last partial batch to64 rows and discards padding.
This keeps the CUDA matrix shape fixed for the exact identical-input response
contract. The original mixed-batch rounding failure and evaluation source are
preserved; no assertion tolerance, training weights or GNN batching was changed.

This opt-in package retains Plan A E0 GraphEncoder (L2/d64, Node21/Edge4)
and its TCP/Left/Right/Peg/Box 5×64 structured readout. A new MLP predicts18
independent object values. Measured Robot76 reconstructs the first15 GT38
fields; peg-head-to-hole is derived from predicted position/rotation/size;
freshness is Clean `[1,0]`. There is no GRU, action head or action loss.

Read `gt38_feature_contract.json` and `target_observability_audit.json` before
using state38. Relative positions are **world-aligned differences**, while
Graph XYZ is in TCP axes. The randomized peg dimensions/hole radius are
predictions, never GT inputs or silently supplied constants. A shape-only
graph cannot uniquely resolve arbitrary symmetric peg orientations, hidden
object states, or invisible hole rims. The API returns estimates in the
official order; numerical compatibility does not establish observability or
closed-loop readiness. Raw orientation errors retain these ambiguities.

`prepare` verifies the archived Task10E encoder and three38D checkpoints,
then freezes120 Full-task parents (96 train,12 validation,12 new Final),
separate parent/offset pairs, and one12,000-update AdamW objective. `collect`
uses unchanged ScriptedExpert through insertion and captures pre-action RGBD,
Robot76 and separate GT NPZ sidecars. `cache` reuses original Plan A and
Task10D CUDA vision / CPU geometry implementations. Each CUDA episode worker
owns two causal camera trackers and an independent four-process geometry pool.
Paired frames restore both tracker memory and observed JPEG prefix files.
The GT sidecar is never passed to graph construction or model.forward.

`train` fits one fresh GNN + Head and a Robot-only control; it does not load
BC weights. Fixed physical target scales differ from archived GRU observation
normalization. First minimum validation episode-macro objective selects the
checkpoint. Final graphs/labels are locked until selection freeze. Real-data
packing, gradients and exact serialized next-update resume are checked first.
Checkpoint includes GNN, Readout, Head, optimizer, scheduler, RNG, train-only
normalization, robot schema and geometry/checkpoint/source identities.

`evaluate` freezes selected weights and evaluates held-out absolute fields,
same-state unseen5/10mm offsets, Raw Geometry and Robot-only controls, oracle
versus predicted assembly, slot response and extraction costs. It never calls
a policy, optimizer, env.step or publishing API.

Run from the isolated source on designated Vast.ai57987772, using an
independent Supervisor service and the prepared root:

The latest steering gives Task10-F approved corrective data priority. The
initial120-episode Full-task plan and12k config remain historical records.
Training now refuses to start until `data_ready_release.json`, SHA-pinned
`final_dataset_manifest.json`, and `final_pretrain_config.json` exist. The
`release` command validates the sealed Task10F import, raw GT, measured robot,
GNN masks, parent splits and exact runtime/camera/geometry contract before
creating these new files. Different Torch runtimes have actually been observed;
they cannot silently pass the release. Dataset import normalization and the
final training release are NOT executed before real Task10F data is ready.

`watch_ready` reuses the existing project's event fingerprint/flock/CLI helpers.
It polls a sealed `gnn_dataset_ready.json` without model calls, verifies every
file and actual HDF5/GT/graph/mask approval, copies only sealed files and checks
SHA/readback locally and on57987772. A verified CLI callback may then claim a
content Event ID once. It defers while the main turn is active, keeps a durable
Invocation Journal, never controls Task10F processes and never launches training
on mere file existence. A single actual local CLI/auth/SSH/write fixture passed;
the real READY-triggered transfer/continuation remains pending until the producer
provides an approved, GT-complete sealed export.

```bash
WANDB_MODE=disabled CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
  python -u -B -m graph.gt_pretrain10f1.train --root "$TASK_ROOT"
python -u -B -m graph.gt_pretrain10f1.evaluate --root "$TASK_ROOT"
```

Frozen inference APIs are `StateEncoder.encode_graph`, `predict_state`,
`extract`, and `contract.assemble_gt38`. `extract` returns both A state38 and
B Readout320/Presence2/Robot76 from the same checkpoint. The independent CLI
reads only whitelisted graph/robot arrays, and needs no GT dataset:

```bash
python -B -m graph.gt_pretrain10f1.extract \
  --checkpoint "$TASK_ROOT/gnn_pretrained.pt" \
  --graphs "$TASK_ROOT/cache/ep2000108/graphs.npz" \
  --receipt "$TASK_ROOT/cache/ep2000108/receipt.json" \
  --schema "$TASK_ROOT/normalizer.json" \
  --out "$TASK_ROOT/extraction_example.npz" --frames 0 1 2 3
```

No future GRU feature cache, policy training, commit or external publishing is
triggered. Existing data/source artifacts remain read-only. `STOP` preserves
the last periodic training checkpoint and ends training; interrupted preparation
files must be archived before resuming. `last.pt` retains optimizer/scheduler/RNG
for an explicitly recorded resume; the real-data resume test verifies next-update
identity. A full primary resume launcher is not implemented.

CPU semantic checks:

```bash
WANDB_MODE=disabled python -B -m unittest graph.gt_pretrain10f1.tests -v
```

The independent `remediation_collect/cache/pipeline/export` workflow records
corrective GT sidecars and explicit per-action GNN/physical masks without
modifying Task10-F's original212 episodes or running processes. Exact-input
Graph reuse must preserve the source **sampling episode identity**: that ID
enters the frozen frame RNG. A new logical capture ID cannot silently replace
the Graph's sampling namespace. The sealed manifest and receipt declare both;
the readiness validator checks the stored scalar against that contract.
`ready_contract.validate(..., marker_path=candidate)` permits complete validation
before the watched marker exists. Export performs two full validations and an
atomic no-overwrite publication only after runtime acceptance. Native-runtime
comparisons use a preregistered protocol and no fitted state/action outputs.
