# GNN training

## Frozen corner-only tracker inputs

Run the preparation and training configs documented at the beginning of
`graph/README.md`. `training.frozen_tracker_gnn:train` uses the corrected
corner-only tracker selected at epoch 374, SHA-256
`ea4df0c535e9ff908c7c2b0b98885128de80310f4a2ee91332991ee285e339cf`.
The tracker runs in strict FP32 evaluation mode with all parameters frozen.
Its causal outputs are cached once; current detections must pass through
exactly, and every parameter/buffer is checked unchanged after generation.
The checkpoint file hash is checked before and after GNN training. No tracker
parameter enters the optimizer. No GNN checkpoint is loaded: the GNN starts
from seed-0 random weights after batch probing is discarded.

Target preparation uses only the original 3600 training and 450 validation
episodes from the 4500-episode release. It joins by seed/mode/frame, verifies
release member checksums, exact frame indices, TCP coordinates and Peg center
agreement with corner GT. The 450 held-out test payloads are excluded from
preparation, cache generation, optimization and selection. The source release
manifest is pinned to SHA-256
`be2db7841d823d89ba4044945a170c80bf5a316a53a58ccc153643f74b0098d8`.
No simulator, RGB-D decoder, surface features or old State36 cache is needed.

Only slots actually observed at least once up to the current frame become
graph nodes. Previously observed missing slots use direct frozen tracker
coordinates; never-seen predictions are omitted. No rigid-body correction
is added. The GNN retains graph2's 20 padded slots, Peg type 0, Box/Hole type 1,
complete directed edges without self-edges, one actual-detection provenance
feature, and no semantic IDs/confidence features. The unchanged
`VertexEdgeGatedMPNN` uses hidden width 64 and two intra-object layers plus
one cross-object layer. The original Object21 loss is reused unchanged.

Full Object21 labels supervise every train/validation frame, including frames
with incomplete graph support. First-observation eligibility applies to input
nodes, rather than masking whole Object21 labels. Validation logs per-mode
loss and Peg/endpoint/Hole position errors; selection averages the five
per-mode frame-mean losses equally. Each strict minimum saves `best.pt`.
Twenty epochs without 0.5% relative improvement lower the learning rate
from 0.0003 to 0.0001, then 0.00003; a further plateau stops training. The
safety limit is 1000 epochs. AdamW uses weight decay 0.01 and clipping at 1.0.

GNN training keeps FP32 tensors and enables TF32 matrix multiplication only
after strict FP32 tracker caching. The shared model/loss and protected
leaderboard dependencies remain unchanged. Batch probes time complete
forward/backward/clipping/AdamW steps with all 20 nodes present. Selection
chooses the largest candidate within 95% of peak eligible throughput and
below 85% of device memory. The initial GNN parameters and RNG are restored
before training. Source-file hashes, target/cache receipts, effective config
and measured batch sizes are recorded. Atomic `last.pt` retains model,
optimizer, plateau and RNG state; no resume launcher is implemented.
The shared `ExperimentLogger` records local metrics and output sidecars;
artifacts are not automatically published.

The existing German RTX 3090 instance `61159793` runs the isolated snapshot
at `/workspace/frozen_tracker_gnn/code`, with outputs under
`/workspace/frozen_tracker_gnn/training`, console log
`/workspace/frozen_tracker_gnn/train.log`, and tmux session `frozen_tracker_gnn`.
Connect using `ssh -p 60777 root@185.113.120.195`, then
`tmux attach -t frozen_tracker_gnn`. The server checkout is detached. Earlier
tracker runs remain separate. Three CPU tests passed locally and on this
server, covering node/edge validity, empty graphs, omitted-point invariance
and a real Object21 backward update. CUDA execution evidence is in the log.

On 2026-11-18 the final expanded probe selected batch **10240**: 36,084
dense-graph frames/second and 20,304,625,664 peak reserved bytes (18.91 GiB).
Batch 12288 exceeded the 85% memory budget; 16384 raised a handled OOM.
Run `19372a661d564b5c927c2de20987ff00` was randomly initialized afterward.
At startup epoch 3 completed in 14.41 seconds with train loss 2.498674 and
mode-balanced validation loss 2.085237; both `best.pt` and `last.pt` existed
and the frozen tracker hash was unchanged. These are startup checks, not final
accuracy results. The earlier 4096-batch check is preserved separately as
`startup_probe_training`/`startup_probe.log`; its weights are not reused.
Nine local tests passed, including the existing first-observation supervision
tests. This workflow does not execute a simulator or publish artifacts.

## Paired GNN test with one frozen tracker

`graph/configs/evaluate_frozen_tracker_gnn_comparison.yaml` compares the original
graph2 `final_gnn.pt` (SHA-256
`e03fd2671fad1f142990f908b5ff7addfac0a3d68c8f341dd5e9670e017b390d`)
against the new corner-only-input GNN selected at epoch 54 (SHA-256
`a47139a38981113a86e31d03dc2f4409086c3958fdadd95396c841ae588e0123`).
Both checkpoints have hidden width 64, two intra-object layers and one
actual-detection provenance feature. Only the GNN weights differ.

The evaluation loads all 450 held-out episodes / 86,179 frames and their
checksum-verified Object21 targets. It checks TCP/frame alignment, Peg center
agreement with GT, release identity and seed separation from train/validation.
The frozen observed-only epoch-374 tracker is executed once. For each frame
batch the evaluator constructs one graph and passes that exact graph and its
provenance tensor to both GNNs. Previously seen missing slots are filled;
never-seen nodes/edges are excluded. There is no surface input or rigid
correction. All inference uses strict FP32 with TF32 disabled, batch 4096
for both GNNs. No optimizer, checkpoint selection or test-driven tuning runs.

The primary table averages five per-mode frame-means equally, matching the
preceding corner-only GNN validation report. The artifact also retains pooled
frame means and each mode's metrics. Object21 uses the unchanged training loss,
including rotation/size and endpoint correction regularization; the three
physical errors are Euclidean 3D distances in millimetres. Full Object21 labels
remain eligible on every frame regardless of incomplete node support.

The isolated detached evaluation checkout is
`/workspace/frozen_tracker_gnn/evaluation_code`; original training source and
checkpoints are preserved. Results, per-frame predictions and the single common
tracker input are under `/workspace/frozen_tracker_gnn/test_comparison`.
The existing `ExperimentLogger` records the effective config and local outputs.
Checkpoint file hashes and GNN parameter/buffer digests must remain unchanged.
New aggregation tests cover complete vector norms, unequal mode lengths,
endpoint regularization and nonfinite rejection. Six local tests and three
server CPU tests passed before actual CUDA evaluation. Protected leaderboard
implementations and dependencies were not modified; no simulator or external
artifact publishing was executed.

On 2026-11-18 CUDA evaluation run `e0452ccb4f0644d3933a7f161a387e4d`
completed on the German RTX 3090. Both GNNs received the identical graph
tensors; their parameter/buffer digests and all checkpoint hashes remained
unchanged. The equal-mode test results were:

| Metric (lower is better) | Existing graph2 GNN | New GNN, epoch 54 | Reduction |
| --- | ---: | ---: | ---: |
| Object21 loss | 3.88193140 | 0.77707746 | 79.98% |
| Peg position mean | 14.032225 mm | 3.439475 mm | 75.49% |
| Hole position mean | 38.910361 mm | 6.993515 mm | 82.03% |
| Peg tip minus Hole vector mean | 28.675327 mm | 9.237956 mm | 67.78% |

The common input archive SHA-256, complete unrounded metrics, per-mode and
pooled results, source hashes and checkpoint identities are in
[results.json](../outputs/frozen_tracker_gnn/test_comparison/results.json).
Per-frame predictions include the endpoint residual required to reproduce
the full Object21 loss. This is a comparison of frozen GNN weights under the
new tracker input distribution; no original-tracker pipeline result is implied.

## Original versus retrained tracker and GNN pairs

`graph/configs/evaluate_original_retrained_tracker_gnn.yaml` uses the same
paired evaluator and metrics as the shared-tracker comparison above, adding
an explicit `existing_tracker` receipt. The existing GNN now receives the
original graph2 epoch-333 tracker (`final_transformer.pt`, SHA-256
`91f745a4e40377695929832aa365bfa2b87edb75fd91c573fb337e25211e8845`).
The new GNN receives its epoch-374 retrained tracker. The two GNN checkpoints
remain the original `final_gnn.pt` and the validation-selected epoch-54 model.
No weights are updated and no checkpoint is reselected.

Each tracker retains its own original train-fitted fallback. Both pairs use
the identical detector observations, physical-slot association, TCP, 450 test
episodes / 86,179 frames, Object21 labels, graph construction, actual-detection
provenance and first-observation eligibility. The tracker predictions differ;
never-seen slots are omitted on both sides. No extra rigid-body correction or
surface feature is added. Object21 labels cover all frames. Strict FP32 with
TF32 disabled, tracker batch 16 and GNN frame batch 4096 apply to both pairs.
The primary metrics are equal means over the five per-mode frame-means;
frame-pooled and per-mode results are retained as well.

The German RTX 3090 uses a separate detached snapshot at
`/workspace/frozen_tracker_gnn/paired_evaluation_code`. Outputs are isolated in
`/workspace/frozen_tracker_gnn/paired_tracker_comparison`, including each
tracker's masked coordinates, each GNN's Object21 and endpoint residual,
common targets, frame identities and checksum receipts. The earlier
shared-tracker comparison remains intact. Seven local CPU tests passed,
including shared/default versus explicit old/new tracker pairing; four
server CPU tests precede CUDA execution. The common model and loss code,
protected leaderboard dependencies, simulator and publishing workflows are
unchanged.

On 2026-11-18 CUDA run `2449e359bd5247ae9c6324e601cac67f` completed both
frozen pairings on all 450 test episodes. The equal-mode results were:

| Metric (lower is better) | Original tracker + original GNN | New tracker + new GNN | Reduction |
| --- | ---: | ---: | ---: |
| Object21 loss | 4.13649812 | 0.77707745 | 81.21% |
| Peg position mean | 15.797320 mm | 3.439475 mm | 78.23% |
| Hole position mean | 39.124535 mm | 6.993515 mm | 82.12% |
| Peg tip minus Hole vector mean | 30.284424 mm | 9.237956 mm | 69.50% |

The [full paired results](../outputs/frozen_tracker_gnn/paired_tracker_comparison/results.json)
retain complete precision, pooled/per-mode metrics and both tracker input
receipts. The newly trained pair agrees with the preceding shared-tracker
evaluation to the displayed precision. This measures the two tracker/GNN
pairings under the common graph/masking protocol described above.

## Three-pipeline comparison by insertion depth

Run the complete-output inference recipe
`python main.py --config graph/configs/evaluate_frozen_tracker_pipeline_object21.yaml`,
then `python main.py --config graph/configs/evaluate_frozen_tracker_stage_comparison.yaml`.
Use `WANDB_MODE=disabled` for local-only logging. The first recipe requires the
same two tracker/GNN pairs and test sidecars as the pipeline comparison below;
the second needs its saved inference payloads, the original tracker/original
GNN's paired-test payload, all pinned report checksums and the existing verified
test corner dataset. Host paths must be adjusted in the configs.

The original pair reuses test run `2449e359bd5247ae9c6324e601cac67f`, using the
epoch-333 original tracker and original `final_gnn.pt`. The new corner-only and
surface pairs retain their validation-selected epoch-54 and epoch-62 GNNs.
Inference run `ac5e5d6ab4a94f73a8d87b164bc45b44` exported their actual Object21
**and endpoint residual** outputs in strict FP32, with TF32 disabled, on the
Korean RTX 3090. The new residual output is required for the existing Object21
endpoint-consistency regularizer; no loss term is omitted or reconstructed.
Earlier prediction artifacts remain unchanged. The optional residual-return
argument preserves the inference helper's previous default interface.

The user selected the existing GT axial insertion-depth split. The definition
already used by the corner tracker is `d <= 0 mm` before insertion, `d > 0 mm`
during axial insertion, with optional entry `(0,20] mm` and deep `>20 mm` bands.
This is a geometry-based slice, not a contact or success label; pre-insertion
includes approach/distant frames as well as alignment. Slice membership uses
only the frozen test dataset's GT depth, identically for all three models.
The 450-episode / 86,179-frame test set contains 79,503 pre-insertion frames
from 449 episodes and 6,676 axial-insertion frames from 254 episodes.

To match the user's reference table, first take frame means within each of the
five modes, then average those mode means equally. Both requested slices have
frames in all five modes. Optional deep-band results use four nonempty modes
because the random mode has no deep frames. Frame-pooled and per-mode metrics
and counts are retained separately. Every Object21 label remains eligible;
first-observation eligibility masks graph inputs, not score-only Object21 GT.
All errors below are Euclidean 3D errors in mm. All reduction percentages use
the original pair as denominator.

| Pre-insertion quantity | Original pair | New corner-only pair | Surface pair |
| --- | ---: | ---: | ---: |
| Object21 loss | 4.169995 | 0.781824 | 0.619517 |
| Peg position mean | 15.877709 | 3.339992 | 2.466347 |
| Hole position mean | 39.285328 | 6.993412 | 5.969528 |
| Tip minus Hole vector mean | 30.442474 | 9.164290 | 7.416670 |

| Axial-insertion quantity | Original pair | New corner-only pair | Surface pair |
| --- | ---: | ---: | ---: |
| Object21 loss | 3.721937 | 0.656668 | 0.282079 |
| Peg position mean | 15.928909 | 4.279209 | 3.069404 |
| Hole position mean | 33.982331 | 6.239079 | 3.837445 |
| Tip minus Hole vector mean | 28.773628 | 9.251112 | 5.289428 |

The full-set reference rows reproduce the supplied table exactly to its shown
precision (original Object21 4.13650 versus corner-only 0.77708). The surface
pair's full-set equal-mode loss is 0.595800, with Peg/Hole/tip-vector means
2.541894/5.827453/7.312801 mm. These differ from frame-pooled figures by design.

CPU scoring run `7dbbaa5ba6aa4950a58816dfdcf8161f` uses the shared
`ExperimentLogger` and unchanged `training.gnn.loss.object21_loss`. It verifies
input report/payload hashes, identical Object21 targets and frame order,
prepared test receipts and GT/TCP consistency. Thirteen CPU tests passed,
including depth boundaries, empty slices, unequal-mode weighting and endpoint
regularization. Local artifact checks independently reproduced all requested
losses and physical means from the saved per-frame outputs and score masks.
No training, checkpoint reselection, simulator execution or publishing ran.

Remote source and results are under
`/workspace/surface_tracker_gnn/object21_pipeline_source` and
`/workspace/surface_tracker_gnn/stage_comparison`. The local report and membership
receipt are in `outputs/frozen_tracker_stage_comparison/results`; inference
receipts and complete new-model outputs are in the sibling `inference` folder.
The three pipelines retain their different trained graph features and training
settings, as explained below; the comparison does not isolate surface context
as the sole changed variable.

## Corner-only versus surface-input pipeline test

`python main.py --config graph/configs/evaluate_frozen_tracker_pipeline_comparison.yaml`
compares the selected corner-only GNN (epoch 54, checkpoint SHA256
`a47139a38981113a86e31d03dc2f4409086c3958fdadd95396c841ae588e0123`)
with the selected surface-input GNN (epoch 62, checkpoint SHA256
`0f4d9d6fd6ce615125698e243b259fad92557a3f5c5848aafe24a85277720241`).
Prerequisites are CUDA PyTorch, NumPy, SciPy, PyYAML, both frozen trackers/GNNs,
and the existing hash-pinned test corner, surface and Object21 sidecars. Host
paths are explicit in the config. `evaluation/frozen_tracker_pipeline_comparison.py`
reuses the existing causal tracker inference, each trained graph constructor,
Object21 vector error calculation and the shared `ExperimentLogger`.

Both pipelines were rerun on the Korean RTX 3090 using strict FP32 with TF32
disabled, tracker batch 16 and GNN batch 4096. Membership, test-target TCP/frame
alignment, seed separation and all checkpoint hashes were checked. Each GNN's
parameter/buffer digest was unchanged after inference; checkpoint files were
also rehashed. No optimizer or model selection used test results. First-seen
eligibility excludes never-observed nodes and edges. Full Object21 targets
remain valid for all 450 episodes / 86,179 test frames.

The primary aggregation pools all frame errors, matching the preceding surface
test report, rather than equally averaging the five mode means. The new common
strict FP32 execution causes small numerical differences from the earlier
surface-only report, whose GNN inference enabled TF32. P90 is the linear 90th
percentile of per-frame Euclidean 3D errors in millimetres.

| Quantity | Corner-only mean | Surface mean | Corner-only P90 | Surface P90 |
| --- | ---: | ---: | ---: | ---: |
| Peg center position | 3.458546 | 2.587934 | 5.429398 | 4.496061 |
| Hole center position | 7.118253 | 5.978879 | 22.017580 | 17.869728 |
| Peg-tip minus Hole vector | 9.357074 | 7.472276 | 27.074194 | 24.250250 |

This is an as-trained pipeline comparison, not an isolated surface-context
ablation. The corner-only GNN uses one provenance feature and groups Box/Hole
under one object type; the surface GNN uses two provenance features and separate
Box/Hole types. Weight decay is 0.01 versus 0.0001. The corner-only run used
mode-balanced validation selection and plateau learning-rate changes, while the
surface run used pooled validation selection and a step scheduler. Training
batch sizes and stopping policies also differ. The result therefore cannot
attribute the whole performance difference to surface points alone.

The isolated execution snapshot is
`/workspace/surface_tracker_gnn/pipeline_comparison_source`; reports and per-frame
predictions are in `/workspace/surface_tracker_gnn/pipeline_comparison`, with a
local copy under `outputs/frozen_tracker_pipeline_comparison/results`. Earlier
training/evaluation sources and outputs are preserved. Nine relevant CPU
contracts passed before both complete CUDA test inferences. Protected historical
leaderboard implementations and dependencies were unchanged. No simulator or
external artifact publishing was executed.

## Frozen surface tracker inputs

`graph/configs/train_frozen_surface_tracker_gnn.yaml` uses the common GNN trainer,
`ExperimentLogger`, Object21 objective, checkpoint handling and early stopping.
The new data/task plugins are in `training/frozen_surface_tracker_gnn.py`;
the Gated MPNN implementation and historical detector-only recipes are unchanged.

Prerequisites: CUDA PyTorch, NumPy, SciPy, PyYAML, prepared Five-mode corner and
Peg-surface datasets, corrected tracker weights and the Object21 target sidecar.
Prepare labels with `python main.py --config
data_collection/configs/prepare_five_mode_object_targets.yaml`, transfer the
output to the training host, then run `WANDB_MODE=disabled python main.py --config
graph/configs/train_frozen_surface_tracker_gnn.yaml` from its source snapshot.

The 4500-episode release keeps its original seed-disjoint split: 3600 train,
450 validation, 450 held-out test. Target preparation reads only train/validation
payloads, verifies original member checksums, and retains TCP for exact frame
alignment checks against the existing corner dataset. No GT enters the GNN inputs.

The surface tracker is the observed-only epoch-443 model, checkpoint SHA256
`f98c4ef1c77d25fa059e9f9cf048fb1f190bcf74302a0fe1367bfe1341236afc`.
It runs in FP32 eval/inference mode with gradients disabled. Its parameter/buffer
hash must remain unchanged after rollout. Measured coordinates must pass through
exactly. The resulting per-frame positions are cached once; GNN epochs do not run
or update the tracker. Never-observed slots are invalid and have no incident
edges; measured versus previously observed missing slots have separate flags.
Surface points influence the GNN only through tracker output. There is no extra
surface-node input or rigid correction. Each frame's Object21 target remains valid
even when some or all corner nodes are unavailable; corner supervision eligibility
is not an Object21 target mask.

The GNN is randomly initialized: hidden width 64, two gated message-passing
layers, two provenance features, no semantic IDs or confidence inputs. AdamW
starts at 0.0003 with weight decay 0.0001, gradient norm clipping at 1, and learning
rate halving every 60 epochs. Early stopping uses validation Object21 loss,
40 stale epochs, absolute minimum improvement 0.0001 and at least 60 epochs;
600 is a maximum guard. Best and resumable last checkpoints are retained.
No test result selects the checkpoint or stop epoch.

The data plugin benchmarks full forward/backward/clipping/AdamW steps on dense
20-node graphs for batches 64–16384. It selects the largest batch reaching at least
95% of peak measured throughput while staying below 85% allocated device memory.
The common trainer reseeds and constructs a new model/optimizer after probing,
so benchmark updates never become GNN initialization. Actual selected batch,
benchmark results, cache hash, input identities and effective config are recorded.

The existing Korean RTX 3090 uses the isolated root
`/workspace/surface_tracker_gnn`; previous tracker runs remain untouched.
Launch under tmux session `surface_tracker_gnn`, with console output in
`/workspace/surface_tracker_gnn/train.log`. Existing caches are reused only after
matching their receipt and all configured input manifest/checkpoint checksums.
Local contract tests run on CPU; actual CUDA startup is checked on the training
host. This workflow does not execute the simulator or publish artifacts.

On 2026-11-18 the Korean RTX 3090 started run
`59ec6322bb104095b2f4025afc5209b2` with batch 8192 selected by the expanded probe.
The first epoch completed with train loss 5.077094 and validation loss 2.377549
in 13.3 seconds; these are startup checks, not a final accuracy claim. Both
`best.pt` and `last.pt` are handled by the common trainer under the run directory.
The earlier short 2048-batch startup probe and its outputs were preserved in
the same root, with its console moved to `startup_probe.log`; the full run was
randomly initialized again and reused only the frozen input cache.
CPU verification passed nine tests (the new graph-input contract and existing
GNN trainer suite), and the graph-input contract also passed on the remote host.
CUDA execution was confirmed through the first train/validation epoch; no
continuous monitoring was started.

### Held-out evaluation of the surface-input GNN

The selected GNN and a linked inference/test package were explicitly published
on 2026-11-18 to
[the original training run on Hugging Face](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/tree/b651e110136acc7ea69f27186e44c4f98aba6964/graph/59ec6322bb104095b2f4025afc5209b2).
The package retains run ID `59ec6322bb104095b2f4025afc5209b2` and contains only
its epoch-62 `best.pt` checkpoint, requested/effective training config, history,
data identity, linked `pipeline_config.json`, `usage.json`, a checksummed UTF-8
source bundle, CPU inference verification and the full/paired/stage test JSON
reports. No last checkpoint or raw test payload was published.

`pipeline_config.json` points to the already-published epoch-443 tracker at
revision `d444f69b37056bf5534fea4dd88d794afce632af`, SHA256
`f98c4ef1c77d25fa059e9f9cf048fb1f190bcf74302a0fe1367bfe1341236afc`,
and the content-addressed GNN file whose SHA256 is
`0f4d9d6fd6ce615125698e243b259fad92557a3f5c5848aafe24a85277720241`.
The config records input units/shapes, physical slot order, causal node
eligibility, graph construction, provenance features, Object21 fields and
precision/aggregation differences between the reports. `inference_source.json`
materializes 23 exact source files including a GT-free inference example;
`usage.json` explains extraction, model loading and episode inference. The
example validates both checkpoint hashes and runs entirely from the extracted
source tree. Its CPU smoke verified finite Object21 outputs, measurement
pass-through, causal node validity and frozen parameters.

The original training records were recovered into a local publication mirror
at `outputs/surface_gnn_publication`; original run metadata/config/metrics are
preserved separately. Publishing used the existing
`python -m tools.publish_experiment --experiments-dir
outputs/surface_gnn_publication/experiments --run-id
59ec6322bb104095b2f4025afc5209b2 --include checkpoint metrics` workflow.
The manifest and receipt record immutable revision
`b651e110136acc7ea69f27186e44c4f98aba6964`. Remote file identities and both
linked checkpoint downloads are verified in the local
`publication_verification.json`. The Korean instance was briefly resumed to
recover the checkpoint after stopped-instance copy failed, then returned to
its prior stopped state. No training or simulator execution ran during posting.

Run `python main.py --config
data_collection/configs/prepare_five_mode_object_test_targets.yaml` to export only
the 450 held-out label payloads; this explicit test-only mode leaves the original
train/validation preparation default unchanged. Transfer the resulting sidecar
to the configured `target_root`, then run `WANDB_MODE=disabled python main.py
--config graph/configs/evaluate_frozen_surface_tracker_gnn.yaml` with the existing
test corner/surface caches and frozen checkpoint files. The evaluator verifies
release membership, disjoint development/test reset seeds, frame counts, TCP
alignment and payload/checkpoint checksums. First-observation eligibility applies
to GNN input nodes; Object21 errors are evaluated on every test frame, including
frames with incomplete or empty node support.

Training run `59ec6322bb104095b2f4025afc5209b2` stopped at epoch 102 after 40
epochs without the required validation improvement. Its selected checkpoint is
epoch 62, validation loss 0.5958381360862066, checkpoint SHA256
`0f4d9d6fd6ce615125698e243b259fad92557a3f5c5848aafe24a85277720241`.
The saved best weights were checked equal to the resumable last checkpoint's
best-model state. Evaluation keeps the epoch-443 surface tracker frozen and uses
the same GNN graph, provenance features and validation precision: strict FP32
tracker rollout, FP32/TF32 GNN. No models are updated or selected on test results.

On all 450 test episodes (90 reset seeds, 86,179 frames), frame-pooled Euclidean
errors were:

| Object21 quantity | Mean (mm) | P90 (mm) |
| --- | ---: | ---: |
| Peg center position | 2.587466 | 4.494915 |
| Hole center position | 5.981179 | 17.826778 |
| Peg-tip minus Hole vector | 7.472465 | 24.245449 |

Peg/Hole positions are the Object21 TCP-relative centers; their position error
equals world-center error because prediction and GT share the same TCP. The
tip-minus-Hole metric is the norm of the difference between two world-axis 3D
vectors, not the difference of their scalar lengths. P90 uses NumPy's linear
percentile over all frame errors, without averaging episode percentiles.
The report also includes per-mode statistics and frame predictions/targets.
Remote results live in `/workspace/surface_tracker_gnn/test_evaluation`; a local
copy lives in `outputs/frozen_surface_tracker_gnn_test`. The evaluation source is
isolated at `/workspace/surface_tracker_gnn/evaluation_source`, preserving the
training source snapshot and all earlier runs. The shared `ExperimentLogger`
records evaluation identity, checkpoint lineage and output receipts. Three CPU
contracts passed locally, two metric contracts passed remotely, and the complete
CUDA test inference finished successfully. No simulator or artifact publishing
was executed.

The Graph-team trainer now supports optimization, validation and checkpoint
resume. It is independent of prediction layout: a task supplies the loss and
scalar metrics. The first supported training recipe is gated MPNN Object21
supervision from the Visual corner detector's output and raw saved GT.

For the matched Baseline/A/B/AB experiment, see
[vertex geometry and readout ablation](#vertex-geometry-and-readout-ablation).

## Run on the training host

### Train from a downloaded cache

After extracting the published `graph_cache.tar` into `/workspace/gnn_cache`, use:

```bash
source /venv/main/bin/activate
cd /workspace/graph-recurrent-VLA
python main.py --config graph/configs/train_gated_mpnn_cached.yaml
```

The dedicated config selects `training.gnn.cached_data:build_data`. Set
`data.cache_dir` to the extraction directory, whose contents must include the
preprocessing namespace and `metadata/data_identity.json` plus
`metadata/cache_inventory.json`. Raw episodes, simulator packages, OpenCV and the
original detector runtime are not required by this data factory. Torch, NumPy
and PyYAML are needed to deserialize batches and execute the trainer.

The loader retains the published episode assignments, never reads reserved test
frames and never regenerates a missing cache entry. Indexing checks metadata
consistency, split overlap, frame counts, file existence and sizes. Each sample
read checks the file's SHA256 against the published inventory, its embedded
source identity, graph schema and target shape. It does not hash original RGB-D
or execute current preprocessing to choose a new namespace. The original
preprocessing versions/hashes are preserved as provenance. Graph schema version
1 is supported; incompatible entries fail validation. Camera, graph and input-
confidence settings must match the archive; replay cannot change cached features.

The existing data factory remains available for raw data and on-demand cache
creation. Cache-only replay uses a distinct dataset identity combining the
original dataset hash, inventory bytes and empty-input policy. Checkpoints from
raw mode cannot silently resume into this mode. Same-cache resume across paths
uses the same identity. DataLoader seed controls shuffling and model RNG, while
the archive's saved split is authoritative; validation/test fraction defaults
are not used to resplit it. `split_file`, frame stride other than 1, and frame
limits are rejected to prevent silently training on a different subset.

Cache-only loader contracts can be checked without training or real cache
generation using `python -m unittest tests.test_gnn_cached_data -v`.

### Train from raw episodes

Use Python 3.11 and install `requirements.txt`,
`visual/corner_detection/requirements.txt`, and optionally
`requirements-tracking.txt`. Extract the approved dataset TAR shards so
`dataset_manifest.json` and its `shard_*/episode_*` paths resolve from one root.
Set `data.root` in `graph/configs/train_gated_mpnn.yaml` to that root, then run
from the repository root:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/train_gated_mpnn.yaml
```

For a single collector directory instead, set `data.manifest: manifest.json`.
The directory must have a COMPLETE manifest, committed episode receipts and
per-frame GT (`save_step_gt: true` during collection). This workflow does not
start the simulator, collect data, extract archives or publish artifacts.
W&B disabled mode retains the shared logger's local records.

`device` selects the GNN device. `detector.device` independently selects corner
extraction (CPU by default; select CUDA on the server if supported). GPU detector
execution requires `data.num_workers: 0`; CPU detection can use worker processes.
The loader constructs graphs on CPU and the batch moves to the selected GNN
device. There is no local training or cache preparation command run automatically
when importing modules, validating config or constructing a dataset.

The gated recipe enables CUDA model compilation:

```yaml
training:
  compile:
    enabled: true
    dynamic: true
```

Other configs default to `enabled: false`. Requested compilation runs only on
CUDA; CPU/MPS keep eager execution. The trainer records the request and actual
activation under `model_compilation` in the shared experiment metadata. It uses
`model.compile(dynamic=True, mode="default", fullgraph=False)` after checkpoint
restoration. This calls `torch.compile` in place, preserving parameter identities
and plain checkpoint keys without an `_orig_mod.` prefix. Both training and
validation use compiled calls; initial calls or new graph shapes can trigger
compilation. Compiler failures propagate rather than silently changing execution.

Gated node encoding, embeddings, attention scores and deterministic message
updates now compute dense candidates and mask their results. Edge-type selection
and empty-edge early returns are retained, including `None` gradients for absent
edge groups. With nonzero dropout during training, message updates retain the
selected-node path to preserve the original dropout draw count/order. Saved
weights remain compatible; exact eager/compiled resume trajectories are not
guaranteed, and compiler settings are recorded but are not locked by the existing
resume signature.

## Modules

| Module | Responsibility |
| --- | --- |
| `training/gnn/trainer.py` | Shared optimization, validation, logger and checkpoint selection |
| `training/gnn/tasks.py` | Task contract, StepResult and Object21 loss/metrics |
| `training/gnn/data.py` | Splits, optional caches, loaders and graph collation |
| `training/gnn/checkpoint.py` | Atomic epoch snapshots and RNG/state restoration |
| `training/gnn/loss.py` | Existing Object21 and paired-delta objective functions |
| `data_collection/episode_dataset.py` | Approved raw frame/GT reading, no GNN dependency |
| `data_collection/vertex_family_manifest.py` | Vertex-family manifest hash and parent-split verification, no training dependency |
| `graph/utils/object21_targets.py` | Raw world GT to Graph-team Object21 targets |
| `graph/vertex_aggregation.py` | Optional global geometry and multi-seed readout |
| `training/gnn/vertex_ablation.py` | Matched four-model training through the existing trainer |
| `evaluation/gnn/vertex_ablation.py`, `vertex_metrics.py` | Paired validation and physical/observation-count diagnostics |

Shared prediction alignment and parent-cluster bootstrap helpers live in
`evaluation.gnn.vertex_metrics`; their previous imports from
`evaluation.gnn.vertex_diagnostics` remain available. Vertex-family manifest
readers use `data_collection.vertex_family_manifest.read_manifest`; the previous
`training.gnn.vertex_family_data.read_manifest` import remains available.
Canonical JSON hashing lives in `data_collection.episode_dataset`, with the
same byte encoding and compatibility import from `training.gnn.data`.

The old `training.gnn_trainer` and `training.gnn_loss` modules moved to the paths
above. Update external imports accordingly. The common entrypoint remains
`python main.py --config ...`, with `trainer: training.gnn.trainer:train`.


## Temporal estimation guides

Frozen-GNN temporal extensions have dedicated feature guides:

- [Temporal estimation](temporal_estimation.md): Hole GRU, Pose GRU and insertion
  GRU/Mamba models, objectives, sequence data and measured comparisons.
- [Hybrid estimation](hybrid_estimation.md): frozen Mamba/Gate composition,
  Peg-head fine-tuning and saved-prediction error analysis.

The rule memory and learned Gate below retain their existing entrypoints.

## Frozen GNN temporal-memory evaluation

```bash
WANDB_MODE=disabled python main.py --config graph/configs/evaluate_temporal_memory.yaml
```

`evaluation/gnn/temporal_memory.py` loads the exact model/data configuration from
a cache-only `best.pt`, checks the cache dataset identity, freezes the model and
evaluates all saved validation frames. Only checkpoint/cache locations, inference
batch size and evaluation settings are overridden. No optimizer, detector, cache
generation, checkpoint overwrite or reserved test access occurs. Local CPU tests
use synthetic inputs; real inference is intended for the training host.

`graph/utils/hole_memory.py` keeps a separate causal memory per episode, replayed
by numeric frame index. Its input gate uses observed BOX/HOLE node count and saved
corner-confidence metadata, never GT rim identities, visibility labels or errors.
Node count does not establish that four aperture corners were detected. Unknown
confidence is rejected by the primary policy. The shipped policies are fixed
before this comparison, rather than selected by a validation threshold sweep:

- `count_only`: update with at least five observed target-side nodes.
- `quality_gated`: require at least five target-side nodes, confidence on all
  of them, mean confidence at least 0.82, and three consecutive qualifying frames.
  Each candidate world position must be within 10 mm of their coordinate-wise
  median; save that median when the window passes.

Consecutive windows clear on rejected quality or missing frame indices. Stored
positions persist without an expiry because this baseline assumes a stationary
hole and a consistent world frame. An accepted, stable later observation replaces
the stored value, including after occlusion. This can still store a stable wrong
prediction; the heuristic confidence is not a calibrated accuracy probability.

The memory stores `predicted_hole_relative + tcp_world`, then returns
`stored_hole_world - current_tcp_world`. Before initialization it returns the
current GNN prediction and marks memory unavailable. Outputs include initialized,
updated, held, quality-accepted and age-in-frames flags. Episode boundaries reset
all state. Changing the position also corrects the endpoint vector by the opposite
amount, preserving the model's estimated peg tip. Hole rotations/radius and all
peg outputs remain current-frame GNN outputs. The exported arrays can be consumed
by a later recurrent workflow; this evaluation does not modify or run a GRU.

`outputs/temporal_memory/<run_id>/` records effective config, checkpoint/source
hashes, per-frame predictions/flags, summary and reacquisition events through
`ExperimentLogger`. GT and optional earlier depth-visibility labels are read for
scoring after replay, not for memory decisions. The evaluator checks stationary
GT position within `stationary_tolerance_m`; it fails if that assumption does not
hold. Diagnostic occlusion labels align by sample ID and describe approximate
7x7 depth support, not actor-segmentation visibility GT.

Metrics include overall and low-observation errors, optional strong-occlusion
errors, endpoint error, equal-episode means, initial memory error/delay, held and
cold-start errors, and wrong-initialization recovery/censoring. `bad_error_mm`
(10 mm by default) is a reporting threshold only. Reacquisition starts when the
observation gate changes from rejected to accepted after memory initialization.
Update latency is measured within that continuous accepted run; short/unstable
runs without updates are reported separately. Before/after error comparisons
refer to those reacquisition events, not independent random frames. All results
are descriptive validation evidence, not an independent test or insertion-success
estimate. Run contracts with
`python -m unittest tests.test_hole_memory tests.test_gnn_temporal_memory -v`.

### Measured validation comparison

On 2026-11-06 KST, Vast.ai instance 59373350 executed CUDA eager inference with
the unchanged source `best.pt` from training run
`4186cdeb0fdb449ea443f28e3bddc936`. Evaluation run
`d5b9bc38666a4d37ab166be778567a64` used 18,751 validation frames from 180 episodes.
The repository base was `3b13a6b`; evaluation/memory additions were uncommitted,
with source hashes recorded in `provenance.json`. No training, detector execution,
cache creation or test evaluation ran. Maximum reconstructed GT hole drift
within an episode was 6.46e-8 m. Gate settings above were fixed before execution.

| Hole position metric, mm | Current GNN | Count-only memory | Quality-gated memory |
| --- | ---: | ---: | ---: |
| All-frame mean | 26.59 | 7.35 | 8.19 |
| Strong-occlusion-proxy mean, 11,127 frames | 37.36 | 7.68 | 9.19 |
| Fewer than three target nodes, 10,035 frames | 41.37 | 7.51 | 9.73 |
| All-frame 90th percentile | 70.42 | 14.16 | 14.17 |

Quality gating improved 163/180 episode means, but initialized only 166 episodes
versus count-only's 179. Its 1,965 cold-start frames averaged 28.63 mm, while its
13,699 held frames averaged 5.85 mm. Thus stricter gates improved stored/held
estimates but reduced coverage; the primary quality gate did not outperform the
simpler policy on total mean error. Initial stored error averaged 4.57 mm, with
5/166 initializations above the diagnostic 10-mm threshold. All five later had a
qualifying update within 10 mm (mean 10 frames to recovery).

After initialized memory saw rejected then accepted quality, the quality policy
had 199 reacquisition events: 107 updated in that accepted run (mean latency
2.14 frames), and 92 did not. Of 14 updated events with prior error above 10 mm,
11 ended within 10 mm. Across all updates, the postprocessing audit found 55
overwrites from <=10-mm memory to >10-mm memory and a longest uninterrupted wrong
hold of 105 frames. Stable observations and heuristic confidence do not guarantee
accurate predictions. In the paired 107 updated reacquisitions, mean error went
from 6.55 to 7.73 mm, so reacquisition did not improve an already good memory on
average. This motivates initialization coverage and overwrite safeguards before
adding a learned recurrent component; these results do not establish insertion
success or independent-test generalization.

Ignored raw artifacts, available on the evaluation host and the local checkout:
[summary](../outputs/temporal_memory/d5b9bc38666a4d37ab166be778567a64/summary.json),
[update audit](../outputs/temporal_memory/d5b9bc38666a4d37ab166be778567a64/memory_update_audit.json),
[comparison plot](../outputs/temporal_memory/d5b9bc38666a4d37ab166be778567a64/temporal_memory_comparison.png).
Local verification passed 55 contracts across memory, cache replay and shared
experiment infrastructure; only synthetic data was used locally. CUDA inference
ran on the server, not on the local Mac. No simulator backend was executed.

## Learned temporal update gate

`graph/temporal_gate.py` adds an independent scalar MLP (30 → 32 → 32 → 1,
ReLU/ReLU/Sigmoid, **2,081 trainable parameters**). It does not change
`graph/gated_mpnn.py`, its State Head, or source checkpoints.
`training/gnn/temporal_gate.py` loads the exact `best_weights` architecture and
state, applies `eval()` and `requires_grad_(False)`, and checks the original cache
identity before inference. Full training and large evaluations require separate
user approval; the commands below describe the prepared workflow.

Prerequisites: Torch, NumPy, PyYAML, Matplotlib; original `best.pt`; matching
extracted cache; and actual source parent identities. No simulator, raw RGB-D,
detector runtime, or network connection is needed. Use the repository root:

```bash
python -m data_collection.temporal_gate_data \
  --cache-dir /workspace/gnn_cache \
  --parent-map /workspace/parent_by_episode.json \
  --output /workspace/gnn_cache/metadata/temporal_gate_splits.json
WANDB_MODE=disabled python main.py --config graph/configs/train_temporal_gate.yaml
WANDB_MODE=disabled python main.py --config graph/configs/evaluate_temporal_gate.yaml
```

Adjust paths in both configs; set evaluation `gate_checkpoint` to the newly
trained run's `best.pt`. `--parent-map` is a JSON mapping such as
`{"shard_000001/episode_000002": "actual_collection_parent_id", ...}`. Include
**all** source train/validation/test episode IDs, even reserved episodes without
cached frames. Published cache metadata contains episode IDs but no parent
lineage. Never assume shard IDs are parent IDs. Only use an episode ID as its
own parent when source provenance confirms independence. The tool writes a new
manifest exclusively (`open('x')`), without touching caches or existing files.
The manifest stores `splits`, `parent_by_episode`, seed and validation fraction.
No real-data manifest was fabricated on this host.

Splits are assigned by parent before any frame is read. Gate train and gate
validation use only the original **GNN train** pool. Parents shared with original
GNN validation/test are excluded. Gate test is the **complete original GNN
validation split**, preserving the previous evaluation episodes; the original
GNN test remains reserved and is never inferred. Gate validation selects the
best gate checkpoint; gate test does not select weights or thresholds. Only
whole episodes are shuffled for training; frames are sorted and must be
contiguous. This protocol provides a final gate test on the established GNN
validation set; it does not claim evaluation on the reserved GNN test set.

The observation feature vector has 17 entries. For BOX and HOLE separately:
valid count, present flag, missing flag, fraction with known confidence,
known-confidence present flag, and known-confidence mean/min/max; finally,
all-vertices-missing. Unknown statistics use zero **with explicit validity and
coverage information**. Padding and invalid confidence never contribute.
The schema does not provide an expected vertex count or a missing-corner
identity, so the implementation does not invent either. Confidence is the
existing heuristic observation score, not calibrated prediction confidence.

Gate input combines those 17 values with current GNN hole world XYZ, previous
memory world XYZ, their signed difference, current TCP world XYZ, and previous
memory initialized flag. Position inputs are divided by a configurable 0.1 m
scale. TCP-relative coordinates use world-aligned axes: `world = relative + tcp`;
TCP quaternion rotation is not appropriate for this representation.

At episode start, previous memory is absent (zero placeholder plus initialized=0
in the MLP input). Effective alpha is **1 on the first frame**, seeding memory
with the current GNN prediction even when vertices are absent. This is an
explicit cold-start policy, independent of node-count/confidence thresholds or
GT. Subsequent alpha is entirely learned in [0,1]. A poor first prediction can
therefore initialize poor memory; initialization failures and subsequent recovery
are scored. Every new episode calls rollout with `memory=None`.

Training uses full-episode autoregressive BPTT with the model's own previous
memory; no teacher forcing, previous GT input, or future observation input.
GT hole world XYZ supervises each step using Huber loss (default delta 0.01 m)
or configurable L1. Loss is the coordinate mean in metres, aggregated over
frames; validation uses the same loss without gradients. First-frame loss is
included but cannot train the forced cold-start alpha. Episode-level optimizer
steps and gradient clipping bound the simple prototype. Large episodes and the
full frozen-output arrays may require substantial RAM/device memory; there is
no streaming cache or resume optimizer support in this initial implementation.

`correct_object21()` replaces [10:13]. Its default also adjusts [7:10] by
`old_hole - new_hole`, reusing the geometric convention of existing
`replace_hole_position()`: the estimated peg tip remains fixed. Every other field
is copied unchanged. `corrected_state38(robot15, object21, freshness2,
hole_relative)` calls the existing `assemble_state38()` with this corrected
Object21; robot and freshness values are supplied unchanged by the caller.
The gate does not fabricate robot/freshness inputs absent from the graph cache.
Evaluation additionally assembles a hole-only variant and reports endpoint-vector
GT errors, preserved-tip residual and the inconsistent hole-only tip shift.

Evaluation runs frozen GNN, the unchanged `HolePositionMemory` with the existing
quality-gated recipe, and the learned gate on identical frames. It reports mean,
median, p90, episode-balanced mean, fewer-than-three BOX/HOLE observations,
zero BOX/HOLE observations, first-frame failures, rule initialization lifecycle,
learned bad initialization recovery, harmful updates, and reacquisition latency.
A harmful update worsens error relative to holding the previous world memory
at the current frame; a separate count records crossings of `bad_error_mm`.
This threshold defines diagnostics only and never controls learned alpha.
Reobservation means returning BOX/HOLE presence after an absent run; recovery
latency is time until error first falls below `bad_error_mm` within that visible
run. Optional existing depth-visibility labels add the same strong-occlusion
proxy and its recovery events; without labels, occlusion availability is false
and no true occlusion result is claimed. GT stationary-hole drift is checked
only after inference.

New UUID output directories contain effective config (source model config,
feature names, split manifest and source/checkpoint hashes), gate-only `best.pt`
and training history, or evaluation summary, predictions, recovery events,
provenance and per-episode PNG traces. Traces align effective alpha with all
three hole-error curves and shade absent BOX/HOLE observations. Evaluation
checks the frozen checkpoint hash, dataset identity, exact split manifest hash,
feature ordering and gate settings against the trained gate. A new logger run
records each invocation through `ExperimentLogger`; artifacts remain local and
are registered through its existing manifest mechanism. No automatic publishing
or new W&B implementation is introduced.

Verification on the current CPU host:

```bash
WANDB_MODE=disabled .venv/bin/python -m unittest \
  tests.test_temporal_gate tests.test_hole_memory \
  tests.test_gnn_temporal_memory tests.test_gnn_cached_data -v
```

The 29 tests pass, including 10 new tests: changing TCP/world coordinates,
causal prefixes, episode resets, GT isolation, gradient flow through previous
predictions, real feature masks, parent leakage prevention, geometry assembly,
frozen-model CPU inference and short workflow smoke tests. One smoke test writes
a tiny actual graph cache and a reduced real VertexEdgeGatedMPNN checkpoint,
then runs both root-entrypoint workflows with one training epoch and final
comparison. Another uses varying synthetic predictions to exercise gate updates.
All fixture outputs are temporary and removed. These are plumbing/correctness
checks, not performance evidence. The specified production checkpoint/cache
were unavailable locally. Full training, the 18,751-frame comparison, CUDA/MPS,
and simulator execution were not performed. No improvement over 26.59 mm or
7.35 mm is claimed.


## Data, splits and cache

Only manifest-listed episodes with valid receipts are read. Partial/rejected
paths and repeated reset seeds are rejected. Splitting is deterministic by seed
and occurs by episode before frame selection. The default reserves 20% validation
and 10% test, each at least one episode; at least one training episode must remain.
Test episodes are reserved, never passed through detection or validation.

An optional `data.split_file` JSON can specify exact `train`, `validation` and
`test` episode-ID lists. Every approved episode must occur exactly once. Keep
this split file for future evaluation. `frame_stride` and optional
`max_frames_per_episode` are explicit sampling controls, not automatic subsampling.

With `data.use_cache: true`, each requested frame is detected and graphed on its
first access, then saved as a compact NPZ below `data.cache_dir`. Subsequent
epochs reuse it. Dataset construction only indexes/hashes source files and does
not write caches. Validation frames are cached when validation first reads them.
`data.use_cache: false` runs detection/graph construction for every access.

Keys include raw frame SHA256, sample ID, camera, complete detector parameters,
backend, graph/confidence settings, library versions and preprocessing code
hashes. Corrupt or mismatched entries raise an error. Entries use no pickle and
are published by atomic rename. Changing settings/code creates a different
namespace. Raw data changing after indexing is rejected on cached access.
No GT corner identities, object sizes or hole pose enter detection or graph
construction. Saved TCP position is treated as proprioception; remaining GT
is used only to construct supervision.

`data.empty_inputs: keep` retains empty detections with valid GT and reports
`empty_input_fraction`. `skip` filters empty samples during collation; metrics
then describe the retained subset. If a split has no usable batches, training
fails instead of reporting a fabricated loss. Padding is excluded from graph
operations. Custom node-level tasks must also mask padding in their losses.

Graph batches are validated during collation; the gated MPNN does not repeat
validation in `forward()`. Direct callers and custom data factories must validate
their graphs before model execution. When semantic IDs are enabled, pass the
model's `num_semantic_ids` to `GraphBatch.validate` to check vocabulary bounds.

## Objective and extension contract

Object21 uses TCP-relative positions with world-aligned axes, world wxyz
quaternions, and the peg +X tip relative to the hole position. It reuses
`object21_loss` without changing its weights. Loss normalization scales and
endpoint consistency weight are recorded under `task.settings`. Paired-delta
loss remains available but is not used in this recipe.

`Task.compute(model, batch)` returns:

```python
StepResult(
    loss=differentiable_scalar,
    metrics={"loss": detached_scalar_tensor, "another_metric": python_float},
    metric_weights={"loss": sample_count, "another_metric": its_denominator},
)
```

Metrics are means with explicit positive denominators, so uneven final batches
are aggregated correctly. Scalar tensors and Python numbers are accepted;
Object21 returns detached tensors. The loop accumulates weighted metrics on the
model device and transfers the epoch means to CPU together for Python-number
logging. CPU/CUDA accumulation uses float64; MPS uses float32. Loss and metric
finiteness are checked together each batch, retaining one host decision before
an optimizer step. Object21 also records position-error norms in metres.
The loop handles backward/optimizer only in training and disables gradients in
validation; it never accesses Object21-specific output keys.

With gradient clipping enabled, `clip_grad_norm_(error_if_nonfinite=True)`
checks gradient finiteness before the optimizer step. Without clipping, the
loop reduces per-parameter finiteness flags on the device and checks the combined
result once. These checks retain host synchronization; they do not make the
entire training loop suitable for full-graph compilation.

Another GNN can use the built-in model selector or `model.factory:
package.module:Constructor`, with the remaining model mapping passed as keyword
arguments (excluding `name`/`factory`). Select `task.factory` and task constructor
arguments in `task.settings`. If targets/data differ, a `data.factory` callable
accepts `(effective_config, shuffle_generator)` and returns `DataBundle` with
train/validation iterables and `identity.dataset_sha256`; batches provide
`.to(device)`. All factories use `module:callable` notation. No trainer rewrite
is needed. BasicGNN currently returns node tensors, so Object21Task is unsuitable
without a compatible readout. Its example config retains `mode: initialize`.

## Outputs and resume

### Loss and geometry metrics

Every epoch records sample-weighted train/validation means through the existing
`ExperimentLogger` and `history.json`. In addition to total loss, its raw
components, position/endpoint mean distance errors (metres), and empty-input
fraction, Object21 records:

| Metric suffix | Meaning |
| --- | --- |
| `peg_rotation_error_deg`, `hole_rotation_error_deg` | Mean shortest quaternion rotation angle in degrees; q and -q are equivalent |
| `peg_half_size_x_error_mm`, `peg_half_size_y_error_mm`, `peg_half_size_z_error_mm` | Mean absolute peg half-size error on each local axis, in millimetres |
| `hole_radius_error_mm` | Mean absolute hole radius error in millimetres |
| `weighted/<loss_component>` | Contribution after the existing loss weight; these sum to total loss |
| `empty_input/<geometry_error>`, `nonempty_input/<geometry_error>` | Validation-only geometry errors, separately for samples with no valid nodes and samples with at least one valid node |

Subset means use each group's own sample count across batches. A group with no
samples is omitted, rather than assigned zero error. With `empty_inputs: skip`,
the empty group is filtered before the task and therefore has no error metrics.
Validation grouping transfers one group count to the host per batch; other
geometry metric calculations remain on the model device until reporting.

Training also records `gradient_norm` (mean global L2 norm before clipping) and,
when clipping is enabled, `gradient_clip_fraction` (fraction of optimizer steps
whose gradients were scaled). These are averaged by optimizer step, independently
of sample-weighted loss metrics. Norm measurement does not modify gradients;
the clipping path reuses the norm returned by the existing clipping operation.

At the existing progress cadence, the logger records cumulative means for the
current phase under `progress/train/*` or `progress/validation/*`, and the
terminal includes cumulative mean `loss`. Means reset at each phase and epoch;
they describe all processed samples in that phase, rather than just the latest
batch. Device scalars are copied together only when a progress report is due.
The objective, loss weights and best-checkpoint selection are unchanged.

### Time and progress

During training, the terminal reports the current/total epoch, train or
validation phase, completed/total phase batches, elapsed time within the epoch,
estimated time remaining within the epoch, and estimated time remaining across
all epochs. `training.log_interval_seconds` defaults to 10 seconds; reports are
emitted at batch boundaries and phase transitions. A slow batch (including
first-access detection/cache creation) can therefore delay the next report.

The same numeric measurements are saved through `ExperimentLogger` under
`progress/*` in `experiments/<run_id>/metrics.json` and W&B when enabled.
`time/epoch_seconds` and each history row's `seconds` record epoch wall time
through checkpoint writing/registration, before writing that history row.
Epoch time includes both training and validation. Timing starts after dataset
indexing, model construction and resume loading; it does not include preparation.

ETAs use measured seconds per loader batch, separately for training and
validation once each has been observed. Initially the observed rate also
estimates the unobserved phase; until a batch has completed, estimates display
`estimating` and ETA scalars are omitted. Future epochs use the mean duration
of completed epochs in this invocation, including checkpoint overhead. Resume
uses the restored epoch number and measures rates afresh. Cache warmup and
variable batch costs can make initial estimates inaccurate. Custom unsized
iterables retain elapsed-time logging but show unknown batch totals and omit
ETAs. Measurements use the host wall clock without adding accelerator
synchronization; CUDA/MPS timing has not been verified on this host.

Each invocation creates a fresh ExperimentLogger UUID and a separate
`output.root/<run_id>/` directory containing effective config, data identity and
splits, epoch history, `last.pt` and `best.pt`. The logger records scalar metrics
and registers outputs through `link_output`; it does not upload checkpoints.

`last.pt` contains model, optimizer, scheduler, completed epoch, earlier best
weights, configuration, dataset hash and Python/NumPy/Torch/accelerator/shuffle
RNG state. Resume sets top-level `resume` to its path and `training.epochs` to a
larger **total** epoch count. The new run has its own history and source sidecar
identity. Data/split/preprocessing hash, objective, optimization settings, seed,
batch settings and device type must match. Best weights survive even if no
resumed epoch improves validation loss.

`best.pt` is a weights-only validation-loss selection snapshot, for later model
loading, not training resume. Exact continuation additionally depends on the
same runtime and deterministic backend behavior; CUDA reproducibility has not
been established here. Best validation loss is not simulator success rate.

## Verification limits

### CUDA masking and compilation measurement

On 2026-11-06, an isolated source snapshot was measured on Vast.ai instance
58752018 (RTX 3090, PyTorch 2.14.1+cu132). The local config at capture used
hidden 64, two layers, dropout 0, batch size 32 and AdamW. The active cache
generator on instance 59288396 (PID 1545) was left running. Only 96 completed
cache files were read from that instance; benchmark code and compiler caches
were placed in a separate directory on the idle instance. Existing remote
repositories and Python environments were not modified.

The samples formed three batches with padded node counts 15/16/12 and edge
counts 1664/1586/1374. Timings cover the copied local training loop with
GPU-resident batches: model, loss, metrics, backward, gradient clipping and
AdamW. Each case used five 30-step windows, repeated in reverse case order
in a second process execution; the table averages both executions. Disk reads,
preprocessing, transfers, checkpointing, logger/progress output and initial
compilation are excluded. These are compute-loop measurements, not end-to-end
epoch times or convergence results.

| Benchmark-only model variant | Eager ms/step | Compiled ms/step |
| --- | ---: | ---: |
| Current masked selection | 28.59 | 27.07 |
| Dense node encoding, pooling and updates; existing edge selection | 24.97 | 21.40 |
| Dense nodes and all edges, with masked message weights | 21.65 | 19.51 |

Compilation used `torch.compile(model, dynamic=True, fullgraph=False)` in default
mode. The first pass over all three shapes, including compilation/backward,
took about 8–14 seconds; variants shared a fresh, isolated compiler cache, so
these are not independent cold-cache comparisons. Current-code execution means
ranged from 27.14–30.04 ms eager and 25.88–28.26 ms compiled, so the small
compile-only benefit overlaps execution variability. Dense nodes plus compilation
reduced mean step time by about 25% (1.34x throughput) versus current eager code.
The current compiled model recorded `aten.nonzero` graph breaks.

All six cases matched reference outputs and gradient values within the recorded
tolerances on these batches. CPU checks additionally covered empty, isolated,
single-object and padded graphs. Dense nodes preserved gradient presence in
these checks. The all-edge candidate changed some absent-edge parameter
gradients from `None` to zero: AdamW can then update parameters it previously
skipped. It is therefore an experimental speed comparison, not a replacement
with equivalent optimizer behavior. Nonzero-dropout RNG equivalence and broader
graph-shape coverage remain unverified by that benchmark. The measured dense-node
candidate was subsequently adopted in the production model, with the original
selected-node update retained for nonzero-dropout training. The all-edge
candidate remains benchmark-only.

Local evidence: [summary](../outputs/gnn_compile_benchmark/summary.json),
[first execution](../outputs/gnn_compile_benchmark/initial_results/),
[reverse-order execution](../outputs/gnn_compile_benchmark/results/), and
[edge cases](../outputs/gnn_compile_benchmark/source/edge_cases.json).
The ignored [source snapshot](../outputs/gnn_compile_benchmark/source/) contains
sample identities/hashes, exact local loop functions, variant generation and
benchmark scripts. From that directory on an idle CUDA host, rerun with
`OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 python benchmark.py --variant dense_nodes --compile`.
The benchmark refuses to start if another CUDA compute process is present.

The original verification did not run training or generate frame caches.
Contract tests check targets, splits, collation, both model output contracts in
validation, settings and checkpoint/RNG round trips using untrained weights:

```bash
python -m unittest tests.test_gnn_trainer tests.test_gnn_training \
  tests.test_graph_builder.ConfidenceTests tests.test_graph_builder.GraphBuilderTests -v
```

The optimizer/resume integration test is disabled by default. On the training
host it can be explicitly enabled with `RUN_GNN_TRAINING_TESTS=1`; it uses synthetic
graph fixtures and does not establish dataset quality. Real-dataset training,
cache materialization and CUDA/MPS training were not run for that original
verification. The supplied data path is a placeholder, not a bundled dataset.
The later vertex-ablation checks below include CPU synthetic optimizer/resume
tests and MPS forward/backward checks.

## Vertex geometry and readout ablation

### Run the matched protocol

Prerequisites: Python 3.11, Torch, NumPy and PyYAML; the original trained gated
MPNN `best.pt`; and the matching extracted published cache, including both
metadata JSON files and all inventoried NPZs. No simulator, detector execution,
remote launcher or artifact publishing is part of this workflow.

Set `reference_checkpoint` and `data.cache_dir` in
`graph/configs/train_vertex_ablation.yaml`. Its `base_config` points to the
existing `graph/configs/train_gated_mpnn_cached.yaml`, whose training recipe is
shared by every variant. Run from the repository root:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/train_vertex_ablation.yaml
# Set manifest in the evaluation YAML to the emitted ablation_manifest.json.
WANDB_MODE=disabled python main.py --config graph/configs/evaluate_vertex_ablation.yaml
```

The matrix trains **all four models from scratch**. The source checkpoint is
audited for architecture/data provenance, never reused as the matrix baseline
score or initialization. The preflight checks the actual saved model settings
and dataset identity against the selected cache. The loader retains the saved
split, checksum checks, input tensors, Object21 targets and empty frames. It
neither reruns the detector nor changes the episode assignments. A different
cache or changed baseline dimensions/semantic/confidence settings is rejected.

The selected recipe uses seed 0, 30 epochs, batch size 32, AdamW (lr 0.001,
weight decay 0.0001), StepLR (10 epochs, gamma 0.5), gradient clipping 1.0 and
the unchanged Object21 loss/normalization. All variants use the same loader
seed/order, epoch/update budget, optimizer, scheduler and minimum-validation-loss
checkpoint selection. Shared original weights start identically; A/AB also
share geometry initialization and B/AB share readout initialization. Optional
modules preserve the later training RNG through separate initialization scopes.
CUDA compilation follows the existing recipe; CPU/MPS remain eager. Equal
settings do not guarantee bitwise accelerator determinism.

`parallel_models` defaults to 1; values 2–4 run independent trainer processes
using the `spawn` context, including on one shared GPU. Each process owns its
model, optimizer, seeded RNG and output directory. `data.num_workers` in the
matrix YAML can override only the loader worker count; it does not alter cached
samples or their saved split. Set these counts to fit host RAM and GPU memory.
Parallel console logs are retained under the matrix's `model_logs/`; completed
checkpoints enter the same audited manifest. Wall time under shared GPU load is
not an isolated throughput benchmark. The existing epoch loop is unchanged.

Each child keeps the existing trainer's UUID output directory, config snapshot,
data identity, history, `last.pt` and `best.pt`. The matrix writes a progressively
updated `ablation_manifest.json` under its own UUID directory, with source and
new checkpoint SHA256s, effective protocol and parameter counts. Reference
training-setting differences are recorded if a copied base recipe changes the
budget. Existing checkpoints are never overwritten. The shared
`ExperimentLogger` records the matrix, four child training runs and evaluation;
no new W&B or publishing implementation is introduced. A failed/incomplete
matrix cannot produce a four-model evaluation table. After a failure, existing
children remain preserved; the matrix launcher currently starts a new matrix
rather than automatically resuming an incomplete one.

Evaluation checks all four checkpoint hashes, flags, dataset identities and
training protocols, then matches sample IDs, targets and observed counts exactly.
It reads the complete saved validation split and leaves the reserved test split
unread. This is descriptive validation evidence: the same validation split also
selects the checkpoints, so it is not an independent test result. For a relocated
but identical cache, an explicit `data.cache_dir` override may be supplied in the
evaluation YAML; inventory/identity checks still apply.

### Existing architecture and actual checkpoint

On 2026-11-07, the local source
`outputs/hybrid_temporal_estimation/inputs/gnn_best.pt` strictly loaded with:

| Item | Saved setting / inspected behavior |
| --- | --- |
| Model | `VertexEdgeGatedMPNN`, hidden **64**, **2** layers per object side |
| Semantic IDs | Vocabulary 16, embedding width 32, **disabled** |
| Confidence | **Disabled**; present nodes have effective confidence 1 |
| Other settings | Edge embedding 16, dropout 0, coordinate scale 0.05 m |
| Size scales | `[0.04, 0.01, 0.01, 0.02]` m |
| Parameters | **234,927**, including stored but disabled semantic embedding |
| Checkpoint SHA256 | `f527ab8f2b70129bfe845cf362a304bc2da195ac24e049d6de4cd4c28432100e` |
| Cache-only dataset SHA256 | `21a03a363ec5d6e1dd391fc014b8e3ad1455c5eaf0489153a9c39eed2d133292` |

The class defaults remain hidden **128**, **3** layers; those are not the saved
experiment settings. Saved tensor shapes independently confirm node encoder
`[64,45]`, message input `[64,149]`, and peg head input `[64,66]`. The source
validation loss is 3.0327817512432054; it is **not** a matched four-model result.
The original cache is absent on this host, so its full frame inventory, node
count distribution and split membership could not be independently replayed
here. Runtime preflight verifies them when that cache is supplied.

The existing flow in `gated_mpnn.py` is:

1. Each node encodes 45 values: object-centered world-axis XYZ / 0.05,
   effective confidence, observed flag, semantic embedding (32), and object-type
   embedding (8). The centroid is confidence-weighted when confidence is enabled;
   unknown confidence is neutral. Disabled/unknown semantics contribute zeros.
2. Each edge encodes 21 values: sender-minus-receiver XYZ / 0.05, distance /
   0.05, observed-pair flag, and a 16-value type embedding. Default
   `GraphBuilder` settings create complete directed edges without self edges.
   PEG, BOX and HOLE retain separate type IDs. Edges within a type use that type;
   edges between different types are CROSS, including BOX↔HOLE. Optional kNN
   selection remains unchanged; tied-neighbor selection is input-order sensitive.
3. Two separate PEG and target-side message stacks each have two layers. The
   target stack handles BOX/HOLE intra-type edges; one shared cross layer then
   processes all CROSS edges. For concatenated sender/receiver/edge features,
   each message is `sigmoid(gate) * message * sender_confidence`. Receivers sum
   messages and divide by `max(1, sum(sender_confidence))`, followed by an MLP,
   residual, dropout and LayerNorm. Unsupported nodes retain their old features.
4. `ConfidenceAttentionPool` computes one scalar attention per node, adds log
   confidence, masks padding, and produces one hidden-width weighted sum per
   side. Each side adds `[present, mean_confidence]`, making 66 head inputs.
   Empty groups have zero pooled features and summary, but heads still produce
   their learned prior. BOX/HOLE share target-side readout; `hole_observed`
   reports actual HOLE nodes separately. The detector's default object_id=1 is
   BOX; HOLE labels require an explicit upstream classification.
5. The unchanged peg head emits position3/quaternion4/halfsize3, and hole head
   emits position3/quaternion4/radius1. Positions offset the observed centroid;
   identity-offset wxyz quaternions normalize to unit norm; dimensions use
   softplus and the saved scales. The existing endpoint residual head remains.

Thus all-to-all relative geometry, distance, object centering and learned
attention already exist. A introduces a different aggregation bias over that
information; it does not make previously unobserved corners available. B tests
the single-vector readout's aggregation, while still projecting to the same
hidden-width head input.

Object21 remains `[peg_pos(0:3), peg_quat(3:7), endpoint_minus_hole(7:10),
hole_pos(10:13), hole_quat(13:17), peg_halfsize(17:20), hole_radius(20:21)]`.
Positions are TCP-relative in world axes; the endpoint vector is also in world
axes, not rotated into the hole frame. Geometry uses the peg's local **+X** tip.
GT is only a loss/evaluation target and never an aggregation input. Recurrent
assembly, controller and action head are unchanged.

The unchanged loss uses SmoothL1 for positions/endpoint with 0.01 m units,
sign-invariant squared quaternion-dot loss, size SmoothL1 with the saved size
units, and endpoint-residual squared regularization. Weights are peg/hole
position 1 each, endpoint 2, rotations 0.5 each, size 0.5, consistency 0.05.
Optional paired-delta loss is not enabled by the saved `Object21Task` recipe.

### A, B and AB mechanisms

| Variant | `model.global_geometry` | `model.readout` | Parameters at hidden=64, layers=2 | Increase |
| --- | --- | --- | ---: | ---: |
| Baseline | `false` | `attention` | 234,927 | 0% |
| A / `geometry` | `true` | `attention` | 239,135 | 1.79% |
| B / `multiseed` | `false` | `multiseed` | 240,237 | 2.26% |
| AB / `combined` | `true` | `multiseed` | 244,445 | 4.05% |

The matrix selects these flags automatically. `aggregation` in its YAML exposes
`geometry_dim: 16`, `readout_seeds: 4`, `readout_dim: 16` and `readout_heads: 4`.
The regular trainer also accepts these six fields under `model` for a single
variant. The baseline's omitted flags preserve the original state-dict keys and
initialization. Resume signatures interpret missing legacy flags as baseline
defaults, while rejecting a change to an enabled module.

**A: AG-Pose-inspired global geometry.** The
[CVPR paper](https://arxiv.org/html/2403.19527v1#S3.SS4) uses an averaged
learned embedding of pairwise keypoint displacements (Eq. 10), an average global
feature (Eq. 13), and node/global/geometry fusion (Eq. 14). The inspected
[official GAFA implementation](https://github.com/Leeiieeo/AG-Pose/blob/bfd97ff398d542bb6a55f584e7886c32c5e17368/model/Net_modules.py#L189)
also adds an embedding of keypoint coordinates before fusion. The repository's
current README identifies that code as the extended version; the global
mechanism was cross-checked against the original CVPR paper.

`GlobalGeometryAggregation` applies once after the existing cross layer and
before readout. For each valid node, it averages a 3→16→16 MLP of displacements
to every observed node in its group, including self, adds a 3→16→16 embedding
of its existing object-centered coordinates, and fuses that geometry with its
node feature and the masked global mean through a bottleneck MLP plus residual
LayerNorm. PEG and target-side BOX/HOLE are processed separately by the **same
module weights**. No cross-group geometry or padding enters either mean.
This is permutation-equivariant and reuses the existing world-axis coordinate
scale. It adapts the paper to observed sparse vertices, SiLU, LayerNorm and a
small bottleneck; it does not add keypoint detection, dense local neighborhoods,
NOCS prediction, reconstruction, GT geometry or new pose/size heads.

**B: GMT-inspired multi-seed pooling.** The
[GMT paper](https://arxiv.org/html/2102.11533v4#S3.SS2) and
[official PMA/MAB implementation](https://github.com/JinheonBaek/GMT/blob/a178b590b158b21423ac0dbc3fcf1b41a94fc52a/models/layers.py#L115)
use learnable seed queries attending to a node multiset. `MultiSeedAttentionPool`
uses four seeds, Q/K/V projections to width 16, four attention heads, a seed
residual, LayerNorm, a residual feedforward transform, and a projection of the
four 16-value tokens back to hidden=64. PEG and target-side pools have separate
weights under this same general design. Confidence supplies the same log prior
as the original pool, or neutral ones when disabled. Unknown quality remains
neutral; zero confidence has the original finite log floor. Invalid nodes get
zero attention, and entirely empty groups return exactly zero pool/summary.

Unlike the official MAB scaling by the total value width, this implementation
uses the usual square root of the per-head width. It reuses existing GNN
features for K/V and omits GMT's additional graph convolutions, self-attention
between pooled tokens and hierarchical coarsening. It is **GMT-inspired**, not
full GMT, and does not claim GMT's injectiveness/WL guarantees. Four seeds are
the initial candidate: at width 16 they concatenate to the existing width 64,
limiting parameter growth. With fewer than four observed vertices, seeds can be
redundant; suitability for the actual node distribution remains an empirical
question. Permuting input nodes leaves the readout invariant; fixed seed order
does not refer to vertex identity.

**AB** applies A and then B, without another module or head. Parameter counts
are close but unequal; architecture effects remain confounded with the added
capacity unless subsequent parameter-matched controls support the conclusion.

### Evaluation and interpretation

`summary.json` reports Mean/Median/P90/P95 for every metric below, with frame
counts. `comparison.md` contains the mean table; each model also saves per-frame
predictions, targets, counts, `endpoint_geom`, `endpoint_residual` and errors in
an NPZ. **Peg half-size X absolute error is the primary metric.** Checkpoints
are still selected by the existing total validation loss, not that diagnostic.

| Metric | Definition |
| --- | --- |
| Peg/Hole position (mm) | L2 position error |
| Peg/Hole rotation (degree) | Sign-invariant quaternion geodesic angle |
| Peg half-size X/Y/Z, hole radius (mm) | Per-component absolute error |
| Endpoint position (mm) | L2 error of `Object21[7:10] + hole_position` |
| Relative 3D (mm) | L2 error of `Object21[7:10]`, the tip−hole vector |
| Lateral / axial (mm) | YZ norm / absolute X of relative-vector error in the **GT hole frame**, for scoring only |
| Endpoint geometry position / relative (mm) | The same two tip metrics before the existing learned residual |
| Endpoint residual magnitude / gain (mm) | Correction norm / geometric relative error minus corrected relative error; positive gain helps |

The old task metric `endpoint_error_m` is the **relative-vector** error, not
physical endpoint position error. This evaluator distinguishes both explicitly.
Residual magnitude alone is not an accuracy metric. Comparing geometric and
corrected relative errors distinguishes improvements in pose/size geometry from
benefit of the existing endpoint correction, without assuming causal attribution.

Groups include exact PEG, BOX, HOLE and combined target vertex counts, absent
PEG/target, empty graphs and configurable high/low support proxies (default:
at least four PEG and four target nodes). High support does **not** establish
full visibility, and low support does not prove occlusion: the input contract
has no expected vertex count, missing-corner identities or occlusion-cause label.
No group is called fully observed without such evidence. Absent groups have
null statistics, not zero error.

Within each group, the evaluator records Pearson associations between half-size
X error and corrected/geometric endpoint errors. Paired comparisons use exactly
the same frames and report baseline-minus-candidate error distributions,
correlations between size-X and endpoint improvements, and the proportion of
size-improved frames whose endpoint also improves. These are descriptive,
frame-weighted associations, not causal effects or independent episode-level
significance tests. A single seed and unequal parameter counts further limit
claims about the bottleneck.

### Verification and real-data result status

2026-11-07, repository source commit `7f1441e` plus this uncommitted change;
Python 3.11, Torch 2.14.1, macOS. The original source checkpoint strictly loaded,
and its CPU outputs matched a saved **pre-change** oracle exactly, including
an empty input. CPU checks cover all variants with confidence on/off, semantic
IDs enabled, reversed vertex order with matched identities/edges, extra padding,
one-sided/empty groups, zero confidence, finite outputs/gradients, quaternion
norms, Object21 geometry consistency, shared initialization/RNG and legacy
resume signatures. Permutation checks use the actual complete-edge recipe;
upstream tied kNN graph selection retains its documented limitation.

The synthetic root-CLI test runs real one-epoch optimizer updates for all four
models, saves checkpoints and evaluates all five validation frames from the
same 10-train/5-validation fixture. A separate retained smoke uses hidden=64,
layers=2 with the same synthetic budget. Neither is a convergence experiment.
The fixture has no observed PEG nodes, so those workflow scores cannot test
the size-representation hypothesis; separate module tests include PEG/BOX/HOLE.
MPS forward/backward checks also passed for all four models at hidden=64,
layers=2. The initial local checks did not execute CUDA, the simulator, or
real-data 30-epoch training. The subsequent authorized Vast run below exercises
CUDA; simulator execution and artifact publishing are not part of it.

Run the relevant suite, including the original synthetic resume/optimizer test:

```bash
WANDB_MODE=disabled RUN_GNN_TRAINING_TESTS=1 .venv/bin/python -m unittest \
  tests.test_vertex_ablation tests.test_gated_mpnn tests.test_graph_builder \
  tests.test_gnn_trainer tests.test_gnn_metrics tests.test_gnn_cached_data \
  tests.test_gnn_training tests.test_gnn_progress \
  tests.test_experiment_logger tests.test_experiment_artifacts \
  tests.test_temporal_hole_gru tests.test_gnn_temporal_memory -q
```

Result: **125 tests passed**, with no skips when the opt-in flag is set. Two
pre-existing validation tests initially expected `forward()` to validate an
invalid graph; they now call `GraphBatch.validate()` explicitly, matching the
existing documented validation boundary. No validation was added to forward.
Local evidence: [test output](../outputs/vertex_gnn_ablation_verification/test_results.txt),
[checkpoint audit](../outputs/vertex_gnn_ablation_verification/checkpoint_audit.json),
[MPS checks](../outputs/vertex_gnn_ablation_verification/mps_smoke.json), and
[retained synthetic run paths](../outputs/vertex_gnn_ablation_verification/smoke_run.json).
These ignored files are local verification evidence, not committed datasets.
The unchanged frozen-GNN memory and hole-GRU workflows also passed their
regression checks. The [real-cache preflight output](../outputs/vertex_gnn_ablation_verification/real_cache_preflight.txt)
records the actual root-entrypoint failure before any real-data training.

### Authorized Vast execution and tmux continuation

On 2026-11-07, the user requested continuation on the existing Vast.ai instance
**59441593 / RTX 3090**. Its original cache was found at
`/workspace/temporal_insertion_data/gnn_cache` (also `/workspace/gnn_cache`):
64,702 train frames / 630 episodes and 18,751 validation frames / 180 episodes;
90 test episodes remain reserved, with no test frames read. The cache-only
identity and original checkpoint SHA256 match the audit above. Actual CUDA
forward/backward passed for all four variants; original-checkpoint CPU/CUDA
maximum output difference was 1.7881393e-7. Remote tests passed 124 cases, with
one optional W&B SDK check skipped (125 total).

The executed source snapshot is Git `4432d99e5d28b55e27aefb25b3e3fe2a08613d53`,
archive SHA256 `13bf3b32abe882addf7caa99fb48b9e9a5fc25f7591df82e4c6111e4057118a8`.
The original local source was `7f1441e` with uncommitted changes; source files
and their hashes are retained with the launch. All models keep the same 30
epochs, batch 32, seed 0, loss, optimizer and scheduler. Loader workers changed
from the original 0 to **4 per model**, and the four independent processes share
one GPU. This runtime change is recorded; cached samples and splits are unchanged.

At the user's request, checkpoint continuation moved each model from the initial
process pool to its own tmux session. The stable `last.pt` restores model,
optimizer, scheduler, best weights and CPU/CUDA/shuffle RNG. Unsaved partial
epochs are replayed; the final budget remains epoch 30. The stopped pool's logs
and checkpoints are preserved. CUDA compiled resume is not claimed bitwise equal
to uninterrupted execution.

| Variant | tmux session | Saved epoch at migration | First resumed epoch |
| --- | --- | ---: | ---: |
| Baseline | `vertex_baseline` | 5 | 6 |
| A | `vertex_geometry` | 4 | 5 |
| B | `vertex_multiseed` | 4 | 5 |
| AB | `vertex_combined` | 4 | 5 |

On the instance, use `tmux attach -t vertex_baseline` (or another name above).
`vertex_ablation_eval` waits for four successful epoch-30 exits, audits the
continuation checkpoints against the original shared protocol, and invokes
the existing root-entrypoint paired evaluator automatically. Model logs and
configs live under
`/workspace/vertex_gnn_ablation/launch_e89855513f8647a4a47b1955650ce977/tmux/`.
The [local migration record](../outputs/vertex_gnn_ablation/launch_e89855513f8647a4a47b1955650ce977/tmux/migration.json)
contains exact checkpoint paths, hashes and commands; the
[launch record](../outputs/vertex_gnn_ablation_verification/remote_launch.json)
identifies source/runtime paths and the retained partial first attempt.

### Completed matched validation results

All four trainers completed epoch **30**, and automatic paired evaluation
completed successfully on 2026-11-07 at **17:17 KST**. Every model and the
evaluation tmux pane exited with status 0. Each model performed **60,660 optimizer
updates**; the actual saved AdamW step counters were checked. The histories
before and after tmux migration cover epochs 1–30 exactly once. Every final
`best.pt` matches the `best_model` stored in its resumable `last.pt`, and each
selected loss matches the minimum of its complete history. The original
published checkpoint and all four migration checkpoints retained their hashes.

Evaluation run `b1bc1a57bb494f4aadb43d91c8eedbc5` uses **18,751 identical frames
from 180 validation episodes**. Sample IDs, targets and PEG/BOX/HOLE counts were
checked for exact equality across all four prediction files. Training and
epoch-level validation used the configured CUDA compilation; this final paired
evaluation uses eager CUDA inference for every model. It evaluates the
minimum-validation-loss `best.pt`, not the epoch-30 weights:

| Model | Continued training run ID | Selected epoch | Best validation loss |
| --- | --- | ---: | ---: |
| Baseline | `38a96f9288f840fab735f48a79343ca0` | 12 | 3.042341 |
| A | `9eed87b0accd47a593de8687e601600f` | 12 | 3.029630 |
| B | `e87597b82ed749e09d221e9062e39918` | 11 | 3.023960 |
| AB | `802284c9d5c247ac807a982c70ab59ff` | 11 | 3.005271 |

Mean errors below are frame-weighted. Distances and sizes are **mm**; rotations
are **degrees**. Smaller errors are better. Residual magnitude is descriptive;
positive residual gain means the existing correction reduced relative error.

| Metric | Baseline | A | B | AB |
| --- | ---: | ---: | ---: | ---: |
| Peg position | 4.527 | 4.662 | 4.877 | 4.321 |
| Peg rotation | 3.173 | 2.966 | 2.882 | 2.702 |
| Hole position | 26.468 | 26.347 | 26.302 | 26.427 |
| Hole rotation | 5.380 | 5.473 | 5.628 | 5.759 |
| **Peg half-size X** | **5.439** | **5.348** | **5.199** | **5.067** |
| Peg half-size Y | 0.716 | 0.627 | 0.879 | 0.774 |
| Peg half-size Z | 0.733 | 0.639 | 0.878 | 0.760 |
| Hole radius | 2.677 | 2.694 | 2.690 | 2.691 |
| Endpoint position | 11.107 | 10.710 | 10.427 | 10.297 |
| Endpoint−Hole relative 3D | 27.295 | 27.213 | 26.961 | 26.870 |
| Lateral | 21.561 | 21.641 | 21.600 | 21.702 |
| Axial | 12.904 | 12.647 | 12.211 | 11.909 |
| Geometric endpoint position, before residual | 10.756 | 10.618 | 10.221 | 10.078 |
| Geometric endpoint relative 3D, before residual | 27.340 | 27.403 | 27.338 | 27.128 |
| Endpoint residual magnitude | 3.142 | 3.523 | 3.688 | 2.819 |
| Endpoint residual relative-error gain | 0.045 | 0.190 | 0.378 | 0.258 |

For the primary mean half-size X error, A improves by **1.67%**, B by **4.41%**,
and AB by **6.84%** against the newly trained baseline. AB is lower than B by
another 0.132 mm (2.54%). This is not a uniform distributional improvement:

| Peg half-size X error, mm | Baseline | A | B | AB |
| --- | ---: | ---: | ---: | ---: |
| Mean | 5.439 | 5.348 | 5.199 | 5.067 |
| Median | 2.829 | 3.013 | 3.220 | 2.873 |
| P90 | 14.220 | 12.938 | 12.440 | 12.590 |
| P95 | 20.290 | 19.978 | 17.484 | 18.057 |

The baseline retains the lowest median, while B has the lowest P90/P95. AB's
physical endpoint position error improves by **7.29%**, but relative 3D improves
only **1.56%**, and lateral error worsens by **0.66%**. Hole rotation and Peg Y/Z
size errors also worsen under AB. Thus AB's primary mean advantage does not
establish an overall Object21 improvement or better insertion control.

### Observation conditions and size/endpoint association

The observed node-count distribution has PEG counts 3–10 and BOX counts 0–11;
there are **no actual HOLE-type inputs**, missing-PEG frames or empty graphs in
this validation split. The model predicts the supervised Hole state from the
BOX target-side observations. Counts above eight further show why detection
count must not be treated as verified coverage of eight physical corners.

| Condition | Frames | Baseline Size X | A | B | AB |
| --- | ---: | ---: | ---: | ---: | ---: |
| High support: PEG ≥4 and target ≥4 | 5,884 | 3.026 | 3.001 | 3.602 | 3.318 |
| Low support: all other frames | 12,867 | 6.542 | 6.421 | 5.929 | 5.866 |
| Target absent | 2,256 | 17.315 | 17.317 | 13.325 | 14.441 |
| Target present | 16,495 | 3.814 | 3.710 | 4.087 | 3.785 |
| PEG ≤4 vertices | 10,352 | 6.349 | 6.420 | 5.723 | 5.798 |
| PEG 5–6 vertices | 6,056 | 4.462 | 4.053 | 4.677 | 4.274 |
| PEG ≥7 vertices | 2,343 | 3.942 | 3.955 | 4.228 | 3.887 |

Values are mean absolute half-size X error in mm; rows overlap. The full summary
retains every exact PEG/BOX/HOLE/target count group and all error quantiles.
These are observation-count conditions, not confirmed full visibility or labeled
occlusion. The compared frames inside each row are identical across models;
different rows are not otherwise matched for pose, size or difficulty.

AB's size-X advantage occurs mainly in low-support frames. On high support it
is **9.68% worse** than the baseline; B is **19.04% worse**. Target-absent frames
account for approximately **93.0% of AB's net summed size-X error reduction**.
For B, the target-absent gain is about twice its net reduction: it is offset by
*worse* error when the target is present (4.087 versus 3.814 mm). A improves
target-present error to 3.710 mm, but barely changes target-absent error. Hence
the overall readout mean advantage cannot be generalized to well-supported
frames or attributed to recovery of unobserved shape.

Mean size-X and physical endpoint errors both decline, but per-frame changes
are not equivalent. The correlations below compare each candidate's paired
baseline-minus-candidate improvements on the same frames:

| Association / conditional fraction | A | B | AB |
| --- | ---: | ---: | ---: |
| Size-X improvement vs endpoint-position improvement, Pearson r | 0.489 | 0.469 | 0.553 |
| Size-X improvement vs relative-3D improvement, Pearson r | 0.094 | 0.085 | 0.116 |
| Endpoint position improves among size-X-improved frames | 67.92% | 66.59% | 69.84% |
| Relative 3D improves among size-X-improved frames | 54.49% | 53.68% | 55.75% |

For AB, size X improves on 9,655 frames; physical endpoint position also improves
on 6,743 of those, and relative 3D on 5,383. This is descriptive evidence of a
moderate size/physical-tip association and a much weaker size/relative-vector
association, not a causal isolation of size from pose.

The existing residual has mixed effects. It reduces mean relative error by the
reported 0.045/0.190/0.378/0.258 mm, but physical endpoint position error is
higher after correction for every model. For example, AB's geometric physical
tip error is 10.078 mm and corrected error 10.297 mm. B's raw geometric relative
error is essentially unchanged from baseline (27.338 versus 27.340 mm), while
its corrected relative error improves to 26.961 mm. Thus its relative-vector
gain includes a stronger effect of the **existing** residual head; it is not
evidence that geometry alone improved by the same amount. No correction head
was added or changed.

### Research conclusions and remaining limits

1. **Global geometry:** A modestly improves mean size X and improves Y/Z size,
   but worsens the primary median and raw geometric relative error. The evidence
   does not establish missing global geometry as the dominant bottleneck.
2. **Multi-seed readout:** B has a larger overall mean size-X improvement than A
   and the best P90/P95, but its benefit is concentrated in target-absent/low-
   support cases. Target-present and high-support size errors worsen.
3. **Combination:** AB has the lowest mean size-X and physical endpoint errors,
   with an additional mean benefit over either component. B remains better on
   primary P90/P95; several other state errors worsen under AB. This is a
   measured tradeoff, not uniform superiority or demonstrated synergy.
4. **Partial observation:** Benefits are not confined to high support; the larger
   aggregate gains occur in low support. True full visibility, occlusion cause,
   absent-PEG behavior and actual HOLE-input behavior are not established by
   this validation population.
5. **Size to endpoint:** Physical endpoint mean improves alongside size, with
   positive paired associations. Relative/lateral improvements are much weaker
   or negative, and the residual analysis distinguishes geometry from correction.
6. **Confounds:** This is one seed, unequal capacity (+1.79%, +2.26%, +4.05%), a
   reused validation split that selects checkpoints, and CUDA compiled resume
   at different saved epochs. All four share data, loss and 30-epoch/60,660-update
   budgets, but these controls do not eliminate capacity or seed effects.
   Multiple seeds and parameter-matched controls would be needed for a stronger
   claim about the architectural bottleneck. No default model was changed.

The original required comparison is now executed; CUDA training and evaluation
ran on the stated Vast host. Simulator execution, controller rollouts, independent
reserved-test evaluation and Hugging Face publishing were not performed.
The tmux panes are retained after successful exit, and the instance remains
available rather than being stopped automatically.

Local retained evidence (ignored artifacts, not committed datasets):

- [Paired summary and Mean/Median/P90/P95](../outputs/vertex_gnn_ablation/launch_e89855513f8647a4a47b1955650ce977/tmux_evaluation/b1bc1a57bb494f4aadb43d91c8eedbc5/summary.json)
- [Conditional analysis and additional AB-vs-A/B paired comparisons](../outputs/vertex_gnn_ablation/launch_e89855513f8647a4a47b1955650ce977/tmux_evaluation/b1bc1a57bb494f4aadb43d91c8eedbc5/condition_analysis.json)
- [Complete histories, checkpoint hashes and selection audit](../outputs/vertex_gnn_ablation/launch_e89855513f8647a4a47b1955650ce977/completion_audit.json)
- [Four-model manifest](../outputs/vertex_gnn_ablation/launch_e89855513f8647a4a47b1955650ce977/tmux_matrix/970bee1dbb684baf8eb229a5f1c0f8a1/ablation_manifest.json)

The evaluation directory also retains all four per-frame prediction NPZs.
`tmux_training/<variant>/<run_id>/` retains the selected and last checkpoints,
effective configs and resumed histories. Earlier epoch histories and migration
records are preserved with their original paths and hashes.

### Follow-up bottleneck diagnostics

The separate [Vertex GNN bottleneck study](vertex_gnn_bottlenecks.md) examines
high-support regressions with actual observed coordinates and train-fitted
constant predictors, compares fresh readout/head probes on identical frozen
Baseline node features, and decomposes geometric/corrected endpoint–hole errors.
It preserves the four checkpoints and the existing Object21 metric definitions.
See that report for provenance, parent-clustered uncertainty and reproducible
root-entrypoint commands; the original four-model comparison above is unchanged.

### BOX-to-Hole observability and GT-vertex diagnostics

The [BOX-to-Hole observability study](box_hole_observability.md) checks the
installed simulator geometry, verifies validation-seed fixture replay, evaluates
frozen checkpoints on full physical-corner references, and separates those
references from unavailable visibility-preserving coordinate replacement.
It includes parent bootstrap intervals, input-distribution checks, anchor/offset
decomposition, 3D cases, source hashes and CPU/CUDA reproduction commands.

### Vertex-family information study

The [vertex-family semantic study](vertex_family_semantics.md) compares the same
AB architecture with XYZ/type only, retained detector-contour tokens, partial
GT-certified family tokens, and redundant type tokens as a capacity control.
It uses a separate original-TRAIN-only development cohort, fresh initialization
for seeds 0/1/2, identical graphs, and the existing loss/trainer/checkpoint policy.
The report separates observed label dependence from a validated performance gain;
neither Oracle labels nor the fixed metadata mapping were added to production.

### Optional early stopping

The GNN trainer accepts `training.early_stopping` with `patience`, `min_epochs`,
and `min_delta` (nonnegative total validation loss units). A new reference loss
is recorded only on an improvement greater than `min_delta`; after `patience`
consecutive epochs without one, training stops once `min_epochs` is reached.
The best checkpoint still uses the exact minimum validation loss, independently
of this threshold. Checkpoints retain the early stopping counter/reference and
resume requires the same stopping settings. Omit this section for the original
fixed-epoch behavior. Vertex-family matrix and evaluation record and accept
actual unequal budgets only when early stopping was enabled.

Early stopping can also be enabled when resuming a checkpoint that predates the
feature. Its reference and patience counter start with the first new validation
result; model, optimizer, scheduler, random states and the historical best are
still restored. Subsequent resumes retain the counter and require the same
settings. The selected-vertex continuation may specify `validation_selection`
to retain only frames passing both physical conditions. The selection must
cover the original validation split exactly, and its hash becomes part of the
new dataset identity. When resuming from the full-validation dataset, the
current checkpoint is evaluated on the selected frames to initialize a new best
loss and stopping reference; historical losses from the full split are not
compared against the new selection.
