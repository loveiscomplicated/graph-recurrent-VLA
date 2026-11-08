# Fixed-ID corner prediction

Fixed-ID histories use GT-assisted offline certificates. For detector-only online
association and the six preserved GNN comparisons, see [detector-only corner
association and GNN](detected_corner_gnn.md). Dataset preparation remains in
[data collection](data_collection.md#fixed-id-per-corner-sequences).

## Shared fixed-ID corner prediction

The current isolated prototype uses the
[fixed-ID corner sequences](data_collection.md#fixed-id-per-corner-sequences),
independent of the [older GNN-based estimators](temporal_estimation.md#earlier-gnn-based-estimators).
With CUDA PyTorch, NumPy,
PyYAML and official `mamba_ssm` installed, set `data.root` and its expected hash,
then run `python main.py --config graph/configs/temporal/fixed_id_corners.yaml`.
The root dispatches to `training.fixed_id_corners:train`; the model is
`graph.temporal.fixed_id_corner.FixedIdCornerPredictor`.

Each batch item is one complete corner history. A single set of weights handles
all IDs, which are metadata only. The 13-D inputs contain the previous frame's
actual detection/availability plus current observed robot state. The causal
Mamba or Transformer outputs displacement from the most recent past actual
detection. There is no GNN, 21D output, association, lifecycle policy, external
memory-slot registry, prediction feedback or synthetic removal. Visible targets
remain actual detector coordinates; missing targets after first sighting use GT.
Both categories receive loss. World positions reconstruct as last detection
plus predicted displacement; visible measurements remain the unchanged values
for a downstream consumer that reports observed and completed positions.

The paired recipe uses width 128, two layers, a common linear input projection,
LayerNorm/linear displacement head, and a zero-initialized head (last-detection
hold). Mamba uses official fused Mamba-1 with state 16, convolution 4 and expansion
2. Transformer uses four-head causal attention, feed-forward width 256, fixed
sinusoidal positions and zero dropout. Matching seeds initialize the common
projection/head identically. Width/depth are matched; parameter counts are
recorded and are not claimed to be identical.

The initial comparison uses 20 epochs, batch 96, AdamW at 0.0003, weight decay
0.01, gradient clipping 1.0 and BF16 autocast, with FP32 error accumulation.
Normalization is fitted only on train context. Loss is mean squared displacement
error over three axes divided by `(0.01 m)^2`, averaged over all valid prediction
frames; it is nonnegative. First sightings and padding receive no loss. Metrics
also report 3-D RMSE in millimetres, separately for visible and missing targets,
and their supervised counts. A last-detection baseline uses the same masks.
Each seed has its own deterministic batch order; its SHA-256 is logged per epoch
so Mamba/Transformer pairing can be checked.

Every run checks the dataset identity, reads train and validation only, retains
complete padded tensors on CUDA, and uses fused AdamW. There is no CPU training
fallback. The inherited calibration split stays unused. Validation selects
`best.pt`; `last.pt` records the latest epoch. Checkpoints also contain optimizer,
normalization, effective config, history, dataset hash and RNG states. Each run
uses the existing `ExperimentLogger` identity and local artifact sidecars;
nothing is uploaded.

To resume, set `resume.checkpoint` to the previous `last.pt` and set
`training.epochs` to the desired **total** epoch count. The source's `best.pt`
must be beside `last.pt`, or specified as `resume.best_checkpoint`. The seed,
model, dataset hash and training settings must match. The model, input
normalization, optimizer and Torch RNG are restored. Historical shuffle orders
are reconstructed from the seed and verified against every saved hash; subsequent
checkpoints additionally store the NumPy generator state. The old history and
best validation checkpoint are carried forward, so the new `best.pt` remains the
best across the complete run. Each continuation gets a new experiment identity,
records source checkpoint hashes and links the source run through its existing
tracking sidecar. Previous outputs are not overwritten. Resume requires a trusted
locally produced checkpoint because it contains serialized optimizer state.

Optional `training.early_stopping: {patience: 10}` stops a run after ten
consecutive epochs without a strictly lower validation loss. An equal loss does
not reset the counter. On resume the counter is the current epoch minus the
best epoch across the full inherited history, so the previous best and its
remaining patience are preserved. `training.epochs` is the maximum total epoch,
not the number of epochs after resume. Every epoch logs the best epoch and
no-improvement count, and `completed.json` distinguishes `early_stopping` from
`maximum_epochs`. If early stopping is null, the fixed epoch budget is retained.

`progress.json` and stdout update at the first batch, every ten batches and the
last batch in each phase; `history.json` and `last.pt` update each epoch. Logs
include loss, epoch/batch progress, visible/missing supervised counts and elapsed
time. `.venv/bin/python -m unittest tests.test_fixed_id_corner_training
tests.test_fixed_id_corner_sequences -v` runs 12 CPU contracts. On the GPU host,
`python -m tests.check_fixed_id_corner_cuda --config <effective-config>` additionally
executes both real backends with BF16 forward/backward and nonzero-output causal
prefix/padding checks before launching training. CUDA is not executed on the Mac.

### Twenty-epoch parallel comparison

On 2026-11-08 KST, Vast instance **59441593 / RTX 3090 24GB** completed all six
paired architecture/seed jobs concurrently in tmux **`fixed_id_corner_train`**.
The runtime was Torch **2.14.1+cu132**, official Mamba **2.2.6.post3**, BF16, batch
96, width 128 and two layers. Mamba has 235,907 parameters; Transformer has
267,395. CUDA preflight and 12 local CPU contracts passed. The source was a dirty
snapshot based on `320b7cc`, with exact file hashes in each effective config.

Every job completed epochs 1–20 and exited with code 0. Each epoch used 672,214
train targets (353,892 visible / 318,322 missing) and 51,510 validation targets
(27,719 visible / 23,791 missing); none had zero supervision. Paired seeds had
identical batch-order hashes for all 20 epochs, identical input normalization
and configs differing only in architecture/output path. Saved final parameters
and all losses were finite; selected checkpoints matched minimum validation loss.

The following are **epoch 20**, three-seed means ± sample standard deviation.
RMSE measures 3-D position prediction, with visible predictions scored against
actual detections and missing predictions against GT. These are development
validation results with supplied fixed IDs, not an evaluation of ID recognition.

| Model | Validation loss | Visible RMSE (mm) | Missing RMSE (mm) |
| --- | ---: | ---: | ---: |
| Last detection hold | 10.4466 | 5.045 | 82.193 |
| Mamba | 0.1787 ± 0.0345 | 2.177 ± 0.256 | 10.474 ± 1.073 |
| Transformer | 0.1065 ± 0.0078 | 2.670 ± 0.097 | 7.798 ± 0.338 |

The validation-selected best checkpoints separately average missing RMSE
9.521 ± 0.132 mm (Mamba) and 7.586 ± 0.110 mm (Transformer). All six runs preserved
both `best.pt` and `last.pt`. Training-loop elapsed time was 74.1–74.4 seconds per
run, excluding startup/data loading; all six overlapped. A simultaneous sample
showed GPU utilization 100% and memory 5,175 MiB. GPU memory was released after
completion.

The server root is `/workspace/fixed_id_corner_training`: six effective recipes
are in `configs/`, stdout loss logs in `logs/`, per-seed checkpoints/history in
`runs/`, and shared `ExperimentLogger` records in `experiments/`. Attach with
`tmux attach -t fixed_id_corner_train`. Its `status` window's periodic `watch`
was stopped at the user's request; no continued monitoring is configured.
The [comparison and completion checks](../outputs/fixed_id_corner_training/comparison.json)
record all six run IDs, checkpoint hashes, final and selected metrics and paired
batch hashes. [Launch metadata](../outputs/fixed_id_corner_training/launch_manifest.json),
[CUDA preflight](../outputs/fixed_id_corner_training/preflight.log), configs,
logs, tracking metadata and all twelve checkpoints were also retained locally
under ignored `outputs/fixed_id_corner_training/`. No artifacts were uploaded.

### Continuation to 40 total epochs

The six original epoch-20 `last.pt` checkpoints are the sources for a matched
continuation to 40 total epochs. The CUDA preflight restored all six models,
optimizer states and normalization values, reproduced the original batch-order
hashes and verified identical epoch-21 order for each architecture pair. It then
ran a finite BF16 backward/optimizer step on both visible and missing targets
for each checkpoint. A CPU contract also compared the next shuffle, Torch random
values, optimizer state and resulting update with an uninterrupted run.

The continuation uses the same dataset, batch 96, learning rate 0.0003, BF16
and other training settings. Each run inherits the prior best checkpoint and
history and gets a new experiment ID. The original files remain under
`/workspace/fixed_id_corner_training`; the continuation source snapshot, six
configs, logs, checkpoints and preflight are under
`/workspace/fixed_id_corner_continuation`. New tmux windows named
`mamba_0_resume` through `mamba_2_resume` and `transformer_0_resume` through
`transformer_2_resume` are in `fixed_id_corner_train`. No watch process or
recurring monitoring was started.

All six continuations finished epoch 40 and exited with code 0. Each contains
the unchanged original 20-epoch history, and paired seed batch-order hashes
match through epoch 40. All epochs have the expected nonzero visible and missing
supervision counts; final weights and all reported losses are finite. Checkpoint
selection was verified against the minimum validation loss across epochs 1–40.

| Model | Epoch 40 validation loss | Epoch 40 missing RMSE (mm) | Best-validation epochs by seed |
| --- | ---: | ---: | --- |
| Mamba | 0.0768 ± 0.0086 | 6.861 ± 0.406 | 36, 39, 38 |
| Transformer | 0.0464 ± 0.0101 | 4.945 ± 0.634 | 40, 40, 39 |

Values are three-seed means ± sample standard deviation. Against epoch 20, the
missing RMSE decreased from 10.474 to 6.861 mm for Mamba and from 7.798 to
4.945 mm for Transformer. These are development-validation position predictions
with supplied IDs; they do not measure automatic ID association. The
[40-epoch curves](../outputs/fixed_id_corner_continuation/learning_curves.png)
show individual seeds, mean train/validation loss and missing-point error, with
the resume boundary at epoch 20. The [audited comparison](../outputs/fixed_id_corner_continuation/comparison.json),
logs, configs, tracking metadata and all twelve continuation checkpoints are
retained locally in `outputs/fixed_id_corner_continuation/` as well as on the
server. No recurring monitor was started and no artifacts were uploaded.

### Early-stopped continuation

The six epoch-40 `last.pt` checkpoints are continued independently on instance
59441593 with patience 10, strictly lower validation loss as improvement, and a
maximum of 1,000 total epochs. The same batch size, precision, learning rate,
model settings, dataset and paired seed shuffles are retained. The current
global best is inherited from all previous epochs; the initial stale counts are
4, 1, 2 for Mamba seeds 0–2 and 0, 0, 1 for Transformer seeds 0–2. Each job
stops independently, preserving both the global `best.pt` and latest `last.pt`.

Fourteen local contracts passed. On CUDA, all six source checkpoints restored
model, optimizer and RNG, reproduced paired epoch-41 order, and executed a finite
backward/update with visible and missing targets. A separate controlled run
verified actual trainer termination after the configured patience and preservation
of the inherited best checkpoint. Effective configs, source snapshot, preflight,
logs and run outputs are under `/workspace/fixed_id_corner_early_stopping`.
The windows `mamba_0_early` through `mamba_2_early` and `transformer_0_early`
through `transformer_2_early` are in tmux `fixed_id_corner_train`. No watch
process or recurring monitor is started.

All six jobs subsequently exited normally through early stopping, exactly ten
epochs after their best validation loss. Mamba seeds 0–2 stopped at epochs
77, 101 and 73, with best checkpoints at 67, 91 and 63. Transformer seeds
0–2 stopped at epochs 90, 103 and 71, with best checkpoints at 80, 93 and 61.
Their inherited 40-epoch histories, source checkpoint hashes, finite losses,
nonzero supervision and paired batch orders during their common training
period were verified. All twelve best/last checkpoints and run metadata were
backed up locally under `outputs/fixed_id_corner_early_stopping/`.

At the checkpoints selected by minimum overall validation loss, the three-seed
mean ± sample standard deviation is:

| Model | Validation loss | Visible RMSE (mm) | Missing RMSE (mm) |
| --- | ---: | ---: | ---: |
| Mamba | 0.03654 ± 0.01115 | 1.219 ± 0.162 | 4.646 ± 0.772 |
| Transformer | 0.02221 ± 0.00396 | 1.578 ± 0.142 | 3.382 ± 0.326 |

Visible scores evaluate the model's prediction against the observed detection;
the downstream observed coordinate remains the original detection. The
missing scores compare prediction against GT. IDs are supplied through the
dataset's GT-assisted certification, so these figures do not evaluate online
ID association. [Learning curves](../outputs/fixed_id_corner_early_stopping/learning_curves.png)
show each seed only until its own stopping epoch; stars mark the checkpoint
chosen by overall validation loss. The [audited results](../outputs/fixed_id_corner_early_stopping/comparison.json)
include per-seed final and best metrics and checkpoint hashes. No artifacts were
uploaded.

### Fixed-ID corner video visualization

`python main.py --config graph/configs/temporal/visualize_fixed_id_corners.yaml`
renders real validation RGB side by side for same-seed Mamba and Transformer
best checkpoints. The recipe needs the checked fixed-ID dataset, its original
full-frame vertex-family dataset, both checkpoint paths, CUDA with official
`mamba_ssm`, OpenCV and FFmpeg/libx264. It validates dataset lineage, checkpoint
and inference source hashes, frame hashes, and the natural detection gap.

Each video overlays a small yellow point for the actual detection, a small pink
point for the model's prediction and a small green point for projected GT. Each prediction is
computed from the prefix available at that frame, containing historical
observations and current robot state. GT is used for offline ID certification,
clip selection, labels and display; it is never a model input. Displayed errors are 3-D Euclidean distances to GT, in
millimetres, not image-pixel errors. A missing certified detection can be a
physical occlusion, detection miss or uncertain ID; markers are projected onto
the image even when the corresponding 3-D point is behind an occluder.

The paired seed-0 examples use PEG ID 4 in validation episode
`shard_000011/episode_000011` (natural gap 21–71) and PEG ID 6 in
`shard_000020/episode_000010` (gap 61–76). These were chosen from natural
validation gaps for reappearance and movement, not model error, and are
illustrative examples rather than an aggregate benchmark. No artificial masks
or online ID association are used. Per-frame world coordinates and errors are
saved alongside each H.264 video; the existing `ExperimentLogger` records the
effective config, source checkpoint identities and video artifacts locally.

The final small-point versions are [the 51-frame gap](../outputs/fixed_id_corner_videos/68b3f6bbf0dd4d189fba123030d6acbf/peg_long_gap.mp4)
and [the 16-frame gap](../outputs/fixed_id_corner_videos/68b3f6bbf0dd4d189fba123030d6acbf/peg_short_gap.mp4).
These replay original ManiSkill camera RGB at 12 frames per second. The colored
markers have a three-pixel radius; points can overlap when their projections are
close. Both clips use Mamba seed 0, best epoch 67, and Transformer seed 0, best
epoch 80. Gap-only 3-D RMSE is 2.00/1.79 mm (Mamba/Transformer) for the long gap
and 7.96/3.52 mm for the short gap. These two cases were selected for motion and
reappearance; their errors are not an estimate of average validation performance.

Each displayed model position was calculated using only the input prefix available
at that frame, rather than a whole-episode forward pass. Independent prefix checks
for the beginning, midpoint, end and reappearance of each gap reproduced the
saved model positions. The videos and per-frame position/error records are saved
under `/workspace/fixed_id_corner_videos/small_points_runs/` and locally in
`outputs/fixed_id_corner_videos/`; no artifacts were uploaded.

For complete episodes with all previously seen corners, run the same entrypoint
with `graph/configs/temporal/visualize_all_fixed_id_corners.yaml`. It uses one
independent temporal history per ID and the same weights for all corners. An ID
appears only from its first certified detection: yellow is its current unchanged
detection, pink is its model-predicted position when no detection is certified,
and green is the GT comparison position for that missing ID. Current detections
are never blended with forecasts. Small points can overlap or project outside
the camera. Frame captions show the number seen, detected and missing; the 3-D
error is the RMSE over missing IDs in that frame.

The videos show only IDs certified by the existing GT-assisted annotations;
unknown detections cannot be assigned to a corner ID and remain unassigned.
Missing certified detections can reflect physical occlusion, detector misses or
uncertain certification. Source episodes and every frame's certified ID set are
checked before rendering, with raw image and dataset hashes preserved. The two
previously selected episodes contain 13 observed canonical IDs each. The full
history is reconstructed from actual prior observations; no online association,
synthetic masking, GT model input or prediction feedback is used.

The complete episodes are [all corners, episode one](../outputs/fixed_id_all_corners_videos/4a922047c78c41c0add998ffb07821c5/all_corners_episode_one.mp4)
and [all corners, episode two](../outputs/fixed_id_all_corners_videos/714a0a78856c4a85a50d86df9075c76d/all_corners_episode_two.mp4).
They contain 104 and 110 original camera frames respectively, replayed at
12 fps. Every frame was checked against the dataset's first-observation masks
and unchanged detection coordinates; the highlighted corner's predictions also
matched the independently checked single-corner video. All output points and
missing-point GT projections were within the camera frame in both examples.
The [all-frame audit](../outputs/fixed_id_all_corners_videos/all_frame_verification.json)
records these checks. There were 106 and 193 unassigned detection events in the
two episodes, respectively; those lack a reliable fixed ID and are not silently
assigned to a track. These example videos are illustrative, not aggregate
evaluation, and all model/checkpoint identities and per-frame records remain
with the videos.

## Fixed-ID corner predictor followed by a GNN

`graph.temporal.fixed_id_gnn.FixedIdCornerGNN` connects a shared per-corner
Mamba or Transformer to `VertexEdgeGatedMPNN(extra_node_dim=1)`. For each
frame, it includes only previously seen, identified corners. It passes the
current detector coordinate unchanged for visible corners and last detection
plus temporal displacement for missing corners. A complete directed graph
uses the existing PEG/BOX/HOLE types, and a single extra node feature marks
detected (1) versus predicted (0). No artificial occlusion or GT enters this
forward path. The caller supplies synchronized TCP, fixed IDs and causal,
right-padded corner histories; the module does not assign IDs or manage tracks.
`GraphBatch.node_valid` here means available graph support, not actual detector
visibility; use the returned `observed` mask for visibility instead.

The model returns Object21 with the existing GNN contract: peg position (3),
peg quaternion (4), peg-tip-to-hole vector (3), hole position (3), hole
quaternion (4), peg half-size (3), and hole radius (1). Positions are relative
to TCP in world-aligned axes, except the peg-tip-to-hole vector; dimensions are
metres and quaternions are wxyz. The model also returns the constructed graph
and selected input indices for inspection.

The training recipe prepares the synchronized Object21 labels from the verified
vertex-family graph cache, using the same episode splits as the corner dataset.
The prepared frame cache includes IDs first seen on the last frame. IDs with no
certificate remain unassigned; duplicate certificates are resolved by the same
offline GT-assisted rule as the corner dataset. GT coordinates never enter the
temporal predictor or GNN inputs. Full causal histories are evaluated once under
`no_grad` with the frozen checkpoint, and observed coordinates override the
predicted values. The GNN alone is optimized with the existing `object21_loss`.
Mamba uses bfloat16 inference on CUDA; small rounding differences can occur
between whole-sequence and prefix inference. There is no synthetic occlusion.

`graph/configs/temporal/prepare_fixed_id_gnn.yaml` creates the shared frame
cache; `graph/configs/temporal/train_fixed_id_gnn.yaml` trains one run using a
configured frozen checkpoint. Mamba and Transformer use paired seeds and the
same GNN initialization, data, optimizer, validation loss and early stopping.
`tools/run_fixed_id_gnn_matrix.py --configs DIR --logs DIR` starts six configured
training processes and runs the comparison recipe once all finish successfully.
The comparison checks cache identity, GNN initialization and settings before
saving per-seed/aggregate 21D metrics and validation learning curves. Validation
selects checkpoints; these are development metrics, not reserved-test results.
Earlier observed-only checkpoints cannot be loaded directly into the GNN with
its extra source channel. No 21D accuracy is established until training finishes.

## Fixed-ID reappearance distance audit

With the fixed-ID dataset and six frozen checkpoints, configure
`graph/configs/temporal/analyze_fixed_id_reappearance.yaml` and run:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/analyze_fixed_id_reappearance.yaml
```

Each event is a certified corner detected after one or more consecutive frames
without a certified detection. Its forecast uses the causal prefix and current
robot state, without the current measurement. Distances are to the actual
reappearing detector coordinate, not GT coordinates. All other valid detector
points of the same detector type are considered competitors, including unknown
IDs and duplicate detections. The audit measures single-corner distances and
does not run one-to-one association. Identity histories are GT-assisted, and
duplicate certificates were resolved using GT during dataset preparation. An
unknown certificate contributes no reappearance event, so the results do not
establish online tracking accuracy. Missing certified detection may result from
physical occlusion, a detector miss, or failed ID certification.

On Vast instance 59441593, 40 development-validation episodes contained 392
events (96 peg, 131 box, 165 hole), with gaps from 1 to 94 saved frames. For
each architecture, the following numbers are averages of three independently
calculated seed statistics, not a pooled event distribution:

| Forecast | Median distance | P90 distance | Within 5 mm | Within 10 mm | Within 20 mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mamba | 1.82 mm | 7.27 mm | 86.6% | 93.1% | 98.7% |
| Transformer | 1.74 mm | 5.40 mm | 89.3% | 95.9% | 99.3% |

For each of the six seeds no other same-type detector point was strictly closer
than the certified target in any event; 30 events had no same-type competitor.
This is conditional on oracle-correct past histories and is not an assignment
success rate. Long gaps are not a representative moving-peg sample: only seven
of the 113 events with gaps of at least 21 frames were peg events. Effective
config, per-event distances, thresholds, and seed summaries are retained in
`outputs/fixed_id_reappearance/c837be49b0544e6ea8295350b43c2a0d/` and its
remote experiment. No validation threshold is claimed as calibrated.
