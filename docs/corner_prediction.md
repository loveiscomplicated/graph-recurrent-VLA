# Fixed-ID corner prediction

## Five-mode spatial/temporal tracker from scratch

The separate `graph.temporal.spatial_temporal_corner.SpatialTemporalCornerTracker`
extracts the graph2 handoff's width-48, four-head, two-temporal-block model.
Its state-dictionary layout and forward computation are preserved, including
same-body spatial attention and exact passthrough of current detections.
The historical fixed-ID/lifecycle implementations and preserved leaderboards
are unchanged.

Run the two config-driven commands near the beginning of `graph/README.md`.
`data_collection.five_mode_corner_tracking:prepare` reads the Five-mode release
and its matching raw archives. It pins the release manifest, verifies archive,
frame and detector-member hashes, derives the 20 GT corners with the existing
evaluation convention, and applies the extracted GT-free physical matcher.
Only `corners` supplies observations; no surface tracks or legacy State36 cache
are read. GT is used for coordinate labels and validation, never for slot
association. Raw RGB/depth pixels are not decoded or modified.

The source release is HF `Graph-Recurrent-VLA/graph-recurrent-rollouts`, revision
`cab29e26c0025430bf8b7aafda437d23dfd6da52`, integration run
`d9ef38b515f44db295ee6d84d0db5279`: 900 reset seeds, five modes per seed,
4500 episodes and 852102 observations. All five episodes of a seed retain their
original split. Training preparation reads only 3600 train and 450 validation
episodes. The 450 test episodes are opened only for the frozen comparison below.
The local source manifest SHA-256 is
`be2db7841d823d89ba4044945a170c80bf5a316a53a58ccc153643f74b0098d8`.

The model starts with random weights and a zero displacement head; no prior
checkpoint is loaded. The fallback for never-observed slots is fitted only on
training GT. Ten-dimensional causal features, full-episode histories,
missing-only Smooth L1 and fixed 20-slot ordering follow the handoff model.
AdamW uses BF16 autocast with FP32 loss/error accumulation and gradient clipping.
The validation selection score averages the missing-point mean 3D error of
the five policy modes equally, so long random episodes do not determine the
score solely by their frame count. Logs also retain pooled mean/P90, body,
last-observation-age and depth-stage diagnostics. Depth stages follow the
historical entrance-plane coordinate; on random trajectories they do not
by themselves certify that the peg is inside the aperture.

Before training, CUDA preflight checks nonzero-head causality, prefix/padding
invariance and exact preservation of detector coordinates. Batch tuning times
full forward/backward/AdamW steps at the longest training sequence. It selects
the largest candidate within 95% of the best measured throughput and below
85% of GPU memory capacity. It restores the original random parameters before
the actual training run. No batch tuning uses validation quality or test data.

The learning rates are 0.0003, 0.0001 and 0.00003. Forty epochs without at least
0.05 mm score improvement reduce the learning rate; the same plateau at the
last rate stops training. The safety limit is 2000 epochs. `best.pt` uses the
strictly lowest validation score, even for improvements smaller than 0.05 mm.
`last.pt` atomically saves model, optimizer and RNG states every epoch; it
retains the dataset/config identity and history. No automatic resume launcher
is implemented. `completed.json` distinguishes plateau stopping from the limit.
The common `ExperimentLogger` records the effective config and local metrics;
publishing is not automatic.

On 2026-11-18 KST, Vast instance `61159793` was rented in Germany with one RTX
3090 and 100 GB disk at $0.2185185/hour including disk (transfer fees separate).
The requested budget is $0.25/hour and China/Hong Kong/Macao offers were excluded.
The deployment uses a source snapshot based on graph2 `b9b61a7`; exact hashes
are retained in `outputs/five_mode_corner_tracking/source_sha256.json` and on
the server. The tmux session is `five_mode_corner_tracker`, under
`/workspace/five_mode_corner_tracking`. No recurring watch is configured.
Training completed normally after 505 epochs (5953 seconds), stopping at the
lowest-learning-rate plateau. The validation-selected checkpoint is epoch 495,
with five-mode balanced missing-point mean 7.5313 mm. Its SHA-256 is
`5b6faab3863efa321862ef6ef3e54eb7a0302a0c57b11aa3580070258a754b7a`.

The prepared payload contains 681842 training and 84081 validation frames.
Local and server CPU contracts passed (three tests), and the server executed
CUDA causality/prefix checks and a finite BF16 backward/optimizer step. Batch
tuning selected 96 episodes: at the longest 287-frame sequence it measured
301.6 episodes/s versus the best 303.4 at batch 64. Batch 96 reserved 17.77 GB;
batch 128 exceeded the 85% memory budget and 192 ran out of memory during the
bounded probe. Parameters were restored before training. The actual training
process was confirmed active in tmux with 99% sampled GPU utilization.

### Training only after first observation

The original handoff and first Five-mode run supervised every current missing
slot, including slots that had never been observed. For the user's tracking
objective, `graph/configs/train_observed_corner_tracker.yaml` instead sets
`training.supervision: after_first_observation`. Launch it with the root
`main.py` as shown near the beginning of `graph/README.md`.

For each episode and physical slot, `ever_seen` is the cumulative OR of actual
detector assignment masks up to the current frame. The supervised mask is
`valid & ever_seen & ~observed`: pre-first-observation frames, never-observed
slots, current detections and padding receive no direct loss. There is no
future-derived eligibility. The same function masks every validation summary,
the five-mode balanced checkpoint-selection score and the plateau scheduler.
Zero-target training batches are skipped; a validation mode without any eligible
targets raises an error instead of silently changing the selection weights.
`supervision.json`, the effective config and logger metadata record included
and excluded point-frame counts. Epoch and validation counts are checked against
the startup audit.

The model architecture, features, train-fitted fallback, 3600/450 train/validation
episodes, optimizer, BF16 precision, learning rates and plateau conditions are
unchanged. The 450 test episodes and their earlier results do not enter training
or checkpoint selection. Weights and optimizer start fresh; batch tuning is
rerun with the new loss and restores the initial random parameters afterward.
Its stress batch includes the longest sequence and at least one supervised
episode, so an empty-target sequence cannot invalidate the backward benchmark.
The historical configs default to `all_missing` to preserve prior experiments.

The rerun uses the same German Vast RTX 3090 instance `61159793`, with a separate
code checkout at `/workspace/observed_corner_tracker/code`, outputs under
`/workspace/observed_corner_tracker/training`, and tmux session
`observed_corner_tracker`. Original code, outputs and the other surface-context
experiment remain separate. Source hashes, launch information and startup
evidence are retained locally under `outputs/observed_corner_tracker/` and on
the server. No recurring watch or automatic instance shutdown is configured.

Startup verification on 2026-11-18 KST passed 13 local CPU contracts and nine
server CPU contracts, followed by CUDA causal checks and BF16 training probes.
The train mask includes 1374969 missing point-frames and excludes 2524585
pre-first-observation point-frames. Validation includes 166754 and excludes
317966 respectively; its eligible Peg/Box/Hole counts are 95842/39062/31850.
The selected batch remains 96, with approximately 11 seconds per epoch and
99% sampled GPU utilization. Initial epochs and atomic best/last saves were
verified, including checkpoint reload and equality of training/validation
target counts with the startup audit. These are launch checks, not final
accuracy results.

#### Held-out evaluation after first observation

The observed-only run completed normally at epoch 382 after 4449 seconds.
Validation selected epoch 374 with five-mode balanced mean 6.1764 mm; checkpoint
SHA-256 is `ea4df0c535e9ff908c7c2b0b98885128de80310f4a2ee91332991ee285e339cf`.
Compare this frozen model with the original graph2 final Transformer using:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/evaluate_observed_corner_tracker_comparison.yaml
```

On 2026-11-18 KST, the same 450 test episodes (90 seeds, 86179 frames) were
processed on the German RTX 3090 with identical detector coordinates, physical
slot masks, TCP and episode boundaries. Each checkpoint retains its own
train-fitted fallback. Both execute FP32 with TF32 disabled, batch 16 and no
Box/Hole rigid correction. The primary mask is now explicitly
`ever_seen & ~observed` for both models; padding is already absent from the
flattened test arrays. It selects 173765 point-frames and excludes 319785
pre-first-observation missing point-frames. First sightings, current detections
and never-observed slots are not scored. Both model results use the same saved
mask hash, with no test-based model or threshold selection.

| Target | Evaluated point-frames | Existing mean / P90 (mm) | Retrained mean / P90 (mm) |
| --- | ---: | ---: | ---: |
| Peg | 96079 | 29.56 / 80.79 | 12.07 / 27.21 |
| Box exterior | 43124 | 9.04 / 16.73 | 3.07 / 5.10 |
| Hole rim | 34562 | 4.80 / 9.33 | 1.92 / 3.28 |
| All | 173765 | 19.54 / 54.66 | 7.82 / 16.07 |

P90s by trajectory mode are existing → retrained:

| Mode | Peg (mm) | Box (mm) | Hole (mm) |
| --- | ---: | ---: | ---: |
| Fixed Expert | 20.69 → 11.65 | 11.00 → 4.11 | 5.34 → 3.14 |
| Fixed Expert → Random | 98.29 → 39.20 | 16.08 → 4.52 | 8.94 → 2.90 |
| Fixed Expert → Random → Expert | 97.18 → 21.79 | 14.17 → 4.46 | 9.07 → 3.05 |
| Fixed Random | 71.87 → 8.15 | 26.00 → 4.32 | 13.96 → 4.56 |
| Moving Expert | 35.18 → 18.88 | 13.64 → 6.37 | 5.73 → 2.87 |

These scores use the observed-only population and must not be substituted for
the earlier all-missing comparison. They measure coordinate tracking, not
Object21 estimation or closed-loop policy success. The original model's values
exactly match its previously reported post-observation subset. An independent
NumPy rescore reconstructs causal histories from each episode's observations,
checks the packed mask hash, and reproduces all body means/P90s from saved
predictions. All four comparison CPU contracts passed, including invariance
to arbitrary errors outside the eligible population.

The evaluation run is `d3263c4b9d2c4102a6f01b77285e8013`.
[Results](../outputs/observed_corner_tracker/test_comparison/results.json) ·
[independent verification](../outputs/observed_corner_tracker/test_comparison/verification.json).
Full per-frame predictions and executed source are preserved on the server in
`/workspace/observed_corner_tracker/test_comparison/`. Models and historical
comparison artifacts were not modified; no simulator, detector rerun or
artifact upload was performed.

### Surface-context training only after first observation

Run `python main.py --config graph/configs/train_observed_surface_corner_tracker.yaml`
for the corrected Peg-surface experiment. It enables the same
`after_first_observation` policy used by the corrected corner-only run, while
retaining the surface model, data, seed, optimizer, stopping criteria and
independent batch-throughput tuning. `ever_seen` is accumulated only from the
physical corner's actual detector-assignment mask. Surface support does not
initialize a corner or make an unseen slot eligible for supervision.

The loss, every validation metric (including surface-support groups), the
five-mode balanced selection score and early-stopping scheduler all use
`valid & ever_seen & ~observed`. Regression tests change excluded GT and errors
by large amounts and verify unchanged loss, zero direct gradient on excluded
predictions, and unchanged validation/model-selection scores even when surface
points are available before the corner's first detection.

The existing Korean RTX 3090 instance `61163897` is reused. The original surface
run `703d64668c584353a5150fef6b9559ea`, its checkpoints, logs and original source
remain under their original paths. The new source checkout is
`/workspace/observed_surface_corner_tracker/source`, with separate training
outputs under `/workspace/observed_surface_corner_tracker/training` and tmux
session `observed_surface_corner_tracker`. It loads no old checkpoint: the
surface variant starts from seed-0 random initialization. Existing train and
validation payloads are reused read-only; test payloads remain unused by this
run. Supervision counts, source hashes, batch probes and launch information are
retained locally in `outputs/observed_surface_corner_tracker/` and on the host.
No recurring watch or automatic instance shutdown is configured.

Startup verification passed 15 local/server CPU contracts plus CUDA causal and
BF16 backward checks. An independent per-episode NumPy calculation exactly
matched the CUDA supervision audit: train retains 1374969 missing point-frames
and excludes 2524585 before first observation; validation retains 166754 and
excludes 317966. Batch tuning selected 64 episodes. The first three epochs and
both best/last checkpoint saves were verified, and hashes of the previous
run's checkpoints, history, completion/config records and original trainer
source remained unchanged. These are startup checks, not final quality results.

### Held-out comparison of the original corner trackers

With the source release/raw archives, prepared test dataset and both frozen
checkpoints available, run these config-driven workflows (adjust host paths):

```bash
python main.py --config data_collection/configs/prepare_five_mode_tracker_test.yaml
WANDB_MODE=disabled python main.py --config graph/configs/evaluate_five_mode_corner_tracker_comparison.yaml
WANDB_MODE=disabled python main.py --config graph/configs/diagnose_five_mode_corner_tracker_comparison.yaml
```

On 2026-11-18 KST, all 450 test episodes (90 reset seeds, 86179 frames) were
evaluated on the existing German RTX 3090 instance. The existing model is the
graph2 final GT-free Transformer, epoch 333, SHA-256
`91f745a4e40377695929832aa365bfa2b87edb75fd91c573fb337e25211e8845`, downloaded
from the revision-pinned handoff. The new model is the epoch-495 checkpoint
above. No model, threshold or association rule was selected or adjusted using
these test results.

Both receive byte-identical current detector coordinates, slot assignments,
observed masks, TCP and episode boundaries. Each retains its own checkpoint's
train-fitted fallback for never-observed slots; no test-derived prior is fitted.
Both use the same extracted architecture, batch 16, full causal sequences and
FP32 with TF32 disabled. Results below score exactly the same missing slots,
before Box/Hole rigid correction. There is no GNN or policy in this evaluation.

| Target | Missing point-frames | Existing mean / P90 (mm) | New mean / P90 (mm) |
| --- | ---: | ---: | ---: |
| Peg | 208945 | 28.93 / 73.03 | 10.65 / 23.88 |
| Box exterior | 210627 | 7.22 / 15.53 | 2.67 / 4.56 |
| Hole rim | 73978 | 26.63 / 64.24 | 19.51 / 53.95 |
| All | 493550 | 19.32 / 55.00 | 8.57 / 22.70 |

The gains differ by trajectory mode. The following are existing → new P90s:

| Mode | Peg (mm) | Box (mm) | Hole (mm) |
| --- | ---: | ---: | ---: |
| Fixed Expert | 15.20 → 15.18 | 5.89 → 4.18 | 52.11 → 53.66 |
| Fixed Expert → Random | 84.58 → 41.61 | 15.47 → 4.18 | 68.09 → 56.26 |
| Fixed Expert → Random → Expert | 74.93 → 25.72 | 10.47 → 4.32 | 63.74 → 57.63 |
| Fixed Random | 82.00 → 6.45 | 22.45 → 5.00 | 69.75 → 51.90 |
| Moving Expert | 25.32 → 18.64 | 11.89 → 4.95 | 52.22 → 48.68 |

Fixed Expert Peg is effectively unchanged and its Hole P90 slightly worsens;
the pooled improvement must not be described as uniform improvement in every
mode. These are coordinate-completion results, not closed-loop success rates.

#### Why the Hole tail remains large

| Missing Hole condition | Point-frames | Existing mean / P90 (mm) | New mean / P90 (mm) |
| --- | ---: | ---: | ---: |
| No observation yet in the causal history | 39416 | 45.76 / 81.13 | 34.56 / 59.46 |
| Observed previously, currently missing | 34562 | 4.80 / 9.33 | 2.35 / 4.09 |

Forty-eight episodes from 11 reset seeds have no Hole observation anywhere in
the episode. Before any Hole observation, missing slots account for 53.3% of
the scored Hole points but 94.4% of the new model's summed Hole distance error.
All 27650 new-model Hole errors above 20 mm occur before first observation.
Thus the dominant remaining issue is initial Hole localization without a
measurement, rather than degradation of previously observed Hole tracks.

The input audit separates detection and association. Of 73978 missing Hole
point-frames, only nine contain an unassigned raw candidate within 5 mm of that
GT corner. During all 9854 uninitialized-Hole frames, Box association is already
initialized; 9847 frames have zero raw Hole candidates within 5 mm of any GT
rim corner, and none have three such candidates even at 15 mm. Observed Hole
coordinates have 3.06 mm P90 error, and the nearest-GT-corner ID mismatch count
is zero. Successful Hole initialization fits its Box-relative center to P90
2.20 mm and radius to P90 1.24 mm. The matcher's 30 mm radius ceiling excludes
none of these test episodes. These findings do not support numbering errors or
that radius limit as the primary source of the large Hole tail.

The fixed-camera geometry supplies a further explanation. At the initial
frame, episodes without any Hole observation have median angle 79.03° between
the outward Hole normal and camera direction, versus 58.60° in the observed
group (90° means edge-on). Raw RGB examples for seeds 24186063, 29577066 and
49991136 show the Hole face as a narrow grazing-view strip. Their actual raw
calibration agrees with the configured camera position. This supports
foreshortening/limited image support as an important detector-input limitation;
it does not prove that every missing frame has the same visibility cause.
In all-unseen frames, a shared center displacement explains 97.35% of the new
model's four-corner squared error: the dominant error is where the unobserved
opening lies, rather than its four-corner shape. Its center error normal to
the Box face is P90 6.74 mm, while the error within that plane is P90 58.95 mm.
The recorded old-model and new-model development seed inventories both have
zero overlap with the 90 test seeds.

Evidence: [paired results](../outputs/five_mode_corner_tracking_test/comparison/results.json),
[independent rescore and diagnostics](../outputs/five_mode_corner_tracking_test/comparison/diagnostics.json),
[raw RGB and corner overlays](../outputs/five_mode_corner_tracking_test/hole_observation_examples.png),
[example calibration/depth audit](../outputs/five_mode_corner_tracking_test/hole_observation_examples.json).
The [per-frame prediction artifact](../outputs/five_mode_corner_tracking_test/comparison/predictions.npz)
was also copied locally and its SHA-256 verified. It remains on the server at
`/workspace/five_mode_corner_tracking_test/comparison/predictions.npz`, with
their SHA-256 in `results.json`. The comparison run is
`ab0aa1c452f64989a6f1935a7ff3ab03`. Five CPU contracts passed and saved
predictions independently reproduced all reported body means/P90s. No
simulator backend, detector rerun, model update or artifact publication was
executed by this comparison.

### Peg surface context in spatial attention

`graph/configs/train_five_mode_surface_corner_tracker.yaml` selects an additive
variant through the same training loop. The original corner-only config and
the already-running baseline deployment remain unchanged. It uses identical
train/validation episodes, corner coordinates/masks, targets, train-only fallback,
width 48, four heads, missing-only loss, optimizer, learning rates and stopping
criteria. Seed 0 initializes every shared parameter identically to the baseline.
The new six-input surface projection is initialized after those shared weights.

`data_collection.five_mode_surface_context:prepare` verifies the original release
and every selected surface NPZ receipt, checks frame indices/track IDs, and
writes a separate dense 32-point sidecar for the 4050 train/validation episodes.
It does not read test members or change existing corner/target bytes. Context
is current world XYZ relative to the train-fitted Peg center and to the current
observed TCP, scaled by 0.1 m. Track IDs only identify padded storage positions;
the model has no learned surface ID or independent surface time history.

After the first corner temporal block, the 20 corner queries attend to corner
keys/values plus the encoded current surface points. Only Peg queries 0–7 may
attend to those points; Box/Hole queries retain their existing same-body mask.
Unavailable surface rows are masked, leaving corner keys available in frames
with no surface support. Only the 20 corner states enter the second temporal
block and prediction head. Directly detected corner coordinates still pass
through exactly. Surface information can persist through the corner temporal
states, without processing surface sequences in a temporal attention block.

The original release has no explicit surface object-ID array. The pinned no-plan
producer and surface audit identify its queries as Peg-only; no Box/Hole points
are inferred from proximity or GT. The cache's visibility/depth/corner-exclusion
gates are inherited. This experiment does not regenerate per-frame object masks
or establish that every surviving track is free of drift. No TAPNext++ weights
are trained and no RGB-D simulator is executed by this offline workflow.

Validation additionally separates missing Peg errors in frames with/without
surface support; the selection score remains the same five-mode average. Batch
tuning is rerun independently using the new model's measured training throughput
and the same 85% memory / 95% peak-throughput thresholds. The new run is randomly
initialized, with no baseline checkpoint loaded, and uses a separate output
directory and tmux session `five_mode_surface_corner_tracker`.

The separate run was launched on 2026-11-18 KST on Vast instance `61163897`
in South Korea: one RTX 3090, 80 GB disk, $0.2503704/hour including disk
(transfer fees separate), below the requested $0.50/hour ceiling. China,
Hong Kong and Macao were excluded. The original baseline instance was not
accessed or changed. Seven CPU contracts passed locally and on the new server;
the CUDA preflight and BF16 training/backward probes also passed.
The sidecar preserves 17474357 train and 2131343 validation surface points.
No held-out test member was read. Batch tuning selected 64; 96 exceeded the
85% allocator memory target and 128 ran out of memory during the bounded probe.
The first two epochs completed with atomic best/last checkpoints and about
11.6–11.9 seconds per epoch, with sampled GPU utilization 99%. The run's
`batch_tuning.json` retains exact candidate throughput and memory measurements.
The different selected batch (baseline 96, variant 64) changes optimizer steps
per epoch and must be reported when comparing learning curves. This launch
verification does not establish final tracking quality. No recurring watch or
automatic instance shutdown was configured.

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

## 6D TCP rotation input

`graph/configs/temporal/fixed_id_corner_rotation_6d.yaml` trains a new
fixed-ID corner predictor using the existing, checksummed sequence dataset.
Set its data root and expected hash before running the command in the
[Graph README](../graph/README.md#corner-prediction-and-tracking). No data
conversion or recollection is required. The configured architecture, optimizer,
seed and device are recorded with the run by `ExperimentLogger`; the input
mean and standard deviation are fitted on valid training rows only and saved
in each checkpoint. This new training workflow starts from scratch; it does
not implement continuation.

The model accepts the existing 13 raw values per frame and converts the TCP
wxyz quaternion to the first two rotation-matrix columns internally. The
network consumes 15 normalized values. It uses the same conversion in
training and inference and ignores padded quaternion values. Invalid or zero
quaternions in valid frames raise an error. Checkpoints have the distinct
`fixed_id_corner_rotation_6d_v1` format and input schema; legacy 13D
checkpoints cannot be loaded by the new loader.

After training, a detector-only tracker can be created with:

```python
from graph.temporal.corner_rotation_6d import load_tracker

tracker, checkpoint = load_tracker("/path/to/best.pt", distance_limit_m=0.03, device="cpu")
# tracker.step(detections_xyz, detector_kinds, robot9)
```

`robot9` remains TCP world XYZ, TCP wxyz, and two finger positions. The
tracker requires a distance gate calibrated for the new model; 0.03 above is
an example, not a validated gate. Recreate the tracker per episode. Existing
leaderboard loaders, evaluation recipes, frozen caches and checkpoints remain
on their original input contract. The new checkpoint is not accepted by those
legacy workflows; downstream evaluation and GNN retraining for the new tracker
need separate integration and validation. No accuracy or success-rate claim is
made until those experiments are run.
