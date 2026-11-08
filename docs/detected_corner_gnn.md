# Detector-only corner association and GNN

This guide covers detector-only association, frozen rollout inputs, downstream
GNN training and saved-result visualization. [Fixed-ID prediction](corner_prediction.md)
supplies the frozen temporal model; [lifecycle tracking](corner_tracking.md) is
a separate experiment.

## Evaluation preservation

The following six [leaderboard results](experiment_leaderboard.md#전체-4102프레임-종합-순위)
are preservation targets for cleanup work. These displayed values are rounded;
the original per-frame arrays and full-precision JSON metrics are the references.

| Training protocol | Architecture | Normalized aggregate error | Hole-position mean |
| --- | --- | ---: | ---: |
| Resumed with early stopping | Transformer | 0.1162 | 1.831 mm |
| Resumed with early stopping | Mamba | 0.1196 | 1.749 mm |
| Fresh 40 epochs | Transformer | 0.1297 | 1.923 mm |
| Fresh 40 epochs | Mamba | 0.1433 | 1.970 mm |
| Initial 20 epochs | Transformer | 0.1694 | 2.304 mm |
| Initial 20 epochs | Mamba | 0.1847 | 2.549 mm |

Before changing a numerical dependency, preserve its source bytes, the selected
checkpoint hashes, frozen temporal checkpoint identities, rollout cache hashes,
effective configs, and reference predictions. The evaluation uses seeds 0/1/2,
the same ordered 4,102 validation frame IDs and Object21 GT, and the original
batch size, dtype, device, package versions and aggregation order. Compare every
per-frame prediction and error array and every full-precision summary exactly.
Rounded table equality or a tolerance-based match does not establish exact
reproduction. Training refactors also need initialization and optimizer/RNG
restoration checks, including the original stopping and checkpoint selection.

When the complete evaluation inputs or original execution environment are
unavailable, retain the affected implementation and its dependencies. Saved
array checks validate existing results; report them separately from new model
inference. Existing checkpoint/source checks remain in force.

For the 2026-11-09 cleanup, the local baseline is in
[baseline.json](../outputs/refactoring_baseline/baseline.json). The local host
has no CUDA and no per-seed frozen rollout caches. The 20-epoch and resumed
results have saved per-frame predictions and GNN checkpoints; the 40-epoch local
copy has summary JSON only. Consequently the six-result numerical dependency
tree is preserved, and full inference reproduction is deferred until the
original inputs and CUDA environment are available. The baseline and other
`outputs/` artifacts are local verification evidence and are excluded from Git.

The [saved-result audit](../outputs/refactoring_baseline/protected_result_verification.json)
confirmed unchanged bytes for 52 source/config files and 581 artifacts. For the
20-epoch and resumed results, all 16 stored error metrics reproduced the original
full-precision aggregate statistics exactly across the 12 prediction archives,
with identical frame IDs and GT. The 40-epoch summaries retained their original
bytes. This audit did not execute model inference.

## Detector-only corner association

`graph.temporal.detected_corner_tracking.DetectedCornerTracker` builds one
internal history per encountered detector point. Before each frame, the frozen
shared temporal model predicts all existing tracks from actual past detection
history and the current robot state. Same-detector-type pairs within the distance
limit are matched one-to-one, maximizing match count before minimizing distance.
Matched outputs are the actual detector coordinates, unmatched old tracks use
the predicted coordinates, and unmatched detections initialize new tracks.
Predictions are never inserted into the history as measurements. Tracks are not
retired, so ID fragmentation and growth are measured rather than hidden.

Tracking receives only detector XYZ, detector peg/box labels and observed robot
state; no GT identity or GT geometry enters matching. The detector does not
provide a GT hole-family label. Offline scoring uses certified IDs and, only
when several detections have the same certificate, the reference vertex selects
one representative for metrics. Unknown-ID detections still participate in
matching but cannot be scored for their physical identity. A reappearance is a
certified detection following one or more frames without that certificate; it may
therefore include physical occlusions, detector misses and certificate failures.

The calibration recipe runs candidate distance gates on calibration episodes.
One common gate for all six model/seed pairs maximizes their mean reappearance
same-ID rate among candidates with at most 1% known-track identity conflicts in
every individual calibration run; ties choose the smaller gate. This rule is an
initial prototype criterion, not a universal safety bound. An identity conflict
means a previously certified track is subsequently certified as another corner.
When no candidate passes, selection fails instead of using validation to tune it.
The evaluation recipe replays separate validation episodes and reports
reappearance retention, new tracks on reappearance, identity switches, known
track conflicts, and growth. The old GT-ID GNN training remains a separate
experiment; this association evaluation does not train or evaluate a GNN.

Set paths and checkpoint hashes in the team YAML files. The matrix launcher
`tools/run_detected_corner_matrix.py --configs DIR --logs DIR` runs six matched
calibration jobs, selects the shared gate, runs six validation jobs, and creates
a comparison only after every preceding stage succeeds. This process waits for
its own child jobs and does not create a background status watcher.

### Detector-only tracking video

`graph/configs/temporal/visualize_detected_corner_tracking.yaml` renders
complete development-validation episodes using the source RGB camera frames,
seed-0 checkpoints, the calibration-selected 30 mm distance gate and the same
online tracker as validation. Configure the dataset, checkpoint and validation
result paths and hashes; CUDA, official `mamba_ssm`, OpenCV and FFmpeg/libx264
are required. The run verifies dataset lineage, input checksums, checkpoint
identity and aggregate replay counts against the recorded validation results.
It writes H.264 videos, a manifest and per-frame track coordinates locally.

Each dot color identifies an internally assigned track within its own panel;
the two panels assign IDs independently. A black rim denotes an actual detector
coordinate and a white rim denotes a prediction for a track not matched in that
frame. All valid detections are included, even those whose physical ID cannot
be certified. The tracker uses no GT identity. Predicted points can project on
top of an occluder, and overlapping dots may hide one another. Selected episodes
illustrate behavior and should not be treated as aggregate accuracy.

### Detector-only rollout RMSE

`graph/configs/temporal/evaluate_detected_corner_rollout_rmse.yaml` replays the
recorded detector-only validation with the selected threshold. It verifies the
source checkpoint, dataset, validation episode identities and per-episode match,
new-track and final-track counts. Configure these paths and hashes first.

Scoring is permutation-invariant: at each frame, all track output coordinates
are optimally paired by minimum total squared 3D distance with GT corner points
whose physical IDs have been certified at or before that frame. Assignment is
separate for detector peg and box types. All available pairs are matched without
a maximum-distance cutoff, and unmatched extra tracks and uncovered GT points
are counted separately. GT identities and coordinates are used only to form
and score this target set, never as input to the tracker. Unknown-ID detections
do participate in tracking and matching, but an otherwise unseen GT corner
without a certificate cannot yet be included in the target set.

The metric is `sqrt(mean(||output_xyz - gt_xyz||²))`, in millimetres, over
matched track-frame pairs. It is reported separately for currently detected
points and unmatched tracks using predictions. The identity-free geometric
pairing is an oracle evaluation, not the tracker's own matching algorithm;
it can associate a point with a physically different corner if this minimizes
distance. Extra unmatched nodes do not contribute to RMSE, so RMSE alone must
not be interpreted as full-set accuracy. Counts and per-frame records are saved.

On instance 59441593, all six rollouts reproduced the corresponding earlier
association counts across 40 validation episodes / 4,102 frames. The target set
contained 52,029 eligible GT corner-frames, all of which had a geometric pair.
The mean of the three individual seed RMSE values was:

| Model | All matched outputs | Currently detected | Predicted / unmatched by detector |
| --- | ---: | ---: | ---: |
| Mamba | 4.052 mm | 3.465 mm | 4.681 mm |
| Transformer | 3.723 mm | 3.472 mm | 4.007 mm |

The mean unmatched-output fractions were 19.50% for Mamba and 19.15% for
Transformer. An unmatched output is not automatically a false positive: it
may correspond to a physical corner whose GT identity has not been certified.
These track-frame outputs are excluded from the displayed RMSE, so the values
are conditional on geometric pairing, not a full graph error or a 21D metric.
See `outputs/detected_corner_rollout_rmse/comparison/300b51596ab4443bb5ff8083b75f4350/comparison.json`
for per-seed results and the corresponding per-episode frame counts.

### Detector-only rollout GNN training

`training.detected_corner_gnn:train` replays each complete source episode using
the calibrated detector-only tracker and the frozen, shared per-corner temporal
checkpoint. It creates graph nodes for every track, including tracks whose
physical ID cannot be certified. Current detections retain their exact observed
coordinates; unmatched tracks use predicted coordinates. Node kinds are only
the detector's peg/box labels, and the extra feature indicates detection versus
prediction. GNN semantic IDs are always unknown; GT corner identity, reference
geometry and GT-derived hole type never enter graph construction. Track IDs are
used internally to maintain history but are not a GNN feature.

The synchronized Object21 target cache and source episodes are checked against
the fixed-ID dataset lineage and sample/TCP order. Existing recorded validation
rollouts must reproduce the same match, new-track and final-track counts before
training proceeds. The full frozen rollout cache is saved with a checksum;
only GNN parameters enter AdamW. The team recipe uses 20 epochs, batch size 96,
learning rate 0.0003 and identical objective, initialization, seed and data
settings for paired Mamba/Transformer runs. Validation loss selects `best.pt`;
`last.pt`, history, progress and source metadata are also retained. No early
stopping is used for this fixed 20-epoch comparison.
Future runs print train/validation loss once per completed epoch; intermediate
batch progress is retained in `progress.json` without printing batch lines.

`evaluation.detected_corner_gnn_comparison:compare` verifies paired initial GNN
weights and settings and evaluates the saved rollout inputs using each selected
GNN checkpoint. It requires exact validation sample IDs, order and Object21 GT
against the existing leaderboard reference, and exports per-frame predictions,
21D error statistics and validation-loss curves. Results use development
validation, which also selects the checkpoint; GT IDs are not used for tracking.
Differences in input information and training settings should still be stated
when adding results beside older leaderboard entries.

On instance 59441593 all six 20-epoch jobs and the comparison completed. The
best GNN epochs were 20 / 19 / 20 for Mamba and 20 / 16 / 20 for Transformer.
Hole-position mean error was 2.549 mm and 2.304 mm respectively, averaged over
the three paired seeds on 4,102 development-validation frames. The nine 21D
metrics, per-frame predictions, seed-specific checkpoint identities and
independent aggregation audit are in the [experiment leaderboard](experiment_leaderboard.md#자동-코너-대응--gnn--20-epoch--검증-전체-4102프레임).
GNN best/last checkpoints, logs, configs, source snapshot and predictions are
backed up locally under `outputs/detected_corner_gnn_training/`; the much larger
per-seed `rollout_inputs.pt` caches remain on the server and can be reconstructed
from the recorded data and frozen temporal checkpoints. No artifacts were uploaded.

#### Fresh 40-epoch run with optimizer checkpoints

`training.detected_corner_gnn_fresh:train` validates and reuses each earlier
run's checksummed frozen rollout inputs, but constructs a **fresh** GNN and a
fresh AdamW. It does not load the earlier GNN weights or optimizer. The initial
GNN hash must match the original matched seed; the temporal checkpoint and all
data/model settings must match the source cache. The cache is hard-linked into
each new run, so the input artifact remains available without a second copy.

The run trains for 40 epochs with the same optimizer hyperparameters and data
split. `last.pt` and validation-selected `best.pt` are written atomically and
include GNN weights, AdamW state, epoch, Python/NumPy/Torch/CUDA RNG state,
effective configuration, source identities and cache hashes. The recipe saves
the state needed for a future resume; it does not itself implement resumption.
Epoch-level train/validation losses are printed, and intermediate batch progress
remains in `progress.json`. The original 20-epoch artifacts remain unchanged.

The detector-only GNN can continue from a completed 40-epoch run with
`training.detected_corner_gnn_resume:train` and a copy of its effective config.
Set `training.epochs` to the maximum total epoch, supply the source `last.pt`
and checksums for both `last.pt` and `best.pt` in `resume`, and set `patience`.
The continuation restores the GNN, optimizer, and random states, retains the
prior best checkpoint and history, and resets the patience counter at resume.
Only a strictly lower validation loss resets patience. Frozen detector-only
rollout inputs are reused, and checkpoints are saved at each epoch. The run
ends at the patience limit or the maximum epoch, whichever comes first.
See `graph/configs/temporal/train_detected_corner_gnn_resume.yaml` for the
resume fields; a full effective configuration is required.

### Object21 validation video

`python main.py --config graph/configs/temporal/visualize_object21.yaml` renders
original camera frames using previously saved Object21 comparison predictions;
fill in the comparison, family dataset, rollout cache and their checksums in a
copy of the config. The two panels use the same validation frame and seed. Peg
wireframes and hole rings depict world poses and dimensions; crosses show the
independently predicted endpoint. Hole axes depict the full rotation. Green is
GT and magenta is prediction. Coordinates are translated by the synchronized
TCP before projection. The visualization uses saved evaluation outputs, not a
new inference, and the selected examples are not aggregate test performance.
