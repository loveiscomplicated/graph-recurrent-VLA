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

## Published checkpoint backup

The 2026-11-09 KST backup preserves every checkpoint found in the automatic
association GNN workflows and their corner-predictor lineage. The private
[complete checkpoint index](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/blob/53587c96ad1a620a152a0bc6a23774a06b2a2dfd/graph/a6a3d5bc7df34172957302eb9524ba25/artifacts/metrics/25baea12a4f5d553282f75ce256ed6b736e2735a5eb317438498793603ba868c/checkpoint_index.json)
contains original run IDs, source paths, sizes, SHA-256 and immutable download
links for each file.

| Workflow | Checkpoints |
| --- | ---: |
| Detector-only GNN, initial 20 epochs | 12 |
| Detector-only GNN, fresh 40 epochs | 12 |
| Detector-only GNN, resumed early stopping | 12 |
| Shared corner predictor, initial training | 12 |
| Shared corner predictor, continuation | 12 |
| Shared corner predictor, early stopping | 12 |
| Corner-predictor preflight | 2 |
| Total | 74 |

Counts include both best/last files for Mamba/Transformer seeds 0/1/2, plus
the two preflight files. The 18 registered frozen rollout-cache files are also
preserved. Existing `tools.publish_experiment` published checkpoints, configs,
metrics and original run metadata to the Model repository; the three comparison
runs' prediction arrays and metrics use the private Dataset repository.
Original experiment IDs and training provenance were retained. Source code was
committed locally as `96cb93d`; this is the backup source revision, rather than
a replacement for the original runs' recorded Git provenance.

Independent remote checks verified 412 published files and matched all 74
checkpoints and 18 caches against server-side LFS SHA-256 and sizes. The index
and its publication files were checked separately. Saved arrays reproduced all
six full-precision result groups, including the fresh 40-epoch results; all 18
archives have identical frame IDs and GT. Model inference was not rerun.

Vast.ai instance `59441593` was stopped only after publication and verification:
the API reported `actual_status=exited`, `cur_state=stopped`, and the instance
remained present with its disk. No destroy operation was requested; other
instances were left alone. Local copies and receipts are under
`outputs/automatic_corner_backup/`, including the [remote verification](../outputs/automatic_corner_backup/remote_verification.json),
[saved-result audit](../outputs/automatic_corner_backup/six_result_verification.json)
and [stop confirmation](../outputs/automatic_corner_backup/instance_stop.json).
Historical notes below about artifacts not being uploaded refer to the state
at the original experiment's completion.

## Detector-only corner association

### Moving Hole visualization

`evaluation.moving_hole:visualize` renders a new scene with the configured
leaderboard Transformer and GNN checkpoints. Run the command in the
[Graph README](../graph/README.md#corner-prediction-and-tracking); set the two
checkpoint paths, device, frame count, translation and output directory in its
YAML. The temporal checkpoint hash is checked against the GNN's recorded source.
The existing detector, tracker, graph construction and experiment logger are
reused. The video shows actual detections and predicted tracks with internal IDs,
plus ground-truth and estimated Hole centers. Results JSON records each center,
error, track count and new-track count and the two checkpoint hashes.

The simulator's Hole is part of the kinematic box mesh, so this experiment moves
the entire box in a straight line; it does not move the hole independently within
the box. It advances the simulation once per frame. Elapsed time is the frame
index divided by ManiSkill's control frequency. The default recipe uses zero
actions; set `policy` to `module:factory` for a policy implementing the existing
`reset(env, seed)` and `act(observation)` interface. Playback FPS is independently
configured.

`motion.start_seconds` and `motion.end_seconds` set the permitted time window.
`continuous` interpolates the configured world translation over
`duration_seconds`; `jump` applies it once at the start. Before each move, the
guard uses simulator GT to measure the Euclidean distance from the peg head to
the Hole center. At or below `minimum_distance_m` it permanently stops motion,
even if the peg subsequently retreats. It also rejects a proposed translation
if any point along its straight path enters the same radius, including for a
jump. This conservative GT-based proximity proxy is checked at control-step
boundaries; it is not an insertion-phase detector or a deployable perception
signal. A policy can bring the peg closer between checks. Frame records include
time, pre-move distance, actual offset and guard status. GT is otherwise used for
display and error, never model input.

The optional `stop_expert_insert` setting additionally latches the guard when
the policy's `phase` is `insert`. It requires a policy with that field; the
existing `data_collection.scripted_policy:make_policy` exposes it and requires
`control_mode: pd_ee_delta_pose`. This expert uses simulator GT for control,
including the current moving goal; it does not use the GNN predictions. Phase
labels and simulator success are recorded per frame, including the terminal
frame. Configure `motion.mode: jump` for the same expert with an abrupt move.

New tracks can arise from newly visible corners as well as failed associations;
track counts alone do not establish identity switches.

On 2026-11-11, the Mac CPU simulator, detector and model ran seed 0 for 60
frames with 60 mm total world-X translation. The Hole-position error averaged
1.27 mm (maximum 2.75 mm); 14 initial tracks grew to 18. This is one illustrated
scene, not an independent accuracy or robustness benchmark. CUDA was not run.
The [video](../outputs/moving_hole/2e4a7b6287b44d81963946a04cd27a31/moving_hole.mp4)
and [per-frame results](../outputs/moving_hole/2e4a7b6287b44d81963946a04cd27a31/results.json)
are ignored local artifacts and may not exist in another checkout.

With the time-window implementation, 2026-11-11 Mac CPU zero-action examples
used 20 Hz, a 0.5–2.5 s window, an 80 mm guard and a 60 mm displacement.
[Continuous motion](../outputs/moving_hole/547ebac312db4cd488594430c74a61e1/moving_hole.mp4)
ran from 0.5 to 2.0 s; mean Hole error was 1.04 mm across 70 frames.
[The jump](../outputs/moving_hole/bb42c59fdce54ee39c9dc6d5df372289/moving_hole.mp4)
occurred at 0.5 s and added six tracks at that frame; mean Hole error after the
jump was 29.66 mm. The peg remained at least 282 mm away in these examples, so
they do not demonstrate proximity stopping during an actual insertion. Guard
behavior has separate tests; these examples are not aggregate benchmarks.

The [expert example](../outputs/moving_hole/94eebbdfa83748e6940242a6342983c8/moving_hole.mp4)
on the same Mac CPU used the expert recipe, seed 0 and a 3.5 s start. Motion
stopped at 14 mm when the expert entered insertion at 4.25 s; the peg head was
still 127 mm from the Hole center, showing why the distance guard alone does
not establish the phase boundary. The offset remained fixed throughout the
insert phase and ManiSkill success became true at 5.1 s. The 80 mm proximity
guard was also reached at 4.65 s. This is one GT-guided episode, not a policy
success-rate measurement. Its [frame records](../outputs/moving_hole/94eebbdfa83748e6940242a6342983c8/results.json)
include the guard, phase and Object21 errors.

The matching [abrupt-motion expert video](../outputs/moving_hole/1bc9da6aab6a484c9a3cd15e36fd9100/moving_hole.mp4)
uses seed 0 and jumps 60 mm at 3.5 s, during the expert's align phase. The
guard froze the offset when insertion began at 4.3 s, and ManiSkill success
was true at 5.25 s. Mean Hole prediction error from the jump through the
terminal frame was 55.58 mm; this GT-guided expert does not consume those
predictions. [Per-frame results](../outputs/moving_hole/1bc9da6aab6a484c9a3cd15e36fd9100/results.json)
record the motion, tracking and phase data. This is a single example.

With the same expert and guards, 120 mm examples are available for
[continuous motion](../outputs/moving_hole/414c77586d65416cb96e8f5d3accb5c5/moving_hole.mp4)
and an [abrupt jump](../outputs/moving_hole/bb5ffae219d148b7ae18fec7ce906cbd/moving_hole.mp4).
Both reached the full 120 mm displacement before the expert entered insertion,
then held the box fixed and reached simulator success on seed 0. Mean Hole
prediction error from 3.5 s to the terminal frame was 59.84 mm and 86.75 mm,
respectively. The controller uses GT, not these predictions. These are single
illustrations; configs retain their exact settings under
`graph/configs/temporal/visualize_moving_hole_expert_120mm.yaml` and
`graph/configs/temporal/visualize_jumping_hole_expert_120mm.yaml`.

For a view without a moving arm crossing the Hole, zero-action 120 mm examples
show [continuous movement](../outputs/moving_hole/4d1a71156d3648e79a6008801509d229/moving_hole.mp4)
and an [abrupt jump](../outputs/moving_hole/ba42aebd6ec3497985938c7fab78d924/moving_hole.mp4).
The Hole is visible in the inspected camera frames, but no per-pixel occlusion
certificate was computed. The actual displacement was 120 mm in both; mean
Hole error from 0.5 s through the end was 6.47 mm and 59.45 mm respectively.
The jump introduced eight new tracks in the jump frame, not necessarily eight
identity errors. Recipes are `visualize_unobstructed_moving_hole.yaml` and
`visualize_unobstructed_jumping_hole.yaml` under `graph/configs/temporal/`.

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

## Random-action diagnostics

`graph/configs/temporal/random_action_diagnostics.yaml` runs the frozen
leaderboard seed-0 Transformer and GNN in the live simulator. For each
configured environment seed it replays an expert until the simulator reports a
successful grasp, the peg has lifted by at least 30 mm, and the expert reaches
alignment. It then compares expert continuation against separately seeded random
6-D arm commands, keeping the gripper closed. A run that never reaches that
switch is recorded rather than silently treated as a successful rollout.

GT is used for the switch check and for **diagnostics only**. Detector error is
the distance to the nearest same-kind GT corner, allowing duplicates and without
asserting visibility. At track birth, the nearest same-kind reference within
10 mm is assigned for diagnostic comparison; unassigned tracks remain in the
actual input and are counted separately. Track-ID switches, ambiguous corners,
unknown detections, and never-observed corners make this an approximate oracle,
not a true detector recall or identity certificate. Matched-node RMSE excludes
unassigned tracks and therefore must be read alongside its coverage.

The GNN counterfactuals replace observed and/or predicted coordinates of
assigned tracks with those GT coordinates while preserving their original
provenance flags and leaving unassigned tracks unchanged. Another drops
unassigned tracks; another changes the assigned tracks' provenance flags to
observed. A final deliberately out-of-distribution condition supplies all 24 GT
corners as observed, including occluded corners. These interventions can interact
and change graph support, and do not establish an additive causal attribution or
a deployable model score. The original GNN input never uses GT.

A 2026-11-13 Mac CPU run used environment seeds 0–4, action seeds 73 and 74,
and 35 post-switch steps per rollout. All 15 runs reached the switch, all 525
post-switch frames retained the simulator grasp, and each expert/random prefix
matched before switching. Across the ten random versus five expert rollouts,
episode-weighted mean Peg position errors were 51.01 versus 18.80 mm, and Hole
position errors were 11.63 versus 7.57 mm. Random rollouts had zero detector
outputs in 107/350 frames; expert controls had zero in 8/175 frames. These are
five paired initial states and two reused action sequences, not ten independent
initial states. The model seed was 0 and no initial visibility filter was applied.

Across random rollout detections, nearest-reference 3D RMSE was 3.21 mm for
1,645 detections; 9 were over 10 mm. For birth-assigned track nodes, observed and
predicted 3D RMSE were 2.53 mm (1,632 nodes) and 18.72 mm (2,483 nodes),
respectively; 326/4,441 total track nodes were unassigned and omitted from those
track errors. Averaged over random episodes, Peg error was 51.01 mm originally,
46.70 mm after replacing assigned predicted positions, 46.73 mm after replacing
all assigned positions, and 5.09 mm using the complete 24-corner GT graph. The
last condition adds hidden corners and changes provenance; the gap is not a pure
GNN error measurement. In one random rollout all 35 frames after switching had
zero detections. Its assigned peg tracks came only from one physical end; fixing
their coordinates still left 128.5 mm mean Peg error, while the complete GT
condition gave 3.9 mm. This points to lost or insufficient graph support and
tracking errors as major failure modes, without proving the training distribution
is their sole cause. Model inference on CUDA and the other two leaderboard model
seeds were not executed.

The [run results](../outputs/random_action_diagnostics/5a5605be3d2c4d71b97b945082601647/results.json),
[main plot](../outputs/random_action_diagnostics/5a5605be3d2c4d71b97b945082601647/diagnostics.png),
[stage plot](../outputs/random_action_diagnostics/5a5605be3d2c4d71b97b945082601647/stage_diagnostics.png)
and per-episode videos are local, ignored verification artifacts.
