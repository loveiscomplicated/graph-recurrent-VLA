# Episode collection

## Fixed-ID per-corner sequences

This dataset-only workflow isolates per-corner temporal prediction before adding
any downstream graph. Set `source.corner_dataset` to a completed, full-frame
`observed_corner_tracks_v1` directory in
`data_collection/configs/fixed_id_corner_sequences.yaml`, then run from the root
with Python 3.11, NumPy and PyYAML:

```bash
WANDB_MODE=disabled python main.py --config data_collection/configs/fixed_id_corner_sequences.yaml
```

IDs are defined in object-local coordinates, independent of detector output
order or world pose. Lexicographic sign combinations `(-1,-1,-1)` through
`(+1,+1,+1)` identify eight corners per family:

| IDs | Frame | Local coordinates |
| --- | --- | --- |
| 0–7 | peg | signs × peg half sizes |
| 8–15 | box | signs × `[peg_half_sizes.x] * 3` |
| 16–23 | hole | signs × `[peg_half_sizes.x, hole_radius, hole_radius]` |

The source certificates attach actual detections to these IDs offline. UNKNOWN
certificates stay unassigned; the adapter does not force them to the nearest
corner. These inherited GT-assisted certificates have imperfect coverage and
are not evidence of perfect visual identity recognition. When certificates
duplicate an ID in one frame, offline annotation retains the actual detection
closest to that ID's GT coordinate, without averaging or snapping its position.
The manifest counts unknown detections and duplicate certificates.

Each `(episode, vertex_id)` sample begins at its first certified detection.
Never-observed IDs and frames before that first sighting are omitted. The first
sighting initializes history; all subsequent frames are prediction targets,
including frames with detections. An ID first seen in the final frame has no
future prediction target and is counted separately. There is **no artificial
corner deletion, random masking or synthetic occlusion** in this workflow.

- Visible observation and target: unchanged actual detector XYZ in metres.
- Missing after first sighting: zero observation with a false observation flag;
  GT world XYZ is a supervision label only.
- `observed` means a certified detection is available. Missing intervals can
  reflect occlusion, detector misses, or an unresolved ID certificate.

`iter_sequences(root, split)` yields independent corner sequences. Their model
inputs have shape `[T,13]`: previous-frame observed XYZ (3), previous-frame
observation flag (1), and current observed robot state (9: TCP world XYZ, wxyz
quaternion, two finger positions). The current detection is shifted to the next
input so that predicting a visible target cannot become copying that target.
`prediction_valid` excludes only initialization. `targets` contains world XYZ;
`target_delta_xyz` subtracts `last_observed_xyz_world`, the most recent detection
strictly before each valid prediction, for displacement training. Predictions
are not fed back as measured observations. A future consumer can return the
unchanged measurement when visible and the predicted position when missing.

ID is sample metadata, not an embedding or numeric input. All corner samples
can reuse one temporal model's weights. Mamba and Transformer can consume the
same samples, labels, masks and parent-separated splits. This command builds
data only; it imports no tracker, GNN or temporal model and starts no training.
It retains train/calibration/validation membership, opens no reserved-test files,
checks input/output hashes and writes a completion manifest only after all
episodes succeed. Experiment identity and the effective config use the shared
`ExperimentLogger`; nothing is uploaded. Run
`.venv/bin/python -m unittest tests.test_fixed_id_corner_sequences -v` to check
first-sighting rules, real-coordinate preservation, ordering invariance,
causality, parent separation and interrupted/corrupt-source behavior.

### Prepared dataset verification

On 2026-11-08 the server converted all 630 episodes from source run
`c9c8da5bc0ac46f8a5cfad4219b082e6`, preserving the previous development splits:

| Split | Episodes | Corner sequences | Visible prediction targets | Missing prediction targets |
| --- | ---: | ---: | ---: | ---: |
| train | 531 | 6,786 | 353,892 | 318,322 |
| calibration | 59 | 753 | 38,430 | 36,769 |
| validation | 40 | 519 | 27,719 | 23,791 |

The dataset is at
`/workspace/fixed_id_corner_sequences/data/278debee127a4c8cad63271e9ff13e22`;
its dataset SHA-256 is
`989ad1a9096d3841a5e0d903eb7afea35a598c9cd57820205ddc9e9e14354d3e`.
The exact server config is `/workspace/fixed_id_corner_sequences/prepare.yaml`.
The tmux session `fixed_id_corner_data` retains `prepare` and `audit` windows;
logs are `/workspace/fixed_id_corner_sequences/prepare.log` and `audit.log`.

The full source/output audit passed for every episode and sequence: 428,099
retained observations match original detector coordinates exactly, all eligible
certified observations remain present, visible and missing targets follow the
declared rule, causal inputs contain no current measurement, and parent splits
are unchanged. Of 428,105 certified source detections, five duplicate certificates
were collapsed to one original measurement each, and one terminal-only first
sighting had no future prediction target. There were zero artificially removed
certified observations. The 60,300 UNKNOWN detections remain unassigned and
7,061 never-seen episode/vertex combinations are excluded. UNKNOWN intervals
must not be interpreted as confirmed physical occlusion.

Canonical ID geometry and observed robot state additionally matched 18 sampled
original raw-GT frames, including world rotations. All seven local contract
tests passed. This preparation/audit ran NumPy on CPU; no simulator, CUDA model
or new training was executed. The implementation was a dirty working snapshot
based on `320b7cc`; exact source-file hashes are recorded in the effective config
and manifest. The original sources and previous model checkpoints are preserved.
Local copies of the [verification result](../outputs/fixed_id_corner_sequences/278debee127a4c8cad63271e9ff13e22/verification.json),
[summary](../outputs/fixed_id_corner_sequences/278debee127a4c8cad63271e9ff13e22/summary.json)
and [effective config](../outputs/fixed_id_corner_sequences/278debee127a4c8cad63271e9ff13e22/prepare.yaml)
are retained as ignored verification artifacts.

## Full episodes for tracked corner completion

`python main.py --config graph/configs/temporal/corner_tracking/prepare.yaml`
packs an existing **full-frame** vertex-family dataset into checksummed temporal
episodes. This preparation adapter also requires OpenCV for the existing label
utilities. Set `source.family_dataset` first. It reuses observed graph coordinates,
BOX physical certificates and the original raw observed TCP/finger state; it
does not run detection or edit RGB-D. PEG supervision reuses the existing unique
coordinate/projection association. Those GT-assisted IDs are labels only and
can remain UNKNOWN. The source certification policies have different strength;
they are not a claim of perfect corner identity.

The adapter rejects selected-frame/noncontiguous cohorts. It preserves the
source development-validation parents and reserves 10% of source training
parents, seed 0, exclusively for calibration. These parents do not receive
optimizer updates. The source dataset is original-TRAIN-only; its development
validation is not a new independent test. Prepared manifests support reserved
test metadata, but training/evaluation here never open those episode files.
Raw files, graph files and certificate archives are checked against source hashes.
Prepared episode bytes and the manifest are also hashed. Every episode carries
an explicit parent and split; overlapping parents are rejected.

For faster preparation, set the source path and `workers` in
`data_collection.temporal.corner_tracks_parallel:prepare` via
`graph/configs/temporal/corner_tracking/prepare_parallel.yaml` and run the same
root entrypoint. Workers process independent episodes, check the same source
hashes and GT/robot alignment, and report progress. The final manifest is written
atomically only after every episode succeeds; failed runs retain partial files
without a completion manifest. `tests.test_corner_tracks_parallel` compares the
prepared episode arrays and splits with the serial adapter and checks that source
bytes remain unchanged and corrupted input cannot publish a manifest.

See [tracked corner completion](corner_tracking.md#tracked-corner-completion-before-gnn)
for synthetic-removal manifests, chronological eligibility, supervision boundaries,
and the shared Mamba/Transformer workflow.

For the separate Graph-team development cohort that preserves selected original
TRAIN frames, detector metadata, physical-role certificates and immutable graph
inputs, see [the vertex-family study](vertex_family_semantics.md). Its
`data_collection.vertex_family:prepare` workflow uses the root config entrypoint
and does not change the collection gate, raw image contract or original splits.

Run commands from the repository root after installing `requirements.txt`.

For the explicitly requested 900-episode expert dataset on a Linux CUDA/Vulkan
host, also install `requirements-tracking.txt`, configure HF write access, and run:

```bash
WANDB_MODE=disabled python -m data_collection.collect_dataset \
  --config data_collection/configs/visible_peg.yaml
```

That config requests native Main/Wrist 512×512 RGB-D, CPU physics, CUDA rendering,
900 successful scripted-expert trajectories, 250-step horizons, raw GT at every
frame, four independent workers and 30-episode archives. Initial Main visibility
is recomputed from every saved initial frame. Every saved frame's raw image schema
and action/observation alignment are checked before publication. Success filtering
changes the reset distribution; no train/validation/test partition is imposed.
`publish: true` explicitly invokes the shared `tools.publish_experiment` trajectory
workflow after collection; `publish: false` keeps the result local. Existing output
directories are refused. A process failure is reported rather than automatically
resuming or overwriting its output. Failed expert rollouts remain local and are
excluded from the approved manifest and published archives. HF failures retain
local archives and the existing experiment run identity for a publishing retry.

`data_collection/scripted_policy.py` extracts the existing GT-guided servo expert
without importing team experiment packages. It uses `pd_ee_delta_pose` and the
installed controller's negative rotational scaling, and returns normalized native
actions matching the actual environment action-space shape. The original team
implementation remains unchanged. This is a privileged expert, not a learned policy.

```bash
python -m data_collection.collect_episodes --help
python -m data_collection.collect_episodes --output outputs/initial_frames \
  --num-episodes 10 --max-attempts 1000 --resolution 512
```

The collector tries consecutive reset seeds starting at `--start-seed` (default
0). It evaluates only the first `base_camera` observation and retains Main and
Wrist raw observations for approved episodes. No action is applied to rejected
states. The attempt budget is finite; insufficient approved episodes produce
`EXHAUSTED` and a nonzero CLI exit. Use a new output directory for each run;
existing output is never overwritten and automatic resume is not implemented.

### Paired Object21 demonstrations for BC

Use `data_collection.collect_object21_bc:collect` and
`data_collection.object21_bc:export` with the root launcher. The first
workflow reuses the raw shard collector and independent verification. Keep the
raw collection settings from `data_collection/configs/visible_peg.yaml`,
change the output directory and seed range, and set `publish: false`.
Provide a JSON exclusion inventory with the original 900 seeds in
`original_collection_seeds`, the 630 Graph development seeds in
`graph_development_seeds`, and their dataset and collection identities in
`graph_dataset_sha256` and `original_collection_run_id`.

```bash
WANDB_MODE=disabled python main.py --config /path/to/collect.yaml
WANDB_MODE=disabled python main.py --config /path/to/export.yaml
```

The collection YAML supplies `trainer`, `collection` (path to the raw settings)
and `excluded_seeds` (path to the inventory). The export YAML supplies
`trainer`, `raw_root`, `output`, `device`, `workers`, `split_seed`,
`split_counts`, and `models`. Set `split_counts` to the intended numbers
of train, validation and test episodes. For each of `models.transformer`
and `models.mamba`, supply `gnn`, its `sha256` and the matching `corner`
checkpoint. Use the resumed seed-0 pairs from the
[checkpoint index](detected_corner_gnn.md#published-checkpoint-backup).
Export needs CUDA, the official Mamba selective scan extension and the
[corner detector dependencies](corner_detection.md#install-and-run).

The exporter may run alongside collection and processes only verified shards.
It fixes split positions by sorted episode seed before inference and stores both
models' predictions for each expert action. Each split contains aligned
`robot_state`, `object21_transformer`, `object21_mamba`, and
`expert_action` arrays. Use `episode_offsets` to recover trajectories. The
Object21 export supplies the 9D robot input used by the corner tracker.

For the Recurrent BC release, run `data_collection.recurrent_bc_dataset:export`
with `raw_root`, `object21_root` and a new `output` in its YAML. It joins
Robot15 by reset seed and step, matching
`recurrent.features.FEATURE_LAYOUT`. The primary inputs are
`state36_transformer` and `state36_mamba`, each `[Robot15, Object21]` with
the same expert action and episode splits. The README includes the legacy
clean-observation State38 adapter, appending `[obs_fresh=1, obs_age=0]`.
The compact archive includes a loader example, split checksums and model
provenance. Publish it after export through
[the shared publisher](huggingface_artifacts.md).

The first [paired Recurrent BC release](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/eca8ba8b122e7d2e7007e7dc20200dad5d3e3725/integration/293fbe2c68764fb8901a40f857a1ade6)
contains 900 fresh successful episodes with 92,566 aligned action rows. The
train/validation/test splits have 720/90/90 episodes and
74,053/9,201/9,312 action rows. Reset seeds range from 200005 to 229115 and
have zero overlap with the previous 900-episode collection, including its 630
Graph development episodes. The archive's SHA-256 is
`aec78bdef35d8f873d0cc8215d1e986050940c9041e7e0948e21081d5531c1f0`.

## Visibility

`data_collection/visibility.py` extracts the historical GT/depth gate, including its calibration
checks, signed depth handling, pixel centers and expert grasp-band calculation.
It imports no Graph, perception, tracker or model packages from this repository.

The default `--visibility-mode ends_and_grasp` means:

- All eight GT peg box corners are in the Main image, in front of the 1 cm near
  plane; projection boundary epsilon is 0.0001 pixels.
- Both end strips (17.5 mm each) and the historical expert grasp band (17.5 mm
  total width) have projected camera-facing surface pixels.
- Every predicted pixel in those regions has valid observed depth within
  1.5 mm of the GT box front surface. Nonpositive signed int16 depth is invalid.

`--visibility-mode full_surface` additionally requires depth confirmation for
all projected camera-facing peg surface pixels. Neither mode requires the
hidden back surface or visibility at later timesteps. These are GT-assisted
dataset selection rules, not a visual detector. Physical collision validity,
expert success and historical diversity/split selection are separate historical
filters and are not included in this extracted visibility collector.

## Trajectories

The default `--steps 0` stores only the approved initial observation. To collect
actual trajectories, provide `--steps N --policy your_module:make_policy`.
The factory takes no arguments and returns an object implementing:

```python
class Policy:
    def reset(self, env, seed):
        # Initialize policy state; do not reset, step, or reposition the env.
        ...

    def act(self, observation):
        # Native batched action matching env.action_space.shape.
        ...
```

The policy receives the same initial observation that passed selection and
controls its own stochastic seed initialization. Collection stops on terminated,
truncated or the requested step count. The policy's native action convention
must match `--control-mode` (omitting it uses the installed task default).

For a simulator/storage smoke check, use explicit zero actions:

```bash
python -m data_collection.collect_episodes --output outputs/collection_smoke \
  --num-episodes 1 --max-attempts 40 --steps 2 --zero-actions
```

This is not an expert demonstration. No old expert, training or model workflow
is imported by this extraction.

To use the standalone scripted expert through the existing collector:

```bash
python -m data_collection.collect_episodes --output outputs/expert_episodes \
  --num-episodes 3 --max-attempts 100 --resolution 512 \
  --steps 250 --max-episode-steps 250 --require-success --save-step-gt \
  --policy data_collection.scripted_policy:make_policy \
  --control-mode pd_ee_delta_pose --render-backend cuda --observation-device cuda
```

`--max-episode-steps` explicitly overrides ManiSkill's default 100-step time limit.
`--require-success` stops a trajectory on the first official success and keeps
unsuccessful attempts in `rejected_seed_*/`, outside the approved manifest.
`--save-step-gt` adds raw `ground_truth/` fields after each action, in addition to
the existing initial GT. These options are opt-in; initial-only defaults remain.

## GPU observations

Add `--render-backend cuda --observation-device cuda` on a supported Linux GPU
host. See [GPU observations](gpu_observations.md) for native rendering versus
output transfer, the runnable device check and backend verification limits.

## Saved data

```text
output/
├── config.json                 # parameters, actual backends, versions, source revision
├── attempts.jsonl              # every seed's first-frame visibility decision
├── manifest.json               # run status and committed episode list
└── episode_000000/
    ├── episode.json            # seed, frame/step counts, stop flags
    ├── visibility.json         # detailed first-frame selection evidence
    ├── frame_000000.npz        # raw initial observation, info and selection GT
    ├── frame_000001.npz        # next observation and action that led to it
    └── ...
```

NPZ keys flatten nested dictionaries with `/`, e.g.
`observation/sensor_data/base_camera/rgb`. Shapes, batch dimensions and dtypes
are preserved. Depth stays signed int16 millimetres; overflow/invalid values
are not repaired. Load with `np.load(path, allow_pickle=False)`. Each post-action
frame includes `action`, `reward`, `terminated`, `truncated`, and `info`.

An episode has T+1 observations and T actions. Every frame is read back before
being exposed. Episode folders are renamed from `.partial` only after their
receipt is written; interrupted/failed folders remain excluded from the manifest.
Exact simulator-state replay, automatic split assignment and video encoding are
not provided. The raw collector itself does not publish; the dataset workflow
above explicitly validates and publishes selected archives through the shared CLI.
Successful-task filtering requires the explicit `--require-success` option.

## Checks

```bash
python -m unittest tests.test_visibility tests.test_data_collection \
  tests.test_observation_device tests.test_maniskill_env -v
python -m unittest tests.test_observation_device_native_gpu -v
```

Recorded results and execution limits: [collection verification](data_collection_verification.md).

## Temporal-gate episode selection

`data_collection/temporal_gate_data.py` selects complete episodes from the existing
published graph cache. Its explicit parent manifest keeps Gate development within
the GNN train split and reserves original GNN validation for final comparison.
It neither collects new simulator data nor modifies cached frames. See
[learned temporal gate](gnn_training.md#learned-temporal-update-gate) for source
parent mapping requirements and the manifest-generation command.

`data_collection/temporal/frozen_episodes.py` shares frozen-frame inference with the
learned gate and GRU estimator. It sorts predictions, targets, observed TCP and
quality by one sample-ID permutation. `FrozenEpisodeDataset` requires complete
zero-based, contiguous episodes and validates row shapes and frame/episode IDs.
Graph/GT pairing is checked against the existing cache identity and byte checksum
before sequence construction. Inputs contain only current GNN outputs and actual
observation metadata; labels are returned separately as Hole world XYZ. GRU uses
the same explicit parent manifest and never reads original GNN test frames.
See [GRU estimator](temporal_estimation.md#gru-temporal-object-state-estimator).

`TemporalPoseDataset` in `data_collection/temporal/pose_sequences.py` extends this path
for Peg/Hole poses. It reuses sample ordering, checksum and parent-split checks,
adds valid PEG/BOX/HOLE quality and causal TCP translation changes, and returns
world-frame Object21 labels separately. Padding is structural and masked; input
normalization is fitted on train observations only. Optional `data.raw_root`
reads only `observation/extra/tcp_pose` and `observation/agent/qpos` from matching
raw frames, verifies every frame against the cache's source SHA256 and checks TCP
agreement. It adds wxyz TCP orientation, causal angular motion and the two Panda
finger positions. There is no fallback to `ground_truth/` fields. Cache-only mode
omits these nine unavailable values. See
[joint pose estimator](temporal_estimation.md#joint-peghole-temporal-pose-estimator).

Insertion-precision GRU retains this same complete-episode input/normalization
contract. `prepare_pose_sequences` can also reuse the independent-cache selector
in `data_collection/temporal/insertion_sequences.py`: a `test_role: independent` manifest must provide
explicit parent identities for both original and external cache episodes. All
external parents must be absent from every original GNN split. Training only
checks external split metadata for leakage; external frame tensors are opened
only for explicit test evaluation. `data.independent_raw_root` is required when
the independent test uses the observed-robot feature schema. The previous GNN
validation comparison remains available and is labeled reused reference data.

`data_collection/temporal/insertion_sequences.py` extends the same frozen-episode path
for insertion Mamba/GRU experiments. It adds observed-support ages and causal
windows, and requires an explicit `gnn_validation_reference` or `independent`
test role. Independent-cache parents must be absent from every original GNN
split. Training validates external metadata without reading independent test
frames. See [Mamba insertion estimator](temporal_estimation.md#mamba-insertion-precision-estimator)
for manifest creation, raw-proprioception joins and the independent-test workflow.

`data_collection/temporal/finetuning_split.py` partitions only the original Mamba
training parents into fine-tuning train/development (seed 0, 80/20 by parent).
The existing 126-episode validation is reserved as a reused development benchmark;
it enters neither gradients, auxiliary-weight calibration nor checkpoint selection.
Pretrained Mamba, Gate and GNN have already seen the new development parents, so
this is held out only from the fine-tuning updates. The original reference test is
also reserved and is not promoted to an independent test. Existing cache checksums,
raw observed robot joins, frame identities and visibility conventions are reused.

## Physical vertex coverage filtering

`data_collection.physical_vertex_selection:prepare` applies a GT-assisted,
frame-level filter to all original TRAIN frames from the 590 parents remaining
after excluding the 40 development-validation parents. Configure paths in
`data_collection/configs/physical_vertex_selection.yaml` and run:

```bash
WANDB_MODE=disabled python main.py --config data_collection/configs/physical_vertex_selection.yaml
```

The detector processes the original RGB-D, with no GT fed to it. For each PEG
candidate, match to a unique physical peg corner within 3 mm in each GT-local
coordinate and 2 pixels in projection; unmatched or ambiguous detections do not
count. The condition requires at least four distinct matched corners, excluding
any set whose corners all share one physical box face. For the Hole condition,
the existing GT/RGB-D topology certificate in `vertex_family_labels.py`
requires three distinct RIM physical IDs; exterior and UNKNOWN candidates do not
count. The coordinate, projection, face-depth and image-support tests are
finite-tolerance operational certificates, not proofs of physical identity.
GT is used in dataset selection and can introduce selection bias; it is not a
reliable deployable selection method without GT. Passing the gate is evidence
under these rules, not a claim about every physically visible corner.

The run verifies source frame hashes, retains the original source IDs, and writes
all decisions to `frames.jsonl` plus aggregate counts and source/config hashes.
It does not alter the input dataset or exclude frames from other workflows.
`max_frames` is only for smoke checks; only an unrestricted 590-parent run can
produce complete population counts. The JSONL can be used as a frame-ID filter
for a later training workflow; no training is initiated here.

The raw and cached entrypoints share frame selection, input validation, ordered
spawn-worker execution, aggregation and output recording in
`data_collection/physical_vertex_selection.py`. The cached entrypoint remains
available at `data_collection.physical_vertex_cache_selection:prepare`; existing
config files and commands still work. Raw workers retain detector UV, while
cached workers project checksum-verified XYZ and exclude padded graph nodes.
Episode auditing reuses the cached workers and preserves the original 960-frame
decisions separately. Cached provenance records both the shared implementation
and the compatibility entrypoint's code hashes.

Run `.venv/bin/python -m unittest tests.test_physical_vertex_selection tests.test_physical_vertex_filter -v`
for parent/ID validation, frame order, population and smoke counts, checksum
rejection, distinct pixel-source decisions, preservation of the original
validation decisions, and actual CPU spawn-worker execution on synthetic native
RGB-D frames. These checks do not rerun the full server dataset or CUDA detector.

For an expanded vertex-family development cohort, `selection.validation_selection`
can point to a prior `selection.json` to preserve its validation episodes while
selecting training parents from the remaining original TRAIN pool. See
[vertex-family semantics](vertex_family_semantics.md#확장-학습-설정).

`selection.frames_per_parent: null` retains all recorded frames for the selected
parents. `raw_storage: hardlink` avoids duplicating raw files when the source and
output are on the same filesystem; the linked raw files share source inodes and
must be treated as immutable. The default `copy` creates independent files.

### Verified cached-graph selection result

On 2026-11-08 KST, the published graph cache was used for the full 590-parent
selection with `data_collection/configs/physical_vertex_cached_selection.yaml`.
Every raw frame and graph archive was hash-checked, and detected UV was recovered
by projecting cached world XYZ through the saved camera calibration. This is not
a fresh detection for all frames: on the first 8,952 frames, independent GPU
re-detection and cached-graph selection produced identical per-frame decisions.
That check does not prove equivalence on every remaining frame. The detector
source files match the cache's recorded source hashes; a 150-frame inspection
showed reconstructed UV differences at most 0.000069 pixels.

| Rule | Frames |
| --- | ---: |
| Peg condition | 21,176 |
| Hole condition | 9,481 |
| Both | 8,093 |

These totals are under the finite-tolerance GT-assisted rules above and apply
to the 60,600 recorded original TRAIN frames in 590 parents only. Local
[summary](../outputs/physical_vertex_selection/7cd80170f5304b69b8384cbf99b457c3/summary.json)
and [frame decisions](../outputs/physical_vertex_selection/7cd80170f5304b69b8384cbf99b457c3/frames.jsonl)
contain all IDs; `both_pass` selects the intersection. No model training was run.

For family preparation, `reuse_family` lists dataset roots with completed or
recovered manifests. Source dataset, label-generating code, settings, archive
hashes and graph/label node ordering are checked before graph/label files are
hardlinked into the new dataset. `reuse_graph_cache` points to the original
cache with metadata/inventory: after obtaining missing detector provenance,
its graph is reused only if XYZ, object order, confidence, TCP and Object21
match exactly. Detector mismatches rebuild that frame's graph rather than
attaching fresh labels to a different node set. Reuse totals are logged and
recorded in the final manifest. An interrupted run's completed frame archives
can be recovered after validating their identities and content; incomplete
archives must be recomputed.

### Physical conditions across complete validation episodes

`data_collection/configs/physical_vertex_episode_selection.yaml` applies the same
GT-assisted physical vertex criteria to every frame represented in a temporal
prediction file. Run `python main.py --config data_collection/configs/physical_vertex_episode_selection.yaml`
after setting the prediction, original 960-frame selection, parent manifest, raw
frame and cache paths. Raw and cached files are checksum verified. New frames use
pixels reprojected from cached detector XYZ; the previously selected 960 retain
the original detector-pixel decisions. The workflow records any disagreements.
For the seed-0 validation episodes, 4,102 frames across 40 parents were checked;
all 960 original decisions agreed with the reprojection audit. The result is a
conservative GT/RGB-D certificate, not a model input or a claim that uncertified
corners are invisible.

### Restore the frozen-M0 temporal experiment on another host

`data_collection/temporal/restore_hf_temporal.py` restores the already published
original graph cache and contiguous raw training-pool episodes, plus explicitly
published temporal checkpoints and the matching runtime/split bundle. It pins
repository revisions, verifies every archive and selected raw frame against its
SHA-256, restores only the original 630-episode training pool, and deletes each
verified raw archive after extraction to limit disk use. The transfer uses a
private Hugging Face dataset and model repository and requires an existing
read-capable token supplied to the process; credentials are not written to the
config, logs or artifact bundle. The six temporal continuations retain the
original frozen 60-epoch M0 checkpoints; the independently continued GNN is not
substituted into them. The restored temporal configurations resume epoch 90 with
Adam and random state, then run 30 additional epochs. Local focused tests cover
selected archive members and checksum rejection.
