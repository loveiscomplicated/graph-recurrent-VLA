# Hybrid temporal estimation

The Graph team's hybrid workflow composes frozen Mamba and Learned Gate outputs,
then supports Peg-head fine-tuning and saved-array diagnostics. Run from the
repository root with Torch, NumPy, PyYAML and Matplotlib, the documented original
checkpoints and checksum-identified evaluation inputs available locally.

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/hybrid/evaluate.yaml
```

Model assembly lives in `graph/temporal/hybrid.py`. Reusable causal replay and
Peg-head feature caching live in `data_collection/temporal/hybrid_features.py`.
Training stays in `training/gnn/temporal/`; scoring and analysis stay in
`evaluation/gnn/temporal/`. `hybrid_inputs.py` validates saved inputs separately
from the evaluation entrypoint.

Historical source manifests keep their original file paths and hashes. The
source check accepts exact current bytes or the original bytes after reversing
only the known import relocations and a final blank line. Model calculation changes still fail SHA256
validation. New audits record current and historical hashes and the check used;
past checkpoints, prediction archives and experiment records are not rewritten.

- [Frozen hybrid evaluation](#frozen-mamba--learned-gate-hybrid)
- [Peg-head fine-tuning](#h1-peg-head-relative-geometry-fine-tuning)
- [Error attribution](#hybrid-h1-relative-geometry-error-attribution)
- [Temporal estimators](temporal_estimation.md)
- [Base GNN and learned Gate](gnn_training.md)

## Frozen Mamba + Learned Gate hybrid

Run the evaluation without training, checkpoint writes, controllers or publishing:

```bash
WANDB_MODE=disabled .venv/bin/python main.py --config graph/configs/temporal/hybrid/evaluate.yaml
.venv/bin/python -m unittest tests.test_temporal_hybrid -v
```

Prerequisites are Torch, NumPy, PyYAML, Matplotlib, all three original checkpoints,
the original validation `predictions.npz` with its `effective_config.json`,
`summary.json` and `provenance.json`, and the published `data_identity.json`.
The config records concrete local input paths, expected checksum, counts, seed,
runtime backend and bootstrap budget. Missing or changed inputs fail explicitly.
Downloading inputs is a separate preparation step; the evaluator does not fetch,
train, overwrite checkpoints or publish. Use `WANDB_MODE=disabled` to keep all
experiment logging local. Results use the existing `ExperimentLogger` run ID and
artifact sidecars under `outputs/hybrid_temporal_estimation/<run_id>/`.

### Implementation and audited contracts

The repository structure is retained: parameter-free composition and streaming
state live in `graph/temporal/hybrid.py`, orchestration in
`evaluation/gnn/temporal/hybrid.py`, and offline diagnostics/plots in
`evaluation/gnn/temporal/hybrid_diagnostics.py`. Config remains in `graph/configs/`.
Existing model and simulator source files were not changed. The audit verifies
their SHA256 against the original Mamba evaluation source and stores the new
evaluation source hashes. No new collection/storage abstraction was introduced.

The audited original locations are:

| Item | Original path / identity |
| --- | --- |
| Frozen Gated MPNN | `/workspace/temporal_insertion_data/published_inputs/best.pt`; source run `4186cdeb0fdb449ea443f28e3bddc936` |
| Learned Gate | `/workspace/temporal_insertion_data/published_inputs/learned_gate.pt`; epoch 15, published run `11dda45dab61418bbe4e695f3e88c8c6` |
| Mamba | `/workspace/temporal_insertion_mamba/training/94c59d23eeba486a81adfa221fa829c3/best.pt`; epoch 19 |
| Validation snapshot | `/workspace/temporal_insertion_mamba/validation/d159dcc6d8464253a76ab9095be5a221/predictions.npz` |
| Split manifest | `/workspace/temporal_insertion_data/temporal_insertion_splits.json`; embedded verbatim in the reference effective config |
| Raw observations / graph cache | `/workspace/temporal_insertion_data/episodes` / `/workspace/temporal_insertion_data/gnn_cache`; not regenerated |

Frozen GNN SHA256:
`f527ab8f2b70129bfe845cf362a304bc2da195ac24e049d6de4cd4c28432100e`.
Gate SHA256:
`262b982a984302a46298bcdf7c0fb812c56689e6e0b780dec41306d57d189c9d`.
Mamba SHA256:
`ca6c6931e421d0bba776b8787038b0d16b87bc203bd82e1c757ac3053ed83069`.
Prediction snapshot SHA256:
`37a4a9ab4063db9542b75a3f0db5b898e76db596f7c3ee65e7c24b151b0c612a`.
Dataset identity:
`21a03a363ec5d6e1dd391fc014b8e3ad1455c5eaf0489153a9c39eed2d133292`.
The original GNN best-weight payload does not store its selection epoch; none is inferred.

The GNN configuration is hidden=64, two gated message-passing layers,
edge_embedding_dim=16, coord_scale_m=0.05 and Size4 scales
`[0.04,0.01,0.01,0.02]`; semantic-ID/confidence model inputs are disabled in
this checkpoint. Its graph builder and normalization remain those recorded in
the original config. The identified Frozen GNN predictions are reused; the GNN
forward is not rerun here.

Its existing `GraphBatch` carries world vertices, observed TCP XYZ, PEG/BOX/HOLE
types, validity/quality masks and typed edge geometry. The actual node encoder
uses 45D: object-centred TCP-relative XYZ / 0.05m, effective confidence,
observed flag, 32 semantic channels (zero in this checkpoint), and 8 object-type
embedding channels. Edges use 21D: relative XYZ / 0.05m, distance / 0.05m,
observed flag and 16 edge-type embedding channels. With confidence disabled,
present nodes have effective weight one. Per-frame `VertexEdgeGatedMPNN.forward`
has no recurrent cache; temporal states start in the two independent branches.

Mamba retains its original 64D feature builder, train-only mean/std buffers,
d_model=128, one layer, d_state=16, expand=2, d_conv=4, head_hidden=64,
0.01m residual position scale, and two pose heads (142,348 parameters).
The Gate retains its 17D BOX/HOLE observation-quality schema and original
30D MLP input: current/previous/innovation/TCP XYZ divided by saved 0.1m scale,
17 quality values and initialization flag. Its hidden width is 32; it does not
use Mamba's feature normalization. Full checkpoint configs and normalizers are
saved in the result audit, without modifying checkpoint bytes.

`StreamingHybridEstimator` sends the same current GNN prediction and TCP to
the original `StreamingInsertionEstimator.step` and `LearnedTemporalGate.step`.
The Gate receives its own quality17, and Mamba receives quality25 plus observed
robot state. The quality subset must match. Outputs are never fed across branches.
Reset clears Mamba caches/TCP/quaternion/age history and Gate world memory;
Gate cold start uses alpha=1. All frames, including frame zero, are scored, with
no warm-up exclusion or 64-frame resets. There is no optimizer or backpropagation;
all pretrained parameters are frozen and inference uses `torch.inference_mode()`.

Hybrid-H1 uses Mamba Peg pose, Gate Hole XYZ, Mamba Hole quaternion and GNN
Size4. H2 replaces only H1's Hole quaternion with GNN's. Both retain the existing
Object21 slices (`0:7`, `7:10`, `10:17`, `17:21`) and optional
`[Robot15,Object21,Freshness2]` State38. Caller freshness is preserved.

### Endpoint, frames and baseline reproduction

Method A uses `peg_position + R_peg * [half_size_x,0,0] - hole_position`.
This follows installed ManiSkill 3.0.1's evaluated `peg_head_pose`, not its
unused `peg_head_pos` convenience property (which omits actor rotation).
Two live CPU resets and subsequent steps with SAPIEN 3.0.3 verified the
reconstruction against `task.peg_head_pose.p`, maximum difference **1.49e-8m**.
See [live endpoint evidence](../outputs/hybrid_temporal_estimation/simulator_endpoint_verification.json)
and its [runnable check](../outputs/hybrid_temporal_estimation/verify_simulator_endpoint.py).

Method B uses `r_mamba + hole_mamba - hole_gate` in world axes. Translation of
both pose positions between TCP-relative/world coordinates leaves this vector
unchanged. The existing Mamba relative vector is pose-derived, not a separate
head; both methods therefore preserve the same endpoint. The measured maximum
A/B difference is **5.96e-8m**, with identical rounded metrics. Endpoint mean
error is **7.06mm** for Mamba and both hybrid methods: Hole replacement cannot
repair Mamba's endpoint itself.

The downloaded original snapshot was checksum-checked against the remote file.
All **126 validation episodes / 12,841 frames** match the split and published
frame inventory exactly. IDs must agree with episode/frame arrays; every episode
starts at zero and is contiguous. Each output is written to its identified row.
There are **zero excluded frames**, and `frame_manifest.json` records each raw
frame checksum and parent. Both branches use the same cached TCP, GNN output and
observations. Original raw-frame/proprioception checksum joins are reused from
the identified evaluation; raw RGB-D files are not reread in this replay.
No independent timestamp field exists in the snapshot; the clock is its original
contiguous saved-frame index. No frame interpolation or separate cache join occurs.

This evaluation ran on the Mac CPU using the explicit `torch_reference` backend
and unchanged saved official-Mamba weights. CUDA and `mamba_ssm` execution were
**not rerun on this host**. Against the original CUDA full-episode predictions,
maximum component differences were GNN **0**, Gate **1.79e-7**, Mamba **4.17e-7**.
Every reported baseline mean reproduced within **1.2e-6mm/degree**. The initial
two-episode smoke also matched independent original Gate and Mamba replay and
Mamba's full-sequence/cache-step outputs. The 64-frame selection protocol is
documented above and is not used for this comparison.

### Measured hybrid validation

Measured 2026-11-07 KST, run `904b37659de34a3f8a0154e77d004df4` from dirty
checkout `850c47be1bb530da4c23dbec8e9bf153b0c539d3`. These are reused development
validation results, not independent-test estimates or insertion success rates.
Frame-weighted means, mm/degrees unless marked percent:

| Metric | Frozen GNN | Learned Gate | Mamba | Hybrid-H1 | Hybrid-H2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Peg Position | 3.93 | 3.93 | 3.56 | 3.56 | 3.56 |
| Peg Rotation | 2.87 | 2.87 | 1.43 | 1.43 | 1.43 |
| Hole Position | 24.62 | 3.67 | 9.01 | 3.67 | 3.67 |
| Hole Rotation | 4.95 | 4.95 | 1.80 | 1.80 | 4.95 |
| Endpoint−Hole Vector | 23.24 | 11.05 | 8.65 | 7.67 | 7.67 |
| Lateral Error | 17.42 | 7.46 | 6.51 | 4.99 | 4.99 |
| Axial Error | 12.14 | 6.84 | 4.64 | 4.90 | 4.90 |
| Relative Orientation | 5.83 | 5.83 | 2.19 | 2.19 | 5.10 |
| Lateral <3mm (%) | 16.34 | 25.82 | 26.79 | 39.85 | 39.85 |
| Relative 3D <3mm (%) | 9.16 | 11.94 | 13.92 | 17.92 | 17.92 |

H1 improves relative 3D mean by **11.41%**, lateral mean by **23.40%** and
Hole position by **59.26%** over Mamba. Axial mean worsens **5.71%**;
axial P95 rises **15.44→16.91mm** and worst relative error **74.02→75.75mm**.
Strict H1 lateral <1/<2/<3mm rates are **5.94/21.08/39.85%**.

| Metric | Mamba mean / median / P90 / P95 | H1 mean / median / P90 / P95 |
| --- | --- | --- |
| Peg position mm | 3.56 / 2.62 / 7.25 / 9.53 | 3.56 / 2.62 / 7.25 / 9.53 |
| Peg rotation deg | 1.43 / 1.02 / 2.89 / 3.94 | 1.43 / 1.02 / 2.89 / 3.94 |
| Hole position mm | 9.01 / 6.76 / 17.60 / 22.35 | 3.67 / 3.24 / 6.50 / 7.65 |
| Hole rotation deg | 1.80 / 1.33 / 3.87 / 5.20 | 1.80 / 1.33 / 3.87 / 5.20 |
| Relative 3D mm | 8.65 / 6.63 / 17.45 / 22.14 | 7.67 / 5.70 / 15.17 / 21.85 |
| Lateral mm | 6.51 / 4.94 / 13.13 / 17.08 | 4.99 / 3.65 / 10.16 / 13.80 |

The main table preserves original GNN/Gate endpoint assemblies. Separate
common-geometric ablations give GNN relative/lateral **23.50/17.49mm** and
Gate **10.43/6.77mm**. They are labeled separately in JSON and are not mixed into
the reported baseline improvement percentages.

Lateral/axial estimation errors rotate vector **error** using GT Hole axes.
H1 and H2 have identical such errors by construction; this does not make Hole
rotation irrelevant. Rotating each model's vector by its own predicted Hole
orientation gives local-vector mean error Mamba **11.05mm**, H1 **10.73mm**,
H2 **22.16mm**. H1's predicted-frame error P95 slightly worsens
**28.12→28.52mm**. Relative orientation is **2.19deg** for H1 versus **5.10deg**
for H2. The inference result explicitly returns predicted-Hole-frame geometry.

The task's 0.003m clearance enlarges randomized aperture half-width relative to
Peg half-width. Actual installed `has_peg_inserted` checks Hole-frame endpoint
`x >= -0.015m`, `abs(y), abs(z) <= hole_radius`, without an orientation flag.
A 3mm estimation-error threshold is neither that physical predicate nor an
Insertion Success Rate; no controller/closed-loop insertion was executed.

### Cancellation, frame changes and paired episodes

Mamba Peg/Hole centre-error cosine mean is **0.184**, with positive dot product
on **63.08%** of frames. Endpoint/Hole errors are more aligned: cosine **0.339**,
positive on **72.92%**, signed XYZ correlations **0.456/0.608/0.078**. The
endpoint error is the relevant quantity for vector cancellation because it also
includes Peg rotation and predicted half-length uncertainty.

For preserved endpoint error `e_E`, old/new Hole errors `e_H/e_G`, the exact
squared-error change is `2*e_E·(e_H-e_G) - (||e_H||²-||e_G||²)`. The mean Hole
squared-error reduction is **129.82mm²**, while lost cancellation is
**108.84mm²**, leaving a net **−20.98mm²** relative squared-error change.
The identity residual is below **8.3e-13mm²**. Cancellation is real and removes
much of the benefit, but it does not erase the aggregate improvement.

Relative 3D improves on **57.48%** of frames (mean improvement within that group
**4.38mm**) and worsens on **42.52%** (mean degradation **3.61mm**). Gate Hole
position improves on **80.24%**, but on **29.45% of all frames** Hole position
improves while relative geometry worsens. In the 5,460 worsened frames,
lost cancellation averages **184.63mm²**, exceeding Hole improvement **95.54mm²**.
Lateral improves on **61.74%** of frames, with **4.27mm** conditional improvement
and **2.91mm** conditional degradation. Raw frame errors and signed comparisons
remain reconstructable from saved predictions/IDs.

H1 improves episode-mean relative error in **89/126** episodes and lateral in
**99/126**. Equal-episode mean paired changes (Hybrid minus Mamba) are
**−0.98mm**, parent-cluster bootstrap 95% CI **[−1.37,−0.60]mm**, and
**−1.50mm**, CI **[−1.86,−1.15]mm**, respectively. Resampling uses seed zero,
10,000 draws and 126 distinct parents; frames are never independent bootstrap
units. Frame-weighted intervals are also retained in diagnostics. These intervals
describe this previously used validation population and do not establish new-test
generalization or account for repeated model/analysis selection.

| Condition (offline definition) | Frames | Mamba relative / lateral mm | H1 relative / lateral mm |
| --- | ---: | ---: | ---: |
| Peg moving (GT change >1mm or >0.01rad) | 6,950 | 9.14 / 6.82 | 7.81 / 4.79 |
| Peg stationary (excludes initial frame) | 5,765 | 8.16 / 6.20 | 7.56 / 5.25 |
| No BOX-or-HOLE observed support | 1,849 | 16.75 / 11.56 | 15.89 / 9.18 |
| Insertion alignment geometry proxy | 782 | 6.01 / 4.34 | 5.65 / 4.11 |
| Fast direction-change proxy | 125 | 9.45 / 6.75 | 7.98 / 5.32 |

Both objects have observed vertex support on 10,992 frames (relative error
**7.29→6.28mm**). This is not a GT both-visible classification. PEG node count
never reaches zero; typed HOLE count is zero throughout, with target support
carried by BOX. Physical Peg occlusion, aperture visibility, grasp phases and
true insertion-phase labels are unavailable. The alignment proxy is the existing
GT endpoint-in-Hole region (`x` between −0.05 and 0.02m, lateral <0.01m).
All GT groups are used only after inference.

The most improved episode is `shard_000002/episode_000010`: relative mean
**18.13→6.55mm**, lateral **15.43→4.76mm**. The largest episode-mean degradation,
`shard_000022/episode_000016`, has relative **7.63→12.01mm** despite Hole
**10.90→7.52mm**; axial **5.49→11.06mm** drives the failure while lateral
improves **4.25→3.78mm**. Worst frame-level change is
`shard_000019/episode_000008/frame_000059.npz`, **13.07→47.82mm**, +34.75mm.
These cases support cancellation loss and axis-dependent tradeoffs rather than
an assumption that smaller absolute Hole error always improves relative geometry.

For a future joint estimator, this evidence supports evaluating endpoint/Hole
error covariance, axial tails and predicted-Hole-frame geometry alongside pose
means, and testing learned uncertainty-conditioned combinations on an independent
parent-disjoint set. Keeping static Hole memory alone cannot reduce the unchanged
7.06mm endpoint error. No such training or architecture decision is part of this run.

### Verification and artifacts

Six new contract tests cover reset/interleaving, frame identity, Object21/State38,
TCP/world transforms, endpoint/sign, quaternion normalization, pose consistency,
GT/future isolation, streaming parity and cluster statistics. The full model and
geometry selection ran **54 tests: 53 passed, one official-CUDA test skipped**.
Shared Logger/Artifact tests ran separately: **36 passed**. An initial combined
test process exposed an existing `imageio` lazy-module interaction with the logger
test's global import mock; isolated infrastructure/model processes pass without
editing those existing tests. Live endpoint checks used CPU simulation/rendering;
CUDA/MPS model inference were not executed on this host.

Final results:
[summary](../outputs/hybrid_temporal_estimation/904b37659de34a3f8a0154e77d004df4/summary.json),
[audit/configs](../outputs/hybrid_temporal_estimation/904b37659de34a3f8a0154e77d004df4/audit.json),
[diagnostics/bootstrap](../outputs/hybrid_temporal_estimation/904b37659de34a3f8a0154e77d004df4/diagnostics.json),
[frame manifest](../outputs/hybrid_temporal_estimation/904b37659de34a3f8a0154e77d004df4/frame_manifest.json),
[prediction arrays](../outputs/hybrid_temporal_estimation/904b37659de34a3f8a0154e77d004df4/predictions.npz),
[model tests](../outputs/hybrid_temporal_estimation/model_tests.log),
[infrastructure tests](../outputs/hybrid_temporal_estimation/infrastructure_tests.log).

Saved plots include
[relative distribution](../outputs/hybrid_temporal_estimation/904b37659de34a3f8a0154e77d004df4/relative_3d_distribution.png),
[lateral distribution](../outputs/hybrid_temporal_estimation/904b37659de34a3f8a0154e77d004df4/lateral_distribution.png),
[frame scatter](../outputs/hybrid_temporal_estimation/904b37659de34a3f8a0154e77d004df4/frame_scatter.png),
[episode comparison](../outputs/hybrid_temporal_estimation/904b37659de34a3f8a0154e77d004df4/episode_comparison.png),
[improved episode](../outputs/hybrid_temporal_estimation/904b37659de34a3f8a0154e77d004df4/largest_improvement_episode.png),
and [degraded episode](../outputs/hybrid_temporal_estimation/904b37659de34a3f8a0154e77d004df4/largest_degradation_episode.png).
Each representative trace includes all five systems and GT world position/wxyz,
Peg/Hole position/rotation errors, relative/lateral/axial errors, and predicted
Hole-frame lateral geometry. Prior evaluation outputs are preserved.

## H1 Peg-head relative-geometry fine-tuning

Run through the root entrypoint after setting the original checkpoint, graph-cache,
observed raw-frame and checksum-identified benchmark paths:

```bash
WANDB_MODE=disabled .venv/bin/python main.py --config graph/configs/temporal/hybrid/finetune_peg.yaml
.venv/bin/python -m unittest tests.test_temporal_hybrid_finetuning tests.test_temporal_hybrid -v
```

The YAML defaults to explicit CPU `torch_reference`. On the already prepared Vast
environment, `device: cuda:0` trains the Peg head on CUDA while
`inference_device: cpu`, `mamba_backend: torch_reference` replay frozen branches
on CPU. Official frozen streaming instead requires `inference_device: cuda:0`
with `mamba_backend: mamba_ssm`; no driver/package upgrade is required.
`action: audit` only verifies/replays H1. `action: run` with
missing training cache/raw roots records `missing_training_data`, without creating
an optimizer or falling back to benchmark training. Output is always a new
`outputs/hybrid_geometry_finetuning/<run_id>/`; pretrained files remain read-only.

### Boundary and freeze policy

The epoch-19 joint Mamba has independent `heads.peg` and `heads.hole` modules.
Implementation **A** fine-tunes only `heads.peg.0.weight` (64×128),
`heads.peg.0.bias` (64), `heads.peg.2.weight` (6×64) and
`heads.peg.2.bias` (6): **8,646 parameters**. No residual adapter is introduced.
The six outputs retain the pretrained residual position/rotation-vector decoder.
Input projection, norms, selective mixers, Hole head, normalization buffers,
Learned Gate and GNN remain frozen. Original GNN predictions come from the same
checksummed graph cache; no detector is retrained. Hole XYZ is always predicted
by the frozen Gate and Hole rotation by the frozen Mamba.

`graph/temporal/hybrid_finetuning.py` contains the model wrapper; orchestration/loss live in
`training/gnn/temporal/hybrid_finetuning*.py`, selection and diagnostics in
`evaluation/gnn/temporal/hybrid_finetuning.py`, parent selection in
`data_collection/temporal/finetuning_split.py`. The inference assembly now constructs
Object21 out of place, preserving its numerical computation while permitting
Method A to backpropagate into the Peg pose. Method A still uses predicted GNN
half-length along Peg local +X; insertion axis remains Hole local +X. Object21
and State38 use the existing inference contracts. GT enters only loss/scoring.

### Fine-tuning data and causal context

Seed 0 partitions the original **504 training parents** into **403 train / 41,270
frames** and **101 fine-tuning development episodes / 10,591 frames**. Sibling
episodes stay with their parent.
The development parents were seen during pretraining; they are not independent
held-out system validation. Original **126 episodes / 12,841 frames** remain a
reused development benchmark with no gradient, calibration or selection access.
The other 180 original reference episodes remain unread. No independently
collected test is inferred or substituted from those historical partitions.

Each branch runs its real per-frame streaming API with a cold reset at episode
frame 0, preserving observed TCP changes, quaternion conventions, support ages,
Gate memory and Mamba selective/convolution caches. A head-input hook caches only
the frozen causal representation. Training shuffles valid pointwise head inputs;
there is no padded supervision and no gradient through a cached backbone. This
deliberately uses **full-episode context**, whereas original Mamba pretraining reset
at 64-frame windows. Final selected heads are replayed through full streaming on
both development and benchmark and checked against cached-head results.
Normalization is restored exactly from pretraining, never refitted.

### Controlled objectives and selection

All three ablations independently start from the same epoch-19 weights, fresh
AdamW, identical shuffle seed, LR 3e-5, weight decay 1e-4, cosine schedule,
gradient clip 1, maximum 15 epochs and patience 5. FT-A uses Peg position and
rotation; FT-B adds relative vector; FT-C also adds lateral vector. Huber errors
are dimensionless with the original scales 0.01 m / 0.1 rad; quaternion rotation
uses sign-invariant geodesic Huber. Relative GT is the actual endpoint−Hole
arrangement, never a zero-distance target. Lateral supervision is error expressed
in GT Hole YZ, matching the existing loss/scorer; GT axis is absent at inference.

Before fitting, a fixed random sample of up to 2,048 **training** frames measures
each loss's Peg-head gradient. Auxiliary coefficients target relative norm 0.10
and lateral norm 0.05 of the combined pose gradient, clipped to [0.001, 0.2]. The
resulting coefficients are fixed, and FT-B/FT-C share the same relative weight.
`gradient_calibration.json` records losses, raw/weighted norms, sample indices and
relative/lateral gradient cosine. Each epoch also logs first-minibatch gradient
contributions. Ratios are an explicit initial budget, not universal coefficients.

Checkpoint selection uses only fine-tuning development. A candidate must stay
within H1 development means +0.2 mm Peg position, +0.1° Peg rotation, +0.2 mm axial
and +0.2 mm moving-Peg position. Its fixed score is relative mean + 0.5×lateral
mean + 0.1×relative P95 + 0.1×lateral P95 (all mm). Epoch 0 is eligible and remains
best if no admissible improvement appears. Benchmark numbers never alter weights,
budgets or this rule. These tolerances are study settings, not physical success
criteria; final motion/lag and tail diagnostics must still be inspected.

### Evidence and verification artifacts

The workflow prints parameter names/count, verifies exact optimizer membership,
stores per-tensor best/last parameter diffs (including frozen buffers), checks all
Gate tensors, and hashes the original three checkpoints before/after the study.
It checks initial H1 parity and bitwise Hole pose invariance, then uses the same
insertion scorer for mean/median/P90/P95, 1/2/3 mm precision, motion/occlusion and
available phase groups. GT-derived alignment/approach groups are offline proxies;
true Grasp/Lift/Transport labels require a separate supplied phase inventory.
Diagnostics preserve endpoint bias, endpoint/Hole error cross term, parent-cluster
paired bootstrap, frame improvement/worsening and the historical failure
`shard_000022/episode_000016`. Estimation precision is not insertion success.

Local CPU baseline audit `43b3ae6ef83b401396ce3fd7655a8e65` reproduced the original
126-episode benchmark without training. New model/geometry/split tests passed
12/12 on local CPU and on the prepared Vast Python environment before adding a
separate CPU-cache/CUDA-head transfer test. Shared logger and
artifact tests passed 36/36 in their own process. A combined broad process exposed
an existing `imageio.plugins` lazy-import interaction with the logger test's global
`importlib` mock; the required infrastructure suite passes independently. CUDA
head-training results and CPU streaming comparisons are recorded below, including
the absence of an independent test. No simulator change is part of fine-tuning.

Selected artifacts store the Peg head only (`kind: hybrid_peg_finetuning`), not
a replacement for the original Mamba checkpoint. Restore the original Mamba/Gate
with `evaluation.gnn.temporal.hybrid.restore_branches`, verify that its three
source checkpoint hashes match the selected artifact's
`config.source_checkpoint_hashes`, then load `payload['peg_head']` into
`mamba.heads['peg']` with `strict=True`. The unchanged
`StreamingHybridEstimator(mamba, gate)` then provides Object21/State38 inference.
`best.pt` and `last.pt` are distinct artifacts, and epoch 0 means the selected
head is the original unadapted head.

After a completed study is copied locally, set `source_run` and the same reference
snapshot in `graph/configs/temporal/hybrid/analyze_finetuning.yaml`, then run:

```bash
WANDB_MODE=disabled .venv/bin/python main.py --config graph/configs/temporal/hybrid/analyze_finetuning.yaml
```

This is a separate tracked analysis, with source input checksums and no model
updates. It adds FT-B versus FT-A / FT-C versus FT-B parent-paired comparisons,
matched episode lag summaries, precision CDFs and development curves. Its GT-only
approach proxy requires the tip to move by >0.5 mm/frame along Hole +X within the
existing alignment window; it is explicitly not a true contact/phase annotation.

### Measured H1 fine-tuning results

On 2026-11-07, Vast.ai **59441593 / RTX 3090** completed study
`cd65c40b572b476aad496c356ffd495a`. The Peg head trained on CUDA; frozen branch extraction and final
full-episode inference used the verified CPU `torch_reference` backend. A two-episode
host timing measured 4.11 ms/frame with maximum Mamba/CUDA-snapshot difference
1.19e-7. The initial all-CUDA attempt was stopped before any optimizer update
because per-frame frozen replay was slower; its files were preserved.
Official `mamba_ssm` CUDA streaming of the updated heads and MPS execution were
not run; CUDA was exercised for Peg-head optimization and the device-transfer test.

**No fine-tuned head was adopted.** All three conditions stopped after 5 epochs;
epoch 0 retained the lowest admissible development score **13.004831**.
Consequently each `best.pt` contains the original H1 Peg head, and the selected
benchmark outputs are unchanged. The following table deliberately reports the
**actual trained, unselected `last.pt` heads at epoch 5**, freshly replayed on the
same 126 reused benchmark episodes / 12,841 frames. These diagnostic numbers were
not used to change the selection rule, loss weights or models.

| Metric | Original H1 | FT-A last | FT-B last | FT-C last |
| --- | ---: | ---: | ---: | ---: |
| Peg position mm | 3.558 | 3.218 | 3.229 | 3.232 |
| Peg rotation deg | 1.431 | 1.307 | 1.307 | 1.307 |
| Hole position mm | 3.669 | 3.669 | 3.669 | 3.669 |
| Relative 3D mm | 7.667 | 7.835 | 7.752 | 7.735 |
| Lateral mm | 4.986 | 5.013 | 4.958 | 4.937 |
| Axial mm | 4.901 | 5.073 | 5.021 | 5.020 |
| Lateral <3mm | 39.85% | 38.93% | 39.99% | 40.51% |
| Relative 3D <3mm | 17.92% | 17.59% | 18.10% | 18.22% |

Hole rotation is unchanged at 1.801538° for every head. Hole XYZ/quaternion
outputs were bitwise identical within each paired replay. The four Peg tensors
changed in every last checkpoint; every other Mamba state tensor and all 2,081
Gate parameters were unchanged. The three original checkpoint SHA256 values
matched before/after. Selected-head cached/full-stream differences were at most
1.19e-7 on both development and the benchmark. GNN inference also asserted exact
parameter/buffer preservation; the detector was not loaded for training.

| Condition | Epochs run | Selected epoch | Last development score | Last guard failures |
| --- | ---: | ---: | ---: | --- |
| FT-A | 5 | 0 | 13.483831 | axial |
| FT-B | 5 | 0 | 13.341005 | axial |
| FT-C | 5 | 0 | 13.315793 | none; score still worse than H1 |

All updated epochs scored worse than epoch 0; even first-epoch scores were
13.105882 / 13.060523 / 13.047452 for A/B/C (exact values are in histories).
The final development Relative 3D means were 7.619 / 7.537 / 7.521 mm, versus
7.340 mm for H1, despite better absolute Peg poses. No benchmark-based override
of this negative development result was made.

#### Loss and gradient evidence

Position/rotation units remained 0.01 m / 0.1 rad. The 2,048 train-frame
calibration fixed the following C weights; A disables both auxiliaries and B
disables lateral. Combined pose gradient norm was 0.453008.

| Term | Initial normalized loss | Raw gradient norm | Weight in C | Weighted norm |
| --- | ---: | ---: | ---: | ---: |
| peg_position | 0.033036 | 0.055132 | 1.00000000 | 0.055132 |
| peg_rotation | 0.039180 | 0.449544 | 1.00000000 | 0.449544 |
| relative_position | 0.131282 | 0.556036 | 0.08147092 | 0.045301 |
| lateral_position | 0.085617 | 0.798353 | 0.02837141 | 0.022650 |

Relative/lateral gradient cosine was **0.974172**, so the two objectives
largely overlap at initialization. Rotation supervision had about 8.15× the raw
position gradient norm. The auxiliary gradient budgets were 10% and 5% of the
combined pose gradient, not universal coefficients of 0.1. Per-epoch first-batch
gradient contributions and normalized term losses are preserved in each history.

#### Tails, precision and cancellation

| Metric mm | H1 | A last | B last | C last |
| --- | ---: | ---: | ---: | ---: |
| Relative P90 | 15.175 | 15.463 | 15.399 | 15.415 |
| Relative P95 | 21.850 | 22.861 | 22.451 | 22.467 |
| Lateral P95 | 13.803 | 13.711 | 13.610 | 13.622 |
| Axial P95 | 16.909 | 17.878 | 17.619 | 17.605 |
| Endpoint mean | 7.062 | 7.188 | 7.152 | 7.152 |
| Endpoint P95 | 22.246 | 23.368 | 23.182 | 23.179 |

C improves lateral <3mm by **0.662 percentage points** and relative <3mm by
**0.296 points**, but its mean relative error and relative/axial tails worsen.
Its mean lateral improvement alone is insufficient evidence of overall success.

The cancellation cross term is `-2 E[e_endpoint · e_hole]` in mm²:
**-14.913 / -15.130 / -15.635 / -15.837** for H1/A/B/C. Cancellation becomes
more favorable, while endpoint mean and tail errors worsen. Thus these results
do not support a simple loss-of-cancellation explanation for the fine-tuning
degradation. C reduces Peg centre Y bias from -0.966 to -0.397 mm, while endpoint
Y bias grows from +0.185 to +0.729 mm. Better centre/angular scalar errors need
not yield a better tip with fixed predicted size; direction and error coupling
matter. A causal attribution specifically to size was not isolated in this study.

| Last head vs H1 | Relative improved episodes /126 | Relative mean delta mm [95% CI] | Lateral improved episodes /126 |
| --- | ---: | --- | ---: |
| FT-A | 46 | +0.1680 [+0.0824, +0.2601] | 58 |
| FT-B | 53 | +0.0857 [+0.0036, +0.1738] | 67 |
| FT-C | 54 | +0.0687 [-0.0134, +0.1567] | 70 |

C improves/worsens relative error on 49.70% / 50.30% of frames. Its axial
mean delta is +0.1194 mm, parent-bootstrap 95% CI [+0.0238, +0.2200]. These CIs
are conditional comparisons on a reused benchmark, not independent-test evidence.

| Added loss, last heads | Relative delta mm [95% CI] | Lateral delta mm [95% CI] |
| --- | --- | --- |
| B minus A | -0.0823 [-0.0940, -0.0706] | -0.0553 [-0.0657, -0.0448] |
| C minus B | -0.0170 [-0.0204, -0.0135] | -0.0213 [-0.0257, -0.0167] |

The added geometry objectives mitigate pose-only degradation and have measurable
effects against their matched controls. They do **not** recover H1 geometry or
satisfy the fixed development selection criterion in this setting.

#### Motion, alignment and failure cases

| Group | Frames | Relative H1→C mm | Lateral H1→C mm | Axial H1→C mm | Lateral <3mm H1→C |
| --- | ---: | --- | --- | --- | --- |
| peg_stationary | 5765 | 7.561→7.629 | 5.254→5.244 | 4.535→4.624 | 39.01%→38.65% |
| peg_moving | 6950 | 7.814→7.884 | 4.788→4.706 | 5.256→5.403 | 40.40%→41.87% |
| insertion_alignment_geometry_proxy | 782 | 5.654→5.727 | 4.111→4.089 | 3.286→3.396 | 48.59%→50.00% |
| target_support_missing | 1849 | 15.894→16.281 | 9.182→9.065 | 11.517→12.071 | 15.90%→19.58% |

Alignment-proxy relative P95 worsens from 14.493 to 15.896 mm despite improved
3mm fractions. Moving-Peg position improves 3.705→3.295 mm. Mean absolute
position lag across the same 126 episodes is 0.03175→0.01587 frames; rotation
absolute lag is 0.88095→0.83333 frames, with six episodes nevertheless worsening
in absolute rotation lag. This is not uniform temporal improvement.

There are no Peg-support-absent frames, while direct Hole-node support is absent
on all benchmark frames; the useful missing-support slice is BOX-or-HOLE absence.
These are observed-support proxies, not physical occlusion labels. True
Grasp/Lift/Transport/contact-phase annotations were unavailable. A separately
defined GT approach proxy is reported for the **selected** heads in comparison.json;
all selected heads are H1 and therefore unchanged on that slice.

Historical failure `shard_000022/episode_000016`:

| Metric mm | H1 | A last | B last | C last |
| --- | ---: | ---: | ---: | ---: |
| Relative 3D | 12.008 | 12.453 | 12.239 | 12.220 |
| Lateral | 3.779 | 3.488 | 3.391 | 3.355 |
| Axial | 11.057 | 11.585 | 11.398 | 11.395 |

C reduces lateral error here, but relative/axial errors still worsen. The
episode was not used as an optimization target or model-selection override.

#### Reproduction and limits

Original checkpoints/results remain preserved; no commit, push, W&B delivery or
HF publication occurred. Training used `ExperimentLogger`; post-selection
analysis has its own linked input hashes and run identity. No simulator/controller
rollout or CUDA physics/render validation is claimed. The installed CPU endpoint
check passed four reset/step cases with maximum 1.49e-8 m difference.

One LR, one calibrated auxiliary-gradient budget and this Peg-only boundary were
evaluated. This negative result does not establish that every possible head-only
weight/schedule fails. There is **no independent test**: fine-tuning development
was previously seen in pretraining, the 126-episode benchmark is reused, and
the reserved 180 reference episodes were not read for training or final scoring.

Local final hybrid/fine-tuning tests: 14 total, 13 passed and one CUDA-host skip.
Remote prepared-environment tests: 13/13 including actual CPU-cache/CUDA-head
transfer. Existing Mamba/precision regressions: 27 total, 26 passed and one CUDA
skip. Shared logger/artifact tests: 36/36 in a separate process.

Reproduce the prepared-host study with its preserved config:

```bash
cd /workspace/hybrid_geometry_finetuning/launch_f3ab893b87514deb84764ec35394f44d/source_cpu_streaming
WANDB_MODE=disabled /workspace/temporal_insertion_validation/8e066415885e4395ae3a824472bc94b1/venv/bin/python main.py \
  --config ../finetune_hybrid_geometry_cpu_streaming.yaml
```

The source base was `850c47be1bb530da4c23dbec8e9bf153b0c539d3` with uncommitted
work. Exact source snapshots and checksums are retained; Git commit alone does
not reconstruct this run. Local analysis uses the documented root entrypoint
with `graph/configs/temporal/hybrid/analyze_finetuning.yaml`.

- [Training status/best epochs](../outputs/hybrid_geometry_finetuning/cd65c40b572b476aad496c356ffd495a/execution_status.json)
- [Freeze/hash audit](../outputs/hybrid_geometry_finetuning/cd65c40b572b476aad496c356ffd495a/audit.json)
- [Train/development/benchmark parent manifest](../outputs/hybrid_geometry_finetuning/cd65c40b572b476aad496c356ffd495a/finetuning_split.json)
- [Gradient calibration](../outputs/hybrid_geometry_finetuning/cd65c40b572b476aad496c356ffd495a/gradient_calibration.json)
- [Selected-head benchmark summary](../outputs/hybrid_geometry_finetuning/cd65c40b572b476aad496c356ffd495a/summary.json)
- [Last-head development and fresh benchmark diagnostics](../outputs/hybrid_geometry_analysis/b6de749c250a43bb9e4ad1d641dd8536/last_head_diagnostics.json)
- [Selected-head tracking/approach comparison](../outputs/hybrid_geometry_analysis/b6de749c250a43bb9e4ad1d641dd8536/comparison.json)
- [Development curves](../outputs/hybrid_geometry_analysis/b6de749c250a43bb9e4ad1d641dd8536/development_curves.png)
- [Source archive and launch provenance](../outputs/hybrid_geometry_finetuning/launch_f3ab893b87514deb84764ec35394f44d/launch.json)


## Hybrid-H1 relative geometry error attribution

2026-11-07 KST, run `807c6a8f83464862be67b85fdc7fce47`. 기존 Hybrid run
`904b37659de34a3f8a0154e77d004df4`의 **126 episodes / 12,841 frames /
126 parents** 전체를 분석했다. 원본 commit은
`850c47be1bb530da4c23dbec8e9bf153b0c539d3`이며 작업 트리는 dirty 상태다.
이는 개발·선택에 반복 사용된 validation이며 독립 Test가 아니다.

**Frozen GNN의 Peg half-length 오차가 Endpoint의 가장 큰 Oracle 개선 대상이다.
삽입 접근 구간의 Lateral 정밀도에는 Peg 회전도 중요하다.** Hole 위치 개선으로
사라지는 error cancellation이 Axial 악화를 설명한다. 전체 mean의 개선을 삽입
접근 구간의 정밀도 개선으로 일반화할 수 없다.

### 실행·구조·입력 검증

Torch, NumPy, PyYAML, Matplotlib, 기존 Hybrid 결과 디렉터리와 원본 snapshot/
metadata, published frame inventory, Frozen GNN / epoch-19 Mamba / epoch-15 Gate
checkpoint를 준비한 뒤 저장소 루트에서 실행한다.

```bash
WANDB_MODE=disabled .venv/bin/python main.py --config graph/configs/temporal/hybrid/analyze_errors.yaml
.venv/bin/python -m unittest tests.test_temporal_hybrid_error_analysis tests.test_temporal_hybrid -v
```

팀 설정은 `graph/configs/temporal/hybrid/analyze_errors.yaml`, orchestration은
`evaluation/gnn/temporal/hybrid_error_analysis.py`, 수학적 분해는
`evaluation/gnn/temporal/hybrid_error_geometry.py`, 그림은
`evaluation/gnn/temporal/hybrid_error_plots.py`, 신규 테스트는
`tests/test_temporal_hybrid_error_analysis.py`다. 기존 `main.py`, `ExperimentLogger`,
Hybrid audit와 geometry scorer를 재사용했다. 이 작업에서 공통 simulator, 모델,
collection, training 구현을 변경하지 않았다. 기존 결과와 checkpoint는 읽기 전용이고
새 결과는 `outputs/hybrid_error_diagnostics/<new_run_id>/`에 저장한다.
실제 checkpoint 기반 forward, optimizer, 학습, checkpoint 저장, 업로드,
commit/push는 이 진단 workflow에 없다. Checkpoint를 읽어 provenance를 검증하고
모델은 폐기한다.

Hybrid snapshot SHA256은
`462fd8328505d0cf2e92503c89ed3d1b182ae83f463d5584926a8cebc45024cd`다.
기존 `load_reference` / `restore_branches`로 원본 checksum, published inventory,
parent split, 세 checkpoint hash와 epoch, train-only normalization, feature contract,
모델 source hash를 검증했다. Observation/GT 배열은 같은 sample ID와 같은 row에서
완전히 일치해야 한다. Manifest의 raw SHA256와 parent도 순서대로 비교한다.
Frame identity checksum은
`7315f6ac706b938bdd6ac9b6c1b8e20ea20ab00a1704e7ff76d591912d93d3dc`다.
제외 프레임은 없고 다른 run 배열을 frame 검증 없이 결합하지 않았다.

Object21은 `[Peg pose7, endpoint-minus-Hole3, Hole pose7, Size4]`다.
Pose 위치는 observed TCP 기준 translation, 축은 world XYZ, 단위는 m,
quaternion은 world-frame wxyz다. World 위치 그림에는 같은 row의 observed TCP를
더했다. 기존 평가에서 두 frozen branch는 독립적으로 frame 0에 reset한 뒤 전체
episode를 causal streaming했다. 이번에는 그 예측을 재사용했다. 별도 timestamp가
없어 시간축은 saved frame index이며 보간하지 않았다.

Method A는 `endpoint = peg_position + R_peg*[half_size_x,0,0]`다.
설치된 ManiSkill 3.0.1 / SAPIEN 3.0.3의 `peg_head_pose`, `has_peg_inserted` 소스를
확인하고 CPU reset 2회 및 후속 step으로 다시 검증했다. Endpoint 차이는 최대
**1.49e-8m**, Hole local vector 차이는 **9.69e-8m**였다. 삽입축은 Hole local
**+X**, lateral은 **Y/Z**다.
[실행 코드](../outputs/hybrid_error_diagnostics/verify_simulator_geometry.py)와
[실측 결과](../outputs/hybrid_error_diagnostics/simulator_geometry_verification.json)를
보존했다. CPU PhysX / `sapien_cpu`를 실행했으며 renderer는 macOS MoltenVK를
사용했다. CUDA/MPS 모델 추론은 이번 호스트에서 실행하지 않았다.

Float64 Method A로 분해했으며 저장 float32 vector와의 최대 차이는 GT
**6.39e-8m**, Mamba **5.24e-8m**, H1 **4.65e-8m**다. 기존 모든 평균 지표를
**0.001mm/degree 이내**로 재현했다. H1 quaternion 재정규화의 최대 **5.96e-8**
component 차이는 공통 Mamba quaternion으로 통일해 Endpoint를 정확히 같게 두었다.
변경한 것은 진단용 배열뿐이며 저장된 예측은 보존했다.

### 정확한 제곱 오차 분해와 cancellation

`E = endpoint_pred-endpoint_GT`, `H = hole_pred-hole_GT`이면
`e_rel = E-H`, `||e_rel||² = ||E||² + ||H||² - 2 E·H`다.
아래는 모든 frame의 항을 계산한 뒤 평균한 **mm²** 값이다. Norm 평균을 더한
기여율로 해석하지 않는다.

| 방법 | Endpoint² | Hole² | E·H | −2 E·H | Relative² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mamba | 97.459 | 147.419 | 61.875 | −123.750 | 121.128 |
| H1 | 97.459 | 17.598 | 7.457 | −14.913 | 100.144 |

모든 프레임에서 식을 검증했으며 최대 residual은 **1.82e-12mm²**다. H1의
Endpoint self term은 Hole self term보다 크지만 cross term 때문에 독립적인 원인
비율은 아니다. Endpoint mean은 양쪽 모두 **7.062mm**다. Cosine mean은
**0.339→0.137**, positive-dot cancellation 프레임은 **72.92→59.86%**,
negative-dot amplification은 **27.08→40.14%**다. GT Hole X/Y/Z signed Pearson
correlation은 **0.622/0.458/0.078→0.259/0.171/0.093**다. 이는 인과계수가 아니다.

Hole squared gain **129.820mm²**에서 lost cancellation **108.837mm²**를 제외한
net relative squared change는 **−20.983mm²**다. Episode별 같은 분해는
`summary.json → episodes_detail → episode → decomposition`에 있다.

| Hole / Relative 변화 | frames | 전체 비율 |
| --- | ---: | ---: |
| 개선 / 개선 | 6,521 | 50.78% |
| 개선 / 악화 | 3,782 | **29.45%** |
| 악화 / 개선 | 860 | 6.70% |
| 악화 / 악화 | 1,678 | 13.07% |

Norm delta tie tolerance는 **1e-6mm**, 실제 tie는 0이다. Relative 개선은
**7,381 / 57.48%**, 악화는 **5,460 / 42.52%**다. Hole 개선/Relative 악화 그룹은
Hole gain **145.39mm²**보다 lost cancellation **261.25mm²**가 커서 Relative²가
**115.86mm²** 증가한다. 이 identity가 기존 29.45% 현상을 정량적으로 설명한다.

### Peg 위치·회전·크기 Oracle

GT 교체는 **diagnostic-only Oracle Intervention**이다. Relative 지표는 Endpoint만
교체하고 동일한 Gate Hole 예측을 고정한 결과다.

| GT 교체 조건 | Endpoint mean mm | Endpoint MSE mm² | H1 Relative mm | Lateral mm | Axial mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| 없음: 예측 Pose + 예측 Size (A) | 7.062 | 97.459 | 7.667 | 4.986 | 4.901 |
| Peg 위치만 GT (B) | 6.072 | 83.337 | 6.886 | 4.316 | 4.553 |
| Peg 회전만 GT (C) | 6.197 | 82.147 | 7.140 | 4.345 | 4.802 |
| Peg Size만 GT (D) | **4.658** | **36.539** | **5.897** | 4.261 | **3.343** |
| Peg Pose GT + 예측 Size | 4.964 | 70.008 | 6.324 | 3.651 | 4.405 |
| Peg 위치 + Size GT | 2.476 | 12.988 | 4.483 | 3.604 | 2.077 |
| Peg 회전 + Size GT | 3.558 | 21.745 | 5.242 | 3.524 | 3.199 |
| Peg Pose + Size 전부 GT (E) | 0 | 0 | 3.669 | 2.845 | 1.817 |

Size Y/Z와 Hole radius는 Endpoint 식에 들어가지 않는다. Peg **half-size X**의
absolute error mean은 **4.964mm**다. Pose를 완벽하게 해도 현재 Size로는 Endpoint
4.964mm가 남는다. 이것이 모든 편향된 pose 예측의 수학적 하한이라는 뜻은 아니다.
Pose를 왜곡해 길이 오차를 상쇄할 수는 있다.

`u`를 Peg +X 단위축, `L`을 half-size X로 두면 정확한 벡터식은
`E = delta_p + L_GT*delta_u + u_GT*delta_L + delta_u*delta_L`다.
Self squared mean은 translation **21.745**, rotation **12.988**, size **70.008**,
rotation-size interaction **0.294mm²**다. 여섯 pairwise cross term의 합은
**−7.127mm²**여서 총 97.459mm²가 된다. 작은 interaction self term만 보고 다른
상호작용도 작다고 판단하면 안 된다. 모든 항과 signed vector를 보존했다.

GT 교체 6개 순서를 모두 평가한 MSE reduction 평균 / 범위는 translation
**17.90 [12.14,23.55]**, rotation **14.12 [12.99,15.31]**, size
**65.43 [60.40,70.35]mm²**다. 이는 순서 평균 Oracle 배분이며 독립적인 인과 기여율이
아니다. Translation은 전체 norm 감소가 약간 크고 rotation은 MSE 감소가 약간 크다.

H1 Relative 상위 5%인 643프레임의 half-length absolute error mean은 **25.00mm**다.
Relative 평균 **28.44mm**는 GT Size만이면 **12.30mm**, GT 위치만이면 **26.09mm**,
GT 회전만이면 **26.01mm**다. Peg 회전 ≥5° 그룹 355프레임에서는 Relative
**20.66→15.62mm**(GT 회전), **15.36mm**(GT Size)로 두 요인이 모두 중요하다.
회전 오차 또는 Endpoint 수식 구현 오류만으로 tail을 설명할 수 없다.

### GT Hole frame의 축별 오차

`a_GT=R_hole_GT*[1,0,0]`, signed axial은 `a_GT·e_rel`, lateral vector는
`e_rel-(a_GT·e_rel)*a_GT`다. 아래 squared 변화는 H1−Mamba다.

| GT Hole 축 | Hole squared gain mm² | Lost cancellation mm² | Relative squared 변화 mm² | Absolute mean Mamba→H1 mm |
| --- | ---: | ---: | ---: | ---: |
| X / Axial | 66.190 | 70.771 | **+4.580** | 4.636→4.901 |
| Y / Lateral | 50.250 | 38.190 | −12.061 | 5.080→4.368 |
| Z / Lateral | 13.380 | −0.123 | −13.503 | 3.078→1.584 |

H1 Relative²의 X/Y/Z 항은 **54.245/41.534/4.365mm²**다. Axial cancellation 손실이
Hole 개선을 초과한다. Lateral 개선은 특히 Z에서 크고 남는 Lateral은 Y에 집중된다.
Predicted Hole frame vector error mean은 **10.728mm**로 GT-frame Relative
7.667mm보다 크므로 Hole 회전 불확실성도 inference geometry에 남아 있다.
3mm는 estimation precision 기준이며 실제 0.003m clearance 및
`x>=-0.015m`, `abs(y),abs(z)<=hole_radius` head-position predicate와 구분한다.
이 simulator predicate에는 별도 orientation/contact flag가 없다.

### 삽입 접근 구간과 통계적 한계

원본 `phase_labels`는 null이다. 식별된 snapshot에는 실제 Before Grasp, Grasp/Lift,
Transport, Contact 라벨이 없고 원본 `/workspace` raw episode 경로도 이 호스트에
없다. 따라서 다음 **순간 GT geometry proxy**를 사용했다. 시간순 state machine이나
실제 phase/contact GT가 아니다. `x,y,z`는 GT Hole 좌표, `d_lat=sqrt(y²+z²)`,
`theta`는 GT Peg/Hole +X 사이 directed angle이다. Fine 조건은
`-50<=x<=20mm`, `d_lat<=10mm`, `theta<=10°`다.

| Proxy / 배타적 우선순위 | 정의 | frames / episodes |
| --- | --- | ---: |
| Head in region | Fine 및 x≥−15mm | 126 / 126 |
| Insertion approach | Fine 및 x<−15mm | 656 / 126 |
| Alignment | −150≤x≤20mm, d_lat≤30mm, theta≤20°, Fine 제외 | 2,913 / 126 |
| Pre-alignment | 상대 거리≤200mm, 위 그룹 제외 | 1,651 / 126 |
| Distant | 나머지 | 7,495 / 117 |

기존 historical alignment proxy **782프레임**도 재현했다. 그룹은 GT로만 나누며
model 입력이나 Gate 갱신에는 사용하지 않는다. Approach cutoff lateral
7.5/10/15mm 및 angle 5/10/15°의 9조합은 모두 동일한 656프레임을 선택했다.
이는 해당 범위의 결과 안정성이지 phase 실재성의 증명은 아니다. Head region은
각 episode에 한 프레임뿐이다.

| 구간 | Endpoint mm | Hole mm Mamba→H1 | Relative mm | Lateral mm | Axial mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| 전체 | 7.062 | 9.008→3.669 | 8.654→7.667 | 6.509→4.986 | 4.636→4.901 |
| Alignment | 5.073 | 8.237→3.357 | 7.079→5.893 | 5.734→3.524 | **3.355→3.959** |
| Approach | 5.206 | 8.020→3.101 | 6.142→5.704 | 4.410→4.153 | 3.591→3.323 |
| Head region | 4.913 | 6.202→3.173 | **5.350→5.390** | 3.965→3.891 | **2.958→3.098** |

H1의 mean / median / P90 / P95는 다음과 같다. 모든 그룹과 두 방법의 전체 통계,
episode별 worst 값 및 해당 sample ID는 JSON에 있다.

| H1 지표 | Alignment | Approach |
| --- | --- | --- |
| Peg position mm | 2.606 / 1.989 / 5.427 / 7.043 | 2.496 / 1.823 / 5.105 / 6.203 |
| Peg rotation deg | 1.050 / 0.820 / 2.036 / 2.565 | 1.327 / 0.791 / 2.440 / 3.977 |
| Hole position mm | 3.357 / 2.962 / 5.308 / 6.962 | 3.101 / 2.818 / 4.992 / 5.630 |
| Hole rotation deg | 1.346 / 1.006 / 2.779 / 3.507 | 1.715 / 1.360 / 3.813 / 4.976 |
| Endpoint mm | 5.073 / 3.450 / 9.903 / 16.489 | 5.206 / 3.316 / 8.158 / 17.652 |
| Relative mm | 5.893 / 4.503 / 10.654 / 15.693 | 5.704 / 4.120 / 8.733 / 17.473 |
| Lateral mm | 3.524 / 2.958 / 6.350 / 7.762 | 4.153 / 3.080 / 7.092 / 9.276 |
| Axial mm | 3.959 / 2.290 / 8.806 / 14.549 | 3.323 / 2.027 / 6.176 / 13.611 |
| Relative orientation deg | 1.380 / 1.047 / 2.751 / 3.731 | 1.511 / 1.334 / 2.909 / 3.662 |

| H1 구간 | Lateral <1 / <2 / <3mm % | Relative <3mm % Mamba→H1 |
| --- | ---: | ---: |
| 전체 | 5.94 / 21.08 / 39.85 | 13.92→17.92 |
| Alignment | 8.24 / 28.08 / 51.05 | 20.67→25.75 |
| Approach | 8.99 / 30.79 / 48.63 | **35.37→30.03** |
| Head region | 6.35 / 28.57 / 48.41 | **37.30→30.16** |

Approach에서는 Lateral mean이 Axial mean보다 크지만 Axial P95가 더 크다.
Lateral 일반 정밀도와 Axial tail을 함께 관리해야 한다. GT Peg 회전 교체 시
Approach Lateral **4.153→2.943mm**, GT 위치만 **3.609mm**, GT Size만 **3.938mm**다.
반면 GT Size는 Approach Axial **3.323→2.331mm**로 줄인다. Alignment에서는
GT Size로 Axial **3.959→2.693mm**, GT 회전으로 Lateral **3.524→2.876mm**다.

Paired delta는 H1−Mamba, 다음은 **episode-equal mean / parent bootstrap 95%
percentile CI**, 단위 mm다. Seed 0, 10,000 draws로 연속 frame을 독립 추출하지 않는다.
Frame-weighted CI도 JSON에 보존했다.

| 구간 | Relative delta [CI] | Lateral delta [CI] | Axial delta [CI] |
| --- | --- | --- | --- |
| 전체 | −0.977 [−1.366,−0.596] | −1.501 [−1.858,−1.147] | +0.263 [−0.024,+0.554] |
| Alignment | −1.178 [−1.828,−0.584] | −2.067 [−2.759,−1.424] | **+0.553 [+0.114,+1.010]** |
| Approach | −0.119 [−0.649,+0.366] | −0.316 [−0.754,+0.102] | +0.213 [−0.315,+0.654] |
| Head region | +0.040 [−0.623,+0.613] | −0.074 [−0.673,+0.579] | +0.140 [−0.707,+0.855] |

Approach의 frame-weighted Axial mean은 개선돼도 episode-equal delta는 악화 방향이다.
긴 episode의 가중치와 outlier 영향이 다르기 때문이다. Approach의 세 CI와 전체
Axial CI는 0을 포함하므로 확실한 차이를 주장하지 않는다. 다중 slice 비교와 반복
model selection을 보정한 CI가 아니므로 독립 Test 일반화나 확증적 유의성은 아니다.

### 실패 episode와 개선·악화 상위 사례

Episode-mean Relative delta로 정렬한 상위 5개씩이며 모든 episode의 시간 그래프를
저장했다. Endpoint/Hole GT·prediction world XYZ, Relative/Lateral/Axial,
Endpoint error, cosine/dot, signed local XYZ, Oracle 및 cancellation을 포함한다.

| episode | Relative Mamba→H1 mm | Hole mm | Lateral mm | Axial mm |
| --- | ---: | ---: | ---: | ---: |
| shard_000002/episode_000010 | 18.126→6.551 | 15.140→3.487 | 15.429→4.758 | 8.769→4.029 |
| shard_000002/episode_000022 | 14.487→8.082 | 15.384→3.813 | 12.622→6.675 | 4.861→3.586 |
| shard_000017/episode_000019 | 23.287→17.970 | 36.682→1.651 | 15.812→8.621 | 15.799→14.436 |
| shard_000018/episode_000029 | 10.311→5.246 | 9.051→2.887 | 8.616→3.652 | 4.898→3.177 |
| shard_000014/episode_000029 | 11.641→6.755 | 12.166→1.779 | 8.129→4.103 | 7.466→4.605 |
| **shard_000022/episode_000016** | **7.630→12.008** | **10.895→7.520** | **4.251→3.779** | **5.487→11.057** |
| shard_000005/episode_000007 | 10.733→14.560 | 11.558→9.347 | 9.569→11.977 | 3.717→7.040 |
| shard_000029/episode_000003 | 4.620→8.361 | 7.267→2.946 | 3.341→7.162 | 2.808→3.581 |
| shard_000011/episode_000019 | 8.102→11.782 | 9.738→7.141 | 5.992→7.924 | 4.320→8.001 |
| shard_000021/episode_000010 | 8.006→10.797 | 11.991→6.234 | 6.695→7.760 | 3.473→6.330 |

필수 실패 episode는 114프레임, Endpoint mean **13.797mm**다. Hole squared gain
**112.787mm²**보다 lost cancellation **212.957mm²**가 커 Relative²가 **100.170mm²**
증가했다. Cosine mean은 **0.664→0.452**다. GT Size만 교체하면 Endpoint
**13.797→6.345mm**, Relative **12.008→6.639mm**, Axial **11.057→5.511mm**다.
GT 회전만으로 Relative는 **11.699mm**여서 rotation만의 문제는 아니다.

Frame 0–19 / 20–59 / 60–113의 Relative mean은 Mamba **5.40/11.54/5.56mm**,
H1 **8.97/17.31/9.20mm**다. 20–59에서 Endpoint **24.02mm**, Axial
**9.21→16.89mm**로 가장 크다. 60 이후 Endpoint는 **5.84mm**로 줄지만 Gate Hole
**7.42mm**와 잔여 편향 때문에 Relative가 악화한다. 이 구간은 phase 라벨이 아니다.

측정된 실패 양상은 (1) 필수 episode의 **axial cancellation 손실**, (2)
shard_000005/episode_000007와 shard_000029/episode_000003의 **lateral 악화와 dot의
음수 전환**, (3) shard_000017/episode_000019의 **큰 Endpoint 잔류**다. 마지막은
Hole이 1.651mm까지 좋아져도 Endpoint 18.300mm로 Relative가 17.970mm 남는다.
개선 최대 episode는 dot이 **−38.66→+9.41mm²**로 변해 amplification도 해소된다.
이는 배타적 분류기나 물리적 접촉 원인 판정이 아닌 결과의 설명이다.

### Fine-tuning 우선 대상과 loss 후보

1. **잔여 Relative는 Endpoint와 frozen half-length를 우선 진단한다.** Endpoint
   term이 Hole term보다 크다. Mamba→H1 악화 프레임의 변화는 cancellation 식으로
   설명된다. Hole을 일부러 부정확하게 만들어 cancellation을 복원하는 것이 목표는 아니다.
2. **전체 translation/rotation의 우열은 단일 비율로 확정하지 않는다.** 위치의
   norm 개선이 약간 크고 회전의 MSE 개선이 약간 크다. Approach에서는 회전 교체의
   Lateral 개선 **1.211mm**가 위치의 **0.544mm**보다 크므로 orientation과
   transverse endpoint precision을 우선 비교할 근거가 있다.
3. **Approach Lateral mean 4.153mm와 Axial P95 13.611mm를 함께 관리한다.**
   Alignment Axial 악화와 Approach/terminal Relative <3mm 하락을 전체 mean으로 숨기지 않는다.
4. **기존 Mamba는 이미 Relative/Lateral/Relative-orientation loss를 학습했다.**
   실제 저장 config는 `pose_relative_full`, geodesic Huber, position unit 0.01m,
   rotation unit 0.1rad, 7개 weight 모두 1이다. Loss를 처음 추가한다는 제안은 맞지 않는다.
   후속 실험에서 supervision geometry를 실제 H1의 **frozen Gate Hole + Mamba Peg +
   GNN Size** 조립에 맞추는 ablation을 우선 비교할 수 있다. GT는 supervision과
   offline grouping에만 사용한다. 이번 작업은 새 모델/데이터 인터페이스를 정하지 않았다.
5. **Mamba Peg head만으로 frozen Size/Gate/Hole-rotation 오차를 직접 고칠 수 없다.**
   Pose 잔여 성분은 줄일 수 있으나 centre translation으로 길이 오차를 흡수하면
   절대 pose가 왜곡될 수 있다. GT Peg Pose에서도 Endpoint 4.964mm가 남는 점을
   고려해 별도 size calibration/refinement 가능성을 먼저 검증할 근거가 있다.
6. **절대 pose·relative·tail·episode 지표를 보존한다.** Peg position **3.558mm**,
   rotation **1.431°**, Gate Hole **3.669mm**, Hole rotation **1.802°**, Relative
   orientation **2.185°**, 전체 H1 Lateral <3mm **39.85%**를 함께 추적한다.
   Approach <3mm, signed X/Y/Z bias, Axial/Relative P95, episode worst,
   predicted-Hole-frame geometry 및 reset/causality도 보존 대상으로 둔다.

후보 objective는 절대 pose anchor를 유지한
`L_pose + lambda_rel*Huber(e_rel/s) + lambda_lat*Huber(e_local_yz/s)
+ lambda_ax*Huber(e_local_x/s) + lambda_endpoint*Huber(E/s)`다.
기존 `s=0.01m` 후보와 실제 정밀도 scale의 ablation은 구분한다. Relative는 GT의
실제 상대 배열을 목표로 하며 Peg를 무조건 Hole 중심으로 옮기는 loss가 아니다.
Size를 조정하는 후속 실험에서는 별도 half-length supervision을 비교할 수 있다.
현재 코드에 axial/endpoint 전용 weight를 구현한 것은 아니다.

초기 비교 후보로 `lambda_rel∈{0.25,0.5,1}`, `lambda_lat∈{0.5,1,2}`,
추가 axial/endpoint term은 각각 `{0,0.25,0.5}`를 제안한다. 기존 rel/lateral이
이미 1인 baseline도 포함한다. 이는 소규모 ablation 후보이며 최적값이 아니다.
Relative loss가 이미 axial/lateral을 포함해 중복 가중될 수 있고 frozen size 편향을
pose에 전가하지 않는지 확인해야 한다. 우선순위는 전체 Cartesian sweep보다
현재 objective 재현 → H1 조립 일치 → 구간별 lateral/axial 가중치 → size 개선의
분리 비교다. 최종 판단에는 별도 parent-disjoint test와 실제 insertion controller가 필요하다.

### 산출물·검증·확정할 수 없는 사항

[Summary](../outputs/hybrid_error_diagnostics/807c6a8f83464862be67b85fdc7fce47/summary.json),
[audit](../outputs/hybrid_error_diagnostics/807c6a8f83464862be67b85fdc7fce47/audit.json),
[effective config](../outputs/hybrid_error_diagnostics/807c6a8f83464862be67b85fdc7fce47/effective_config.json),
[frame manifest](../outputs/hybrid_error_diagnostics/807c6a8f83464862be67b85fdc7fce47/frame_manifest.json),
[frame diagnostics](../outputs/hybrid_error_diagnostics/807c6a8f83464862be67b85fdc7fce47/frame_diagnostics.npz)에
전 프레임 errors/dot/cosine, Oracle vectors, squared terms, proxy masks,
episode 분해·worst frames·parent bootstrap를 보존했다. `endpoint_position`은 실제
Endpoint 위치 오차다. 기존 scorer의 `endpoint`는 역사적으로 relative-vector norm을
뜻하여 `relative_3d`와 중복이므로 둘을 혼동하지 않는다.

[개요](../outputs/hybrid_error_diagnostics/807c6a8f83464862be67b85fdc7fce47/error_attribution_overview.png),
[episode·구간 비교](../outputs/hybrid_error_diagnostics/807c6a8f83464862be67b85fdc7fce47/episode_and_geometry_comparison.png),
[필수 실패 episode](../outputs/hybrid_error_diagnostics/807c6a8f83464862be67b85fdc7fce47/episode_shard_000022_episode_000016.png)를
비롯해 개선/악화 10개 episode 시간 그래프를 저장했다. 개요와 필수 사례는 시각 점검했다.

최종 최적화 없는 테스트 **26개 통과**: 신규 analytic 9개, 기존 Hybrid 6개,
기존 insertion geometry 11개다.
[테스트 로그](../outputs/hybrid_error_diagnostics/model_tests.log)를 보존했다.
최초 기존 suite 전체 실행은 27개가 통과했으나 두 테스트
`test_independent_cache_cli_path_uses_unseen_parents`와
`test_tiny_all_ablation_cli_training_evaluation_and_checkpoint_preservation`에 임시
합성 fixture 학습이 포함돼 있었다. 학습 금지 요청에 맞지 않는 테스트 선택 실수였고
실제 데이터나 기존 checkpoint를 사용·변경하지 않았다. 이후 이 둘을 제외하고
quaternion-roundoff regression을 추가했다. 실제 checkpoint 3개와 기존 평가 입력은
실행 전후 SHA256이 동일하다. Commit/push는 하지 않았다.

GT는 scoring, 기준축, offline geometry grouping, Oracle에만 사용했다. Oracle
감소량은 실현 가능한 학습 개선량이나 삽입 성공률이 아니다. 실제 phase/contact GT,
closed-loop insertion 결과, 독립 Test 일반화, 최적 weight, Size 오류의 센서/가림/
학습 원인, 특정 fine-tuning 효과는 확정할 수 없다. CUDA/MPS 추론 및 Frozen GNN
재실행도 하지 않았다. 이 작업은 원본 snapshot을 검증해 재사용한 진단이며 실제
데이터로 새 모델을 학습하지 않았다.
