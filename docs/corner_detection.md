# Corner detection

`visual.corner_detection` extracts current-frame, image-supported peg and box
corners from native 512×512 RGB-D. It is independent of the existing tracking,
pose, graph and controller workflows. No simulator, object GT, previous frame,
other camera or hidden per-episode object dimensions enter inference.

## Initial Peg corners under robot occlusion

An exploratory 2026-11-14 CPU audit rendered the initial states of simulator
seeds 200–1399 from the current factory base camera at `[0.65, -0.30, 0.65]` m.
The existing detector ran without GT masks; GT, actor segmentation and depth
were used only for evaluation. Peg occlusion is the fraction of the predicted
GT Peg silhouette whose valid depth is at least 1.5 mm in front of the Peg
surface and whose segmentation belongs to a robot link. This is an image-space
approximation, not the fraction of the full 3-D surface.

Of 1,200 frames, 13 had at least 1% robot occlusion and four had at least 5%;
the maximum was 18.8%. The detector returned at least two distinct physical Peg
corners within 3 mm in all 13 frames, matching 41 of 62 independently supported
visible reference corners. The support criterion is an approximation: an actor
pixel within a 7×7 image neighborhood must be within 6 mm of the 3-D corner.
The corresponding aggregate match rate was 5,447/7,330 in the 1,183 frames
with no measured robot occlusion, without adjustment for initial-state or other
differences. This is not a causal comparison.

Among the 13 occluded frames, matched corners occurred at both Peg ends in
five; eight had matches only at one end. Twelve of the 13 had reference support
at both ends. In the strongest observed case (18.8% occlusion), both matches
were at the unoccluded end. These are initial observations only: the audit does
not establish behavior under heavier occlusion or during grasping and motion.
The sparse 13-frame group is not enough to claim a robust success rate.

[Example overlays](../outputs/peg_robot_occlusion/occluded_examples.png),
[aggregated results](../outputs/peg_robot_occlusion/summary.json),
[per-frame results](../outputs/peg_robot_occlusion/results.json),
[additional frames](../outputs/peg_robot_occlusion_expanded/results.json), and
[end identity audit](../outputs/peg_robot_occlusion_ends/results.json) are local
ignored artifacts. The detector ran on CPU, with CPU physics and SAPIEN/MoltenVK
rendering on the Mac host; CUDA was not evaluated. This diagnostic reused
`ExperimentLogger` and did not publish artifacts or change detector code.

## Initial-state evaluation on the main camera

On 2026-11-15, the CPU detector evaluated all 1,200 reset seeds (200–1399)
from the restored Peg and Hole audits. The saved JSONs supplied seeds and
occlusion labels. Raw RGB-D was re-rendered on the Mac with CPU physics and
SAPIEN/MoltenVK. Every reset was checked against the stored Peg pose and
dimensions and the four stored Hole front-rim projections and depths. The
live camera was checked at 512×512, with a 90° vertical FOV, eye
`[0.65, -0.30, 0.65]` and target `[0.065, 0.297, 0.11]` m.

The detector takes RGB-D and calibration. GT and actor segmentation determine
reference eligibility: a same-actor depth pixel within a 7×7 image window
must be within 6 mm of the vertex. Scores use one-to-one 3-D assignment at
3 and 5 mm, with both Hole rims and the Box exterior sharing the Box assignment.
The front rim is the four negative-local-X opening corners used in the
[Hole occlusion audit](data_collection.md). Evaluation covers reset frames.

| Supported reference | Matched within 3 mm | Recall at 3 mm | Recall at 5 mm | Frames with at least two matches at 3 mm |
| --- | ---: | ---: | ---: | ---: |
| Peg | 5,502 / 7,412 | 74.2% | 83.4% | 1,142 / 1,200 |
| Box exterior | 4,333 / 5,756 | 75.3% | 93.3% | 1,185 / 1,200 |
| Hole front rim | 1,395 / 4,657 | 30.0% | 47.0% | 448 / 1,200 |

Across the 1,193 frames with supported Hole front corners, 462 had zero
front-rim matches within 3 mm. In the 34 frames with at least two Hole
corner pixels blocked by the robot, only one frame had a front-rim match
within 3 mm and none had two matches even at 5 mm. The other 1,166 frames
still had 436 with zero front-rim matches at 3 mm. Both Peg and Hole front
rim had zero matches in 17 of 1,200 frames at 3 mm. The rear rim had zero
supported references in this cohort.

The [per-seed results](../outputs/initial_corner_detection/7cdbe76f362e4f97baa50646ae3b001d/results.json)
record source checksums and the effective configuration. The
[Graph evaluation command](../graph/README.md#initial-state-corner-detector-evaluation)
runs through the root launcher with local experiment tracking.

## Main-camera Hole corner calibration

The 2026-11-15 CPU sweep used 180 randomly selected seeds from the 1,200
reset audit. Selection maximized 3 mm Hole matches with at most 0.15
additional unmatched Box predictions per frame and at least 99% of
baseline Box exterior matches. The depth support, quadrant, two-edge
and subpixel-search variants used the existing 512×512 main camera.

Relaxing the depth, quadrant or two-edge checks yielded no additional
3 mm Hole matches with the original merge. A 0.65 px offset search added
four matches but also 557 unmatched Box predictions on the 180 seeds.
A 5 mm duplicate radius for accepted concave RGB candidates was selected
from 3, 5 and 7 mm trials. It added 13 Hole matches and 22 unmatched Box
predictions on the selection set.

| Metric on 400 separate reset seeds | Baseline | 5 mm Hole merge |
| --- | ---: | ---: |
| Supported Hole recall at 3 mm | 442 / 1,550 (28.5%) | 475 / 1,550 (30.6%) |
| Supported Hole recall at 5 mm | 705 / 1,550 (45.5%) | 742 / 1,550 (47.9%) |
| Frames with zero Hole matches at 3 mm | 171 | 160 |
| Unmatched Box predictions at 3 mm | 1,464 | 1,493 |

Opt in on the CPU detector with
`TorchCornerDetector(device="cpu", hole_rgb_merge_radius_m=0.005)`.
The opt-in acts at the duplicate check for accepted concave RGB
candidates. The detector default uses the original merge. The calibration
covers CPU reset frames. Motion, MPS and CUDA were outside the run.

[Candidate and gate sweep](../outputs/main_camera_corner_calibration/9983f96b071e473d92b88002739230cc/results.json)
and [merge sweep and separate-seed scores](../outputs/main_camera_corner_calibration/6cd0f50d820b4ed7b9bb86627772a950/results.json)
record the effective variants and per-frame metrics.

## Install and run

The existing Visual Linux/Python 3.12/CUDA environment remains defined by
`visual/pyproject.toml` and its lockfile. For a standalone detector environment,
install only the detector dependencies from the repository root:

```bash
python -m pip install -r visual/corner_detection/requirements.txt
python examples/detect_corners.py --input /path/to/frame.npz \
  --output outputs/corners.npz --device cpu
```

The input NPZ contains `rgb`, `depth`, `intrinsic_cv`, `extrinsic_cv` and optional
`masks`. The example explicitly exports result tensors to CPU for NPZ storage
and refuses to overwrite the output file. Load the result with
`np.load(path, allow_pickle=False)`; candidate metadata and parameters are JSON
strings stored in scalar string arrays.

## Python API

```python
from visual.corner_detection import detect_tensors, detect, detect_cpu, Parameters

# NumPy arrays and Torch tensors are accepted; result tensors stay on the device.
result = detect_tensors(rgb, depth_raw_mm, intrinsic_cv, extrinsic_cv,
                        masks=masks, device="auto")
world_xyz = result.xyz

# Explicit CPU export of the same Torch calculation, including 2-D masks.
exported = result.as_dict()
exported = detect(rgb, depth_raw_mm, intrinsic_cv, extrinsic_cv, device="cpu")

# Explicit independent NumPy reference, never an automatic fallback.
reference = detect_cpu(rgb, depth_raw_mm, intrinsic_cv, extrinsic_cv)
```

For repeated calls, instantiate `TorchCornerDetector(device, parameters)` and
call it with the same input arguments. A detector instance holds mutable
per-call workspace and should not be used concurrently. `detect_tensors` caches
one instance per device when parameters are omitted. Construct separate instances
for concurrent streams. Parameter values and candidate merge order are retained
from the extracted source; this migration does not retune the detector.

| Input | Meaning |
| --- | --- |
| `rgb` | `[512,512,3]` RGB uint8, not BGR or normalized RGB |
| `depth_raw_mm` | Raw millimetre depth, normally signed int16; `[512,512]` or singleton channel |
| `intrinsic_cv` | `[3,3]` CV camera intrinsics |
| `extrinsic_cv` | `[3,4]` CV world-to-camera transform |
| `masks` | Optional bool `[2,512,512]`, slot 0 peg and slot 1 box |

Inputs are unbatched; remove the simulator's leading environment dimension.
Nonpositive depth and depth at or above 4 m are excluded. Integer pixel centers
are used for unprojection; adding a half-pixel offset changes this detector's
geometry and must not be borrowed from another projector.

Without masks, fixed task-material HSV rules supply an RGB-only baseline.
This is not general object segmentation or a claim about SAM2 performance.
Explicit empty masks remain empty. Plane fitting, contour geometry, current
depth support, contact/occlusion checks and physical-edge evidence suppress
unsupported or clipped corners. The box recovery stage includes an exterior
cube prior whose scale is estimated from observed faces; it does not accept
instance dimensions as GT input. This is a task-specific geometric detector.

| Native result | Meaning |
| --- | --- |
| `xyz` | `[N,3]` float32 world coordinates in metres |
| `camera_xyz` | `[N,3]` float32 CV camera coordinates in metres |
| `uv` | `[N,2]` float32 subpixel image coordinates |
| `object_id` | `[N]` int64; 0 peg, 1 box |
| `masks` | `[2,512,512]` bool material/input masks |
| `metadata`, `diagnostics`, `parameters` | Candidate evidence and current execution settings |

Tensor fields remain on the requested device. `detect`/`as_dict` explicitly
export them to CPU. Historical candidate method labels are retained in metadata
for exact result comparisons; they are not module names or new experiment IDs.

## Graph consumer

The Graph team's [graph input](graph_input.md) guide describes conversion of
`TensorResult` to `graph.build_graph.GraphBatch`, including unknown semantic IDs,
quality scoring, explicit PEG/BOX/HOLE labels, edges and padding. These transforms
live in `graph/build_graph/`; the detector does not construct graphs. Build outside
`torch.inference_mode()` before GNN training so inference tensors are cloned into
ordinary training inputs. Detector IDs remain `0=peg`, `1=box`.

## Backends

`device="auto"` selects CUDA, then MPS, then Torch CPU. An explicitly requested
unavailable GPU raises an error. The Torch path never silently invokes the NumPy
reference detector. A runtime backend error is propagated.

- CPU/CUDA use portable Torch geometry.
- MPS uses fused Metal kernels and requires a PyTorch build exposing
  `torch.mps.compile_shader`. This capability is checked when compiling the
  Metal library. A CPU-only or CUDA call does not compile Metal shaders.
- OpenCV HSV, connected components, contour topology and RGB subpixel features
  remain on CPU. Small descriptors and merge decisions also cross the host
  boundary. Dense XYZ is not downloaded to run a NumPy detector. This is a
  hybrid implementation, not a completely GPU-resident pipeline.

The original NumPy float64 and Torch float32 implementations are not generally
bit-identical across backends. The extraction verification below compares old
and new code **on the same backend**.

## Module layout

```text
visual/corner_detection/
├── __init__.py             public API
├── detector.py             Torch execution and candidate merge
├── candidates.py           material masks, subpixel features, contour proposals
├── geometry.py             NumPy geometry, visibility and edge checks
├── numpy_reference.py      explicit NumPy reference assembly
├── planar_reference.py     reference planar core
├── box_recovery.py         reference box recovery
├── evaluation.py           optional GT references and matching, never inference
├── backends/
│   ├── torch_ops.py        portable Torch geometry and device selection
│   └── metal_ops.py        fused MPS kernels
└── requirements.txt        standalone dependencies
```

The extraction source was the preserved ignored directory
`local_corner_probe_10i_20261104/` in the original checkout. Its default Torch
detector and kernels became `detector.py` and `backends/`. Common numerical
helpers shared by its CPU recovery revisions were verified identical before
being extracted to `geometry.py`. Its NumPy core and recovery became
`planar_reference.py`, `numpy_reference.py` and `box_recovery.py`. RGB/contour
helpers moved to `candidates.py`. No imports refer to that original directory;
it is a provenance record, not a runtime dependency. Original sources and raw
artifacts remain preserved.

## Checks and recorded verification

```bash
python -m unittest tests.test_corner_detection tests.test_corner_backends -v
```

On 2026-11-05 KST, Apple M4 Pro/macOS, Python 3.11.13, Torch 2.14.1,
NumPy 2.4.6 and OpenCV 5.0.0:

- 45 tests ran: **38 passed**, **7 CUDA tests skipped**. Tests include crop,
  invalid depth, explicit masks, hidden corners, world transforms, device output,
  CPU execution and fused Metal versus portable CPU geometry.
- On **56 saved frames** (28 states, Main/Wrist separately), extracted and frozen
  source Torch outputs matched **exactly on CPU**, and **exactly on MPS**:
  XYZ, UV, camera XYZ, object IDs, masks, candidate metadata and diagnostics.
- The extracted NumPy reference also matched original corners, masks and
  diagnostics on all 56 frames. RANSAC parameters, merge order and Metal shader
  source were preserved. No inference input GT was added.
- These are same-input extraction checks, not a new held-out accuracy estimate,
  multi-view fusion, continuous-live test or fresh latency benchmark.

CUDA/NVIDIA hardware and the existing locked Python 3.12/Torch 2.8 CUDA
environment were **not executed on this Mac**. CPU and MPS were executed in the
existing Python 3.11 environment. Shared simulator code was not changed or run
for this extraction.

The preserved source-package hash (SHA256 of the sorted filename/hash JSON) is
`8cb25e3413d5e52c919a8d2b6601cde494626811e0b5a7af310fb5af48a7cd5f`.
The saved capture manifest SHA256 is
`aa01ff0c784040f2f8b76e6b9df75fb3566264404f164f2d094c5c7deb213b17`.

The optional regression script requires a trusted external frozen Python source
package and saved capture manifest; ordinary inference and unit tests do not:

```bash
WANDB_MODE=disabled python scripts/verify_corner_detection.py \
  --reference-source /path/to/frozen/source_package \
  --capture /path/to/capture --output outputs/corner_verification --device mps
```

Add `--include-numpy-reference` to compare the NumPy implementation too. This
driver verifies frame/source hashes, loads no GT and records results through the
existing `utils.experiment_logger.ExperimentLogger`; it creates no custom logger
or publishing workflow.

Local ignored verification artifacts:
[unit test log](../outputs/corner_detection_verification/tests.log),
[CPU summary](../outputs/corner_detection_verification/cpu/summary.json),
[MPS summary](../outputs/corner_detection_verification/mps/summary.json), and
[source snapshot hashes](../outputs/corner_detection_verification/source_hashes.json).
These raw links require the local preserved artifacts; source hashes are also
recorded in each summary's experiment configuration.

## Pilot trajectory corner detection evaluation

The current RGB-D detector was evaluated on the unfiltered mixed-trajectory
pilot with the current base camera on 2026-11-15. The source contains 100 reset
seeds, three modes per seed and 62,395 observations. The fixed sample takes
four equally spaced trajectory frames including the last observation per
trajectory, plus the reset frame once per seed: 1,300 frames in total. Samples
from the same initial state and shared expert prefixes are correlated. The pilot
is a development dataset that has already been inspected, not an independent
test or a uniformly random sample of all frames.

Run on CPU after collecting the pilot, with the shared simulator packages and
saved raw RGB-D and segmentation:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/pilot_corner_detection.yaml
```

The detector takes only RGB-D and calibration; simulator GT and segmentation are
used for scoring, not detector inference. Before evaluation the workflow resets
all 100 states and verifies saved segmentation, robot joints and Peg, Box and
Hole poses against the live simulator; the observed Peg/Box actor segmentation
IDs are 18/19. The saved reset RGB, depth and segmentation also matched between
the three modes for all 100 seeds. Every evaluated frame is hash checked.

Reference corners are produced by `visual.corner_detection.evaluation.corners`.
A vertex is eligible when its projection is inside the image and a same-actor
segmentation pixel in a 7×7 neighbourhood has valid depth within 6 mm of the
3-D vertex. This is an approximate RGB-D support rule, not a pixel-perfect or
independent manual visibility certificate. Predictions are matched one-to-one
within each object class, maximizing the number within the stated 3-D threshold.
Box exterior and Hole rim matches compete in one Box assignment. A prediction
near an unsupported vertex can be counted as unmatched, so precision is relative
to this supported reference set, not necessarily the true physical false-positive
rate. Hole rim here means the eight physical mouth-corner references, not Hole
center estimation.

| Supported reference family | Eligible corners | Matched within 3 mm | Recall at 3 mm | Recall at 5 mm |
| --- | ---: | ---: | ---: | ---: |
| Peg | 7,208 | 4,968 | 68.9% | 78.6% |
| Box exterior | 5,871 | 4,519 | 77.0% | 92.9% |
| Hole rim | 4,618 | 1,101 | 23.8% | 41.3% |

At 3 mm, supported-reference precision was 82.2% (4,968/6,047) for Peg
predictions and 54.9% (5,620/10,231) for the combined Box predictions. Mean
nearest same-object GT vertex distance over every prediction was 2.96 mm for
Peg and 5.34 mm for Box; this separate nearest-reference statistic permits
multiple predictions near one vertex and includes unsupported vertices.

Of 1,229 sampled frames with at least one supported Hole rim reference, 613 had
no rim match within 3 mm; across the 100 initial seeds, 43 had no such match
in any sampled frame. These samples repeat related states, so neither figure
implies an independent Bernoulli trial per frame. The low rim support may
contribute to Hole estimation difficulties but does not by itself establish the
cause of downstream GNN errors. At 5 mm, 435 of those 1,229 frames still lacked
a rim match. During recorded grasp frames, Peg recall was 63.7% (1,457/2,288),
compared with 71.4% (3,511/4,920) for non-grasp frames; those groups differ in
other respects and are not a causal estimate of occlusion.

[Per-frame scores, effective config and grouped results](../outputs/pilot_corner_detection/f5546623ea134ee4adfa13c8818ab24c/results.json)
are local ignored artifacts. The evaluation reused `ExperimentLogger`, ran CPU
detector inference and CPU physics, and used the host SAPIEN/MoltenVK renderer
for reset checks. CUDA was not tested. No model or detector implementation was
changed.

## Hole opening recovery

Use `TorchCornerDetector(device="cpu", recover_hole_rim=True)` to enable
depth-based opening proposals. The detector finds enclosed gaps on observed
Box planes, projects the boundary between pixel cells onto the plane, and
fits four sides in metric coordinates. A perpendicular observed face supplies
the fitting axes. The width and location come from the current frame. The
validation checks side support, approximate squareness, depth behind the
plane, and near foreground occlusion. The detector appends verified corners
after the existing candidate stages and deduplicates them at 3 mm. The
default detector remains available for existing Graph evaluations.

The [paired evaluation commands](../graph/README.md#hole-opening-recovery-evaluation)
use one observation for both detector settings. The reset cohort re-renders
the original 1,200 seeds and verifies the saved geometry and camera. The pilot
replays the 1,300 checksummed sample frames from the previous evaluation. The
paired evaluator checks that baseline match counts reproduce per frame.
Pilot frames include 100 resets and related observations from 300
trajectories, so the evaluation measures performance on an inspected
development cohort.

Candidate, verification, and post-merge coverage count supported GT rim
vertices with at least one Box point within 3 mm at each stage. All three
use the same GT denominator. Final recall uses one-to-one assignment with
Box exterior references in the same assignment. Unmatched Box predictions
use this approximate supported reference set.

Paired CPU results on 2026-11-15

| Cohort | Rim proposal coverage | Rim verified coverage | Rim after merge | Rim 3 mm recall | Box exterior 3 mm recall | Unmatched Box predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial states (4,657 rims) | 52.7% → 68.3% | 36.5% → 52.3% | 30.0% → 46.0% | 30.0% → 46.0% | 75.3% → 75.3% | 4,465 → 5,821 |
| Pilot sample (4,618 rims) | 49.2% → 66.4% | 30.5% → 48.9% | 23.8% → 43.2% | 23.8% → 43.2% | 77.0% → 77.0% | 4,611 → 5,691 |

- Initial states. Opening-only coverage passed through 1,963 → 1,621 → 1,621 reference vertices at proposal, shape verification and depth verification. [Per-frame results](../outputs/hole_rim_recovery/bb7e7f1227f64272a5ce00cec3305f9c/results.json).
- Pilot sample. Opening-only coverage passed through 2,000 → 1,592 → 1,592 reference vertices at proposal, shape verification and depth verification. [Per-frame results](../outputs/hole_rim_recovery/8e2f5a69f4ff41219e7e80029ef80572/results.json).

Box exterior match counts agreed frame by frame. Appending more points
improved rim recall and reduced combined Box precision. The added recovery
requires an enclosed opening on the detected face, which limits recovery
when occlusion breaks the boundary. The paired runs used the Mac CPU and
MoltenVK for reset rendering. MPS and CUDA inference were outside these runs.

The refined recovery applies a 0.25 px right/down correction to the opening
predictions at their fitted depth. This empirical adjustment compensates a
systematic localization bias on the current renderer and camera. It applies
only to the opt-in opening path. A verified opening point also takes precedence
over legacy Box candidates marked as concave within 8 mm. This lets a
jointly fitted rim replace older points around the same corner while keeping
non-concave candidates in their original merge priority.
