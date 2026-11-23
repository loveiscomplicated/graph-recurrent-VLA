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
from visual.corner_detection import TorchCornerDetector, detect_tensors, detect, detect_cpu

# NumPy arrays and Torch tensors are accepted; result tensors stay on the device.
result = detect_tensors(rgb, depth_raw_mm, intrinsic_cv, extrinsic_cv,
                        masks=masks)
world_xyz = result.xyz

# Explicit CPU export of the same Torch calculation, including 2-D masks.
exported = result.as_dict()
exported = detect(rgb, depth_raw_mm, intrinsic_cv, extrinsic_cv, device="cpu")

# Explicit independent NumPy reference, never an automatic fallback.
reference = detect_cpu(rgb, depth_raw_mm, intrinsic_cv, extrinsic_cv)
```

The public helpers and `TorchCornerDetector.main()` default to CUDA, then
MPS when its shader compiler is available, then Torch CPU. Each uses the main
recovery settings for Peg, Box exterior and single-frame Hole rims. For a model
rollout, construct one detector per stream and call it with the current RGB-D
frame at every step.

The main preset checks Box occlusion at the apparent Box-facing Peg prism end
using the current depth frame. To replay the previous main preset, use
`TorchCornerDetector.main(device="cpu", reject_peg_box_cut=False)`.
The [Expert comparison video](../outputs/corner_detector_video/corner_detector_comparison.mp4)
uses this same preset with MPS.
When MPS is selected, the main preset uses the optimized Metal RANSAC,
frame-level candidate validation, Box tail quantiles, and bounded Hole
component processing. `detect()` and `detect_tensors()` select the same MPS
preset. Use `TorchCornerDetector.main(device="mps", optimized=False)` to
compare the individual validation path with the same recovery settings.
The default CUDA preset uses the 2-pixel grid plane fit and TorchInductor
validation. It compiles the tensor validation for contour boundaries, three-plane
junctions, observed face envelopes, RGB Box candidates, and fitted Box vertices.
It includes physical edge validation inside those graphs. OpenCV proposal
extraction and Python candidate assembly use the usual paths. Representative
frames are needed for warmup because tensor shapes and layouts can specialize
the graphs. See [CUDA validation scope comparison](#cuda-검증-컴파일-범위-비교).

Use `TorchCornerDetector.main(device="cuda", compile_cuda_validation=False)`
for the grid fit with eager validation, or add `plane_fit_method="ransac"` to
replay the previous CUDA default. Passing `compile_cuda_edges=True` selects
the edge-only compilation preset and defaults to the RANSAC plane fit.
CPU and MPS default to RANSAC. CUDA with `optimized=False` defaults to
RANSAC and eager validation.

The main CUDA grid preset enables the grazing-view Hole RANSAC fallback. The
detector first runs the normal grid path. Only when that path has no
depth-verified Hole opening does it refit the Box planes with deterministic
RANSAC and queue verified opening corners. CPU and MPS already use RANSAC for
the complete plane fit by default, so they do not run this secondary fit. Pass
`hole_rim_ransac_fallback=False` to the main CUDA detector when reproducing the
historical grid-only output. The fallback requires the Hole recovery path
enabled by `TorchCornerDetector.main()`.

Use `TorchCornerDetector.main(device="cuda", sample_cuda_box_planes=True)` to
compare the experimental Box edge kernel. It tests plane membership at six
pixels on each side of each candidate leg. One Triton launch handles the
legs of all eight Box vertices across the three candidate plane orientations.
Box plane selection and vertex coordinate refinement use the existing path.

Pass `device="cpu"` to pin a CPU evaluation or use `device="cuda"` or
`device="mps"` to require a specific accelerator. An unavailable explicit device
raises an error. Runtime detector errors propagate. The low-level
`TorchCornerDetector(device, parameters)` constructor retains its explicit
recovery options for historical replays. Episode-local Hole rim memory and
longer-span replacement are applied by the trajectory replay workflow.

A detector instance holds mutable per-call workspace. Construct a separate
instance per concurrent stream. `detect_tensors` caches an instance per device
when parameters are omitted.

### 사선 Hole 입구의 RANSAC fallback 진단

2026-11-19에 Hole을 끝까지 한 번도 관측하지 못한 보존 사례
`case_0146_521142120`의 251개 원본 RGB-D 프레임을 재생했다. 기존 평가의
CUDA 검출 후보는 그대로 두고, 2픽셀 CUDA grid 경로가 Hole 입구를 찾지
못했을 때 RANSAC fallback이 새로 낸 후보만 뒤에 추가했다. 따라서 아래
차이는 전체 검출기 재실행 차이가 아니라 새 후보의 효과를 분리한 결과다.
Tracker와 GNN은 각각 체크섬이 고정된 epoch 443과 epoch 62 가중치를
사용했고 둘 다 동일 RTX 3090에서 baseline과 fallback을 다시 실행했다.

```bash
WANDB_MODE=disabled python main.py \
  --config graph/configs/temporal/hole_rim_ransac_fallback_cuda.yaml
```

fallback은 251/251프레임에서 네 점을 냈고, 1,004개 점의 GT Hole rim
일대일 대응 오차는 평균 2.275 mm, P90 2.640 mm, 최대 3.234 mm였다. 모든
프레임에서 6 mm 이내 점이 최소 세 개여서 물리 matcher가 프레임 0부터 Hole
ID를 초기화했다. 기존 입력은 251프레임 모두 Hole 미관측이었다.

| CUDA 동일 에피소드 251프레임 | 기존 고정 파이프라인 평균 / P90 | fallback 평균 / P90 |
| --- | ---: | ---: |
| GNN Hole 위치 | 92.620 / 92.620 mm | 0.753 / 0.752 mm |
| GNN tip−Hole 벡터 | 92.323 / 92.323 mm | 1.323 / 1.322 mm |

보존된 원본 출력과 CUDA baseline GNN 출력의 Object21 원소별 최대 차이는
`1.79e-7` m였다. 앞서 실행한 CPU fallback과 CUDA fallback은 관측 마스크와
프레임별 후보 수가 완전히 같았고, 최종 Object21 원소별 최대 차이는
`2.68e-7` m였다. CUDA 실행은 Vast.ai 인스턴스 `61278571`의 RTX 3090,
PyTorch `2.14.1+cu132`에서 수행했으며 결과 회수 후 인스턴스를 삭제했다.

이 결과는 원본 RGB-D가 남은 한 개의 대표 실패 에피소드에 한정된다. 나머지
62개 끝까지 미관측 사례의 일반화 및 정상 시점에서의 오검출 변화는 아직
평가하지 않았다.

[CUDA 수치 결과](../outputs/hole_rim_ransac_fallback_case_0146_cuda/results.json) ·
[CUDA 프레임별 출력](../outputs/hole_rim_ransac_fallback_case_0146_cuda/predictions.npz) ·
[실행 영수증](../outputs/hole_rim_ransac_fallback_case_0146_cuda/execution.json) ·
[CPU 수치 결과](../outputs/hole_rim_ransac_fallback_case_0146/results.json)

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
- Optimized resident RGB input uses the GPU for HSV, material components,
  and grayscale conversion. CUDA material components use Triton when installed.
  Host input uses OpenCV for HSV and components. OpenCV contour topology and
  subpixel features and the greedy candidate merge run on CPU.

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

In the first opening implementation, 1,259 inserted points were unmatched
on the initial cohort and 1,026 on the pilot. The nearest physical vertex
was 3–5 mm away for 1,031 and 835 points, and 5–10 mm away for 122 and 92.
Another 104 and 99 were within 3 mm of a supported vertex already claimed
by a different prediction. Two initial-state points were near physical
vertices outside the supported set. These point-level counts differ from
the net increase in unmatched Box predictions because some inserted
points take over an existing match. [Initial point audit](../outputs/hole_rim_recovery/171c176815c44300a9410eca1442adaa/results.json)
and [pilot point audit](../outputs/hole_rim_recovery/e801541dbc534846a962af97fc74b2b4/results.json).

The correction and replacement radius were selected using saved points
from these cohorts. The comparison measures the effect on fixed frames
and leaves generalization to different cameras, renderers and scenes open.

| Same-frame CPU replay | Base detector | First opening recovery | Refined opening recovery |
| --- | ---: | ---: | ---: |
| Initial rim 3 mm recall | 29.95% (1,395/4,657) | 45.97% (2,141/4,657) | 65.79% (3,064/4,657) |
| Initial unmatched Box points | 4,465 | 5,821 | 3,797 |
| Pilot rim 3 mm recall | 23.84% (1,101/4,618) | 43.22% (1,996/4,618) | 59.98% (2,770/4,618) |
| Pilot unmatched Box points | 4,611 | 5,691 | 3,817 |

Box exterior matches were identical per frame across the three variants.
Totals were 4,333/5,756 for the initial frames and 4,519/5,871 for the
pilot. The refined path had fewer rim matches than the first opening path
on 19 initial frames and zero pilot frames. [Initial replay](../outputs/hole_rim_recovery/c4308705b47c486fb068572b68c70d53/results.json)
and [pilot replay](../outputs/hole_rim_recovery/b568f8b5d2a34d4ead0dc46bbb26952a/results.json).

The fixed-parameter holdout used the 150 frozen validation seeds for resets
and the first 36 frozen test seeds for trajectories from the mixed-trajectory
protocol. The evaluator checked seed disjointness against both tuning runs
and the pilot split before opening either detector. It used the same 512×512
main camera, CPU physics and Mac MoltenVK renderer as the tuning evaluation.
It replayed expert, random and expert-to-random actions without filtering
successful or visible episodes, scoring the four trajectory quartiles
including the final observation. Reset frames and non-reset trajectory
frames came from separate seeds. Both variants received the same RGB-D and
calibration per frame, with GT and actor segmentation used for scoring.
The values remained fixed at 0.25 px and 8 mm.

| Fixed holdout | Previous rim recall | Refined rim recall | Unmatched Box points | Box exterior recall |
| --- | ---: | ---: | ---: | ---: |
| 150 resets | 263/579 (45.4%) | 366/579 (63.2%) | 685 → 451 | 546/722 (75.6%) for both |
| 432 frames from 108 trajectories | 587/1,432 (41.0%) | 905/1,432 (63.2%) | 2,005 → 1,304 | 1,426/1,904 (74.9%) for both |

Box exterior match counts were identical on every frame. The number of
supported trajectory frames with zero rim matches stayed at 114. Three
reset frames and zero trajectory frames lost rim matches. The result
tests new scene seeds under the same camera and renderer. The detector
ran on CPU. MPS and CUDA inference were outside this evaluation.

[Seed selection](../outputs/hole_rim_holdout/1e9075d317324c218a05f13f296d34fb/selection.json),
[reset scores](../outputs/hole_rim_holdout/1e9075d317324c218a05f13f296d34fb/reset.json)
and [trajectory scores](../outputs/hole_rim_holdout/1e9075d317324c218a05f13f296d34fb/trajectory.json)
include the frozen source hashes, frame identities and sampled trajectory
frame checksums.

## Small geometry recovery

For a CPU development comparison, enable both
`recover_hole_rim=True` and `recover_small_geometry=True` on
`TorchCornerDetector`. The Peg path fits a prism from two perpendicular
observed planes and the depth silhouette. It requires observed Peg depth
within 6 mm of each proposed vertex and replaces existing Peg candidates
within that distance. The Hole path accepts smaller enclosed depth regions,
requires at least two samples on each fitted side and fits a square in
metric plane coordinates. Both paths use current-frame RGB-D.

The 150 resets and 432 trajectory frames from the earlier holdout were
reused to develop these changes, so the results below are development
measurements. The detector ran on Mac CPU. MPS and CUDA were outside the
evaluation. [Reproduction command](../graph/README.md#small-geometry-recovery-evaluation).

| Cohort | Peg 3 mm recall | Peg unmatched points | Hole rim 3 mm recall | Box unmatched points |
| --- | ---: | ---: | ---: | ---: |
| Resets | 721/944 → 837/944 (76.4% → 88.7%) | 114 → 143 | 366/579 → 381/579 (63.2% → 65.8%) | 451 → 433 |
| Trajectories | 1,513/2,272 → 1,816/2,272 (66.6% → 79.9%) | 396 → 646 | 905/1,432 → 957/1,432 (63.2% → 66.8%) | 1,304 → 1,243 |

Box exterior matches were identical on each evaluated frame. The trajectory
Peg result is two matches short of 80%. Hole recovery still requires an
enclosed gap on an observed plane. On the trajectory cohort, 226 of 1,432
supported Hole references had no 3 mm opening-path proposal before this
change. Relaxing the foreground/behind-plane depth check alone recovered
no additional matches in the 216-frame development sweep. This motivates
a different candidate source for fragmented boundaries.

[Reset results](../outputs/small_geometry/9e09494904f44e8f81d6eb406155cfd6/reset.json)
and [trajectory results](../outputs/small_geometry/9e09494904f44e8f81d6eb406155cfd6/trajectory.json).

## Corner detector benchmark

The frozen benchmark replaces the previous holdout as the reference dataset
for Peg and Hole rim comparisons. Its seed selection takes the remaining
114 seeds in the mixed-trajectory test split after excluding the 36
trajectory seeds used to develop small geometry recovery. All 150
previous validation resets, the pilot split and the named tuning results
are excluded. The first 78 available seeds supply reset observations.
The remaining 36 supply expert, random and expert-to-random rollouts.
The evaluator samples four non-reset observations per rollout at the
trajectory quartiles, including the final observation. The camera,
visibility rule and one-to-one 3 mm scoring follow the previous evaluation.

The dataset stores raw RGB-D, actor segmentation, camera calibration and
scoring GT for all 510 observations. The per-cohort JSONs record paths,
hashes and actor IDs. The [replay command](../graph/README.md#corner-detector-benchmark)
uses those saved arrays. To regenerate the collection using the frozen
protocol and tuning results, run:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/corner_detection_benchmark.yaml
```

The first CPU benchmark results on 2026-11-15 compare the refined recovery
with the optional Peg prism and small-opening recovery enabled.

| Cohort | Peg 3 mm recall | Hole rim 3 mm recall | Box exterior 3 mm recall | Unmatched Box points |
| --- | ---: | ---: | ---: | ---: |
| 78 resets | 75.2% → 89.1% | 68.2% → 67.9% | 72.8% → 72.8% | 227 → 227 |
| 432 trajectory frames | 71.6% → 85.3% | 60.2% → 61.9% | 78.6% → 78.6% | 1,124 → 1,096 |

[Seed selection](../outputs/corner_detection_benchmark/69de5607c1b74eee968a991918275d19/selection.json),
[reset manifest and scores](../outputs/corner_detection_benchmark/69de5607c1b74eee968a991918275d19/reset.json)
and [trajectory manifest and scores](../outputs/corner_detection_benchmark/69de5607c1b74eee968a991918275d19/trajectory.json)
record the frozen inputs. Mac CPU inference and CPU physics with MoltenVK
rendering ran for this evaluation. MPS and CUDA inference were outside
this run.

### Fragmented Hole rim recovery

On CPU, set `recover_fragmented_rim=True` along with
`recover_hole_rim=True` and `recover_small_geometry=True`. If no enclosed
opening passes verification, the detector searches connected regions behind
an observed Box plane. It fits only the component boundary samples adjacent
to observed face pixels, so a gap in the Box mask need not close the contour.
This option also allows 2.8 mm median side residual and 19 mm difference
between the two fitted spans before square fitting, and applies a 0.375 px
image correction. The existing detector options retain their previous behavior.

| Fixed benchmark | Previous Hole 3 mm recall | Fragmented recovery | Unmatched Box points |
| --- | ---: | ---: | ---: |
| 78 resets | 209/308 (67.9%) | 255/308 (82.8%) | 227 → 236 |
| 432 trajectory frames | 944/1,526 (61.9%) | 1,115/1,526 (73.1%) | 1,096 → 1,104 |

Peg and Box exterior match counts agreed frame by frame. Among trajectory
frames with supported rim references, zero-match frames fell from 138 to 95.
Three reset and ten trajectory frames lost matches. The new option was
selected on a subset of this benchmark, so the table measures development
performance. CPU inference ran on Mac using saved observations. MPS and CUDA
inference were outside this run.

[Reset replay](../outputs/corner_detection_benchmark_replay/01301ae6b7c746d1be7a08d5be676afd/reset.json)
and [trajectory replay](../outputs/corner_detection_benchmark_replay/01301ae6b7c746d1be7a08d5be676afd/trajectory.json).

### Trajectory Hole rim recovery

The trajectory evaluator uses an empty `StaticRimMemory` per episode.
It retains the existing fragmented-rim detections and considers extra square
fits with the shorter and longer observed side spans. It can also recall
past opening points in world coordinates. Each supplemental point requires
current Box material-mask depth within 6 mm in a 7×7 neighborhood, and
must be at least 2 mm from every already emitted Box point. Update memory
with the observed fits after scoring each frame, in chronological order.
The stationary Box assumption held at all scored frames of the 108 benchmark
episodes. Use a new memory whenever an episode or Box changes.

On the same 432 benchmark frames, Hole rim 3 mm recall rose from
1,115/1,526 (73.1%) to 1,239/1,526 (81.2%). Peg and Box exterior match
counts agreed frame by frame. Unmatched Box predictions rose from 1,104
to 1,229. Settings were compared on this benchmark, so these are development
results. The replay consumed the four saved non-reset frames per episode.
It had no reset observation, future frame or GT as an inference input.
CPU inference ran on Mac. MPS and CUDA inference were outside the run.

[Frame scores](../outputs/trajectory_rim_recovery/f69747ee1304491bb7447a9a68a2ea09/trajectory.json)
and [replay command](../graph/README.md#trajectory-hole-rim-recovery).

A causal joint-square comparison aligned verified four-corner observations,
took their per-corner median and projected the result onto a square before
checking current depth. It reached 1,216/1,526 (79.7%) with 1,188 unmatched
Box predictions. The temporal point reuse still provides the higher recall
at 1,239/1,526 (81.2%) with 1,229 unmatched predictions. Peg and Box exterior
match counts agreed on every frame. The comparison uses the existing
trajectory replay command and is recorded under `consensus`.
[Joint-fit replay](../outputs/trajectory_rim_recovery/cb9365ad49b44770bc07bbbc80c490ae/trajectory.json).

On the same benchmark, Peg matched 580/740 (78.4%) during grasp and
1,425/1,611 (88.5%) outside grasp. These groups differ in pose and visibility.

The next paired replay tested two experimental variants. Peg size memory
learned dimensions from earlier RGB-D bounds and extended shortened bounds
from a current observed plane. Substituting those vertices dropped grasp
matches from 580 to 566. Keeping existing predictions and adding distinct
size-conditioned vertices yielded 581/740 grasp matches, with unmatched
grasp Peg points rising from 248 to 290. Non-grasp matches stayed at
1,425/1,611 while unmatched points rose from 281 to 288. Hole ranking kept
the primary predictions and used distance to the joint square to prioritize
supplemental candidates before 3 mm deduplication. It yielded 1,238/1,526
matches and 1,225 unmatched Box points versus 1,239 and 1,229 for temporal
reuse. Box exterior matches agreed frame by frame. These variants remain
evaluation comparisons. [Replacement replay](../outputs/trajectory_rim_recovery/b73debc0b93d4d1b80afb355fff33c36/trajectory.json)
and [supplemental replay](../outputs/trajectory_rim_recovery/9bf89be6d9ca4479badaff4445e308ee/trajectory.json).

A current-frame boundary comparison tested opt-in evaluation variants.
Peg silhouette refinement used half-pixel transitions into measured background
to estimate bounds unsupported by a fitted plane. It improved non-grasp
matches from 1,425 to 1,449 of 1,611, but reduced grasp matches from 580 to
477 of 740. Hole ranking prioritized supplemental points near measured
Box-to-deeper-depth image transitions before 3 mm deduplication. It retained
1,239/1,526 Hole matches and reduced unmatched Box points from 1,229 to
1,226. Box exterior matches stayed at 1,517. Both variants remain in the
CPU trajectory evaluation. [Boundary comparison](../outputs/trajectory_rim_recovery/68cf6bec0981441ba54d3b4c10b902ae/trajectory.json).

### Peg end visibility audit

On the frozen 432-frame trajectory benchmark, silhouette refinement lost
112 one-to-one Peg reference identities during grasp and gained nine.
For each lost identity, the audit compares the original matched point to the
nearest surviving refined point in the GT Peg basis. All 112 nearest
surviving points were beyond 3 mm. Median absolute axial error rose from
1.04 to 1.39 mm and median transverse error rose from 2.23 to 2.98 mm.
No refined axial error exceeded 3 mm, while 53 transverse errors did.
The median signed inward shift was approximately zero. Only two losses
had robot occlusion on the corresponding cap under the 7×7 cap-ray check.
This check detects foreground robot depth at sampled cap locations and
does not measure side occlusion or certify cap visibility.

| Peg 3 mm recall | Base prism | Silhouette | Silhouette with detected end plane |
| --- | ---: | ---: | ---: |
| Grasp, 159 frames | 580/740 (78.4%) | 477/740 (64.5%) | 534/740 (72.2%) |
| Non-grasp, 273 frames | 1,425/1,611 (88.5%) | 1,449/1,611 (89.9%) | 1,416/1,611 (87.9%) |

The end-plane gate is an evaluation variant. It uses the normal of a fitted
current-frame Peg plane aligned with the prism's long axis. GT and segmentation
enter the audit and scoring only. This Mac CPU run reused the saved frames.
MPS and CUDA inference were outside the evaluation.

With the frozen benchmark and shared ManiSkill environment, run
`WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/peg_end_visibility.yaml`.
[Per-frame audit](../outputs/peg_end_visibility/9cabe174bf6d40bc829a774651799c45/results.json).

The direct bound audit records support-plane offsets before corner filtering.
During grasp, all 104 changed local −Z bounds moved inward by a median
0.88 mm and worsened. Median absolute support-plane error rose from 1.79
to 2.67 mm. The local +Y bounds improved in 113 of 120 changes, with median
error falling from 1.12 to 0.48 mm. The camera-direction experiment suppresses
inward shifts of transverse bounds whose outward normal has a CV-down
component greater than 0.5. This covered all 104 observed bad −Z shifts
and zero of the 120 +Y shifts. It uses no GT at inference.

| Same-trajectory Peg comparison | Grasp recall / unmatched | Non-grasp recall / unmatched |
| --- | ---: | ---: |
| Base prism | 580/740 (78.4%) / 248 | 1,425/1,611 (88.5%) / 281 |
| Silhouette | 477/740 (64.5%) / 365 | 1,449/1,611 (89.9%) / 262 |
| Suppress lower-face inward shifts | 562/740 (75.9%) / 273 | 1,461/1,611 (90.7%) / 245 |

The camera rule remains an experimental comparison. Its grasp result
remains below the base prism. [Bound-level audit and comparison](../outputs/peg_end_visibility/f5eec4133f724350baec722cef7e3aa0/results.json).

### Box exterior corner recovery

The 2026-11-15 replay used the frozen 78 initial frames and 432 trajectory
frames. Before the correction, 346 of 371 initial exterior references had a
raw candidate within 3 mm. This fell to 301 after validation and 270 after
merging. On trajectory frames the counts were 1,833, 1,627 and 1,517 of
1,929. Those stage counts use nearest-candidate distance. Final recall uses
one-to-one matching at 3 mm.

The initial replay showed a median signed reprojection error of roughly
−0.5 pixel on both image axes for several exterior vertices. The
`TorchCornerDetector(device="cpu", correct_box_exterior_pixel_center=True)`
option shifts selected output positions by +0.5 pixel on both axes and
maps the shift to camera X and Y at the existing depth. Selection requires
a non-concave Box output within 12 mm of a current cube proposal. Fitted
Hole opening points are excluded. The correction runs after merging and
leaves the candidate and output counts fixed. The half-pixel offset was
measured on this camera and renderer. The detector's shared unprojection
continues to use integer pixel coordinates.

| Cohort | Box exterior 3 mm recall | Unmatched Box predictions | Hole rim 3 mm recall | Peg 3 mm recall |
| --- | ---: | ---: | ---: | ---: |
| Initial, 78 frames | 270/371 (72.8%) → 331/371 (89.2%) | 236 → 175 | 255/308 (82.8%) | 434/487 (89.1%) |
| Trajectory, 432 frames | 1,517/1,929 (78.6%) → 1,754/1,929 (90.9%) | 1,226 → 989 | 1,239/1,526 (81.2%) | 2,005/2,351 (85.3%) |

The trajectory comparison supplies identical depth-ranked temporal rim
points to both paths. Peg and Hole match counts agree frame by frame.
Refitting exterior bounds directly from the silhouette was also tried and
reduced exterior recall to 239/371 (64.4%) on initial frames and 1,191/1,929
(61.7%) on trajectories. These are development results for the frozen
benchmark on Mac CPU. MPS and CUDA inference were outside the replay.

[Replay command](../graph/README.md#box-exterior-corner-recovery),
[initial scores](../outputs/box_exterior_recovery/73973d83a922420cb53de266c00df159/reset.json),
[trajectory scores](../outputs/box_exterior_recovery/73973d83a922420cb53de266c00df159/trajectory.json).


#### Fixed-offset holdout

The holdout used the prior Hole recovery evaluation's 150 validation reset
seeds and 36 test trajectory seeds. None overlapped the Box exterior tuning
benchmark. The same camera protocol reproduced the initial observations, and
432 saved observations from 108 trajectories were replayed in episode order.
The evaluator checksum-verified the frames and retained +0.5 pixel on both
axes and the 12 mm cube proximity rule. Each pair used the same inputs and
the same supplemental Hole rim predictions.

| Holdout cohort | Box exterior 3 mm recall before → after | Unmatched Box predictions before → after | Hole rim 3 mm recall, both paths | Peg 3 mm recall, both paths |
| --- | ---: | ---: | ---: | ---: |
| Initial, 150 frames | 546/722 (75.6%) → 662/722 (91.7%) | 425 → 309 | 462/579 (79.8%) | 837/944 (88.7%) |
| Trajectory, 432 frames | 1,426/1,904 (74.9%) → 1,722/1,904 (90.4%) | 1,373 → 1,077 | 1,200/1,432 (83.8%) | 1,816/2,272 (79.9%) |

Exterior matches increased on 83 initial frames and 240 trajectory frames
and decreased on one initial frame and two trajectory frames. Hole rim and
Peg match counts agreed on every frame. Unmatched Box counts include all
Box outputs, including opening predictions. The holdout tests different
seeds using the same simulated camera and CPU detector. Other cameras and
MPS/CUDA inference were outside the evaluation.

[Holdout command](../graph/README.md#box-exterior-corner-recovery),
[frame selection](../outputs/box_exterior_holdout/383430b0c51643988544597a53abb2fd/selection.json),
[initial scores](../outputs/box_exterior_holdout/383430b0c51643988544597a53abb2fd/reset.json),
[trajectory scores](../outputs/box_exterior_holdout/383430b0c51643988544597a53abb2fd/trajectory.json).

### Peg trajectory position correction

The CPU option `correct_peg_position=True` requires
`recover_small_geometry=True`. After merging, it shifts the ray of every
Peg output by +0.25 pixel on both image axes and adds 0.5 mm to its camera
depth. It exports the corrected image and 3-D coordinates together. The
offsets were chosen on the development benchmark. The replay does not use
grasp labels, GT pose or segmentation when correcting predictions. Grasp
labels are used for reporting.

The comparisons used the base Peg prism, the Box exterior correction and
the same depth-ranked temporal Hole predictions on both sides. The holdout
reused the Box exterior holdout's 432 saved frames from 36 seeds, disjoint
from the 36 development trajectory seeds.

| Cohort | Peg 3 mm recall before → after | Unmatched Peg predictions before → after |
| --- | ---: | ---: |
| Development, all | 2,005/2,351 (85.3%) → 2,179/2,351 (92.7%) | 529 → 355 |
| Holdout, all | 1,816/2,272 (79.9%) → 2,010/2,272 (88.5%) | 646 → 452 |
| Holdout, grasp | 502/734 (68.4%) → 622/734 (84.7%) | 321 → 201 |
| Holdout, non-grasp | 1,314/1,538 (85.4%) → 1,388/1,538 (90.2%) | 325 → 251 |

Holdout Peg matches increased on 89 frames and decreased on one. Hole rim
and Box exterior match counts and unmatched Box counts agreed frame by
frame on both cohorts. These offsets were validated on the same simulated
camera on Mac CPU. Other cameras and MPS/CUDA inference were outside the
evaluation.

[Replay command](../graph/README.md#peg-position-correction),
[development scores](../outputs/peg_position_correction/eb3a7029e9514739938962a17571f925/trajectory.json),
[holdout scores](../outputs/peg_position_holdout/bd222e0dced042f7b5563b71aa610f48/trajectory.json).

#### Centered Peg prism bounds

After the Peg position correction, 112 of 734 grasp references remained
unmatched on the same holdout. The stage audit applies the same ray and depth
offsets to the intermediate candidates for comparison. Of the misses, 30 had
a validated candidate within 3 mm before merge and six had only an unvalidated
candidate within 3 mm. Of the 76 remaining misses, 17 had a final prediction
within 6 mm and 59 lacked one within 6 mm.

The old plane-to-bound calculation evaluates the plane along an axis through
the camera origin. A plane can be slightly tilted relative to the corresponding
orthogonal prism axis, so this introduces a bound error that depends on the
object's translation. The opt-in `center_peg_plane_bounds=True` option takes
the intersection along the prism axis through the center of the observed Peg
cloud bounds. The local-depth gate and output merge run on the resulting
vertices. The post-merge Peg position correction is also enabled on both
paths in the replay. The fit does not use grasp labels.

| Holdout cohort | Peg 3 mm recall before → after | Unmatched Peg predictions before → after |
| --- | ---: | ---: |
| All, 432 frames | 2,010/2,272 (88.5%) → 2,178/2,272 (95.9%) | 452 → 297 |
| Grasp, 160 frames | 622/734 (84.7%) → 672/734 (91.6%) | 201 → 152 |
| Non-grasp, 272 frames | 1,388/1,538 (90.2%) → 1,506/1,538 (97.9%) | 251 → 145 |

Grasp matches increased on 19 frames, and no grasp frame had an increase
in unmatched Peg predictions. Six non-grasp frames each gained one unmatched
Peg point. Hole rim and Box exterior match counts and unmatched Box counts
agreed frame by frame. A synthetic tilted-plane test verifies translation
invariance. The replay used Mac CPU and the same simulated camera.

[Replay command](../graph/README.md#peg-position-correction),
[paired trajectory scores](../outputs/peg_plane_center_holdout/85151decce144ddab9dfce22033401cc/trajectory.json).

#### Fresh trajectory validation

On 2026-11-15, the fixed correction was replayed on 24 new scene seeds
(1711150000–1711150023). The scene seeds were absent from the saved protocol
and the prior local evaluation inventories. The same camera and control
protocol ran expert, expert-random and random rollouts, with four non-reset
samples per rollout. Both paths used +0.25 pixel on each axis and +0.5 mm
camera depth. Only the improved path enabled centered prism bounds.

| Fresh cohort | Peg 3 mm recall before → after | Unmatched Peg predictions before → after |
| --- | ---: | ---: |
| Grasp, 105 frames | 430/490 (87.8%) → 465/490 (94.9%) | 136 → 105 |
| Non-grasp, 183 frames | 1,009/1,081 (93.3%) → 1,070/1,081 (99.0%) | 128 → 65 |
| All, 288 frames | 1,439/1,571 (91.6%) → 1,535/1,571 (97.7%) | 264 → 170 |

Grasp matches increased on 12 frames and decreased on none. One grasp
frame gained an unmatched Peg prediction. Hole and Box match counts and
unmatched Box counts agreed on every frame. A 10,000-draw percentile
bootstrap resampling scene seeds gave a 95% interval of 92.2–97.2% for
post-change grasp recall. Inference ran on Mac CPU with the same
simulated camera used for development.

[Fresh evaluation command](../graph/README.md#peg-position-correction),
[cohort selection](../outputs/peg_plane_center_fresh/944a5ca9b1954caf80f62fc66f53be9e/selection.json),
[paired trajectory scores](../outputs/peg_plane_center_fresh/944a5ca9b1954caf80f62fc66f53be9e/trajectory.json).

#### Depth verification of rejected Hole rim fits

On 2026-11-15, 155 of the 165 missed Hole references in the 288 fresh
trajectory frames occurred in frames without a verified opening or a
recalled rim point. In the 432-frame development replay, the raw enclosed
contour square candidates already covered 104 rim references within 3 mm
on frames without those outputs.

The CPU option `verify_rejected_rim_squares=True` requires
`recover_fragmented_rim=True`. When both contour and depth-component
verification fail, it checks the rejected square fits against current RGB-D.
It samples 5 and 8 mm to either side of each fitted edge, looking for
Box-plane support outside and deeper depth inside. It retains the strongest
fit that meets the support requirements on all four sides. The verified
corners enter the existing merge and causal rim memory.

| Paired trajectory cohort | Hole rim 3 mm recall | Unmatched Box predictions |
| --- | ---: | ---: |
| Development, 432 frames | 1,239/1,526 (81.2%) → 1,354/1,526 (88.7%) | 989 → 1,011 |
| Fresh, 288 frames | 811/976 (83.1%) → 949/976 (97.2%) | 559 → 579 |

In the fresh cohort, rim matches increased on 42 frames and agreed on the
other 246. Peg and Box exterior match counts agreed frame by frame on both
cohorts. Unmatched Box counts cover exterior and opening outputs combined.
The comparison used the existing simulated camera and the Mac CPU detector.

[Replay command](../graph/README.md#hole-rim-depth-verification),
[development scores](../outputs/hole_depth_fallback_dev/1aa2e79f3db2419e8ec1dde403b77f0a/trajectory.json),
[fresh scores](../outputs/hole_depth_fallback/374151b3f35d43a49b14e2f10ffe358c/trajectory.json).

#### Fresh validation of fixed Hole rim verification

The fixed four-edge depth check was evaluated on 24 new scene seeds
(1711150100–1711150123) absent from the saved protocol and earlier local
evaluation inventories. The same camera and control protocol generated
72 trajectories across expert, expert-random and random modes. Both
settings scored the same 288 checksum-verified samples.

| Metric | Before | After |
| --- | ---: | ---: |
| Hole rim 3 mm recall | 791/1,011 (78.2%) | 835/1,011 (82.6%) |
| Unmatched Box predictions | 630 | 665 |
| Peg 3 mm recall | 1,541/1,590 (96.9%) | 1,541/1,590 (96.9%) |
| Box exterior 3 mm recall | 1,219/1,340 (91.0%) | 1,219/1,340 (91.0%) |

The fixed check added 44 matches across 21 frames. Peg and Box exterior
match counts agreed on every frame. A 10,000-draw bootstrap resampling
scene seeds gave a 95% interval of 70.9–92.6% for improved rim recall.
The point estimate fell below the 90% target on this Mac CPU cohort.

[Replay command](../graph/README.md#hole-rim-depth-verification),
[cohort selection](../outputs/hole_depth_fallback_fresh/eaf1f267cd534e4fb84f1e7b229a3255/selection.json),
[paired scores](../outputs/hole_depth_fallback_fresh/eaf1f267cd534e4fb84f1e7b229a3255/trajectory.json).

#### Depth-verified longer-span replacement

Of the 176 baseline misses on the 288-frame fresh validation cohort, 135
had no primary Box candidate within 3 mm. Another 21 had a candidate
within 3 mm and none passed primary validation. The remaining 20 had a
validated primary candidate within 3 mm and were unmatched after output
merging and temporal recovery. The counts compare final one-to-one matches
with distances to intermediate primary candidates.

Fit the longer observed span on both enclosed contour and depth-component
proposals, then apply the existing four-edge depth test. When a rim
already exists, each new corner must lie within 7 mm of an emitted rim
point. The verified square replaces the opening and recalled rim points.
It skips output corners within 3 mm of a retained Box point.

On the same frames, Hole rim 3 mm recall rose from 835/1,011 (82.6%) to
918/1,011 (90.8%), and unmatched Box predictions fell from 665 to 504.
Hole rim matches increased in 67 frames and decreased in 22. Peg
(1,541/1,590) and Box exterior (1,219/1,340) match counts agreed on every
frame. This cohort informed the replacement design and serves as an
engineering comparison on Mac CPU.

[Replay config](../graph/configs/temporal/hole_rim_replacement.yaml),
[paired scores and stage counts](../outputs/hole_rim_replacement/50ad7416c5f24702b5335d4d7ede2dbd/trajectory.json).

#### Independent validation of longer-span replacement

The fixed replacement was evaluated on 24 additional scene seeds
(1711150200–1711150223). These seeds were disjoint from prior local
evaluation inventories. The same camera and control protocol generated
72 trajectories, with 288 checksum-verified frames paired across settings.

| Metric | Before replacement | After replacement |
| --- | ---: | ---: |
| Hole rim 3 mm recall | 875/1,002 (87.3%) | 913/1,002 (91.1%) |
| Unmatched Box predictions | 645 | 524 |
| Peg 3 mm recall | 1,456/1,513 (96.2%) | 1,456/1,513 (96.2%) |
| Box exterior 3 mm recall | 1,180/1,260 (93.7%) | 1,180/1,260 (93.7%) |

Rim matches increased on 47 frames and decreased on 38. Peg and Box
exterior match counts agreed on every frame. A 10,000-draw bootstrap
resampling scene seeds put the 95% interval for improved rim recall at
85.6–96.0%. Inference ran on Mac CPU with the existing simulated camera.

[Replay config](../graph/configs/temporal/hole_rim_replacement_fresh.yaml),
[paired scores](../outputs/hole_rim_replacement_fresh/a63dd475b8224b5b819388dbd8451174/trajectory.json).

### Peg rear-end recovery during insertion

On the CPU detector, enable `recover_peg_rear_end=True` together with
`recover_small_geometry=True`. For a short observed Peg, two perpendicular
side planes define a longitudinal axis that must agree with the direction
toward the observed Box. The recovery considers only the end away from Box.
That end needs a fitted plane or two side faces whose observed edges terminate
at the same longitudinal bound. A recovered vertex must have local Peg depth
support within 5 mm of the calibrated position. A closer Box pixel on the
projected ray rejects that candidate. Accepted vertices use the existing Peg
merge. Inference uses the current RGB-D frame and calibration.

The 2026-11-15 paired CPU replay used the original expert episodes with scene
seeds 1711150300–1711150307. It compared the last three alignment frames and
every insertion frame against the same saved RGB-D inputs and reference
corners. Both variants used the existing centered Peg bounds and position
correction. The depth support limit was selected on these frames.

| Cohort | Peg 3 mm recall before → after | Unmatched Peg predictions before → after |
| --- | --- | --- |
| Last three alignment frames per episode (24) | 128/130 (98.5%) → 128/130 (98.5%) | 26 → 26 |
| All insertion frames (148) | 550/572 (96.2%) → 568/572 (99.3%) | 317 → 317 |
| Last insertion frame per episode (8) | 22/28 (78.6%) → 28/28 (100%) | 10 → 10 |

Box exterior and Hole rim match counts and unmatched Box counts agreed frame
by frame. The recovery requires at least two observed side planes and still
depends on the RGB material mask. This replay did not include newly collected
episodes after tuning.

[Saved frame manifest](../outputs/peg_insertion_holdout/9f1b5679c01d4584a5adbdd3677c1c22/trajectory.json)
and [paired replay](../outputs/peg_rear_end/482483eed0b2496f85deb4f28673d554/trajectory.json)
are local artifacts.

The independent CPU replay used fixed recovery conditions on expert scene seeds
1711150400–1711150415. Fifteen of sixteen episodes reached insertion. The
last-insertion cohort includes the final observed insertion frame of the episode
that timed out during insertion.

| Independent cohort | Peg 3 mm recall before → after | Unmatched Peg predictions before → after |
| --- | --- | --- |
| All insertion frames (416) | 1593/1648 (96.7%) → 1635/1648 (99.2%) | 828 → 825 |
| Last insertion frame per episode (15) | 41/55 (74.5%) → 55/55 (100%) | 13 → 13 |

On the last frames, one episode gained an unmatched prediction and another
lost one. Three insertion frames with visible Peg corners and zero baseline Peg
outputs all recovered at least one match.

[Independent replay with checksummed frame references](../outputs/peg_rear_end_holdout/5daec1f2b2334394a6f266b36c11f6b5/trajectory.json).

## Combined Peg, Box exterior and Hole rim measurement

The 2026-11-15 CPU replay enabled Peg position correction, centered plane
bounds and rear-end recovery, the Box exterior pixel correction, and
depth-verified longer-span rim replacement with episode-local rim memory.
Reset scoring used the single-frame detector and counted only the Hole
entrance rim. Trajectory scoring kept the existing visibility rule for both
rims. Each replay used its saved RGB-D frames in chronological order. The
insertion cohort starts from the last three saved alignment frames.

| Saved cohort | Frames | Peg 3 mm recall | Box exterior 3 mm recall | Hole rim 3 mm recall | Unmatched Peg | Unmatched Box |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent resets | 150 | 941/944 (99.7%) | 662/722 (91.7%) | 496/579 (85.7%) | 38 | 335 |
| General trajectories | 288 | 1468/1513 (97.0%) | 1180/1260 (93.7%) | 913/1002 (91.1%) | 166 | 524 |
| Insertion | 416 | 1635/1648 (99.2%) | 1420/1580 (89.9%) | 1056/1087 (97.1%) | 825 | 733 |
| Last insertion frame per episode | 15 | 55/55 (100%) | 52/58 (89.7%) | 16/17 (94.1%) | 13 | 20 |

The general trajectories cover 24 scene seeds with Expert, Random and
Expert–Random rollouts. Compared with the previous replay of these frames,
Peg gained 12 matches and its unmatched count stayed at 166. Box exterior
and Hole rim match counts and unmatched Box counts agreed per frame.
Unmatched Box predictions are counted jointly because both corner families
share the same 3 mm assignment.

[Replay configuration](../visual/configs/combined_corner_detection.yaml)
and [per-frame scores](../outputs/combined_corner_detection/e752443d833f4351a9f6a70d61371406/scores.json).

### GPU execution of Peg rear-end recovery

`TorchCornerDetector.main(peg_rear_end_device="mps")` and
`TorchCornerDetector.main(peg_rear_end_device="cuda")` send the observed Peg
and Box clouds, planes, depth, and Box mask to the selected device.
Candidate geometry, calibrated depth support, and Box occlusion use tensor
operations on that device. Four candidate slots and their validity flags
return to the CPU merge. Frame preparation, the original Peg prism, Box
exterior recovery, and single-frame Hole rim recovery execute on CPU.

The paired benchmark replays the same 416 checksum-verified insertion
frames, including 15 final insertion frames. It alternates execution order
after warm-up. End-to-end timings include both transfers and exclude file
loading and scoring. The separate synchronized pass reports stage times,
including dispatch and synchronization overhead. The 6 mm depth gate is
timed with candidate generation and the 5 mm calibrated gate is timed
separately.

On Apple M4 Pro, mean frame time was 86.0 ms on CPU and 95.0 ms with MPS
rear-end recovery. On the Vast RTX 3090 host, mean frame time was 107.2 ms
on CPU and 109.7 ms with CUDA rear-end recovery. Each comparison uses a CPU
baseline on the same host. Both accelerators matched the corresponding
CPU scores on the 416 insertion frames. Each run found 1635/1648 Peg
corners, 1420/1580 Box exterior corners and 115/1087 Hole rim corners.
The Hole counts are for the frame detector. The earlier combined
measurement also ran episode-local Hole rim memory and longer-span
replacement.

| Stage, mean ms per insertion frame | M4 Pro CPU | MPS rear end | Vast CPU | RTX 3090 rear end |
| --- | ---: | ---: | ---: | ---: |
| Upload | — | 2.867 | — | 0.492 |
| Candidate generation and initial depth gate | 0.146 | 2.870 | 0.313 | 1.869 |
| Projection and calibration | 0.003 | 0.504 | 0.012 | 0.169 |
| Calibrated depth support | 0.006 | 0.411 | 0.017 | 0.084 |
| Box occlusion | 0.004 | 0.476 | 0.012 | 0.202 |
| Download | — | 0.979 | — | 0.048 |
| Total rear-end recovery | 0.165 | 8.169 | 0.371 | 2.969 |

CPU validation ran only when candidates passed the initial gate. The table
divides stage totals by all 416 frames. The rear-end-only offload benchmark
sets the main geometry device to CPU explicitly.

[MPS frame scores and timing](../outputs/peg_rear_end_mps/49b2cfdadcab4b08bf254ff9938de71b/scores.json)
and [CUDA frame scores and timing](../outputs/peg_rear_end_cuda/65836ce8b1874a1e8ef925d59c2e495a/scores.json).

### GPU-resident frame geometry

`TorchCornerDetector.main(device="mps")` and
`TorchCornerDetector.main(device="cuda")` use the main recovery settings
with GPU RGB material color classification, depth validation and
unprojection, point selection, plane fitting, and Peg prism and rear-end
checks. The rear-end stage consumes the existing device tensors and passes
its candidates to the usual merge without the separate rear-end upload
or download. The CPU extracts OpenCV contours and RGB features, fits Hole
openings, and runs greedy candidate selection. Those stages use host images or
readbacks of plane descriptors, depth, and candidate data as needed. In the
416-frame measurement below, material connected components also ran on the CPU.
The resident-input migration is measured separately below.

The paired replay used the same 416 saved insertion frames on Apple M4 Pro
and the RTX 3090 host. The total includes sensor uploads, required intermediate
readbacks, and conversion of the final results to a CPU dictionary. File
loading and scoring are outside the timer. The separate stage timing pass
synchronizes before and after each measured region. Nested stage times
overlap their inclusive parent timings.

| Host | CPU mean / p50 / p95 (ms) | GPU mean / p50 / p95 (ms) |
| --- | --- | --- |
| Apple M4 Pro | 101.2 / 101.1 / 116.3 | 47.8 / 48.4 / 63.7 |
| RTX 3090 server | 107.7 / 107.5 / 120.9 | 74.4 / 65.9 / 87.1 |

Each route matched 1635/1648 Peg, 1420/1580 Box exterior and 115/1087
Hole rim references over the 416 frames. Both GPUs matched all 55 last-frame
Peg references. Peg unmatched counts agreed frame by frame. Box exterior
matches changed by one in each direction on two frames per host. Joint Box
unmatched counts changed from 697 to 694 on Mac and from 694 to 697 on the
RTX 3090 server.

The largest timed improvement was dense plane masks. Mean time dropped from
78.40 to 1.10 ms with MPS and from 57.37 to 0.82 ms with CUDA. The point cloud
selection and plane fit region itself took longer on either GPU. The main
preset now selects the resident GPU path automatically on supported hosts.

[MPS frame scores and stage timings](../outputs/corner_geometry_mps/a040d40978824dd483772c6ee2686e8f/scores.json)
and [CUDA frame scores and stage timings](../outputs/corner_geometry_cuda/34023f82e1064451b7243c2d61f9e882/scores.json).


## 지난주 검출기와 현재 메인 성능 비교

2026-11-15에 지난주 기준으로 2026-11-13 커밋 `f4dcbaa7d572017ad08ec99a6c8fa6e5f138859d`을 선택했다. 이 커밋의 검출기 소스는 11월 5일의 최초 추출 커밋 `922744a`와 동일하다. 당시 `TorchCornerDetector(device="cpu")`와 현재 `TorchCornerDetector.main(device="cpu")`를 Mac에서 동일한 RGB-D 프레임에 재실행했다. CPU 고정은 로직 변경의 영향을 비교하기 위한 설정이다. 현재 배포 기본값은 지원되는 GPU를 자동 선택한다.

가시 기준점은 올바른 물체의 깊이 픽셀이 7×7 이웃에서 GT 꼭짓점 6 mm 이내에 있을 때 평가에 포함했다. 검출 성공은 같은 물체에서 3 mm 이내 일대일 매칭으로 계산했다. 초기 상태의 Hole은 입구 림을 평가했고, 궤적에서는 관측 기준을 만족한 양쪽 림을 평가했다. 미매칭은 기준점과 짝을 이루지 못한 **출력 점** 수다. Box 외곽과 Hole 림은 같은 Box 출력에서 매칭되므로 미매칭 Box에 합산했다.

- 초기 상태는 기존 독립 리셋 150개를 사용했다.
- 일반 궤적은 24개 장면 시드의 Expert, Expert → Random, 처음부터 Random 세 모드에서 각각 96프레임을 사용했다. Expert → Random은 사전 배정된 전환 단계와 난수 지연에 도달한 뒤 랜덤 액션을 실행한다. 표의 전환 후 행은 20개 에피소드에서 랜덤 액션이 적용된 뒤의 관측 프레임 72개다.
- Box 이동은 Expert가 움직이는 새 시드 8개에서 측정했다. Box를 시뮬레이터 x축으로 2초 동안 60 mm 이동시키고, 실제 위치가 바뀐 320프레임을 채점했다. 수집은 기존 근접 가드 80 mm를 사용했다.
- 삽입 전용은 별도의 Expert 평가에서 삽입에 진입한 15개 에피소드의 연속 416프레임이다. 말미는 각 에피소드의 마지막 삽입 프레임이다. 일반 궤적의 정책별 행과 삽입 말미 행은 상위 행에 포함된다.

### 3 mm 재현율과 미매칭 점

각 셀은 **지난주 → 현재** 순서다. 재현율 괄호에는 매칭 수 변화와 공통 가시 기준점 수를 표시했다.

| 조건 | 프레임 | Peg 3 mm | Box 외곽 3 mm | Hole 림 3 mm | 미매칭 Peg | 미매칭 Box |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 초기 상태 | 150 | 76.4% → 99.7% (721 → 941/944) | 75.6% → 91.7% (546 → 662/722) | 29.4% → 85.7% (170 → 496/579) | 114 → 38 | 529 → 335 |
| 일반 궤적 전체 | 288 | 74.4% → 97.0% (1,126 → 1,468/1,513) | 74.2% → 93.7% (935 → 1,180/1,260) | 24.7% → 80.3% (247 → 805/1,002) | 160 → 166 | 967 → 580 |
| Expert | 96 | 69.0% → 94.2% (310 → 423/449) | 74.2% → 93.6% (314 → 396/423) | 24.3% → 77.3% (75 → 239/309) | 68 → 69 | 308 → 182 |
| Expert → Random 전체 | 96 | 71.6% → 96.8% (340 → 460/475) | 74.0% → 94.7% (318 → 407/430) | 25.7% → 82.2% (88 → 281/342) | 43 → 67 | 331 → 192 |
| Expert → Random 전환 후 | 72 | 69.7% → 95.9% (253 → 348/363) | 71.7% → 94.2% (236 → 310/329) | 21.8% → 82.8% (57 → 216/261) | 29 → 44 | 250 → 131 |
| 처음부터 Random | 96 | 80.8% → 99.3% (476 → 585/589) | 74.4% → 92.6% (303 → 377/407) | 23.9% → 81.2% (84 → 285/351) | 49 → 30 | 328 → 206 |
| Box 실제 이동 중 · Expert | 320 | 58.6% → 98.2% (903 → 1,513/1,540) | 82.6% → 92.3% (1,344 → 1,502/1,627) | 11.5% → 82.8% (146 → 1,053/1,272) | 129 → 259 | 959 → 730 |
| 삽입 전체 | 416 | 55.2% → 99.2% (909 → 1,635/1,648) | 81.1% → 89.9% (1,282 → 1,420/1,580) | 0.6% → 10.6% (7 → 115/1,087) | 315 → 825 | 807 → 697 |
| 에피소드별 마지막 삽입 | 15 | 58.2% → 100.0% (32 → 55/55) | 70.7% → 89.7% (41 → 52/58) | 0.0% → 0.0% (0 → 0/17) | 13 → 13 | 31 → 20 |

일반 궤적에서 Peg는 342개, Box 외곽은 245개, Hole 림은 558개를 더 맞혔다. Box 이동 중에는 Peg 재현율이 58.6%에서 98.2%로 올랐고 미매칭 Peg도 130개 늘었다. 전체 삽입에서는 Peg 재현율이 55.2%에서 99.2%로 올랐으며, 미매칭 Peg 역시 315개에서 825개로 늘었다. 삽입 말미 Peg는 가시 기준점 55개를 모두 맞혔고 미매칭은 13개였다.

### 궤적 이력 기반 Hole 림 복구

현재 단일 프레임 메인과 함께, 같은 **고정 Box** 궤적의 기존 이력 기반 평가 결과도 확인했다. 이 경로는 에피소드 내 림 이력과 깊이 검증을 거친 긴 변 후보 교체를 사용한다. Peg와 Box 외곽 매칭 수는 아래 집계에서 단일 프레임 결과와 같았다.

| 조건 | 단일 프레임 → 이력 기반 Hole 림 3 mm | 미매칭 Box |
| --- | ---: | ---: |
| 일반 궤적 전체 | 80.3% → 91.1% (805 → 913/1,002) | 580 → 524 |
| Expert | 77.3% → 91.3% (239 → 282/309) | 182 → 167 |
| Expert → Random 전체 | 82.2% → 90.1% (281 → 308/342) | 192 → 173 |
| Expert → Random 전환 후 | 82.8% → 90.8% (216 → 237/261) | 131 → 116 |
| 처음부터 Random | 81.2% → 92.0% (285 → 323/351) | 206 → 184 |
| 삽입 전체 | 10.6% → 97.1% (115 → 1,056/1,087) | 697 → 733 |
| 에피소드별 마지막 삽입 | 0.0% → 94.1% (0 → 16/17) | 20 → 20 |

Box가 움직이는 조건의 표에는 단일 프레임 메인 결과를 사용했다. 고정 Box 위치를 전제로 저장한 림 이력은 이 구간에 적용하지 않았다.

### 현재 검출기 CPU·GPU 처리 시간

앞서 동일한 삽입 프레임 416개에서 측정한 시간표를 그대로 사용했다. 전체 시간은 입력 전송과 중간 데이터 회수, 최종 CPU 출력 변환을 포함하며 파일 읽기와 채점은 제외한다. CPU와 GPU는 같은 장비 안에서 비교했다. 단위는 ms다.

| 장비 | CPU 평균 / p50 / p95 | GPU 평균 / p50 / p95 |
| --- | ---: | ---: |
| Apple M4 Pro · MPS | 101.2 / 101.1 / 116.3 | 47.8 / 48.4 / 63.7 |
| RTX 3090 서버 · CUDA | 107.7 / 107.5 / 120.9 | 74.4 / 65.9 / 87.1 |

단계별 평균은 별도의 동기화 계측에서 나온 ms/프레임이다. 전처리 소계에 그 아래 세부 항목이 포함된다. 전체 프레임 시간은 동기화 계측을 끈 상태로 측정했다.

| 구간 | Mac CPU | MPS | 서버 CPU | CUDA |
| --- | ---: | ---: | ---: | ---: |
| **RGB-D 전처리 전체** | **84.73** | **17.71** | **71.75** | **16.69** |
| ↳ 입력·마스크 GPU 업로드 | — | 1.89 | — | 0.91 |
| ↳ RGB 색상 분류 | 0.14 | 1.75 | 0.45 | 0.56 |
| ↳ 깊이 역투영 | 1.32 | 0.43 | 3.19 | 0.23 |
| ↳ 점군 추출·평면 피팅 | 2.62 | 8.68 | 6.53 | 10.91 |
| ↳ **영상 전체 평면 마스크 생성** | **78.40** | **1.10** | **57.37** | **0.82** |
| ↳ 색상 마스크·평면 정보 CPU 회수 | — | 2.40 | — | 1.21 |
| 윤곽 후보 처리 | 5.15 | 8.21 | 14.04 | 22.17 |
| Box 외곽 / Hole 림 처리 | 6.99 / 2.37 | 5.51 / 2.59 | 15.18 / 3.58 | 17.13 / 3.89 |
| Peg 기본 프리즘 / 뒤쪽 끝면 복구 | 0.16 / 0.16 | 1.43 / 2.21 | 0.33 / 0.36 | 1.22 / 2.15 |
| 후보 병합·좌표 보정 | 0.31 | 2.80 | 0.78 | 1.47 |
| 최종 CPU 출력 변환 | 0.02 | 0.39 | 0.05 | 0.29 |

Peg 가림 필터를 켠 공개 기본 API를 Apple M4 Pro MPS에서 영상과 동일한
Expert 에피소드 140프레임으로 다시 측정했다. RGB-D 입력부터 텐서 출력까지
평균 43.95 ms, p95 54.58 ms였다. CPU 딕셔너리 변환까지 포함하면 평균
44.36 ms, p95 55.06 ms였다. 위 표의 17.71 ms(MPS)와 16.69 ms(CUDA)는
RGB-D 전처리 소계다. [재측정 결과](../outputs/corner_detector_video/main_latency.json)

### 재현 자료

- [쌍별 채점 및 이동 Box 프레임 목록](../outputs/corner_detector_comparison/a7570138297342a7bc9f5c8cfe7e227a/scores.json)
- [재실행 코드](../outputs/corner_detector_comparison/replay.py) · [설정](../outputs/corner_detector_comparison/replay.yaml)
- [궤적 이력 기반 Hole 결과](../outputs/combined_corner_detection/e752443d833f4351a9f6a70d61371406/scores.json)
- [MPS 시간 원본](../outputs/corner_geometry_mps/a040d40978824dd483772c6ee2686e8f/scores.json) · [CUDA 시간 원본](../outputs/corner_geometry_cuda/34023f82e1064451b7243c2d61f9e882/scores.json)

저장소 루트에서 매니페스트의 로컬 프레임과 ManiSkill CPU 환경을 준비하고 아래 명령을 실행한다.

```bash
WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=outputs/corner_detector_comparison .venv/bin/python main.py \
  --config outputs/corner_detector_comparison/replay.yaml
```

### 삽입 중 Box 가림 경계에서 생긴 Peg 오검출

2026-11-15에 위 비교와 동일한 삽입 416프레임을 다시 채점했다. 개선 전 미매칭 Peg
825개 중 712개(86.3%)는 기본 프리즘, 112개는 기존 윤곽 경로(`v2_core`),
1개는 뒤쪽 끝면 복구에서 나왔다. 지난주 검출기의 기존 경로는 미매칭 315개
(`v2_core` 287개, 기타 경계·평면 후보 28개)를 출력했다. 기존 경로에
귀속되는 미매칭은 203개 줄었고 프리즘과 뒤쪽 복구에 713개가 더해져 전체가
510개 늘었다. 프리즘 미매칭 712개 중 620개는 GT Peg의
길이 방향으로 양쪽 실제 끝면에서 각각 6 mm 넘게 떨어져 있었다. Box가 Peg
앞부분을 가려도 관측 길이가 120 mm 이상 남으면 프리즘은 점군이 끊긴 위치를
끝면 경계로 사용할 수 있다. 잘린 위치에도 Peg 측면의 깊이점이 남아 있어 기존
6 mm 근접 검사만으로는 그 경계를 통과한 점을 구별하기 어려웠다.

메인 검출기는 병합과 좌표 보정 후 Box 쪽 프리즘 출력만 검사한다. 관측된
끝면 평면이 후보를 지지하면 유지한다. 그 외에는 후보 전후의 투영점에서 더
가까운 Box 깊이가 확인되거나, 같은 끝의 후보 두 개 이상에서 Box 가림이
일관되고 그 너머에 Peg 깊이 지지가 없을 때 제거한다. 검출에는 현재 프레임의
RGB-D와 카메라 보정값을 사용한다.

| 같은 프레임에서의 비교 | 프레임 수 | Peg 3 mm 재현율 | 미매칭 Peg |
| --- | ---: | ---: | ---: |
| 삽입 전체 · CPU / MPS | 416 | 1,635/1,648 (99.2%) → 1,635/1,648 (99.2%) | 825 → 471 |
| 삽입 전체 · CUDA | 416 | 1,635/1,648 (99.2%) → 1,635/1,648 (99.2%) | 825 → 470 |
| 에피소드별 마지막 삽입 · CPU / MPS / CUDA | 15 | 55/55 (100%) → 55/55 (100%) | 13 → 9 |
| 별도 일반 궤적 · CPU | 288 | 1,468/1,513 (97.0%) → 1,468/1,513 (97.0%) | 166 → 162 |

세 백엔드 모두 삽입의 각 프레임에서 Peg 매칭 수를 유지했다. Box 외곽과
단일 프레임 Hole 림은 각각 1,420/1,580(89.9%), 115/1,087(10.6%)로 유지됐다.
미매칭 Box도 각 백엔드의 삽입 프레임별 전후가 같았다. CPU와 CUDA는 697개,
MPS는 694개였다. 삽입 프레임으로 필터 조건을 정했고 별도 일반 궤적을 회귀
검증에 사용했다. CPU에서 남은 미매칭 471개는 기본 프리즘 358개,
기존 윤곽 112개, 뒤쪽 끝면 복구 1개로 구성된다.

CPU와 MPS에서 삽입 전체 미매칭은 42.9% 줄었다. 지난주 검출기의 315개와
비교하면 CPU 기준으로 156개가 더 남아 있다. 고정된 필터의 성능은 별도
일반 궤적과 아래 독립 삽입 에피소드에서 평가했다.

[지난주 검출기 후보 경로별 집계](../outputs/peg_insertion_precision/historical_methods.json) ·
[프레임별 CPU 삽입 결과 및 후보 진단](../outputs/peg_insertion_precision/cpu_paired.json) ·
[MPS](../outputs/peg_insertion_precision/mps_paired.json) ·
[CUDA](../outputs/peg_insertion_precision/cuda_paired.json) ·
[별도 일반 궤적](../outputs/peg_insertion_precision/general_paired.json) ·
[재생 코드](../outputs/peg_insertion_precision/audit.py)

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python outputs/peg_insertion_precision/audit.py \
  outputs/peg_rear_end_holdout/5daec1f2b2334394a6f266b36c11f6b5/trajectory.json \
  outputs/peg_insertion_precision/cpu_paired.json cpu
```

#### 고정 Peg 가림 필터의 독립 삽입 검증

2026-11-15에 사용한 새 Expert 장면 시드는 `1711150600–1711150615`다.
기존 카메라와 Expert 정책으로 수집한 16개 에피소드가 모두 삽입에 진입했다.
삽입에 성공한 15개 에피소드의 283프레임과, 스텝 제한으로 종료된 에피소드의
137프레임을 합쳐 전체 420프레임을 평가했다. 튜닝 데이터와 초기 상태 지문,
프레임 체크섬이 겹치지 않았다. 메인 검출기에서 `reject_peg_box_cut`만 껐다
켰으며, 필터의 거리와 깊이 조건을 고정했다. 이전과 같은 가시 기준점 및
일대일 3 mm 매칭으로 채점했다.

CPU·MPS·CUDA에서 삽입 전체 미매칭은 973개에서 410개로 57.9% 줄었다. 3 mm
재현율은 1,584/1,604(98.75%)에서 1,578/1,604(98.38%)로 내려가
99%에 미치지 못했다. 필터 적용 전보다 0.37%p 낮았다. 다섯 프레임에서
잃은 매칭 여섯 개는 모두 Peg 앞쪽 끝면의 꼭짓점이었다.

| 평가 구간 · CPU / MPS / CUDA 공통 | 프레임 | Peg 3 mm 재현율 전 → 후 | 미매칭 Peg 전 → 후 |
| --- | ---: | ---: | ---: |
| 전체 삽입 | 420 | 98.75% (1,584/1,604) → 98.38% (1,578/1,604) | 973 → 410 |
| 성공한 에피소드의 전체 삽입 | 283 | 98.44% (1,133/1,151) → 98.09% (1,129/1,151) | 566 → 391 |
| 스텝 제한에 도달한 에피소드 | 137 | 99.56% (451/453) → 99.12% (449/453) | 407 → 19 |
| 에피소드별 마지막 삽입 | 16 | 93.10% (54/58) → 93.10% (54/58) | 17 → 8 |

미매칭은 16개 에피소드 중 9개에서 감소했다. 스텝 제한에 도달한 한 에피소드가
전체 감소 563개 중 388개를 차지했다. 성공한 에피소드만 집계해도 175개
감소했다. 전체 삽입에서 Box 외곽과 단일 프레임 Hole 림의 3 mm 매칭 수,
미매칭 Box 수는 각 백엔드의 필터 적용 전후 모든 프레임에서 같았다. Box
외곽은 CPU·CUDA에서 1,618/1,716, MPS에서 1,617/1,716이었다.
Hole 림은 모두 124/1,146이었다. 미매칭 Box는 CPU 505개, MPS 503개,
CUDA 502개였다.

[새 삽입 프레임 매니페스트](../outputs/peg_box_cut_holdout/938390740331455b8187d1e0cedfeab8/frames.json) ·
[CPU](../outputs/peg_box_cut_holdout/938390740331455b8187d1e0cedfeab8/cpu_scores.json) ·
[MPS](../outputs/peg_box_cut_holdout/938390740331455b8187d1e0cedfeab8/mps_scores.json) ·
[CUDA](../outputs/peg_box_cut_holdout/938390740331455b8187d1e0cedfeab8/cuda_scores.json) ·
[실행 설정](../outputs/peg_insertion_precision/holdout.yaml)

```bash
WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=outputs/peg_insertion_precision .venv/bin/python main.py \
  --config outputs/peg_insertion_precision/holdout.yaml
```

## 검출 실행 최적화

### 기본 실행과 비교 옵션

`detect()`와 `detect_tensors()`는 최신 최적화 메인을 사용한다. 장치는
CUDA, MPS, Torch CPU 순으로 자동 선택한다. CUDA 기본 경로는 2픽셀 격자
평면 피팅과 확대 검증 컴파일을 사용한다.

| 실행 경로 | 사용 방법 |
| --- | --- |
| 최신 GPU 기본 경로 | `TorchCornerDetector.main()` |
| 이전 CUDA 기본 경로 | `TorchCornerDetector.main(device="cuda", plane_fit_method="ransac", compile_cuda_validation=False)` |
| 기존 실행 경로로 비교 | `TorchCornerDetector.main(device="mps", optimized=False)` |
| CUDA 경계 검증 컴파일 | `TorchCornerDetector.main(device="cuda", compile_cuda_edges=True)` |

`optimized=False`는 기존 실행 경로에서 현재 검출 규칙과 보정값을 사용한다.
`reject_peg_box_cut=False`로는 Peg 가림 필터를 별도로 비활성화할 수 있다.

### 적용한 변경

- **GPU 전처리와 기하 계산.** 깊이 역투영, 점군 추출, 평면 피팅과
  마스크 생성, Peg 프리즘과 뒤쪽 끝면 검증을 GPU에서 실행한다. 호스트
  RGB-D가 들어오면 OpenCV에서 필요한 입력을 재사용해 전송을 줄인다.
  [GPU 상주 처리와 단계별 계측](#gpu-resident-frame-geometry)
- **기본 GPU 경로 최적화.** Box RGB 특징은 마스크 주변에서 탐색하고,
  서브픽셀 보정은 전체 영상 좌표에서 한다. 평면 윤곽을 재사용하고, MPS
  평면 피팅은 동일한 표본과 투표 기준으로 병렬 집계한 뒤 재피팅을 묶어
  실행한다.
- **후보 처리 추가 최적화.** 윤곽 꼭짓점 인덱스를 재사용하고 교차점을
  일괄 계산한다. Box RGB 후보의 Hole 판정은 병합 때 읽고, 좌표 보정은
  병합에서 정한 물체별 인덱스를 사용한다. Peg 프리즘 기저는 Box 가림
  검사에서 재사용하며, 평면별 경계 계산을 묶은 뒤 기존 순서대로 적용한다.
- **CUDA 컴파일 옵션.** 윤곽과 Box 후보의 두 경계를 검증하는 Torch 함수에
  `torch.compile(dynamic=True, fullgraph=True)`를 적용한다. 첫 호출은
  최초 실행에서 약 19.8초, 뒤이은 실행에서 약 4.9초가 걸렸다. CUDA
  롤아웃에 사용할 때는 대표 프레임으로 워밍업한다.

### 동일 프레임 전후 비교

모든 시간은 ms/프레임이다. 입력 전송부터 CPU 딕셔너리 출력까지 포함하며
파일 로딩과 채점을 제외한다. 각 행은 같은 장비와 프레임에서 실행 순서를
교대로 바꿔 얻은 쌍별 비교다. 실행마다 기준 시간도 다시 측정했다. CUDA
컴파일 행의 정상 실행 시간에서는 첫 컴파일 호출을 제외했다.

| 변경 | 장비와 입력 | 프레임 | 평균 전 → 후 | p95 전 → 후 |
| --- | --- | ---: | ---: | ---: |
| 기본 GPU 경로 최적화 | M4 Pro · MPS · 독립 삽입 | 420 | 36.90 → 33.09 | 49.08 → 42.60 |
| 기본 GPU 경로 최적화 | M4 Pro · MPS · 일반 궤적 | 288 | 39.12 → 35.37 | 57.68 → 48.50 |
| 기본 GPU 경로 최적화 | RTX 3090 · CUDA · 독립 삽입 | 420 | 67.66 → 62.73 | 73.85 → 69.28 |
| 후보 처리 추가 최적화 | M4 Pro · MPS · 독립 삽입 | 420 | 39.37 → 37.38 | 54.31 → 52.54 |
| 후보 처리 추가 최적화 | M4 Pro · MPS · 일반 궤적 | 288 | 44.34 → 41.18 | 58.69 → 56.63 |
| 후보 처리 추가 최적화 | RTX 3090 · CUDA · 독립 삽입 | 420 | 67.52 → 64.46 | 92.09 → 87.98 |
| CUDA 경계 검증 컴파일 | RTX 3090 · CUDA · 독립 삽입 | 420 | 59.51 → 52.94 | 64.47 → 58.23 |

기본 경로와 후보 처리 최적화는 각 비교의 모든 프레임에서 좌표, 출력 순서,
물체 ID, 마스크, 메타데이터가 일치했다. Peg·Box 외곽·Hole 림의 3 mm
매칭과 미매칭 수도 프레임별로 같았다. CUDA 컴파일 비교에서도 420프레임
전체의 검출 출력과 마스크가 같았다.

앞선 삽입 416프레임의 **RGB-D 전처리 소계**는 MPS 17.71 ms와 CUDA
16.69 ms였다. [CPU·GPU 단계별 계측](#현재-검출기-cpugpu-처리-시간)

### MPS RANSAC 평면 종료 판정

최적화된 Metal 경로는 피팅된 평면의 inlier 수를 별도로 CPU에 복사하는
대신, 다음 평면을 찾기 위해 이미 만드는 잔여 점군의 길이 차를 사용한다.
마지막 재피팅이 집계한 inlier와 잔여 점군에서 제외되는 점은 같은 마스크를
사용한다. 이 변경은 `graph`의 `b81d0e3`에서 가져왔다.

2026-11-24 Apple M4 Pro, PyTorch 2.14.1에서 `f1947b1`을 기준으로
Expert·Random·Expert→Random 48개 저장 에피소드마다 등간격 5프레임을
재생했다. Torch와 OpenCV는 CPU 스레드를 하나씩 사용했다. 두 구현을
워밍업한 뒤 프레임마다 실행 순서를 교대했다. 전체 시간에는 NumPy 입력을
받아 CPU 딕셔너리를 내보내기까지가 포함되며 파일 로딩은 제외했다. 평면
피팅은 에피소드별 가운데 프레임의 별도 호출에서 MPS를 동기화해 측정했다.

| 구간 | 프레임 | 평균 전 → 후 (ms) | p50 전 → 후 (ms) | p95 전 → 후 (ms) |
| --- | ---: | ---: | ---: | ---: |
| 전체 검출 | 240 | 30.36 → 29.06 | 28.08 → 26.91 | 41.43 → 41.01 |
| 평면 피팅 합계 | 48 | 5.11 → 4.06 | 4.97 → 4.09 | 6.98 → 4.89 |

NumPy 240프레임과 MPS 상주 48프레임에서 픽셀·카메라·월드 좌표,
물체 ID와 순서, 마스크, 메타데이터, 진단 정보가 일치했다. CUDA는
이 호스트에서 실행하지 않았다. [입력 매니페스트](../outputs/corner_detection_holdout/frames/manifest.json)

### MPS 단일 프레임 10 ms 목표 진단

2026-11-24 M4 Pro, PyTorch 2.14.1, OpenCV 5.0.0에서 위 48개 에피소드의
512×512 RGB-D를 각각 등간격 4프레임씩 재생했다. 워밍업 후 동일 프레임에서
모드 실행 순서를 돌려가며 측정했다. 일반 지연은 CPU 딕셔너리 출력까지이며,
입력 파일 읽기, 채점, 시뮬레이터, ID 매칭을 제외했다. 두 비교는 각각의
기본 검출 시간을 다시 쟀다. 3 mm 평가는 가시 GT와 같은 물체 ID 내에서
일대일 대응했다. 아래 분모는 Peg 1,100, Box 외곽 844, Hole 림 694다.

| 경로 | 평균 / p95 (ms) | Peg / Box 외곽 / Hole 림 3 mm 매칭 |
| --- | ---: | ---: |
| 기본 MPS, NumPy 입력 | 23.95 / 28.18 | 1,058 / 760 / 573 |
| 기본 MPS, GPU 상주 입력 | 22.47 / 26.96 | 1,058 / 760 / 573 |
| 기본 검출에서 평면 피팅만 사전 계산 | 20.18 / 23.92 | 1,058 / 760 / 573 |
| 윤곽 후보 검증 생략 | 20.94 / 24.61 | 1,028 / 668 / 560 |
| Box 외곽 후보 검증 생략 | 20.66 / 24.25 | 1,058 / 595 / 573 |
| 후보 검증 일부 생략 및 RANSAC 40회 | 18.39 / 22.23 | 993 / 669 / 560 |
| 기본 검출, RANSAC 40회 | 31.38 / 37.96 | 1,058 / 760 / 573 |
| Torch 2픽셀 격자 피팅, MPS | 35.67 / 46.73 | 959 / 760 / 517 |
| Torch CPU | 88.83 / 113.39 | 1,058 / 763 / 573 |

사전 계산 진단은 같은 프레임에서 얻은 기본 경로의 평면을 재사용하며,
192프레임의 전체 검출 결과가 일치했다. 실제 운용에서 이전 프레임의
평면을 재사용하는 방법과는 조건이 다르다. 후보 검증을 생략한 행은
병목을 확인하기 위한 진단이다.

별도 동기화 호출 48프레임에서 전처리·평면 준비는 평균 10.31 ms였으며
그 안에 평면 피팅 4.13 ms, CPU 재질 마스크 1.58 ms, 평면 정보 회수
0.90 ms가 포함된다. 윤곽 후보 5.35 ms, Box 외곽 3.99 ms, Hole 림
1.76 ms, Peg 뒤쪽 복구 1.85 ms, 병합 1.27 ms였다. 구간 타이머는 GPU
작업 사이에 대기를 추가하므로 이 값을 일반 호출의 지연과 합산하지 않는다.

추가 24프레임에서 장치 경계를 계측했다. 프레임당 사용자 정의 Metal
커널 호출이 평균 34.6회였고, 이 중 RANSAC 투표·선택·재피팅이 19.4회였다.
명시적 MPS→CPU 읽기는 13회, RANSAC의 불리언 인덱싱은 6.46회였다.
후보 병합의 첫 읽기와 불리언 인덱싱에서 관측된 호출 시간은 각각 1.38,
1.29 ms/프레임이다. 읽기에는 앞서 제출된 GPU 작업을 기다리는 시간도
포함된다. 이 호스트에서 `xctrace`를 사용할 수 없어 GPU 점유율은 계측하지
않았다.

별도 Metal 시제품은 모든 가설을 한 번에 채점하고, 선택한 평면들을 별도
스레드그룹에서 재피팅했다. 단일 투표 집합에서 큰 면이 표본을 지배할 수
있고, 반복 제거 전략이 빠져 있어 기본 RANSAC과 검출 결과가 달라진다.
Box 극단 분위수의 대안은 전체 정렬 대신 양쪽 꼬리의 `topk`와 선형 보간을
사용했다.

| 두 번째 비교 | 전체 평균 / p95 (ms) | 평면 피팅 평균 (ms) | Peg / Box 외곽 / Hole 림 3 mm 매칭 |
| --- | ---: | ---: | ---: |
| 기본 MPS | 23.66 / 25.69 | 4.01 | 1,058 / 760 / 573 |
| Metal 병렬 피팅, 가설 160개 | 22.54 / 26.79 | 1.59 | 512 / 751 / 468 |
| Metal 병렬 피팅, 가설 320개 | 21.51 / 25.53 | 1.73 | 517 / 762 / 488 |
| Box 분위수에 `topk` 사용 | 24.92 / 25.47 | — | 1,058 / 760 / 573 |

`topk`는 192프레임의 좌표, 순서, 마스크, 메타데이터 및 진단을
보존했지만 평균 지연은 줄지 않았다. 시제품은 진단 코드에서만 실행했다.
CUDA와 closed-loop 평가는 이 비교에 포함하지 않았다.

10 ms 단일 프레임 목표로는 다음 GPU 경로를 우선 검증할 수 있다.

1. **전처리와 평면 준비.** RGB-D, 재질 마스크, 점군을 GPU에 유지하고 Peg와
   Box를 한 번에 처리한다. 유효 점군과 평면은 최대 용량 버퍼 및 GPU
   카운터로 넘긴다. 기존 순차 inlier 제거를 유지하되 다음 패스를 GPU에서
   이어가고, 작은 Peg 평면의 탐색을 큰 면의 점수와 분리한다. 정리된 깊이
   영상에서는 타일별 법선·평면 투표도 후보 생성에 쓸 수 있다.
2. **후보 생성과 검증.** 윤곽 선분, Box 깊이·에지, Peg와 Hole의 기하
   후보를 `(물체, 평면, 후보)` 단위로 묶는다. 여러 후보가 공유하는 깊이
   패치와 평면 마스크를 재사용하고, 가시성·경계 검증은 동일한 GPU 버퍼에
   기록한다. Box 외곽 검증을 통째로 건너뛰면 위 표처럼 외곽 매칭이 줄어든다.
   OpenCV 윤곽과 Hole 피팅의 출력을 보존해야 한다면 CPU 작업을 독립 GPU
   검증과 겹치는 경로부터 비교한다.
3. **출력.** 후보 플래그·좌표·필요한 ID를 한 버퍼로 묶어 CPU가 필요할 때
   한 번 읽는다. GPU 소비 경로에서는 중간 좌표 복사를 피한다. 입력부터
   최종 소비자가 읽을 수 있을 때까지 측정해 비동기 제출 시간과 실제
   단일 프레임 지연을 구분한다.

[A/B 계측](../outputs/corner_latency_investigation/results.json) ·
[Metal 시제품 비교](../outputs/corner_latency_investigation/parallel_results.json) ·
[장치 경계 계측](../outputs/corner_latency_investigation/boundaries.json) ·
[Apple 간접 디스패치](https://developer.apple.com/documentation/metal/specifying-drawing-and-dispatch-arguments-indirectly) ·
[Apple GPU 타임라인](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-a-visual-timeline)

저장 RGB-D 매니페스트가 있는 MPS 호스트에서는 저장소 루트에서 재생한다.

```bash
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python outputs/corner_latency_investigation/benchmark.py
.venv/bin/python outputs/corner_latency_investigation/benchmark_parallel.py
.venv/bin/python outputs/corner_latency_investigation/trace_boundaries.py
```

### 꼭짓점 직접 LK 추적과 출력 보존 비교

2026-11-24 M4 Pro에서 Main 카메라의 저장된 Expert 4개와
Random-after-lift 4개 에피소드의 연속 782프레임을 재생했다. 전체 검출
프레임에서 얻은 꼭짓점 픽셀을 시작점으로 양방향 pyramidal LK를 적용했다.
중간 프레임은 현재 RGB 재질 마스크와 5×5 깊이 이웃을 확인해 3D로
복원했다. 세 프레임 간격의 보수적 변형은 이전 Peg·Box 후보를 각각 85%
이상 유지하지 못하면 현재 프레임을 다시 검출한다. GT는 채점에만 썼다.

| 경로 | 평균 / p95 (ms) | Peg / Box 외곽 / Hole 림 3 mm 매칭 |
| --- | ---: | ---: |
| 매 프레임 기본 검출 | 30.96 / 39.06 | 4,148 / 4,105 / 2,625 |
| 검출 간격 2, 중간 프레임 LK | 16.34 / 37.57 | 3,545 / 4,105 / 2,607 |
| 검출 간격 3, 중간 프레임 LK | 11.25 / 36.17 | 3,241 / 4,101 / 2,596 |
| 검출 간격 3, 후보 손실 시 재검출 | 12.41 / 37.41 | 3,368 / 4,103 / 2,595 |

매칭은 동일 물체의 전체 GT 꼭짓점과 3 mm 이내 일대일 대응으로 계산했다.
LK 구간은 평균 0.88–0.98 ms였다. 표의 하이브리드 시간은 측정한 LK 및
초기화 시간에 해당 프레임의 검출 시간을 더해 산출했다. 보수적 변형은
69프레임에서 추가 재검출했다. Box 매칭 총계가 같아도 프레임별 매칭은
달랐다. 간격 2의 LK 프레임 중 기본 검출이 매칭한 Box 외곽 2,047개 가운데
2,039개가 동일 GT ID를 유지했다. 이 실험은 연속 영상 재생이며 closed-loop
제어 성공률과 moving-hole 정확도는 검증 범위 밖이다.

[LK 비교 결과](../outputs/corner_latency_investigation/corner_lk_results.json) ·
[재생 스크립트](../outputs/corner_latency_investigation/benchmark_corner_lk.py) ·
[원본 프레임 매니페스트](../outputs/main_camera_held_out/raw/manifest.json)

같은 날, MPS 검출기에 NumPy RGB-D를 넣는 경로의 재질 연결 성분 처리를
색상 마스크의 foreground를 감싸는 사각형으로 줄였다. 사각형 밖은 모두
배경이므로 8방향 연결 성분과 픽셀 수 기준은 같다. 연산한 마스크는 원래
영상 크기로 되돌린다. 앞의 192프레임 A/B 조건에서 평균 **29.69 → 27.89 ms**,
p95 **37.48 → 35.26 ms**였다. 192프레임 모두 좌표, 후보 순서, 마스크,
메타데이터와 진단 정보가 일치했다. Box 외곽의 사각형 계산도 따로 비교했으나
재질 마스크 변경 위에 추가했을 때의 개선 방향이 두 실행에서 달라 진단으로
남겼다. [출력 보존 A/B](../outputs/corner_latency_investigation/exact_results.json) ·
[A/B 스크립트](../outputs/corner_latency_investigation/benchmark_exact.py)

### MPS 후보 전송과 평면 마스크 회수 일괄 처리

2026-11-24에 기존 RGB-D 검출 규칙으로 MPS 입력과 회수 경계를 줄였다.
RANSAC의 결정적 표본과 세 점 가설 인덱스를 한 번에 올린다. 윤곽 경계,
세 평면 교차점, 관측 면 외곽, Box RGB 후보는 검증 호출마다 입력 배열을
한 번씩 올리고 Metal 커널은 같은 타입의 텐서 뷰를 읽는다. 평면 계수,
유효 플래그, 비트 압축한 평면 마스크는 한 버퍼로 회수한다. CPU에서
지지 마스크를 이미 가지고 있으면 재사용한다. 평면 라벨링과 팽창은
16×16 타일과 한 픽셀 겹침 영역으로 한 번에 실행하고, 두 침식 결과를
다음 커널에서 함께 계산한다. 이 경로의 마스크 처리 호출은 평면 준비당
4회에서 2회로 줄었다. 별도 24프레임 계측에서 프레임당 명시적 MPS→CPU
읽기는 13회에서 9회, 자체 Metal 커널 호출은 평균 34.6회에서 32.6회로
줄었다. [변경 후 장치 경계 계측](../outputs/corner_latency_investigation/batched_boundaries.json)

M4 Pro, PyTorch 2.14.1에서 체크섬을 검증한 기존 48개 에피소드 전체를
재생했다. 프레임별 전후 실행 순서를 교대했고 파일 읽기와 첫 호출 워밍업을
제외했다. 시간은 검출 호출에서 CPU 딕셔너리 출력까지다.

| 입력 | 프레임 | 평균 전 → 후 (ms) | p50 전 → 후 (ms) | p95 전 → 후 (ms) |
| --- | ---: | ---: | ---: | ---: |
| NumPy RGB-D | 2,073 | 24.19 → 21.01 | 22.42 → 18.94 | 34.12 → 31.18 |
| MPS 상주 RGB-D | 48 | 22.40 → 18.73 | 21.90 → 18.25 | 26.53 → 22.18 |

검출 개수와 순서, 물체 ID, 카메라·월드·픽셀 좌표의 float32 비트,
마스크, 메타데이터와 진단 정보가 모든 프레임에서 일치했다. NumPy 재생의
2,073프레임 중 10 ms 이내 호출은 0건이었다. 후보 병합에서 플래그와 좌표를
한 번에 읽는 시제품은 192프레임의 출력을 보존했으나 반복 비교의 평균
이득이 약 0.1 ms로 작아 진단 코드로 남겼다. 이 재생에서는 CUDA와 폐루프
제어를 실행하지 않았다.

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python outputs/corner_latency_investigation/verify_batched_mps.py --all
```

[전체 프레임 측정](../outputs/corner_latency_investigation/batched_mps_full.json) ·
[후보 회수 시제품](../outputs/corner_latency_investigation/fusion_isolation_results.json) ·
[평면 마스크 회수 비교](../outputs/corner_latency_investigation/plane_readback_results.json) ·
[마스크 커널 병합 비교](../outputs/corner_latency_investigation/morphology_results.json)

### MPS 프레임 단위 후보 검증

최적화된 MPS 검출기는 윤곽 경계, 세 평면 교차점, 관측 면 외곽, Box의
RGB·큐브 후보를 순서대로 모아 하나의 Metal 커널로 검사한다. 각 후보는
자신의 평면, 지지 마스크, 기본·복구 플래그를 참조한다. 후보별 검증식은
개별 커널과 공유한다. 커널 출력을 원래 후보 묶음 순서대로 등록하므로
겹치는 후보에 적용하는 기존 병합 우선순위를 유지한다. 별도 24프레임
계측에서 이 검증 단계의 Metal 호출은 프레임당 평균 6.25회에서 1회로
줄었다. [커널 호출 계측](../outputs/corner_latency_investigation/frame_batch_boundaries.json)

192프레임에서 Metal 기본 스레드그룹 크기와 32·64·128을 비교했다.
128에서 같은 출력을 유지하며 평균 17.59 → 17.10 ms,
p95 20.18 → 19.27 ms를 기록해 이 크기를 적용했다.
[스레드그룹 비교](../outputs/corner_latency_investigation/frame_batch_workgroups.json)

2026-11-24 M4 Pro에서 직전 MPS 전송 최적화 소스를 기준으로 한 쌍별
비교다. 파일 읽기와 워밍업을 제외한 검출 호출부터 CPU 딕셔너리 출력까지
측정했다. 단위는 ms/프레임이다.

| 입력 | 프레임 | 평균 전 → 후 | p50 전 → 후 | p95 전 → 후 |
| --- | ---: | ---: | ---: | ---: |
| NumPy RGB-D | 2,073 | 24.73 → 23.22 | 22.89 → 21.31 | 36.41 → 35.13 |
| MPS 상주 RGB-D | 48 | 22.22 → 20.73 | 20.79 → 18.92 | 31.42 → 28.01 |
| NumPy RGB-D, 사선 이동 Hole 사례 | 251 | 25.22 → 23.94 | 24.28 → 23.16 | 30.31 → 28.94 |

모든 재생 프레임에서 후보 수와 순서, 물체 ID, 픽셀·카메라·월드 좌표의
float32 비트, 마스크, 메타데이터, 진단 정보가 일치했다. CUDA와 폐루프
제어는 이 비교에서 실행하지 않았다. 이동 Hole은 체크섬을 확인한
`case_0146_521142120`의 RGB-D를 재생했다.

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python outputs/corner_latency_investigation/verify_frame_batch.py --all
```

[전체 프레임 측정](../outputs/corner_latency_investigation/frame_batch_full.json) ·
[이동 Hole 측정](../outputs/corner_latency_investigation/frame_batch_moving_case.json) ·
[이동 Hole 재생 스크립트](../outputs/corner_latency_investigation/verify_moving_case_frame_batch.py)

### MPS Box 외곽 분위수와 Hole 성분 처리

Box에서 기존 회전 좌표의 0.1%와 99.9% 분위수를 계산한다. Metal 커널은
좌표 열을 정렬하지 않고 각 분위수 보간에 필요한 인접 순위 두 개만 찾는다.
순위 계산은 float32를 사용하며, 보간은 설치된 PyTorch MPS의 텐서 가중치
`lerp`와 같은 FMA를 사용한다. 0, 비정규 수 또는 비유한 경계가 나오면
기존 `torch.quantile`을 호출한다. CPU 분할 시제품은 일반 프레임에서는
출력을 맞췄으나 극단 입력에서 MPS와 보간 반올림이 달랐다.
[PyTorch 2.14.1 MPS 보간 소스](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/kernels/BinaryKernel.metal#L664-L688)

Hole 림은 연결 성분의 경계 상자와 1픽셀 여유 영역에서 경계 중점을 구하고,
깊이 지지는 경계 상자 내부의 성분 픽셀만 순서대로 읽는다. 기존 4방향
연결 성분 판정과 경계 중점, 사각형 피팅, 깊이 게이트를 유지한다.

2026-11-24 M4 Pro에서 프레임 단위 후보 검증을 적용한 직전 검출기를
기준으로 비교했다. 한 프레임씩 실행 순서를 교대하고 CPU 딕셔너리 출력까지
측정했다. 파일 읽기와 워밍업은 제외했다.

| 입력 | 프레임 | 평균 전 → 후 (ms) | p95 전 → 후 (ms) |
| --- | ---: | ---: | ---: |
| NumPy RGB-D | 2,073 | 20.65 → 19.76 | 31.77 → 30.59 |
| MPS 상주 RGB-D | 48 | 18.50 → 17.30 | 29.05 → 26.21 |
| NumPy RGB-D, 사선 이동 Hole 사례 | 251 | 20.48 → 19.10 | 30.08 → 26.97 |

모든 프레임에서 검출 개수·순서, 물체 ID, 픽셀·카메라·월드 좌표의 float32
비트, 마스크, 메타데이터와 진단이 일치했다. CUDA와 폐루프 제어는 이
비교에서 실행하지 않았다.

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python outputs/corner_latency_investigation/verify_box_bounds.py --all
```

[전체 프레임 측정](../outputs/corner_latency_investigation/box_bounds_full.json) ·
[사선 이동 Hole 측정](../outputs/corner_latency_investigation/box_bounds_moving_case.json) ·
[Box 분위수 단독 비교](../outputs/corner_latency_investigation/box_bounds_only_full.json) ·
[Metal 선택 시제품](../outputs/corner_latency_investigation/box_fused_bounds_probe.json) ·
[Hole 성분 범위 비교](../outputs/corner_latency_investigation/hole_component_bounds_probe.json)

### 원본과 계측 기록

- 기본 GPU 경로 최적화 [MPS 삽입](../outputs/corner_detector_optimization/mps_holdout_final.json) · [MPS 일반 궤적](../outputs/corner_detector_optimization/mps_trajectory.json) · [CUDA 삽입](../outputs/corner_detector_optimization/cuda_holdout_final.json) · [비교 기준 소스](../outputs/corner_detector_optimization/original/corner_detection/detector.py)
- 후보 처리 추가 최적화 [MPS 삽입](../outputs/corner_detector_refinement/mps_final.json) · [MPS 일반 궤적](../outputs/corner_detector_refinement/mps_trajectory.json) · [CUDA 삽입](../outputs/corner_detector_refinement/cuda_final.json) · [비교 기준 소스](../outputs/corner_detector_refinement/original/corner_detection/detector.py)
- CUDA 컴파일 [420프레임 비교](../outputs/corner_detector_cuda_compile/full.json) · [첫 실행](../outputs/corner_detector_cuda_compile/probe.json)

### Vast.ai CUDA 구간별 기준 시간

2026-11-15에 커밋 `6d1a06d4aa243ab6e1cd38456dbfc5c66c65f7af`의
`TorchCornerDetector.main(device="cuda:0")`을 Vast.ai 인스턴스
`60669483`에서 측정했다. RTX 3090, Core i7-9700K, NVIDIA 드라이버
580.126.09, PyTorch 2.14.1+cu130, OpenCV 5.0.0을 사용했다. Torch와
OpenCV는 각각 CPU 스레드 1개로 고정했다. 경계 검증 컴파일 옵션은 기본값인
꺼짐으로 두었다.

독립 삽입 16개 에피소드의 기존 512×512 RGB-D 420프레임을 체크섬 검증 후
재생했다. 대표 프레임 16개로 워밍업한 뒤 입력 경로와 계측 방식의 실행 순서를
프레임마다 회전했다. NumPy 입력은 검출 호출 안의 전송을 포함한다. CUDA 상주
입력은 RGB, 깊이, 카메라 보정값을 호출 전에 올렸다. 파일 읽기, 시뮬레이션,
렌더링, 채점은 측정 범위에서 제외했다.

전체 지연은 구간 타이머를 끈 호출에서 측정했다. 단위는 ms/프레임이고 각
셀은 평균 / p50 / p95다.

| 출력 범위 | NumPy 입력 | CUDA 상주 입력 |
| --- | ---: | ---: |
| 검출 호출 → CUDA 텐서 | 64.88 / 61.78 / 88.43 | 64.19 / 61.31 / 87.70 |
| CPU 딕셔너리 변환만 | 0.27 / 0.26 / 0.33 | 0.27 / 0.25 / 0.32 |
| 검출 호출 → CPU 딕셔너리 | 65.15 / 62.06 / 88.75 | 64.46 / 61.57 / 88.07 |

구간은 별도 호출에서 `utils.timer.Timer`로 시작과 종료에 CUDA를
동기화해 측정했다. CPU 작업과 전송 대기를 포함한다. 들여쓴 행은 상위
구간에 포함된다. 계측을 켠 CPU 딕셔너리 출력까지의 평균은 NumPy
66.13 ms, CUDA 상주 65.21 ms였다.

| 구간 | NumPy 평균 / p95 | CUDA 상주 평균 / p95 |
| --- | ---: | ---: |
| **RGB-D 전처리 및 평면 준비** | **20.01 / 25.86** | **18.93 / 25.25** |
| ↳ 입력 깊이·보정값 준비 | 0.41 / 0.65 | 0.04 / 0.10 |
| ↳ 깊이 역투영 | 0.23 / 0.32 | 0.23 / 0.34 |
| ↳ OpenCV용 RGB 호스트 뷰 | 0.01 / 0.01 | 0.23 / 0.24 |
| ↳ RGB 색상 분류 | 0.68 / 0.71 | 0.59 / 0.68 |
| ↳ 색상 마스크 CPU 회수 | — | 0.16 / 0.17 |
| ↳ 재질 마스크 연결 성분 처리 | 3.72 / 3.95 | 3.27 / 3.55 |
| ↳ 마스크 업로드 | 0.35 / 0.39 | 0.28 / 0.30 |
| ↳ 점군 추출 및 평면 피팅 | 12.22 / 17.12 | 12.04 / 17.39 |
| ↳ 영상 평면 마스크 생성 | 0.94 / 1.33 | 0.92 / 1.30 |
| ↳ 평면 정보·마스크 CPU 회수 | 0.65 / 0.92 | 0.69 / 0.85 |
| **윤곽 후보 처리** | **20.61 / 24.83** | **20.66 / 24.95** |
| **Box 외곽** | **15.14 / 18.43** | **15.04 / 18.29** |
| Hole 림 | 3.56 / 4.35 | 3.86 / 4.70 |
| Peg 기본 프리즘 | 1.16 / 1.33 | 1.15 / 1.32 |
| Peg 뒤쪽 끝면 복구 | 2.28 / 2.50 | 2.27 / 2.54 |
| 후보 병합 | 0.80 / 1.31 | 0.76 / 1.31 |
| 좌표 보정·월드 변환 | 0.68 / 1.09 | 0.70 / 1.09 |
| Peg Box 가림 필터 | 1.01 / 1.41 | 1.01 / 1.45 |

CUDA 상주 입력의 윤곽 후보와 Box 외곽을 합하면 35.70 ms로, 구간
계측을 켠 전체 평균의 약 55%다. 전처리 안에서는 점군 추출 및 평면 피팅이
12.04 ms로 가장 길었다.

[프레임별 구간 시간 및 호출 수](../outputs/corner_cuda_timing/timing.json) ·
[재생 스크립트](../outputs/corner_cuda_timing/benchmark.py) ·
[입력 프레임 매니페스트](../outputs/peg_box_cut_holdout/938390740331455b8187d1e0cedfeab8/frames.json)

검출기 의존성을 설치한 CUDA 호스트에서 저장소 루트를 기준으로 실행한다.

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python outputs/corner_cuda_timing/benchmark.py \
  outputs/peg_box_cut_holdout/938390740331455b8187d1e0cedfeab8/frames.json \
  outputs/corner_cuda_timing/timing.json --commit "$(git rev-parse HEAD)"
```

### GPU 상주 RGB의 재질 마스크와 회색조 변환

같은 장치에 상주하는 RGB 텐서를 기본 최적화 경로에 넘기면 8방향 연결
성분 계산과 Box 특징용 RGB→회색조 변환을 GPU에서 실행한다. 연결 성분은
CUDA의 Triton 커널과 MPS의 Metal 커널에서 같은 영역 기준을 적용한다.
Peg와 Box의 기본 마스크는 35픽셀, Box 복구 마스크는 8픽셀 이상인 성분을
유지한다. 깊이도 텐서로 들어오면 기본·복구 Box 마스크가 같은지 나타내는
플래그만 CPU에서 읽는다. OpenCV의 Box 특징 추출에는 1채널 회색조 영상을
전달한다. CUDA 환경에 Triton이 없으면 연결 성분에 기존 OpenCV 경로를 사용한다.

CPU는 OpenCV 윤곽 및 서브픽셀 특징 추출, Hole 림 피팅, 후보 병합을
계속 처리한다. 윤곽 선분 교차점과 게이트의 Torch 이관도 시험했다. CUDA에서
교차점만 이관하자 상주 입력 평균이 59.86 ms에서 60.12 ms로 늘었다.
MPS에서는 교차점과 회색조를 함께 이관한 실험에서 420프레임 중 2프레임의
후보 개수가 달라졌다. 교차점 계산은 기존 경로를 사용한다.

기준 코드는 `bb01149`의 검출기다. 위와 같은 삽입 420프레임의 입력
체크섬을 검증한 뒤 장치와 입력 유형별로 같은 프레임에서 기존·변경 호출
순서를 교대했다. 16개 프레임 워밍업 뒤 결과를 집계했다. 측정은 검출 호출부터
CPU 딕셔너리 변환까지이며, GPU 상주 입력 준비, 파일 I/O, 커널 첫 컴파일,
시뮬레이션, 채점은 측정 구간 밖이다. MPS 비교는 기존 TAPNext++ MPS 작업이
종료된 뒤 실행했다. Torch와 OpenCV 각각 스레드 1개, CUDA 경계 검증
컴파일은 꺼짐으로 고정했다.

| 장치와 입력 | 평균 전 → 후 (ms) | p50 전 → 후 (ms) | p95 전 → 후 (ms) |
| --- | ---: | ---: | ---: |
| Vast.ai RTX 3090 · CUDA 상주 | 63.33 → 59.57 | 62.40 → 58.57 | 70.75 → 66.89 |
| Apple M4 Pro · MPS 상주 | 27.27 → 26.37 | 25.07 → 23.84 | 37.53 → 36.48 |
| RTX 3090 · NumPy 입력 | 64.12 → 64.26 | 63.34 → 63.16 | 72.26 → 71.87 |
| M4 Pro · NumPy 입력 | 30.65 → 30.76 | 30.13 → 30.60 | 41.04 → 41.52 |

두 입력 유형에서 물체 ID, 후보 수와 순서, 픽셀·3D 좌표, 메타데이터,
마스크, 진단 정보가 장치별 420프레임 전체에서 일치했다. MPS에서는 별도
일반 궤적 288프레임에서도 두 입력 유형의 출력이 일치했다. 별도 합성
마스크로 빈 마스크, 포화 마스크, 무작위 밀도, 대각선 연결 성분을 OpenCV와
비교했고, 두 기기에서 경계와 영역 기준 회귀 테스트도 통과했다.

[CUDA 프레임별 결과](../outputs/corner_gpu_migration/cuda_final.json) ·
[MPS 삽입 결과](../outputs/corner_gpu_migration/mps_final.json) ·
[MPS 일반 궤적](../outputs/corner_gpu_migration/mps_general.json) ·
[A/B 스크립트](../outputs/corner_gpu_migration/compare.py)

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python outputs/corner_gpu_migration/compare.py \
  outputs/peg_box_cut_holdout/938390740331455b8187d1e0cedfeab8/frames.json \
  cuda outputs/corner_gpu_migration/cuda_final.json
```

MPS에서는 장치 인수를 `mps`로 바꾼다. 비교 스크립트가 읽는 기준 모듈은
로컬 `outputs/corner_gpu_migration/original/corner_detection/`에 있다.

### CUDA와 MPS의 지연 차이

2026-11-15에 위 삽입 420프레임에서 균등하게 뽑은 96프레임을 동일한
GPU 상주 입력 형식으로 프로파일링했다. GPU 상주 입력 준비와 파일 읽기는
측정 밖이다. CUDA는 Vast.ai RTX 3090 및 Core i7-9700K, MPS는 Apple
M4 Pro에서 실행했다. 두 장치에서 같은 소스의 기본 설정을 사용했다.
MPS에서 동시 TAPNext++ 작업과 겹친 계측은 제외하고 작업 종료 후 다시
측정했다.

상위 구간만 동기화한 계측의 평균은 다음과 같다. 일반 호출에서는 구간
타이머를 끄고 CPU 딕셔너리 출력까지 측정했다. 단위는 ms/프레임이다.

| 구간 | CUDA | MPS |
| --- | ---: | ---: |
| 일반 호출 전체 | 57.45 | 33.22 |
| RGB-D 전처리 및 평면 준비 | 13.72 | 12.46 |
| 윤곽 후보 | 19.72 | 8.27 |
| Box 외곽 | 14.40 | 5.17 |
| Hole 림 | 3.83 | 2.60 |

세부 계측의 윤곽 구간에서는 OpenCV 윤곽 추출보다 경계·교차점·외곽
후보 검증의 차이가 컸다. Box 구간에서도 OpenCV 특징 추출보다
`box_cube`와 `box_rgb`의 차이가 컸다. MPS는 이런 검증과 평면 피팅의
여러 연산을 커스텀 Metal 커널로 묶는다. CUDA eager는 후보의 투영, 패치
샘플링, 기하 검증을 여러 Torch 연산으로 실행한다. 세부 동기화는 CUDA
계측 전체 시간도 늘렸으므로, 세부 항목은 병목 위치 확인에 사용했다.

기본 CUDA 일반 호출 3프레임의 PyTorch 추적에서 프레임당 GPU 커널이
약 3,631개, GPU 복사가 약 193개, CUDA 스트림 동기화 호출이 154개였다.
장치 커널 시간의 합은 8.86 ms, 장치 복사 시간의 합은 0.77 ms였다.
호스트 `cudaLaunchKernel` API 호출 시간의 합은 17.96 ms였다. 추적의
이벤트 시간은 CPU/GPU 동시 실행과 프로파일러 개입 때문에 전체 지연에서
단순 차감할 수 없다. 해당 96프레임의 별도 워밍업 실행에서 경계 검증
컴파일을 켜자 CUDA 일반 호출 평균이 58.09 ms에서 51.36 ms로 줄었다.
이 비교의 컴파일 시간은 포함하지 않았다.

[CUDA 상위 구간](../outputs/corner_backend_profile/cuda_coarse.json) ·
[MPS 상위 구간](../outputs/corner_backend_profile/mps_coarse_idle.json) ·
[CUDA 세부 구간과 추적 집계](../outputs/corner_backend_profile/cuda.json) ·
[MPS 세부 구간](../outputs/corner_backend_profile/mps.json) ·
[CUDA 컴파일 비교](../outputs/corner_backend_profile/cuda_compiled.json) ·
[계측 스크립트](../outputs/corner_backend_profile/measure.py)

### CUDA 검증 컴파일 범위 비교

2026-11-15 Vast.ai RTX 3090에서 PyTorch 2.14.1+cu130과 RANSAC 평면 피팅으로
`compile_cuda_edges=True`와 `compile_cuda_validation=True`를 비교했다.
GPU에 미리 올린 동일한 삽입 420프레임을 체크섬으로 검증하고, 32개 대표
프레임으로 워밍업한 뒤 프레임마다 두 경로의 실행 순서를 교대했다.
전체 지연은 검출 호출부터 CPU 딕셔너리 출력까지다. 입력 업로드와
파일 읽기는 계측 밖이다.

확대 경로는 윤곽 경계와 교차점, 관측 면 외곽, Box RGB 후보와 Box 큐브
검증의 그래프 안에 경계 검증을 포함한다. 최초 시도에서 Box 큐브의
좌표 계산까지 컴파일하자 두 프레임에서 카메라 좌표의 최대 축 차이가
1.92 mm와 2.96 mm였다. 최종 경로는 Box 큐브에서 평면 선택과 RGB
특징에 맞춘 후보 좌표 계산을 기존 Torch 연산으로 수행하고, 그 이후의
깊이·가시성·물리 경계 검증을 묶는다.

| 구간 | 경계만 컴파일 | 검증 범위 확대 |
| --- | ---: | ---: |
| 전체 평균 | 49.01 ms | 35.98 ms |
| 전체 p50 | 48.31 ms | 35.41 ms |
| 전체 p95 | 54.14 ms | 40.72 ms |
| 윤곽 후보 평균 | 16.21 ms | 7.16 ms |
| Box 외곽 평균 | 10.18 ms | 6.25 ms |

윤곽과 Box 구간은 같은 입력 중 48개를 골라 구간 경계에서 장치를
동기화해 별도로 측정했다. 전체 측정 중 새 컴파일 그래프는 생기지
않았다. 앞선 실험의 디스크 캐시가 있는 상태에서 프로세스별 첫 호출은
5.40초와 31.82초였다. 첫 호출과 추가 워밍업 비용은 표에서 제외했다.

420프레임 모두 후보 개수와 순서, 물체 ID, 마스크, 후보 메타데이터 및
진단 정보가 일치했다. 좌표는 비트 단위로는 차이가 있었고, 최대 절대
축 오차는 0.000031픽셀, 카메라·월드 좌표 모두 0.18 µm 이하였다.
최종 변경은 CUDA 선택 옵션으로 제공한다.

[초기 420프레임 결과](../outputs/corner_cuda_validation/full.json) ·
[최종 420프레임 결과](../outputs/corner_cuda_validation/final.json) ·
[A/B 스크립트](../outputs/corner_cuda_validation/compare.py)

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python outputs/corner_cuda_validation/compare.py \
  outputs/peg_box_cut_holdout/938390740331455b8187d1e0cedfeab8/frames.json \
  cuda outputs/corner_cuda_validation/final.json
```

비교 스크립트는 이 변경 직전 검출기 소스의 로컬 스냅샷
`outputs/corner_cuda_validation/original/corner_detection/`을 기준으로 읽는다.

### 픽셀 격자 기반 평면 피팅

`TorchCornerDetector.main(device="cuda", plane_fit_method="grid")`는
RGB 재질 마스크의 유효 깊이에서 격자 기반 평면 후보를 만든다. 2픽셀
간격으로 중앙차분 법선을 구하고, 마스크 경계나 깊이 불연속을 가로지르는
표본을 제외한다. 방향과 물체 중심 기준 오프셋을 양자화해 표본을 묶고
유사 후보를 합친다. 점군을 가장 가까운 후보에 배정한 뒤 그룹별 공분산의
고유분해를 한 번씩 배치로 실행한다. `plane_fit_method="ransac"`을
지정하면 기존 평면 피팅 경로를 사용할 수 있다.

2026-11-15 Vast.ai RTX 3090에서 두 방식에 동일하게
`compile_cuda_validation=True`를 적용했다. GPU 상주 512×512 RGB-D 프레임마다
실행 순서를 교대했다. 전체는 별도 구간 계측 없이 CPU 딕셔너리 출력까지
측정했다. 평면 피팅은 별도 호출에서 CUDA를 동기화해 프레임 내 모든 피팅을
합산했다. 파일 읽기, 업로드와 채점은 두 시간에서 제외했다.

독립 삽입 420프레임에서 평균 평면 피팅은 10.31 → 4.19 ms,
전체 검출은 37.22 → 31.71 ms였다. 동일 물체의 가시 3D 기준점에
3 mm 이내 일대일 매칭을 적용했다. Box 외곽과 Hole 림은 Box 출력에서 함께
매칭했다. Peg는 1,578 → 1,549 / 1,604, Box 외곽은 1,618 → 1,639 /
1,716, Hole 림은 124 → 95 / 1,146이었다. 미매칭 출력은 Peg 410 → 404,
Box 502 → 549였다. [삽입 프레임별 원본](../outputs/grid_plane_benchmark/561db8bc546c443baa9c0f2da795a21f/results.json)

직전 통합 파이프라인의 133프레임(Expert 장면 12615147, 프레임 16–148)에서는
평균 평면 피팅이 12.52 → 4.39 ms, 전체 검출이 39.25 → 31.01 ms였다.
3 mm 매칭은 Peg 694 → 688 / 701, Box 외곽 461 → 459 / 595,
Hole 림 449 → 449 / 485였다. 이 에피소드의 Peg/Box 분할 ID 18/19는
저장된 첫 프레임의 물체 중심을 투영해 확인했다. 두 세트에서 워밍업 이후
추가 컴파일은 없었다. 격자 모드의 전체 검출은 CUDA에서 검증했다.
[파이프라인 프레임별 원본](../outputs/grid_plane_benchmark/3120a4e8fd9244859372aac0a2b1f52b/results.json) ·
[비교 스크립트](../outputs/grid_plane_benchmark/benchmark.py)

```bash
PYTHONPATH=.:outputs/grid_plane_benchmark OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  WANDB_MODE=disabled python main.py --config outputs/grid_plane_benchmark/cuda.yaml
```

다른 호스트에서는 YAML의 `manifest`를 저장 프레임 옆에 있는 매니페스트의
경로로 바꾼다.

#### 격자 간격 비교

2026-11-15 같은 RTX 3090에서 2, 3, 4, 6, 8 px를 비교했다.
법선을 계산하는 이웃 거리는 2 px로 고정했다. 평면 피팅과 전체 검출의
평균은 250 ms를 넘긴 호출이 있던 프레임을 모든 간격에서 공통 제외해
계산했다. 시간 집계는 삽입 418/420, 연속 장면 132/133프레임이며
3 mm 매칭은 모든 프레임을 포함한다.

| 간격 | 삽입 피팅 / 검출 (ms) | 삽입 Peg 매칭 / 1,604 | 연속 장면 피팅 / 검출 (ms) | 연속 장면 Peg 매칭 / 701 |
| ---: | ---: | ---: | ---: | ---: |
| 2 px | 4.48 / 32.73 | 1,549 | 4.66 / 32.95 | 688 |
| 3 px | 4.40 / 29.37 | 1,258 | 4.62 / 32.91 | 675 |
| 4 px | 4.17 / 25.94 | 717 | 4.61 / 31.70 | 638 |
| 6 px | 3.90 / 23.00 | 205 | 4.52 / 28.43 | 501 |
| 8 px | 3.76 / 22.13 | 44 | 4.35 / 27.61 | 394 |

Hole 림은 각 세트 내에서 간격에 따른 매칭 수 변화가 없었고, Box 외곽
재현율 변화는 2 px 대비 0.35%p 이내였다. 삽입 420프레임에서 균등 추출한
60프레임의 CPU 진단에서 Peg 법선 표본 중앙값은 2/3/4 px에서
49/21/12점이었다. Peg 평면 후보가 하나도 없는 프레임은 각각
0/4/27개였다. 기본 격자 간격은 2 px다.

[삽입 420프레임](../outputs/grid_stride_benchmark/ea728f6e981b426692cfd29e90476bca/results.json) ·
[연속 133프레임](../outputs/grid_stride_benchmark/a432ef60acb3488387369f0f7b944ed8/results.json) ·
[실행 설정](../outputs/grid_stride_benchmark/cuda.yaml)

#### 레가시 CUDA PCA5 법선으로 평면 확정

`TorchCornerDetector.main(device="cuda", plane_fit_method="pca_normals")`는
`d222fe2`의 모서리 검출에서 쓰는 CUDA PCA5 커널로 RGB 마스크 안의
국소 법선을 구한다. 유효한 점·법선을 2픽셀 격자 경로와 같은 방향·오프셋
그룹화와 평면별 배치 피팅에 넘긴다. 평면 마스크부터 Peg·Box·Hole의 검증과
병합까지는 기존 검출 흐름을 사용한다. 법선 이외의 레가시 모서리 특징 계산과
그래프 생성은 이 실험에 포함하지 않는다. 평면 피팅 시간에는 PCA5 법선
계산과 그룹화, 정밀 피팅을 포함한다.

2026-11-15 Vast.ai RTX 3090에서 메인 카메라의 동일 GPU 상주 프레임으로
2 px 격자 방식과 비교했다. 양쪽 모두 확대 CUDA 검증 컴파일을 적용했다.
파일 읽기와 입력 업로드, 3 mm 채점을 계측 밖에 두었고, 전체 검출은
CPU 딕셔너리까지 포함했다. 워밍업 이후 새 컴파일 그래프는 없었다.

| 측정 | 연속 133프레임 격자 → PCA5 | 독립 삽입 420프레임 격자 → PCA5 |
| --- | ---: | ---: |
| 평면 피팅 평균 | 4.39 → 4.51 ms | 4.21 → 4.18 ms |
| 전체 평균 | 31.37 → 32.30 ms | 31.83 → 31.06 ms |
| 전체 p95 | 35.60 → 34.12 ms | 36.88 → 33.74 ms |
| Peg 3 mm | 688 → 684 / 701 | 1,549 → 1,569 / 1,604 |
| Box 외곽 3 mm | 459 → 461 / 595 | 1,639 → 1,620 / 1,716 |
| Hole 림 3 mm | 449 → 449 / 485 | 95 → 123 / 1,146 |

삽입 시간 집계에는 격자 경로 357.97 ms가 관측된 프레임 한 개가 포함돼
있다. 그 프레임을 두 경로에서 함께 제외하면 전체 평균은 둘 다 31.05 ms다.
3 mm 매칭은 모든 프레임을 포함한다. PCA5 경로는 CUDA 실험 옵션으로 두었다.

[연속 133프레임](../outputs/pca_plane_benchmark/536a5d3fa9d247e8a421a74ca66b998a/results.json) ·
[삽입 420프레임](../outputs/pca_plane_benchmark/d38f0a18c886408ba8e2049e81805802/results.json) ·
[실행 설정](../outputs/pca_plane_benchmark/cuda.yaml)

#### 피팅된 평면 마스크의 윤곽 관심 영역

평면 피팅과 마스크 생성 뒤, 최적화 경로의 윤곽 추출은 각 평면 마스크의
지원 픽셀을 감싸는 사각형에서 실행한다. OpenCV의 좌표 오프셋으로 윤곽점을
원본 영상 좌표로 복원한다. Vast.ai RTX 3090에서 2 px 격자와 확대 CUDA
검증 컴파일을 사용한 동일 133프레임의 전체 평균은 31.01 → 30.89 ms,
윤곽 후보 구간은 7.38 → 7.38 ms였다. 꼭짓점 좌표, 순서, 후보 메타데이터와
출력 마스크는 133프레임 모두 일치했다. 이 비교의 ROI 적용 범위는 CPU 윤곽
추출이다. GPU 평면 마스크 생성과 Box 검증은 기존 전체 영상 구조를 사용했다.
[프레임별 결과](../outputs/plane_roi_benchmark/98bc6d91d50d48ec9c237ee100290e17/results.json)

#### 국소 윤곽 선분 일괄 추출

최적화 경로는 윤곽에서 국소 후보 검증에 쓰는 양쪽 선분을 간격별로 한꺼번에
추출하고 기존 후보 순서로 조립한다. 2026-11-15 RTX 3090의 동일 메인
카메라 133프레임에서 윤곽 후보 평균 7.52 → 6.84 ms, 전체 평균
31.66 → 30.99 ms, 전체 p95 35.90 → 34.76 ms였다. 두 경로 모두 2 px
격자와 확대 CUDA 검증 컴파일을 사용했다. 프레임별 꼭짓점과 마스크는
모두 일치했다. 이 일괄 추출 경로와 2 px 격자, 확대 검증 컴파일을 메인
CUDA 기본값으로 사용한다.
[프레임별 계측](../outputs/contour_segment_benchmark/a83f8d27a92e4df4af6c6eec381c8775/results.json)

#### Box 경계 픽셀 샘플링

2026-11-15 Vast.ai RTX 3090에서 Expert 궤적 133프레임과 별도 삽입
420프레임을 검증했다. 두 경로 모두 메인 카메라의 동일한 CUDA RGB-D
입력을 사용했고, 프레임별 실행 순서를 교대했다. 전체 시간은 CPU 결과
내보내기까지 포함한다. 구간 시간은 CUDA를 동기화한 별도 실행에서 쟀다.
워밍업 이후 추가 컴파일 그래프는 없었다.

| 구간 | 메인 CUDA 평균 / p95 (ms) | 샘플링 커널 평균 / p95 (ms) |
| --- | ---: | ---: |
| Box 큐브 경계 검증 | 0.507 / 0.585 | 0.322 / 0.359 |
| Box 외곽 전체 | 6.090 / 6.545 | 5.908 / 6.368 |
| 전체 검출 | 31.275 / 35.720 | 30.649 / 35.674 |

전체 검출 중앙값은 30.632 → 30.454 ms였다. 메인 CUDA의 전체 시간에는
290.05 ms 단일 지연이 포함됐다. 동일한 피팅 기하를 넣은 Box 꼭짓점
승인 판정은 553프레임에서 일치했다. 독립적으로 검출기를 실행했을 때
꼭짓점과 마스크는 549프레임에서 정확히 일치했다. 나머지 네 프레임의
카메라 좌표 최대 축 차이는 0.0027 mm 미만이었고, 3 mm 매칭은 모두
유지됐다. 두 경로의 합계는 Peg 2,237/2,305, Box 2,098/2,311,
Hole 544/1,631이었다. `sample_cuda_box_planes=True`로 실험 경로를
선택할 수 있다.
[프레임별 계측](../outputs/box_plane_sampling_benchmark/0e7b1ed08a424749a7f8781a2a47c779/results.json)

마스크 안의 표본만 모으는 별도 실험에서는 평면 피팅이 0.17–0.18 ms
줄었다. 삽입 420프레임에서 3 mm 집계는 프레임별로 같았으나 3프레임의
원시 출력이 달라 기본 구현에 반영하지 않았다.
[삽입 계측](../outputs/compact_grid_benchmark/e2abc3343958491b902be618aa72d5b1/results.json)

## 새 궤적에서 CUDA와 MPS 평가

같은 초기 상태에서 세 정책을 각각 실행한다. Expert는 GT 기반
`ScriptedPegPolicy`를 사용한다. Random은 첫 스텝부터 EE 이동·회전의 정규화
성분을 각각 `[-0.5, 0.5]`에서 균등 추출하고 15스텝마다 그리퍼 열기 또는
닫기를 균등 추출한다. Expert→Random은 해당 시드의 Expert 길이가
`T`일 때 `[1, T-1]`의 정수 스텝을 균등 추출해 같은 Random 정책으로 전환한다.
수집기는 세 정책의 초기 GT, RGB-D와 분할 해시, 혼합 정책의 Expert prefix
action을 확인한다. 초기 관측, 5스텝 간격 관측, 마지막 관측과 전환 직전·전환
스텝·직후 관측을 저장한다. 에피소드 상한은 250스텝이다.

수집에는 Python 3.11, 루트 `requirements.txt`, 혼합 궤적 모듈이 불러오는
`matplotlib`, CPU PhysX와 SAPIEN 렌더러가 필요하다. 평가에는
`visual/corner_detection/requirements.txt`, PyYAML과 해당 가속기를
지원하는 PyTorch가 필요하다. 수집한 `outputs/corner_detection_holdout/frames/`를
그대로 다른 호스트에 복사한 뒤 두 평가 설정을 각각 실행한다.

```bash
WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python main.py --config visual/configs/corner_detection_holdout_collection.yaml
WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python main.py --config visual/configs/corner_detection_holdout_mps.yaml
WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python main.py --config visual/configs/corner_detection_holdout_cuda.yaml
```

양쪽 평가에서 같은 SHA-256의 512×512 Main RGB-D와 캘리브레이션을 가속기
메모리에 올리고 `TorchCornerDetector.main()`을 사용한다. 분할과 GT는
채점에만 사용한다. 물체 분할의 7×7 픽셀 근방에 기준 꼭짓점과 6 mm 이하로
떨어진 깊이 점이 있을 때 가시 기준점으로 집계한다. Peg와 Box 출력을
물체별로 3 mm 및 5 mm 이내에서 일대일 매칭하며, Box 매칭에는 외곽과 Hole
림 기준점이 함께 들어간다.

각 에피소드에서 골고루 고른 3프레임으로 먼저 워밍업한 뒤 모든 저장
프레임을 채점한다. 검출 시간은 GPU 동기화까지 포함한다. CPU 출력까지의
시간은 검출과 `as_dict()`의 꼭짓점 및 마스크 내보내기를 합친 값이다.
렌더링, 파일 읽기, GPU 입력 전송, 채점은 계측 범위 밖이다.

2026-11-15, `graph` 작업 트리에서 실행했다. 측정에 사용한 검출기 소스는
커밋 `af4dca2`와 해시가 일치한다. 새 리셋 시드 16개는 저장소 Git 이력과 접근 가능한 과거
실험·데이터 매니페스트의 기록에서 일치 항목이 없었다. CPU PhysX와
SAPIEN/MoltenVK에서 총 48개 궤적을 생성해 2,073프레임을 두 장치에서
동일하게 재생했다. 혼합 정책의 전환 시점은 12–115스텝이었고,
첫 Random action이 적용되기 전까지의 관측을 전환 전으로 집계했다.
Expert는 16/16회 성공했다. 순수 Random은 Peg가 초기 위치에서 10 mm
이상 이동한 궤적이 0/16개였고, 혼합 정책은 13/16개였다.

| 정책 구간 | 프레임 | CUDA / MPS Peg | CUDA / MPS Box 외곽 | CUDA / MPS Hole 림 |
| --- | ---: | ---: | ---: | ---: |
| Expert | 402 | 87.9% / 94.7% | 91.5% / 91.5% | 67.8% / 76.4% |
| Random | 816 | 87.4% / 97.1% | 89.4% / 89.5% | 75.5% / 80.9% |
| Expert→Random 전체 | 855 | 90.0% / 95.6% | 91.4% / 91.4% | 76.2% / 84.4% |
| 전환 전 | 209 | 86.6% / 93.3% | 92.2% / 92.2% | 70.9% / 81.6% |
| 전환 후 | 646 | 91.1% / 96.4% | 91.1% / 91.1% | 78.0% / 85.3% |
| 전체 | 2,073 | 88.5% / 96.0% | 90.6% / 90.7% | 74.3% / 81.5% |

위 표는 3 mm 재현율이다. 전체 5 mm 재현율은 CUDA / MPS 순으로
Peg 95.0% / 96.1%, Box 외곽 95.2% / 95.2%, Hole 림 85.9% / 94.4%였다.
3 mm 정밀도는 Peg 85.2% / 92.4%, Box 출력(외곽·림 합산)
70.6% / 72.8%였다. 프레임당 미매칭 출력은 Peg 0.87 / 0.45개,
Box 2.79 / 2.61개였다.

| 정책 | RTX 3090 CUDA 평균 / p50 / p95 (ms) | Apple M4 Pro MPS 평균 / p50 / p95 (ms) |
| --- | ---: | ---: |
| Expert | 28.08 / 28.50 / 30.97 | 32.52 / 31.75 / 44.05 |
| Random | 40.45 / 28.74 / 31.36 | 29.31 / 28.18 / 42.21 |
| Expert→Random | 28.15 / 28.23 / 30.93 | 30.00 / 28.88 / 43.84 |
| 전체 | 32.98 / 28.53 / 31.04 | 30.21 / 29.30 / 43.42 |

시간은 GPU 상주 입력부터 CPU 출력까지다. CUDA는 Linux, Core i7-9700K,
PyTorch 2.14.1+cu130에서 2픽셀 격자 피팅과 TorchInductor 검증을 썼다.
MPS는 macOS, M4 Pro, PyTorch 2.14.1에서 RANSAC과 Metal을 썼다.
첫 CUDA 실행에서 144프레임으로 워밍업하는 동안 15개 그래프가 컴파일됐다.
측정 중 Random 프레임 두 개에 새 그래프가 각 한 개씩 생겼고, 두 호출의
총 검출 시간은 각각 5.30초와 4.05초였다. 별도로 468.71 ms 호출 한 건에선
새 그래프가 관측되지 않았다. 위 집계에는 2,073프레임을 모두 포함했다.

두 프레임을 워밍업에 추가한 CUDA 재실행은 17개 그래프를 준비했고
측정 중 추가 그래프는 0개였다. 전체 평균 / p50 / p95는
28.23 / 28.32 / 30.89 ms였다. 정책별 평균 / p95는 Expert
27.85 / 30.45 ms, Random 28.63 / 30.79 ms, Expert→Random
28.01 / 31.01 ms였다. Random의 374.05 ms 호출 한 건도 시간에 포함했다.
두 CUDA 실행의 프레임별 출력 점 수와 3 mm·5 mm 매칭 수는 모두 같았다.
두 프레임의 출력 좌표는 최대 축 차이 0.00068 mm만큼 달랐다.

[입력 매니페스트](../outputs/corner_detection_holdout/frames/manifest.json) ·
[CUDA 원본](../outputs/corner_detection_holdout/cuda/72135dfee99f47c7a52307bf5ce9c546/results.json) ·
[추가 워밍업 CUDA 원본](../outputs/corner_detection_holdout/cuda_warm/f5d9170dacae42ae8d123e3dce123ddd/results.json) ·
[추가 워밍업 설정](../visual/configs/corner_detection_holdout_cuda_warm.yaml) ·
[MPS 원본](../outputs/corner_detection_holdout/mps/fd66d4509c764c8dad04d13a5bbebee4/results.json) ·
[검출 소스 스냅샷](../outputs/corner_detection_holdout/detector_source.tar.gz)

전체 집계와 원본 체크섬은 저장소에 기록했다. RGB-D와 프레임별 결과는
`outputs/corner_detection_holdout/`의 로컬 아티팩트다.
[전체 정밀도 집계 JSON](../visual/benchmarks/corner_detection_holdout.json)
