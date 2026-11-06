# Corner detection

`visual.corner_detection` extracts current-frame, image-supported peg and box
corners from native 512×512 RGB-D. It is independent of the existing tracking,
pose, graph and controller workflows. No simulator, object GT, previous frame,
other camera or hidden per-episode object dimensions enter inference.

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
