# uv environment migration

The old environment inherited torch, NumPy and Pillow from its parent Python
through `--system-site-packages`; its freeze was not a complete install recipe.
The cu128 torch constraints did not configure the wheel source. Separate tracker
environments also installed incompatible implementations of the `sam2` module.

The replacement is one isolated `visual/.venv`, Python 3.12, with a fully resolved
`uv.lock` and explicit official cu128 index for torch/torchvision. `setup.sh`
synchronizes with `uv sync --locked`, checks imports and executes a CUDA operation.
It does not change drivers, system CUDA, or global Python packages. Python downloads,
if needed, go into the visual cache. Existing environments are not deleted.

## Dependency decisions

Retained the original core versions: torch 2.8.0+cu128, torchvision 0.23.0+cu128,
NumPy 2.1.2, Pillow 11.0.0, ManiSkill 3.0.1, SAPIEN 3.0.3, Transformers 5.19.0,
Hydra 1.3.7, iopath 0.1.10, OpenCV 5.0.0.93, SciPy 1.18.1, matplotlib 3.11.2,
ImageIO 2.38.1, imageio-ffmpeg 0.6.0, trimesh 5.1.1 and transforms3d 0.4.2.
Constraints retain HF Hub 1.33.0, tokenizers 0.23.2 and pytorch-kinematics 0.7.6.
Triton 3.4.0 is explicit because `pc_gpu.py` imports it directly and torch 2.8
requires that version. Transitive dependencies are resolved and locked, rather
than copied wholesale from the old freeze.

Grounding DINO is supplied by Transformers, not a second source installation.
Transformers also supplies DINOv2 for `reanchor_v1.Dino`. SciPy supplies the
existing pose estimator; Torch/Triton supply the existing GPU point cloud path.
ManiSkill's required mplib remains transitively installed, but the pipeline uses
its existing scripted policy, not the previously failing mplib planner.

Excluded torchaudio (unused), duplicate opencv-python-headless (same `cv2`
namespace), SAMURAI, EfficientTAM, DAM4SAM, SAM2Long, loguru/VOT experiment
dependencies, Ultralytics and optional depth-model experiments. Packages such as
accelerate that were only present in the freeze are not direct dependencies.
Historical requirements, scripts, results, and checkpoints remain untouched.

SAM2 comes from the original upstream commit
`2b90b9f5ceec907a1c18123530e92e794ad901a4`. The project config explicitly disables
its optional CUDA extension (`SAM2_BUILD_CUDA=0`), matching the old setup, and
pins its isolated build torch to 2.8.0+cu128. This retains dependency resolution
and build isolation. No editable source tree or competing `sam2` package is used.
The default model is **SAM2.1 Small**, configuration
`configs/sam2.1/sam2.1_hiera_s.yaml`, checkpoint
`checkpoints/sam2.1_hiera_small.pt`. Causal streaming, conditioning/re-anchor,
GPU point cloud and pose implementations have not changed. The only existing
Python script changed is the smoke test: Small only, safe CUDA diagnostics,
and nonzero exit status when a stage fails.

## Validation on 2026-10-23

Host: Linux x86_64, Python 3.12.15, uv 0.13.4, RTX 3090, NVIDIA driver 595.91.07.
The installed torch reports CUDA runtime 12.8; the driver's advertised CUDA 13.2
capability does not replace that runtime.

| Check | Final configuration result |
| --- | --- |
| `uv lock --check` | Passed, 116 resolved packages including the project |
| `uv sync --locked` / `bash setup.sh` | Passed; fresh isolated install and final synchronization |
| Repeat `bash setup.sh` | Passed; checked 115 installed packages in 3 ms, no reinstalls |
| `uv pip check` | Passed; installed packages compatible |
| Essential imports / CUDA allocation and operation | Passed, torch 2.8.0+cu128 on RTX 3090 |
| Activation with inherited Python/Conda variables | Passed; local interpreter, user site disabled, PYTHONHOME/PYTHONPATH cleared |
| Existing smoke test | Passed; simulator RGB-D/render, Grounding DINO inference, SAM2 Small image inference and video initialization |
| `python scripts/test_pose_v1.py` | Passed both existing pose checks |
| GPU point cloud / Triton FPS | Passed on a synthetic 64x64 depth/mask input, finite 1024x3 CUDA output |
| Minimal live vision pipeline | Passed one seed (600), `e5_loop.py --cond vision --vcfg v1 --fast-pose`, 111 steps, success, no runtime error |
| Shell syntax / `git diff --check` | Passed |

The simulator backend was inspected at runtime: **physx_cpu** simulation and
**sapien_cuda** rendering. CUDA model inference and point cloud kernels were
executed. PhysX CUDA/vectorized simulation, other GPUs/platforms, and compiled
20 Hz ablations were not tested. One successful episode is an integration check,
not a reproduction of the historical success rates or performance claims.

Commands for the main validation, from `visual/` after activation:

```bash
uv lock --check
uv sync --locked
uv pip check
python scripts/check_install.py
OMP_NUM_THREADS=8 python scripts/smoke_test.py --out cache/validation/final-smoke
python scripts/test_pose_v1.py
OMP_NUM_THREADS=8 python scripts/e5_loop.py --cond vision --vcfg v1 --fast-pose \
  --seeds 600 --name uv_validation --out cache/validation/final-e5
```

Detailed logs and generated outputs are kept in ignored `cache/`:
`setup-final.log`, `setup-idempotent.log`, `final-smoke.log`, `final-e5.log`, and
`validation/`. No old result files were overwritten. No existing checkpoints were
found; the Small checkpoint was downloaded only after that check. HF assets were
cached during model validation. These assets and `.venv` are not committed.

## Remaining limits

No unresolved dependency/import/runtime incompatibility was observed in the
executed default pipeline. The simulator smoke still reports int16 depth overflow
at its default camera settings, as already documented by the shared environment;
this migration does not change depth handling. SAM2's optional connected-component
postprocessing extension remains unavailable by design, as in the old setup.
Model checkpoints/HF revisions are external assets and are not pinned by uv.lock;
this lock guarantees Python dependencies, not bitwise model/driver reproducibility.
Legacy tracker and depth comparison scripts require their historical environments
and are unsupported here. GPU/driver/Vulkan/EGL failures on other containers still
require system-level provisioning; setup does not install those prerequisites.
