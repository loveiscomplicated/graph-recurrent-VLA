# Visual: RGB-D → 물체 추적 → point cloud / pose

ManiSkill3 `PegInsertionSide-v1`에서 **RGB-D → Grounding DINO(초기 검출) → SAM2(추적) → 물체별 point cloud와 6D pose**를 만드는 Vision prototype이다. 공용 코드(`envs/`)와 다른 팀 폴더는 건드리지 않는다.

## TAPNext++ 포인트 추적

공식 PyTorch 구현은 `visual/third_party/tapnet` Git 서브모듈에서 불러온다.
가중치 다운로드와 두 프레임 추론 예제는 [TAPNext++ 가이드](../docs/tapnextpp.md)를 참고한다.

동결 Peg 에피소드와 기존 GT query NPZ, 256 체크포인트 및 CoTracker 서브모듈을
준비한 뒤 보조점 네 조건을 비교할 수 있다. 입력 위치와 장치는 YAML에서 지정한다.

```bash
PYTHONPATH=visual/third_party/tapnet:visual/third_party/co_tracker WANDB_MODE=disabled \
  python main.py --config visual/configs/tapnextpp_support.yaml
```

좌표·시각·채점 검증과 산출물은 [보조점 감사](../docs/tapnextpp.md#보조점-구현-감사-및-네-조건-비교)를 참고한다.

## CoTracker3 Online 포인트 추적

공식 구현은 `visual/third_party/co_tracker` Git 서브모듈에 있다.
CUDA PyTorch 환경과 기존 TAPNext++ 실행의 `tracks.npz`, `summary.json`,
같은 ManiSkill RGB 프레임이 필요하다. 다른 롤아웃을 비교하려면
`visual/configs/cotracker_online.yaml`의 `source`와 `reference_run`을 함께 변경한다.

```bash
git submodule update --init visual/third_party/co_tracker
mkdir -p visual/checkpoints/co_tracker
curl -fL https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth \
  -o visual/checkpoints/co_tracker/scaled_online.pth
PYTHONPATH="visual/third_party/co_tracker${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE=disabled \
  python main.py --config visual/configs/cotracker_online.yaml
```

초기 좌표는 TAPNext++ 실행에서 그대로 읽는다. CoTracker3는 16프레임
윈도우를 8프레임씩 이동하며, 프레임별 최초 출력과 출력이 가능해진 입력
프레임 번호를 저장한다. GT 배열은 비교 결과의 채점에 사용된다.

## Apple Silicon SAM2 추적

MLX/Metal에서 peg와 box를 추적하고 카메라 해상도의 GPU 마스크를 반환한다.
Apple Silicon, MLX 환경, 변환한 SAM2.1 Small 가중치와 RGB·초기 박스가 담긴
로컬 NPZ가 필요하다. [환경 설치 및 입력 형식](../docs/sam2_mlx.md)을 참조한다.

```bash
WANDB_MODE=disabled .venv/bin/python main.py --config evaluation/configs/sam2_mlx_replay.yaml
```

## SAM2 CUDA TF32 스트림

`visual.sam2_cuda.CudaObjectStream`은 CPU의 uint8 NumPy RGB 프레임(HWC)을
JPEG로 인코딩·디코딩한 뒤 CUDA에서 SAM2.1 Small로 추적한다. 기본 내부 해상도는
512이고 공간 메모리는 3개다. [Visual 환경과 체크포인트](#installation-single-supported-environment)를
준비하고 저장소 루트에서 `source visual/env.sh`로 환경을 활성화한다.

```python
from visual.sam2_cuda import CudaObjectStream

stream = CudaObjectStream("visual/checkpoints/sam2.1_hiera_small.pt")
try:
    first_masks = stream.start(rgb0, {1: [20, 40, 100, 80]})
    next_masks = stream.track(rgb1, timestep=1)
finally:
    stream.reset()
```

첫 프레임에는 물체의 정수 ID와 카메라 픽셀 좌표의 XYXY 박스를 전달한다. 이후
프레임은 순서대로 전달한다. 반환값은 물체 ID별 카메라 해상도의 `torch.bool`
CUDA 텐서다. `reset()` 후에는 모델을 다시 로드하지 않고 새 영상에 사용할 수 있다.
프레임 호출마다 PyTorch의 TF32 설정을 복원한다. 호출 중 설정은 프로세스 전체에
적용되므로 다른 정밀도 설정을 사용하는 CUDA 작업과 겹치지 않게 한다.

## 독립 꼭짓점 검출기

`corner_detection/`은 현재 RGB-D 한 프레임에서 관측으로 지지되는 peg/box 꼭짓점을
검출하는 별도 패키지다. 기존 tracking/pose 실행과 독립적으로 호출한다. Torch CPU/CUDA
계산과 선택적 MPS Metal 커널, 명시적으로 호출하는 NumPy 기준 구현을 포함한다.
[사용법·구조·검증 결과](../docs/corner_detection.md)를 따른다.

삽입 말미 Peg 뒤쪽 끝면 복구를 비교하려면 PyTorch CPU 환경과 설정에 지정된
저장 프레임 및 매니페스트가 필요하다.

```bash
WANDB_MODE=disabled .venv/bin/python main.py --config visual/configs/peg_rear_end.yaml
```

[끝면 복구 조건 및 비교 결과](../docs/corner_detection.md#peg-rear-end-recovery-during-insertion)를 참고한다.

같은 삽입 프레임에서 CPU 메인 검출기와 Peg 뒤쪽 끝면만 GPU로 계산하는 경로를
비교할 수 있다. 두 설정은 저장된 삽입 매니페스트와 원본 RGB-D 파일이 필요하다.
MPS는 Apple Silicon PyTorch, CUDA는 NVIDIA GPU와 CUDA 지원 PyTorch 환경에서 실행한다.

```bash
WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python main.py --config visual/configs/peg_rear_end_mps.yaml
WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python main.py --config visual/configs/peg_rear_end_cuda.yaml
```

색상 분류, 깊이 역투영, 평면 피팅과 Peg 검사를 같은 GPU에서 실행하는
옵션은 `TorchCornerDetector.main(device="mps")` 또는 `device="cuda"`다.
같은 삽입 매니페스트에서 CPU 메인과 비교하려면 아래 설정을 사용한다.

```bash
WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python main.py --config visual/configs/corner_geometry_mps.yaml
WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python main.py --config visual/configs/corner_geometry_cuda.yaml
```

고정된 초기·궤적·삽입 프레임에서 Peg, Box 외곽, Hole 림을 함께 재측정하려면
설정에 지정된 세 매니페스트와 원본 RGB-D 파일을 같은 저장소 위치에 준비한다.

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 .venv/bin/python main.py --config visual/configs/combined_corner_detection.yaml
```

## 결론

- 실시간 추적은 된다: causal streaming + GPU point cloud로 **p95 45.1 ms (22.5 FPS)**, GPU 0.61 GB.
- 삽입에 필요한 상태 정밀도는 **위치 ≲ 1 mm, 축 ≲ 0.3°**다(GT에 노이즈를 넣은 곡선: 1 mm 82.5%, 2 mm 52.5%, 회전 1° 37.5%).
- vision 추정만으로 스크립트 정책을 돌리면 삽입 성공률이 **52.5–62.5%**(GT 100%)다. 로봇 손 위치(TCP), "다가가서 보고 잡기", "삽입 직전 다시 보기"를 더하면 처음 보는 seed 세트 3개에서 **82.5–85%**다.
- 남은 병목: 손에 쥔 peg 추정(그리퍼에 가려 카메라만으로는 축 방향이 안 정해짐), mm 단위 정렬, box가 움직였을 때(60%).
- RGB-only depth는 정답 scale을 줘도 수 cm 오차라 이 작업에는 depth 센서가 필요하다. relevance score는 추정 pose + 기하 규칙으로 충분하다(AUROC 0.94 / 1.00).

수치, 근거, 회의 안건은 [`REPORT.md`](REPORT.md)에 있다.

데모 영상(v0 10개, v1 107개, 약 300 MB: 최종 설정 삽입 성공·실패, box 이동 대응, re-anchor 전후 비교 등)은 용량 때문에 저장소에 넣지 않고 따로 보관하고 있다. 필요하면 공유한다.

## 문서

| 문서 | 내용 |
|---|---|
| [`REPORT.md`](REPORT.md) | v1 결과: 요약, 설계 결정과 근거, 병목 분석, 실험별 결과(E0–E9), 열린 질문 |
| [`docs/plan_v1.md`](docs/plan_v1.md) | v1 실험 설계: 목표, 평가 원칙, 실험 목록과 우선순위, 기본 결정 |
| [`docs/v0.md`](docs/v0.md) | 첫 prototype(Step 1–8)과 v1에서 뒤집힌 결론 |
| [`docs/known_issues.md`](docs/known_issues.md) | 환경 문제와 우회법 (Vulkan, mplib, SAMURAI 등) |

## 폴더

```
visual/
├── scripts/       실험 코드
├── results/       표(md), 수치(json), 대표 그림. step1–8 = v0, v1/E0–E9 = v1
├── vulkan/        EGL Vulkan ICD (docs/known_issues.md)
├── env.sh         환경 변수와 venv 활성화
└── requirements.lock.txt, requirements.step7.lock.txt, constraints.txt
```

git에 넣지 않는 것(다시 만들 수 있음): `data/`(수집 데이터, E9 export), `results/v1/**/runs/`(mask npz), `checkpoints/`, `third_party/`의 로컬 소스(SAM2, SAMURAI, EfficientTAM, DAM4SAM, SAM2Long 소스), `cache/`, `logs/`.

## 주요 스크립트

| 파일 | 역할 |
|---|---|
| `collect_v1.py`, `run_e0_collect.sh`, `e0_stats.py`, `init_v1.py` | 데이터 수집(정책 4종, 교란), 통계, 초기 검출 |
| `track_v1.py`, `run_e1.sh`, `eval_v1.py`, `gallery_v1.py` | offline tracking(SAM2 / SAMURAI / EfficientTAM, re-anchor), 평가 |
| `stream_v1.py`, `pc_gpu.py`, `e2_stream.py`, `e2_summary.py` | causal streaming tracker, GPU point cloud, 속도 측정 |
| `reanchor_v1.py`, `e3_run.py`, `e3_calib.py`, `e3_fit_gate.py`, `e3_sim_gate.py` | re-anchor v1과 수락 게이트 학습 |
| `pose_v1.py`, `test_pose_v1.py`, `e4_pose.py` | peg / hole pose 추정기와 평가 |
| `e5_loop.py`, `run_e5.sh` | vision-in-the-loop 정책 |
| `e6_relevance.py`, `e7_depth.py`, `e8_domain.py`, `e9_export.py` | E6–E9 |
| `track_dam4sam.py`, `track_sam2long.py`, `stream_sam2long.py` | 추가 tracker 비교 |
| `collect.py`, `grounding.py`, `track_sam2.py`, `pointcloud.py`, `run_pipeline.py`, `track_compare.py`, `reanchor.py` | v0 |

## Installation (single supported environment)

The default environment supports **Linux x86_64, Python 3.12, PyTorch
2.8.0+cu128 / torchvision 0.23.0+cu128, ManiSkill 3.0.1, SAPIEN 3.0.3,
and SAM2.1 Small**. Dependencies are resolved in `pyproject.toml` and `uv.lock`;
the old requirements files are historical snapshots, not installation inputs.

Prerequisites:

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/) >=0.9
  and Git. Validation used uv 0.13.4.
- An NVIDIA driver compatible with CUDA 12.8 and GPU passthrough in the
  container (`nvidia-smi` must work). PyTorch supplies CUDA runtime wheels;
  installing a system CUDA toolkit or nvcc is not required for this setup.
- System Vulkan loader, NVIDIA Vulkan/EGL driver libraries, and OpenGL runtime
  (`libGL.so.1`, used by OpenCV). On Debian/Ubuntu the loader packages are
  typically `libvulkan1`, `libegl1`, `libgl1`; NVIDIA libraries come from the
  driver/container runtime. Enable graphics capabilities when launching the
  GPU container. Python package installation cannot fix missing driver libraries.
- Network access to PyPI, download.pytorch.org, GitHub, and (for models)
  Hugging Face / Meta checkpoint hosting, plus sufficient disk for CUDA wheels.

```bash
cd visual
bash setup.sh
source env.sh
python scripts/smoke_test.py --out results/step1
```

`setup.sh` uses `uv sync --locked` and the local `.venv`, with no system site
packages. uv finds or downloads Python 3.12; no global Python packages or drivers
are changed. Repeat runs synchronize the existing environment. Exit code 1
indicates setup/sync failure, 2 indicates import failure, and 3 indicates installed
packages but unavailable/failing CUDA. Setup validates imports and GPU execution;
it does not download checkpoints or prove renderer/model inference works.

Activation always selects `visual/.venv`, even from another venv or Conda shell.
It clears inherited `PYTHONHOME` / `PYTHONPATH`, disables user site packages,
and preserves `VISION_ROOT`, `VISION_CACHE`, `HF_HOME`, `PIP_CACHE_DIR`,
`TORCH_HOME` and Vulkan ICD behavior. Set `VISION_CACHE` before sourcing to move
caches; model/data paths use `VISION_ROOT` (default: this directory). An explicit
`VK_ICD_FILENAMES` wins over the bundled NVIDIA EGL workaround.

### Model assets

First check `checkpoints/sam2.1_hiera_small.pt`. If absent, obtain the
[SAM2.1 Small checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
and place it there. Do not substitute SAM2.0 weights. The pinned upstream SAM2
source supplies `configs/sam2.1/sam2.1_hiera_s.yaml`; no `third_party/sam2`
checkout or editable install is needed. The optional connected-components CUDA
extension is disabled as in the original environment. Its isolated build uses
the same torch version as runtime.

Grounding DINO uses `IDEA-Research/grounding-dino-tiny` through Transformers;
re-anchor v1 also uses `facebook/dinov2-small`. Their existing loaders download
missing assets into `HF_HOME` on first use. For offline operation populate that
cache first, then set `HF_HUB_OFFLINE=1`. Setup itself does not fetch model assets.

### Validation and troubleshooting

Run `uv lock --check`, `uv sync --locked`, and
`python scripts/check_install.py` from `visual/` to check packaging separately
from the full smoke test. The smoke test initializes Grounding DINO, SAM2 Small
image/video predictors, and the simulator, writes JSON/images under `--out`,
and exits nonzero if a stage fails. It may download missing HF assets.

- Resolution/download failures: check Git and network/TLS access to the indexes;
  do not install the old requirements with pip or inherit parent torch.
- CUDA unavailable: check driver compatibility and container GPU passthrough.
  `nvidia-smi` reports driver capability, not the installed PyTorch runtime.
- Vulkan/EGL failure: inspect `VK_ICD_FILENAMES` and the driver libraries;
  see [known issues](docs/known_issues.md). The bundled ICD requires
  `libEGL_nvidia.so.0`; it is not a portable replacement for a driver.
- Existing `.venv` with system packages: move it aside manually and rerun setup.
- Missing checkpoints/data: install the assets above and collect episodes using
  the existing collection scripts before offline tracking/export.

See [migration validation](docs/uv_migration.md) for checks actually executed on
this host. Historical experiment results in REPORT.md are unchanged and are not
validation results for this environment.

### Legacy experiments

SAMURAI, EfficientTAM, DAM4SAM, SAM2Long and YOLO/depth comparisons are
**legacy and unsupported by this environment**. Their scripts, results, and old
requirements remain for reference. In particular `run_step7.sh` and `run_e1.sh`
still orchestrate old separate environments; they are not default setup or smoke
commands. Use `track_v1.py --backend sam2 --model small` for the supported tracker.
The supported streaming and re-anchor implementations are unchanged. Historical
base+/tiny options are retained for reference; only Small is validated/supported.

Experiment commands in [REPORT.md](REPORT.md#재현) describe the original environment;
comparison/depth commands need their historical dependencies and assets.
