# TAPNext++ point tracking

Five-mode에서 TAPNext++의 용도는 선택적 비꼭짓점 표면점 추적에 한정한다.
꼭짓점 좌표와 가시성 조건은 현재 RGB-D 프레임의 꼭짓점 검출기 출력으로
정한다. 아래 파일럿 캐시와 3D 꼭짓점 복원 절차는 별도 실험 기록이다.
[Five-mode 입력 규칙](data_collection.md#five-mode-graph-input)

The Visual prototype uses the upstream PyTorch tracker in the `visual/third_party/tapnet`
submodule. The submodule pins the same upstream revision used by the original
local benchmark. Run the commands below from the team repository root.

```bash
git submodule update --init visual/third_party/tapnet
python -m pip install numpy torchvision einops
mkdir -p visual/checkpoints/tapnextpp
curl --fail --location --retry 3 \
  https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt \
  -o visual/checkpoints/tapnextpp/tapnextpp_256_official.pt
```

Use a Python environment with a compatible PyTorch and torchvision pair.
The checkpoint is about 2.5 GB and lives in a Git-ignored directory.
Its SHA-256 is
`cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da`.

Run two frames to check loading, query initialization, and recurrent state.
The upstream wrapper takes BGR images and returns pixel coordinates in the
original frame as `(x, y)`, along with visibility and the next state.

```bash
PYTHONPATH="visual/third_party/tapnet${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import numpy as np
from tapnet.tapnextpp.votsp2026.model import TAPNextPP

tracker = TAPNextPP.from_checkpoint(
    "visual/checkpoints/tapnextpp/tapnextpp_256_official.pt", device="cpu"
)
frame = np.zeros((256, 256, 3), dtype=np.uint8)
queries = np.array([[128, 128]], dtype=np.float32)
xy, visible, state = tracker.track_frame(frame, queries, autocast=False)
xy, visible, state = tracker.track_frame(frame, state=state, autocast=False)
assert xy.shape == (1, 2) and visible.shape == (1,) and np.isfinite(xy).all()
print(xy, visible)
PY
```

For the [512×512 checkpoint](https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt),
download `tapnextpp_512.ckpt` into the same directory and pass its path and
`input_resolution=512` to `from_checkpoint`.
See the [upstream tracker guide](https://github.com/google-deepmind/tapnet/blob/ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b/tapnet/tapnextpp/votsp2026/README.md)
for additional usage.

Other checkouts can add this repository's `visual/third_party/tapnet` directory to their
`PYTHONPATH` and pass an absolute path to the checkpoint in
`visual/checkpoints/tapnextpp/`.

## 파일럿 추적 캐시

`visual/configs/tapnextpp_cache.yaml`은 혼합 궤적 파일럿의 RGB-D와 카메라
캘리브레이션 파일을 해시로 확인하고 저장 순서대로 재생한다. RGB-D 검출기가
찾은 Peg 마스크 내부에서 32개 표면점을 고르고, MLX 이동 영역 방식으로 추적한다.
각 궤적에서 모델 상태를 초기화한다.

NPZ의 `frame_indices`는 원본 프레임 번호다. `anchor_frame`은 32개 표면점을
초기화한 첫 프레임이며, `query_xy`는 초기 좌표다. `surface_xy`는 `(T, 32, 2)`
크기의 원본 이미지 픽셀 좌표 `(x, y)`, `surface_visible`은 `(T, 32)` 크기의
가시 여부다. 초기화 전에는 좌표를 `NaN`, 가시 여부를 `false`로 기록한다.
초기화할 수 없는 궤적도 보존하고 `anchor_frame=-1`로 표시한다.

매니페스트에 원본 매니페스트, 모델 가중치, 캐시 파일의 SHA-256을 저장한다.
CUDA에서는 `visual/configs/tapnextpp_cache_cuda.yaml`을 사용한다. 원본을
다른 호스트에서 전송 중이라면 매니페스트를 먼저 복사하고
`source_wait_timeout_seconds`로 새 궤적을 기다리는 시간을 지정할 수 있다.
수집된 프레임 전체가 도착한 궤적부터 처리한다.
실행 명령과 환경 준비는 [Visual README](../visual/README.md#tapnext-포인트-추적)를
참고한다.

### 꼭짓점 검출 추가

`visual/configs/tapnextpp_corner_cache_cuda.yaml`은 원본 RGB-D와 TAPNext++
캐시의 해시 및 프레임 순서를 확인한다. 기존 표면 추적 배열을 복사하고
`TorchCornerDetector.main()`이 각 프레임에서 검출한 꼭짓점을 별도 실행 폴더에
저장한다. 프레임 `i`의 검출은 `corner_offsets[i:i+2]`가 가리키는 반열린
범위에 있다. 각 검출에는 물체 ID, 원본 픽셀 좌표, 월드 좌표와 카메라 좌표,
검출 방법이 있다. 물체 ID 0은 Peg, 1은 Box다.

평가는 GT와 물체 분할로 깊이가 뒷받침되는 기준 꼭짓점을 고른다. 파일럿의
물체 ID는 `evaluation.pilot_corner_detection.confirm_actor_ids`로 100개
리셋에서 검증한 뒤, 원본 매니페스트의 SHA-256과 함께 `actor_ids.json`에 둔다.
`episode_metrics.jsonl`은 진행 중인 궤적별 점수를, `metrics.json`은 전체 및
정책별 3·5 mm 정밀도와 재현율, GPU 상주 입력 기준 검출 시간과 호스트 출력
시간을 기록한다. 실행 때 생성된 CUDA 컴파일 그래프 수를 함께 기록한다.

## 보조점 구현 감사 및 네 조건 비교

저장소 루트에서 두 공식 서브모듈, 256×256 체크포인트, PyTorch·torchvision·OpenCV,
그리고 config에 지정된 로컬 149프레임 에피소드와 동결 입력 NPZ를 준비한다.

```bash
PYTHONPATH=visual/third_party/tapnet:visual/third_party/co_tracker WANDB_MODE=disabled \
  .venv/bin/python main.py --config visual/configs/tapnextpp_support.yaml
```

TAPNext ICCV 2025 보충자료의 Appendix F Table 8은 TAPNext-B의 DAVIS first
평가에서 관심점 하나와 9×9 local, 4×4 global을 같이 넣는다. arXiv의 같은
내용은 Appendix C에 있고, 양방향 strided 절차는 arXiv Appendix F 및 ICCV
Appendix I에 있다. 이 실험에는 TAPNext++ 256 체크포인트와 PegInsertion
에피소드를 사용했다. 50px extent는 CoTracker 공식 평가 코드에서 가져온
설정이다. VOTSp2026 TAPNext++ 제출은 512 입력에서 별도 8×8 local 설정을 쓴다.
[TAPNext ICCV 보충자료](https://www.openaccess.thecvf.com/content/ICCV2025/supplemental/Zholus_TAPNext_Tracking_Any_ICCV_2025_supplemental.pdf),
[TAPNext arXiv](https://arxiv.org/html/2504.05579v2),
[TAPNext++ VOTSp2026](https://github.com/google-deepmind/tapnet/blob/ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b/tapnet/tapnextpp/votsp2026/README.md)

### 좌표 생성

공식 CoTracker 평가 경로는 영상을 `align_corners=True`로 resize하고,
관심점 xy에 `(Wm−1)/(W−1), (Hm−1)/(H−1)`을 곱한다. Local grid는
그 결과를 중심으로 `get_points_on_a_grid(size, (50, 50), (y, x))`를 호출한다.
즉 50은 **resize된 모델 영상에서의 영역 너비**다. 공식 기본 크기는 local
8×8, global 5×5이며, 여기서는 TAPNext의 9×9와 4×4를 적용했다. 공식 함수는
너비의 1/64만큼 margin을 두고 균등하게 배치한다. Local의 실제 한 축 범위는
중심 ±24.21875 모델 px이고, Global 각 축은 4, 86⅔, 169⅓, 252 모델 px다.
CoTracker 공식 평가의 local/global 보조점 시각은 전체 입력 영상 기준 0이다.
여기서는 관심점이 있는 프레임에서 스트림이 시작하므로 원본 영상에서 모든 보조점의
시각은 관심점과 같다.
[CoTracker 평가 predictor](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/cotracker/models/evaluation_predictor.py),
[CoTracker grid 함수](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/cotracker/models/core/model_utils.py)

이전 로컬 실험의 `outputs/tracker_support/appendix_c.py`는 CoTracker의 좌표
배율과 TAPNext++ 래퍼의 좌표 배율을 섞었다. 512→256에서 최외곽 Global 축에
0.49411 모델 px 차이가 생겼다. 다음은 해당 부분의 수정 전후다.

```python
# 이전
image_scale = np.array([(w - 1) / (resolution - 1), (h - 1) / (resolution - 1)])
x, y = target[0] / image_scale
local = get_points_on_a_grid(9, (extent, extent), (float(y), float(x)))[0]
points = np.concatenate((target, local.numpy() * image_scale,
                         global_grid.numpy() * image_scale))
# track_frame subsequently applies [resolution / w, resolution / h].

# 수정
model_xy = utils.display_to_model(target, h, w, resolution)
x, y = model_xy[0]
local = get_points_on_a_grid(9, (extent, extent), (float(y), float(x)))[0]
local[40] = torch.from_numpy(model_xy[0])
model_xy = np.concatenate((model_xy, local.numpy(), global_grid.numpy()))
display_xy = utils.model_to_display(model_xy, h, w, resolution)
```

래퍼는 RGB resize에 `align_corners=False`를 쓰고, 좌표에는 `256/W, 256/H`
및 역수를 쓴다. 이 스케일 계약을 초기 GT와 출력에 동일하게 적용했다. 정수 pixel
center 기반 보간 인덱스의 반 픽셀 이동과는 별개의 좌표 계약이다. 512×512
원본에서 채점하는 4px는 모델 입력 2px에 해당한다. 실제 forward tensor에서
Local 중앙 점은 관심점과 정확히 동일하며 Local 81개에 포함된다.

Local은 원래 관심점을 중심으로 유지하고 면 밖 좌표를 그대로 보낸다. 이
에피소드에서는 (anchor, corner) = (41, 1), (47, 1)에서 각각 9개씩
프레임 밖이다. Clamp하면 경계에 중복점이 생기고 균일 간격도 바뀐다.
모델의 query positional embedding은 `grid_sample`의 기본 zero padding을
사용하며, 경계 위치에 따라 부분 보간 또는 0이 발생할 수 있다. 프레임 밖 좌표에서도
모델은 visibility와 trajectory를 출력한다.
[TAPNext++ 좌표 래퍼](https://github.com/google-deepmind/tapnet/blob/ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b/tapnet/tapnextpp/votsp2026/utils.py),
[TAPNext PyTorch 모델](https://github.com/google-deepmind/tapnet/blob/ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b/tapnet/tapnext/tapnext_torch.py)

### 모델 호출과 채점

- 첫 inner forward를 hook으로 기록했다. `[1,Q,3]`의 순서는 `[t,y,x]`이고
  1·82·17·98개 점 모두 상대 시각 0이다. D의 첫 ViT self-attention은 같은
  `[1,1122,768]` tensor를 Q/K/V로 받아 마스크 없이 패치 1024개와 점 98개를
  처리한다. 모델은 concat 이후 같은 self-attention을 반복하고 recurrence에
  모든 점의 토큰을 유지한다. 좌표와 visibility는 프레임별 head 출력으로 저장한다.
- 이전 평가는 `xy, _, state`를 받고 채점 프레임의 `xy[0]`만 저장했다.
  새 평가는 `xy, vis, state`를 받고 전체 프레임에 대한 모든 query의 좌표와
  visibility를 저장한 뒤 `trace["trajectory_source_xy"][scored - anchor, 0]`만
  채점한다. 기존 채점과 동일하게 모델 visibility로 분모나 성공 여부를 조정하지
  않는다. 각 점의 예측이 유한한지 검사하고 GT와의 원본 px L2 오차 ≤4를 센다.
- 매 stream의 첫 호출은 `state=None`, 이후에는 같은 stream의 캐시를 전달한다.
  모델 hook에서 이후 query 인자가 없고 state의 query tensor가 첫 호출과
  같은 객체임을 검사했다. D의 별도 reverse smoke는 실제 프레임 34→33을
  새 state로 실행했고 34→35와 동일한 초기 tensor가 기록됐다. 비교 실험의
  frozen workload는 전부 anchor 이후에 채점하므로 forward만 포함한다.
  CoTracker 공식 evaluator의 strided는 영상 뒤집기, `T−1−t`로 query
  remap, 예측을 다시 뒤집어 결합하는 별도 경로다.
  [CoTracker 공식 evaluator](https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/cotracker/evaluation/core/evaluator.py)
- 기존 단일 관심점 `appendix_c/single` 스케줄의 9개 (anchor, corner) 조합과
  동일하다. 더 이전 `tap_none`은 여러 관측 GT 꼭짓점을 anchor마다 함께
  넣었다. 새 A는 보조점 효과를 비교하기 위해 관심점 하나로 고정했다. 새 실행은
  동결 NPZ에서 입력 배열만 읽었고, GT 생성기로 재계산한 5개 배열과 프레임 전체가
  일치했다. 각 소스 파일의 SHA-256도 확인했다.
- 전체 240건은 가림으로 판정된 기존 채점 요청이다. phase별 원본 0-based 구간과
  분모는 reach 0–42 (7), descend 43–55 (27), close 56–63 (24), lift
  64–74 (33), align 75–127 (83), insert 128–148 (66)이다. 타이밍은
  프레임당 모델 래퍼 호출과 장치 동기화를 포함한다. RGB→BGR 변환,
  준비 실행, 채점 및 저장은 제외한다. 각 호출을 실제 처리 프레임의 phase에
  배정하고, 호출이 없었던 프레임은 타이밍 분모에 포함하지 않는다.

각 `<variant>_frame_<anchor>_corner_<id>.npz`에는 원본 query xy, 실제 초기
`model_query_tyx`, 모든 점의 trajectory와 visibility, 소스 프레임 번호,
채점 프레임 및 raw latency가 있다. 같은 이름의 PNG는 왼쪽에 시작 프레임의
배치, 오른쪽에 전체 trajectory를 보여 준다. 노랑은 관심점, 청록은 Local,
보라는 Global이다. `comparison.json`은 전체 및 phase별 성공률, 중앙 위치
오차와 평균·P95 추론 시간을 담는다. 장치, 체크포인트와 동결 입력 해시 및
attention hook 결과는 ExperimentLogger 실행 메타데이터에 있다.

### 새 실행 결과

2026-11-15, Apple Silicon MPS, PyTorch 2.14.1, seed 12615147.
Run ID `3b96116880834fedb6b494c4a7f40bae`. 네 조건 각각 9개 독립 stream,
257 forward 호출을 새로 실행했다. CUDA/CPU 벤치마크와 CoTracker 모델 자체 실행은
이 감사에 포함되지 않았다. CoTracker grid 함수는 CPU에서 실행했다.

| 조건 | 전체 ≤4px | 중앙 오차 (원본 px) | 호출당 평균 / P95 (ms) |
| --- | ---: | ---: | ---: |
| A 관심점만 | 33/240 | 14.46 | 115.77 / 116.80 |
| B Local | 33/240 | 12.22 | 124.42 / 126.58 |
| C Global | 33/240 | 14.55 | 118.22 / 122.98 |
| D Local + Global | 33/240 | 12.75 | 125.78 / 126.61 |

같은 프레임 34와 관심점, 같은 98개 토큰 수를 유지하고 마지막 Global
보조점만 모델 xy에서 (−20, −30) 이동시켰다. 첫 self-attention 모듈의
관심점 출력 벡터에서 최대 절대 변화가 0.00127381이었다. 동일 입력 반복의
최대 변화는 0이었다. 이 별도 점검은 위 측정이 끝난 뒤 수행했으며
`attention_perturbation.json`에 기록했다. 저장된 36개 stream NPZ에서 원본
채점 mask와 query 좌표, 4px 통계 및 구간별 timing을 다시 검증했다.

- [비교 JSON](../outputs/tapnextpp_support/3b96116880834fedb6b494c4a7f40bae/comparison.json)
- [관심점 attention 개입 점검](../outputs/tapnextpp_support/3b96116880834fedb6b494c4a7f40bae/attention_perturbation.json)
- [D의 경계 밖 점 배치와 전체 trajectory 예시](../outputs/tapnextpp_support/3b96116880834fedb6b494c4a7f40bae/d_frame_041_corner_1.png)

### 프레임 0부터 표면점 추적과 3D 꼭짓점 복원

```bash
uv pip install --python .venv/bin/python 'mlx==0.32.4' 'mlx-metal==0.32.4'
PYTHONPATH=visual/third_party/tapnet:visual/third_party/co_tracker WANDB_MODE=disabled \
  .venv/bin/python main.py --config visual/configs/tapnextpp_surface_tracking.yaml
```

이 명령은 Apple Silicon의 [관측 기반 MLX 이동 영역](#현재-관측으로-이동하는-mlx-작업-영역)을 사용한다.
아래 PyTorch MPS 전체 화면 재생 결과를 재현하려면
`visual/configs/tapnextpp_surface_tracking_mps.yaml` 복사본에서
`tap_native_patch_crop_xyxy`, `tap_mps_fp16_vit`,
`tap_mps_fp16_recurrent_mlp`, `tap_mps_skip_past_query_embeddings`를 제거한다.

CUDA 기본 설정은 [관측 기반 이동 영역과 CUDA Graph](#cuda-이동-영역과-graph-재사용)를 사용한다.

```bash
PYTHONPATH=visual/third_party/tapnet:visual/third_party/co_tracker WANDB_MODE=disabled \
  python main.py --config visual/configs/tapnextpp_surface_tracking_cuda.yaml
```

동일한 149프레임 에피소드에서 CPU 꼭짓점 검출기를 매 프레임 실행한다.
프레임 0의 검출 Peg mask를 침식한 뒤 유효한 깊이가 있는 내부 픽셀에서
서로 떨어진 표면점 32개를 고른다. TAPNext++ 256 MPS 모델의 state는
프레임 0에서 한 번 초기화하며 꼭짓점 query와 가림 시점 재초기화를 사용하지 않는다.

현재 프레임의 TAPNext++ 가시성, RGB 기반 Peg mask, 깊이를 통과한 점들을
카메라 3D로 역투영한다. 초기 표면점과의 대응에서 5mm 이내 합의점
4개 이상을 요구하는 강체 정합으로 이동을 얻는다. 초기 검출 꼭짓점을 그 변환으로
옮기고, 현재 검출에 3D 거리 12mm 이내 대응이 있으면 해당 관측을 채택한다.
검출된 보이는 꼭짓점은 별도로 같은 프레임 GT와 4px 일대일 매칭으로 채점한다.
가림 결과의 물리 ID는 채점할 때만 동결 GT로 연결한다. 미해결 요청은 실패로 세고
중앙 오차는 유한한 예측에서 계산한다. detector, TAPNext++ 및 융합 시간은
프레임 1–148에서 계산하며 TAPNext++ 측정에는 MPS 동기화를 포함한다.

2026-11-15 실행 `5afa7855b9d443d89489991467c153b7`에서 초기 query 32개
모두 Peg actor 위였음을 추론 후 segmentation으로 확인했다. 149프레임에
걸쳐 강체 자세를 얻었고 합의점은 최소 4개, 중앙값 24개였다. 보이는 검출은
803/806, 가림 복원은 199/240, 꼭짓점 6의 프레임 83 이후 가림 요청은
50/61이 4px 이내였다. 가림 실패 41건은 모두 삽입 구간에 있었다.
MPS 추적 평균 / P95는 119.76 / 128.44ms였고, CPU 검출과 강체
정합을 포함한 직렬 처리 시간은 195.03 / 224.02ms였다.
단일 에피소드의 저장 RGB-D replay이며, 실시간 센서 입력과 CUDA에서의
실행은 검증하지 않았다.
[요약](../outputs/tapnextpp_surface_tracking/5afa7855b9d443d89489991467c153b7/summary.json) ·
[프레임별 궤적과 시간](../outputs/tapnextpp_surface_tracking/5afa7855b9d443d89489991467c153b7/tracks.npz)

기본 장치 선택으로 다시 실행한 `1157daba67354279bdcece6947f5c7fd`에서는
`TorchCornerDetector.main(device="auto")`가 이 Mac에서 `mps:0`을
선택했다. 위 CPU 검출 비교와 초기 32개 query, 채점 프레임 및 성공 건수가
일치했다. 프레임당 평균 / P95는 검출 44.67 / 56.14ms, TAPNext++
118.76 / 122.27ms, 전체 직렬 처리 164.76 / 176.67ms였다.
[기본 GPU 경로 실행 요약](../outputs/tapnextpp_surface_tracking/1157daba67354279bdcece6947f5c7fd/summary.json)

RTX 3090에서 검출기와 TAPNext++를 모두 `cuda:0`으로 지정한 실행
`3f9cdebe50b14c8da1aab4b919317e06`은 PyTorch 2.14.1+cu130을 사용했다.
MPS 실행과 영상·동결 입력·체크포인트 해시를 대조했고, 저장된 초기 32개
query와 채점 mask가 일치했다. 4px 성공 건수는 동일했다.

| 단계 | MPS 평균 / P95 (ms) | RTX 3090 CUDA 평균 / P95 (ms) |
| --- | ---: | ---: |
| 꼭짓점 검출 | 44.67 / 56.14 | 74.53 / 80.96 |
| TAPNext++ | 118.76 / 122.27 | 39.86 / 41.46 |
| CPU 3D 정합 및 융합 | 1.34 / 2.02 | 1.72 / 2.35 |
| 전체 직렬 처리 | 164.76 / 176.67 | 116.11 / 121.58 |

검출기 벽시계 시간에는 OpenCV CPU 단계와 장치 간 데이터 이동이 포함된다.
위 숫자는 호스트가 다른 두 실제 실행에서 얻었다.
[CUDA 실행 요약](../outputs/tapnextpp_surface_tracking/3f9cdebe50b14c8da1aab4b919317e06/summary.json) ·
[CUDA 프레임별 궤적과 시간](../outputs/tapnextpp_surface_tracking/3f9cdebe50b14c8da1aab4b919317e06/tracks.npz)

### 입력 해상도와 CUDA FP16 벤치마크

2026-11-15에 위 149프레임을 네 조건으로 재실행했다. 체크포인트, 동결
채점 입력과 에피소드 해시, 초기 표면점 32개의 좌표 및 채점 마스크가
네 실행에서 일치했다. MPS는 전체 화면을 192×192로 리사이즈했다.
CUDA는 같은 RTX 3090에서 256×256을 유지하고 공식 래퍼의 FP16 autocast를
활성화했다. 검출기는 각 장치의 기준 실행과 동일한 설정으로 실행했다.

위 표면점 재생 명령에서 MPS 최적화 필드 세 개를 제거한 config 복사본을 사용한다. MPS는
`tap_input_resolution: 192`를 추가한다. CUDA는 `device: cuda:0`과
`detector_device: cuda:0`으로 설정하고 `tap_cuda_autocast: true`를 추가한다.
기준 실행은 각각 해상도 256과 autocast 비활성화를 사용한다.

| 조건 | TAPNext++ 평균 / P95 (ms) | 전체 평균 / P95 (ms) | 가림 ≤4px | 삽입 ≤4px |
| --- | ---: | ---: | ---: | ---: |
| MPS 256 FP32 | 117.73 / 119.32 | 155.27 / 163.56 | 199/240 | 25/66 |
| MPS 192 FP32 | 73.01 / 74.89 | 110.99 / 119.37 | 92/240 | 0/66 |
| CUDA 256 FP32 | 39.67 / 41.31 | 111.71 / 115.35 | 199/240 | 25/66 |
| CUDA 256 FP16 autocast | 24.55 / 25.18 | 95.87 / 98.33 | 199/240 | 25/66 |

시간은 프레임 1–148에서 측정했다. 삽입 구간(128–148)의 추적 평균 / P95는
MPS 118.39 / 119.76 → 73.05 / 74.33ms,
CUDA 39.86 / 41.48 → 24.55 / 24.98ms였다.
MPS 192에서는 삽입 가림 66건 모두 좌표를 복원하지 못했고, 전체 자세를
얻은 프레임도 149개에서 87개로 줄었다. CUDA FP16은 149프레임 모두 자세를
얻었으며 삽입 가림의 중앙 오차는 4.5966px에서 4.5980px로 변했다.

- [비교 JSON](../outputs/tapnextpp_surface_tracking/input_resolution_and_precision.json)
- [MPS 256](../outputs/tapnextpp_surface_tracking/22a9d57f25ea41cda3f57b6e9feba5bc/summary.json) · [MPS 192](../outputs/tapnextpp_surface_tracking/4ccdf21ced9048ab95aaeb7a3efdd618/summary.json)
- [CUDA FP32](../outputs/tapnextpp_surface_tracking/c8b2683a37f344a09ab8f7a110d22b56/summary.json) · [CUDA FP16](../outputs/tapnextpp_surface_tracking/1c300c35c5a349888c626b5860db16ec/summary.json)

위 표의 CUDA 두 실행은 [검출 지연 최적화](corner_detection.md#검출-전체-지연-최적화)
직전의 검출기 사본을 사용했다. 최신 기본 검출기로 같은 CUDA FP16 재생을
반복한 결과 검출 평균 / P95는 69.62 / 72.42 → 68.02 / 71.20ms,
전체는 95.87 / 98.33 → 95.26 / 98.34ms였다. 저장된 추적 좌표와
가림 복원 좌표가 일치했다. 위 MPS 두 실행은 최적화 적용 이후 실행했다.
[최신 CUDA FP16 재실행](../outputs/tapnextpp_surface_tracking/bfd4c9896c5344179f2704a23d3c1ccf/summary.json)

### 고정 ROI와 CUDA autocast 캐시

2026-11-15에 최신 꼭짓점 검출기를 사용해 같은 149프레임을 다시 비교했다.
실행 config에 `tap_crop_xyxy: [0, 145, 320, 465]`를 추가하면 해당 원본 RGB
영역만 TAPNext++에 전달한다. 오른쪽과 아래는 범위에서 제외한다. 이 ROI는
이 에피소드의 영상을 기준으로 정했고 좌표는 기존 원본 RGB-D 픽셀로 되돌린다.
전체 화면을 쓰려면 필드를 생략한다. CUDA에서 `tap_cuda_autocast: true`와
`tap_cache_cuda_autocast: true`를 함께 쓰면 프레임 간 FP16 가중치 캐시를
재사용한다. 두 옵션의 기본값은 `false`다.
MPS에서 `tap_mps_fp16_vit: true`를 사용하면 각 공간 Transformer 블록을
FP16으로 실행하고 그 출력을 다음 recurrent 블록에 FP32로 전달한다.
이 설정도 기본값은 `false`다.

| 조건 | TAPNext++ 평균 / P95 (ms) | 가림 ≤4px | 삽입 ≤4px |
| --- | ---: | ---: | ---: |
| MPS 256 전체 화면 | 117.73 / 119.32 | 199/240 | 25/66 |
| MPS 256 Transformer FP16 | 112.22 / 114.19 | 199/240 | 25/66 |
| MPS 224 전체 화면 | 96.80 / 98.55 | 107/240 | 0/66 |
| MPS 192 고정 ROI | 72.18 / 73.65 | 145/240 | 9/66 |
| MPS 224 고정 ROI | 95.68 / 97.74 | 154/240 | 2/66 |
| CUDA 256 FP16 기존 | 25.54 / 25.82 | 199/240 | 25/66 |
| CUDA 256 FP16 캐시 | 22.68 / 23.71 | 199/240 | 25/66 |

CUDA 저장 출력은 추적 좌표, 가시성, 복원 좌표까지 일치했다. 같은 프로세스에서
검출기 없이 원본 149프레임을 캐시 끔 → 켬 → 켬 → 끔 순서로 재생한 결과,
평균은 캐시 끔 24.30 / 24.19ms, 켬 21.66 / 21.56ms였다. 전체 파이프라인
한 쌍에서 평균은 95.26 → 94.08ms였고 P95는 98.34 → 106.83ms였다.
검출 시간 변동이 포함된 값이며, 전체 P95 개선은 관측되지 않았다.

MPS Transformer FP16은 240건의 개별 성공·실패 판정과 표면점 가시성이
기준과 일치했다. 삽입 중앙 오차는 4.5966 → 4.6058px였고, 전체 지연
평균 / P95는 155.27 / 163.56 → 150.00 / 156.29ms였다.
MPS 선형층 가중치만 FP16으로 바꾼 실험은 118.81ms였고, 전체 backbone을
FP16으로 바꾼 실험은 100.55ms였으나 삽입 성공이 0/66으로 떨어졌다.

[구간별 비교 JSON](../outputs/tapnextpp_surface_tracking/additional_tracker_optimization.json) ·
[CUDA 캐시 실행](../outputs/tapnextpp_surface_tracking/dd1165b85b8843e0910268a399980db0/summary.json) ·
[MPS Transformer FP16 실행](../outputs/tapnextpp_surface_tracking/fab68806f62e4194942b1b27508ae4df/summary.json)

### MPS 네이티브 패치 축소

2026-11-15에 M4 Pro에서 기존 149프레임과 초기 표면점 32개를 고정했다.
검출 좌표, 입력 체크섬과 채점 기준은 실행 간 일치했다. 공간 Transformer FP16의
블록별 동기화 계측에서 12개 블록의 프레임당 소계는 공간 Transformer 40.14ms,
recurrent MLP 38.62ms, recurrent 상태 처리와 프로젝션 28.19ms였다.
블록별 동기화는 병목을 확인할 때만 사용했다.

입력은 전체 화면을 기존과 같이 256×256으로 변환한다. 그 안에서 지정한
패치만 projection에 전달하고 해당 영역의 원래 이미지 위치 임베딩을 그대로 쓴다.
Query 임베딩과 출력 좌표계는 전체 영상 기준이다. `tap_native_patch_crop_xyxy`의
순서는 32×32 패치 격자의 왼쪽, 위, 오른쪽, 아래이며 오른쪽과 아래는 제외한다.
`tap_mps_fp16_recurrent_mlp`는 recurrent MLP 내부만 FP16으로 바꾸고
recurrent 입력과 상태 업데이트는 FP32로 유지한다.

```bash
PYTHONPATH=visual/third_party/tapnet:visual/third_party/co_tracker WANDB_MODE=disabled \
  .venv/bin/python main.py --config visual/configs/tapnextpp_surface_tracking_mps.yaml
```

| 조건 | 이미지 토큰 | 추적 평균 / P95 (ms) | 전체 평균 / P95 (ms) | 가림 ≤4px | 삽입 ≤4px | 삽입 중앙 오차 (px) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 전체 화면 FP32 | 1024 | 117.73 / 119.32 | 155.27 / 163.56 | 199/240 | 25/66 | 4.60 |
| 전체 화면 Transformer FP16 | 1024 | 112.22 / 114.19 | 150.00 / 156.29 | 199/240 | 25/66 | 4.61 |
| 전체 화면 Transformer·recurrent MLP FP16 | 1024 | 104.74 / 106.14 | 140.89 / 148.30 | 199/240 | 25/66 | 4.60 |
| 넓은 패치 영역, Transformer FP16 | 576 | 68.87 / 70.03 | 105.50 / 111.80 | 215/240 | 41/66 | 3.49 |
| 넓은 패치 영역, Transformer·recurrent MLP FP16 | 576 | 65.06 / 65.83 | 102.49 / 109.48 | 211/240 | 37/66 | 3.55 |
| 좁은 패치 영역, Transformer·recurrent MLP FP16 | 440 | 55.23 / 56.37 | 91.72 / 97.41 | 200/240 | 26/66 | 5.42 |

PyTorch MPS 설정 파일은 넓은 패치 영역 `[0, 7, 24, 31]`과 두 FP16 옵션을 사용한다.
이 에피소드에서 삽입 성공 수가 가장 높았던 조건을 재현하려면
`tap_mps_fp16_recurrent_mlp: false`로 바꾼다. 좁은 영역은 `[0, 8, 20, 30]`이다.
패치 범위는 이 카메라와 에피소드에서 선택했다. 넓은 영역의 두 FP16 조건은
전체 화면 대비 가림 성공 19건을 얻고 7건을 잃었다. 전체 화면 FP16 두 블록은
240건 각각의 성공·실패 판정을 유지했다. 다른 에피소드와 CUDA의 새 패치 경로는
이번에 실행하지 않았다.

[프레임 검증 및 전체 비교](../outputs/tapnextpp_surface_tracking/mps_native_patch_optimization.json) ·
[넓은 영역 실행](../outputs/tapnextpp_surface_tracking/fc8d8ce76e8a409ba248a25e3f817eb5/summary.json) ·
[좁은 영역 실행](../outputs/tapnextpp_surface_tracking/7534999d95f34437a15b03e42652b487/summary.json)

PyTorch MPS 설정의 `tap_mps_skip_past_query_embeddings: true`는 프레임 0에서
표면점을 처음 주입할 때 기존 위치 임베딩을 사용한다. 이어지는 프레임에는
모든 query 시점이 이미 지났으므로 위치 임베딩을 샘플링하는 대신 TAPNext++의
mask token을 바로 사용한다. `false`로 설정하면 모든 프레임에서 기존 계산을 한다.

2026-11-15에 M4 Pro에서 동일한 149프레임을 끔 → 켬 → 켬 → 끔 순으로
재생했다. 꼭짓점 검출기는 `bb01149` 소스로 고정했고, 체크포인트와 입력
체크섬이 일치했다. 아래 시간은 각 조건에서 초기 프레임을 제외한 두 실행의
프레임별 측정값을 합쳐 집계했다.

| 조건 | 추적 평균 / P95 (ms) | 삽입 구간 추적 평균 / P95 (ms) | 전체 평균 / P95 (ms) | 가림 ≤4px | 삽입 ≤4px |
| --- | ---: | ---: | ---: | ---: | ---: |
| 매 프레임 query 위치 임베딩 | 65.22 / 66.67 | 65.34 / 66.40 | 102.51 / 110.56 | 211/240 | 37/66 |
| 첫 프레임 이후 query 위치 임베딩 생략 | 62.67 / 63.74 | 62.60 / 63.36 | 100.30 / 108.46 | 211/240 | 37/66 |

네 실행의 시간 외 추적 배열이 모두 바이트 단위로 일치했다. 삽입 구간의
가림 복원 중앙 오차는 3.5488px였다.

[끔 1](../outputs/tapnextpp_surface_tracking/f9a92eb028f34b779dbf41dbbd31a025/summary.json) ·
[켬 1](../outputs/tapnextpp_surface_tracking/9550361e75b84749b4a77eba2f9638f8/summary.json) ·
[켬 2](../outputs/tapnextpp_surface_tracking/a763b3720f20464b881e9a58aa356a4d/summary.json) ·
[끔 2](../outputs/tapnextpp_surface_tracking/05a81d5309664ba9a44a28aeb3df2505/summary.json)

### MPS recurrent 블록 컴파일

PyTorch MPS 설정의 복사본에 `tap_mps_compile_recurrent: true`를 추가하면
기존 FP16/FP32 구성을 유지한 채 recurrent residual 블록을 `torch.compile`로
실행한다. 측정 전에 초기 프레임과 state가 있는 프레임을 각각 한 번 실행하며,
그 시간을 실험 metadata의 `tracker_compile_warmup_ms`에 기록한다.
이 옵션의 기본값은 `false`다.

2026-11-15에 M4 Pro, PyTorch 2.14.1에서 동일한 149프레임을 끔 → 켬 → 켬 → 끔
순으로 재생했다. 양쪽 모두 첫 프레임 이후 query 임베딩을 생략하고 576패치를
사용했다. 꼭짓점 검출기는 `bb01149` 소스로 고정했다. 아래 시간은 조건별 두
실행의 프레임 측정값을 합쳐 집계했으며, 초기 프레임과 워밍업은 제외했다.

| 조건 | 추적 평균 / P95 (ms) | 삽입 추적 평균 / P95 (ms) | 전체 평균 / P95 (ms) | 가림 ≤4px | 삽입 ≤4px |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPS 기본 | 62.69 / 63.28 | 62.83 / 63.16 | 98.69 / 104.38 | 211/240 | 37/66 |
| recurrent 블록 컴파일 | 51.53 / 52.15 | 51.57 / 52.15 | 88.02 / 93.65 | 211/240 | 37/66 |

240개 가림 샘플 각각에서 복원 가능 여부와 4px 판정이 일치했다. 삽입 복원
오차의 중앙값은 3.5488px에서 3.5494px로 바뀌었고 삽입 복원 좌표의 최대
차이는 0.2212px였다. 빈 Inductor 캐시로 시작한 컴파일 실행의 워밍업은
2.13초였다. 결과는 이 MPS 호스트와 단일 에피소드에서 확인했다.

[기본 1](../outputs/tapnextpp_surface_tracking/90f7fcaae59c430792b4c8bcc0a98cb0/summary.json) ·
[컴파일 1](../outputs/tapnextpp_surface_tracking/e4227796908443e6a8150b84828e2c9d/summary.json) ·
[컴파일 2](../outputs/tapnextpp_surface_tracking/2d6096a10a524f09acbc3b29bee857d4/summary.json) ·
[기본 2](../outputs/tapnextpp_surface_tracking/56a6cef420284826a7778ff61adcd82c/summary.json)

별도 프로토타입에서 single-step recurrence·정규화·MLP 게이트를 Metal 커널로
처리하고 마지막 Transformer에서 표면 query의 출력만 계산했다. 같은 576패치에서
평균 49.33ms, P95 50.16ms로 240개의 4px 판정을 유지했다. 384패치
`[0, 11, 24, 27]`와 recurrent 컴파일 조합은 평균 38.39ms, P95 39.35ms였다.
삽입에서 기존 성공 3건을 잃고 12건을 얻어 46/66을 기록했다. 패치 범위는
같은 에피소드의 영상에서 선택했으며 다른 에피소드에서 검증하지 않았다.

[Metal 조합](../outputs/tapnextpp_surface_tracking/c6683ceb696e4a1ab58a1cd3d5c54bc2/summary.json) ·
[384패치](../outputs/tapnextpp_surface_tracking/a9f53e2122474011b7d48aeb26f4c436/summary.json)

### MLX 온라인 추적

Apple Silicon의 MLX 0.32.4에서 공식 PyTorch 체크포인트를 시작할 때 한 번
변환한다. 256px 네이티브 모델에 256×256 또는 512×512 BGR 영상을 받으며,
512×512에서는 2:1 bilinear 리사이즈를 사용한다. 첫 프레임에 query를 주입하고
이후에는 mask token을 사용한다. Temporal convolution 이력과 RG-LRU 상태는
FP32로 MLX GPU에 유지한다. 공간 Transformer와 recurrent 채널 MLP는 FP16이다.

```bash
uv pip install --python .venv/bin/python 'mlx==0.32.4' 'mlx-metal==0.32.4'
PYTHONPATH=visual/third_party/tapnet:visual/third_party/co_tracker WANDB_MODE=disabled \
  .venv/bin/python main.py --config visual/configs/tapnextpp_surface_tracking.yaml
```

`visual/configs/tapnextpp_mlx_block.yaml`은 컴파일된 MPS 이전 실행의
`tracks.npz`에서 query를 읽어 동일한 149프레임의 첫 블록 입력을 준비한다.
첫 블록의 query 출력 상대 RMSE는 0.043%, 마지막 RG-LRU 상태의 상대 RMSE는
0.000081%였다. 이 오차는 모델 내부 값이며 원본 영상 픽셀 오차와 단위가 다르다.
[첫 블록 기록](../outputs/tapnextpp_mlx_block/a1ea5e25b27046d8a9848cd8bfc69ad4/summary.json)

전체 재생은 2026-11-15에 M4 Pro에서 MLX → MPS → MPS → MLX 순으로 진행했다.
두 경로에 동일한 체크포인트와 576패치, 초기 표면점 32개를 사용했고 검출기는
`bb01149` 소스로 고정했다. 두 실행의 프레임별 시간에서 워밍업과 초기 프레임을
제외하고 합산했다.

| 경로 | 추적 평균 / P95 (ms) | 삽입 추적 평균 / P95 (ms) | 전체 평균 / P95 (ms) | 가림 ≤4px | 삽입 ≤4px |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPS recurrent 컴파일 | 51.72 / 52.87 | 51.70 / 52.88 | 89.25 / 96.55 | 211/240 | 37/66 |
| MLX 12블록 | 46.91 / 47.36 | 46.91 / 47.30 | 83.28 / 89.90 | 211/240 | 37/66 |

가림 샘플 240건의 복원 가능 여부와 4px 판정이 모두 일치했다. 표면점 가시성
판정도 일치했다. 삽입 복원 오차의 중앙값은 MPS 3.5494px, MLX 3.5554px였고
두 경로의 삽입 복원 좌표 최대 차이는 0.1470px였다. 검증 범위는 이 호스트와
동일한 149프레임 에피소드다.

[MLX 1](../outputs/tapnextpp_surface_tracking/d6120996f0704231a62fbbff56300663/summary.json) ·
[MPS 1](../outputs/tapnextpp_surface_tracking/b36d384a12df4308a70d179d759f57ec/summary.json) ·
[MPS 2](../outputs/tapnextpp_surface_tracking/c305e49668454aa4bc3c325c1297ba76/summary.json) ·
[MLX 2](../outputs/tapnextpp_surface_tracking/033d2b9d8b2c440d86c579b1d8a6e74b/summary.json)

#### MLX 연산 및 상태 배치 최적화

아래 고정 영역 비교에서는 다음 최적화를 함께 적용했다. 입력 해상도, 576패치,
32개 표면 query, 모델 가중치 및 FP16/FP32 구성을 유지한다.

- 마지막 공간 Transformer는 query 행의 attention 출력과 MLP만 계산한다.
  이미지 토큰도 K/V를 제공하고 모든 블록에서 recurrent 상태를 갱신한다.
- Recurrent `linear_x`와 `linear_y`의 FP32 가중치를 로드할 때 묶어
  프레임마다 하나의 행렬곱으로 계산한다.
- 4-tap temporal convolution은 두 쌍의 합으로 계산해 곱셈 후 reduction용
  중간 버퍼를 없앤다. 이력은 시간축별로 연속 저장하되 외부에 전달하는
  `[tokens, time, width]` 형태는 유지한다.
- BGR 입력은 한 번 전송하고 GPU에서 crop, RGB 변환, 리사이즈, 패치 임베딩을
  컴파일한다. 필요한 crop만 리사이즈하며 기존 전체 화면 리사이즈 후 crop과
  같은 임베딩을 생성한다. 프레임마다 수행하던 NumPy RGB 복사를 제거했다.

2026-11-15 M4 Pro, MLX 0.32.4에서 커밋 `3b4d536`의 MLX 구현과 비교했다.
동시에 실행 중인 앞선 실험이 종료된 뒤 기존 → 최적화 → 최적화 → 기존
순서로 같은 149프레임을 재생했다. 검출기는 `bb01149` 소스로 고정하고
`ExperimentLogger`에 실행별 config, 소스 SHA-256, 체크포인트·에피소드·평가
참조 해시를 기록했다. 아래 시간은 조건별 두 실행에서 워밍업과 프레임 0을
제외한 296개 측정값을 합산한 값이다. 삽입 구간은 프레임 128–148이다.

| 경로 | 추적 평균 / P95 (ms) | 삽입 추적 평균 / P95 (ms) | 전체 평균 / P95 (ms) | 가림 ≤4px | 삽입 ≤4px |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 MLX | 47.17 / 48.24 | 47.10 / 48.18 | 83.10 / 89.85 | 211/240 | 37/66 |
| 최적화 MLX 고정 영역 | 44.19 / 44.84 | 44.07 / 44.34 | 80.43 / 86.41 | 211/240 | 37/66 |

추적 평균은 6.32%, P95는 7.03% 감소했다. 초기 query, GT, 평가 mask,
검출 결과, 표면점 가시성 및 복원 가능 여부가 같고, 가림 240건과 삽입 66건의
개별 4px 판정이 모두 같다. 각 경로의 두 반복에서 시간 이외 출력 배열도
일치했다. 다만 행렬곱과 합산 순서가 바뀌어 기존 경로와 좌표가 완전히 같지는
않다. 표면점 최대 차이는 0.06195px, 복원 좌표 최대 차이는 0.36419px,
삽입 복원 좌표 최대 차이는 0.22730px다. 삽입 오차 중앙값은
3.55536 → 3.57386px였다.

256/512px 입력 × 전체/576패치 crop × query 1/7/32개, 총 12조합의 초기 및
연속 3프레임도 확인했다. 이미지 임베딩과 가시성은 기존과 일치했고, reset은
해당 구현의 최초 결과를 정확히 재현했다. 기존 구현에서 만든 이력의 이어받기,
FP32 상태 형태, 잘못된 입력의 거부도 확인했다. CUDA와 PyTorch MPS는 이번
변경에서 재실행하지 않았다. 정확도 보존 결과는 이 149프레임 에피소드에
대한 검증이며 다른 영상에서의 정확도를 보장하지 않는다.

기존 MLX의 함수별 동기화 계측에서는 12블록 합계가 공간 Transformer 22.49ms,
recurrent 채널 MLP 19.20ms, recurrent 상태·프로젝션 11.90ms였다.
구간마다 동기화한 값이므로 이를 전체 지연 시간에 단순 합산하지 않는다.
중간 추론 텐서는 MLX GPU에 있고 NumPy는 입출력 경계에서 사용한다.
추가 FP16 프로젝션, 행렬곱 padding, 전체 모델 단일 compile도 시험했으나
채택한 조합보다 느렸다. 8-bit 가중치 행렬곱의 별도 계측에서도 유효한
전체 MLP 가속 근거를 얻지 못해 기본 경로에 포함하지 않았다.

[기존 1](../outputs/tapnextpp_surface_tracking/6df323e609d247b0912ee75ab54868ab/summary.json) ·
[최적화 1](../outputs/tapnextpp_surface_tracking/3448a78de1174ed1ad10fe7a5017de79/summary.json) ·
[최적화 2](../outputs/tapnextpp_surface_tracking/8e2bb65870d34265a11e0a9a8398e4f4/summary.json) ·
[기존 2](../outputs/tapnextpp_surface_tracking/f9940940f7f8452d912c4f2b1c66a8d2/summary.json) ·
[재생 비교](../outputs/tapnextpp_mlx_verification/dab1851e89904e03bd0eae62138b4ffd/replay_comparison.json) ·
[입력·상태 검증](../outputs/tapnextpp_mlx_verification/dab1851e89904e03bd0eae62138b4ffd/summary.json) ·
[구간별 계측](../outputs/tapnextpp_mlx_profile/24cf3430186842d3b23b1ca4802b5f5e/summary.json)

#### MLX Transformer·recurrent MLP의 FP16 GELU

기존 GELU를 FP16 입력의 65,536개 비트 패턴에 대해 시작할 때 한 번 계산한다.
Transformer MLP는 기존 erf GELU 테이블을 조회한다. Recurrent MLP는 Metal
커널 하나에서 gate/value bias를 더하고 기존 tanh GELU 테이블을 조회한 뒤
두 값을 곱한다. 두 경로 모두 기존 FP16 반올림 위치와 행렬곱을 유지한다.

2026-11-15 M4 Pro, MLX 0.32.4에서 `deb5233` 기준과 변경 구현을
기준 → 변경 → 변경 → 기준 순서로 재생했다. 동일한 149프레임, 576패치,
표면점 32개와 `bb01149` 검출기를 사용했다. 조건별 두 실행에서 워밍업과
프레임 0을 제외하고 추적 시간을 합산했다.

| 경로 | 추적 평균 / P95 (ms) | 삽입 추적 평균 / P95 (ms) | 전체 평균 / P95 (ms) | 가림 ≤4px | 삽입 ≤4px |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 MLX | 44.22 / 45.14 | 44.09 / 44.50 | 81.67 / 90.51 | 211/240 | 37/66 |
| GELU 테이블 + 게이트 융합 | 43.55 / 43.95 | 43.42 / 43.81 | 82.82 / 89.98 | 211/240 | 37/66 |

네 실행에서 시간 외 배열이 바이트 단위로 일치했다. 삽입 복원 오차
중앙값은 3.57386px로 같았다. 256/512px 입력 × 전체/576패치 crop ×
query 1/7/32개에서도 좌표·가시성, reset 및 기존 이력 이어받기를 확인했다.
검출 및 RGB-D 융합을 포함한 전체 평균은 이번 두 쌍에서 개선되지 않았다.

동일 입력에 대해 각 블록을 따로 동기화해 측정한 12블록 합계는 recurrent
MLP 21.79 → 20.64ms, 공간 Transformer 25.24 → 24.97ms였다. 각 구간의
동기화 비용이 포함되므로 이 합계는 전체 추적 시간과 별도로 비교한다.

Attention을 별도 행렬곱과 softmax로 분리한 후보는 평균 44.65ms였다.
두 MLP의 중간 폭을 3,072에서 2,560으로 줄인 후보는 39.89ms였으나
삽입 4px 판정이 37/66에서 0/66으로 떨어져 제외했다.

[기준 1](../outputs/tapnextpp_surface_tracking/7d3de9fcd7be448e9ba5f578fc0bfa7c/summary.json) ·
[변경 1](../outputs/tapnextpp_surface_tracking/41472e7ee0834f468bd286e2fb4e9d46/summary.json) ·
[변경 2](../outputs/tapnextpp_surface_tracking/2c784708cbd44e7c8e6201c3683d127f/summary.json) ·
[기준 2](../outputs/tapnextpp_surface_tracking/cf344330a22b4dc38df6f5a8ae205271/summary.json) ·
[재생 비교](../outputs/tapnextpp_mlx_verification/ad38ee23447f437f9c55d7f478f5eaed/replay_comparison.json) ·
[블록별 계측](../outputs/tapnextpp_mlx_layer_profile/c168e1a102364351af0e1987c25deef3/summary.json) ·
[입력·상태 검증](../outputs/tapnextpp_mlx_verification/ad38ee23447f437f9c55d7f478f5eaed/summary.json)

#### MLX 작업 영역 축소

저장된 expert 에피소드의 오프라인 재생 비교에는
`visual/configs/tapnextpp_surface_tracking_mlx_workspace.yaml`을 사용한다.
기존 576패치 대신 `[2, 15, 16, 25]`의 140패치를 사용한다. 좌표는 32×32
네이티브 패치 격자 기준이며 오른쪽·아래쪽 경계는 제외한다. 512px 원본
영상에서는 x=32–255, y=240–399에 해당한다. 모델 해상도, 체크포인트,
query 32개, 원본 좌표계 및 프레임마다 recurrent 상태를 갱신하는 방식은 같다.

```bash
PYTHONPATH=visual/third_party/tapnet:visual/third_party/co_tracker WANDB_MODE=disabled \
  .venv/bin/python main.py --config visual/configs/tapnextpp_surface_tracking_mlx_workspace.yaml
```

140패치 범위는 첫 관측에서 정하지 않았다. 같은 expert 에피소드의 후반
프레임과 복원 결과도 확인하고 범위를 골랐다. 아래 수치는 이 에피소드에
맞춘 오프라인 재생 결과다. Closed-loop에서 실행 정책이 다른 경로를 만들면
Peg가 이 범위를 벗어날 수 있으므로 rollout 설정으로 사용하지 않는다.
Closed-loop 평가에는 현재·과거 관측만으로 영역을 정하고, 가림이나 영역
이탈 시 재탐색까지 포함해서 별도 rollout에서 지연 시간과 복원을 측정한다.

2026-11-15 M4 Pro, MLX 0.32.4에서 `6fd015b`의 구현과 `bb01149`의
검출기를 고정했다. 같은 149프레임에서 기존 → 140패치 → 140패치 → 기존
순으로 재생했고, 각 조건의 첫 프레임을 제외한 296개 시간 표본을 합산했다.

| 경로 | 추적 평균 / P95 (ms) | 삽입 추적 평균 / P95 (ms) | 검출 시작 → 추적 완료 평균 / P95 (ms) | 가림 ≤4px | 삽입 ≤4px |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 576패치 | 44.28 / 45.07 | 44.21 / 45.35 | 81.38 / 87.79 | 211/240 | 37/66 |
| 작업 영역 140패치 | 17.25 / 23.05 | 16.71 / 19.07 | 52.88 / 62.51 | 236/240 | 62/66 |

140패치는 기존 4px 성공을 모두 유지했고, 삽입 구간에서 25건이 추가로
성공했다. 삽입 오차 중앙값은 3.57 → 2.88px였고 전체 가림 오차 중앙값은
0.87 → 0.95px였다. 좌표와 가시성은 576패치와 달라졌다. 입력 query, GT,
채점 mask 및 검출 결과는 일치했고 각 설정의 두 반복에서 시간 이외 배열도
일치했다. RGB-D 융합까지 포함하면 140패치는 평균 54.05ms, P95 63.45ms였다.
PyTorch MPS 및 CUDA 경로는 이번에 다시 실행하지 않았다.

[기준 1](../outputs/tapnextpp_surface_tracking/4f65d1cefce047499f094d2439f83184/summary.json) ·
[140패치 1](../outputs/tapnextpp_surface_tracking/6eb7ce1f16ce40adbbe3648e928888dd/summary.json) ·
[140패치 2](../outputs/tapnextpp_surface_tracking/300d42d1764a4677901c68f2fb384ffa/summary.json) ·
[기준 2](../outputs/tapnextpp_surface_tracking/49d8c7f0837a45c6838e67dd9d69aa60/summary.json) ·
[재생 비교](../outputs/tapnextpp_surface_tracking/6eb7ce1f16ce40adbbe3648e928888dd/workspace_comparison.json) ·
[패치 범위 탐색](../outputs/tapnextpp_surface_tracking/6eb7ce1f16ce40adbbe3648e928888dd/workspace_screening.json)

#### 현재 관측으로 이동하는 MLX 작업 영역

MLX 기본 설정은 모델에 들어갈 패치를
매 프레임 관측에서 정한다. 시각 모듈은 첫 query, 현재 선택적 물체 mask와
직전의 가시 query만 받는다. 재생에서는 기존 RGB-D 검출기의 Peg mask를
전달한다. Expert 단계, 이후 프레임, GT를 영역 선택에 사용하지 않는다.
Mask 또는 query의 경계에 2패치 여유를 더하고 12, 16, 24, 32 중 필요한
크기의 정사각형을 고른다. 기존 영역이 같은 크기이고 새 경계를 포함하면
그대로 쓴다. 관측이 없으면 이전 가시점 주변에서 검색을 넓히고, 신뢰할
가시점이 없거나 빈 mask가 3프레임 이어지면 전체 화면을 처리한다.

선택 영역의 픽셀만 MLX로 옮겨 원래 위치 임베딩으로 투영한다. 이동 시
겹치는 절대 패치 좌표의 recurrent 이력과 query 이력을 이어받고,
새로 들어온 패치의 상태는 초기화한다. 이 동작은 패치와 query의 행 순서가
달라져도 상태를 다른 위치나 다른 query로 옮기지 않는다.

```bash
PYTHONPATH=visual/third_party/tapnet:visual/third_party/co_tracker WANDB_MODE=disabled \
  .venv/bin/python main.py --config visual/configs/tapnextpp_surface_tracking_mlx.yaml
```

2026-11-15 M4 Pro, MLX 0.32.4에서 검출기를 `bb01149`로 고정하고 같은
149프레임을 기준 → 이동 영역 → 이동 영역 → 기준 순으로 재생했다. 아래 시간은
각 조건의 초기 프레임을 제외한 두 실행을 합산했다.

| 경로 | 추적 평균 / P95 (ms) | 삽입 추적 평균 / P95 (ms) | 전체 평균 / P95 (ms) | 가림 ≤4px | 삽입 ≤4px |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기준 576패치 | 43.47 / 43.81 | 43.41 / 43.78 | 78.87 / 86.85 | 211/240 | 37/66 |
| 관측 기반 이동 영역 | 17.97 / 24.23 | 18.40 / 25.60 | 53.80 / 65.81 | 236/240 | 62/66 |

이 재생의 모든 프레임에서 144패치를 썼고, 영역은 프레임 86, 104, 119에서
이동했다. 삽입에서 기존 성공 2건이 실패로 바뀌고 27건이 새로 성공했다.
영역 선택과 결과 배열은 두 반복에서 일치했다. 고정 영역의 결과는 변경 전
기록과도 일치했다. 256/512px 입력에서 별도로 관측을 이동시키거나 끊어,
겹치는 패치와 query 상태, 새 패치 초기화, 전체 화면으로 확장하고 다시
관측된 영역을 따라가는 동작을 확인했다.

이 결과는 순차 재생 평가다. 이동 창이 실제 제어 행동에 미치는 영향과
검출이 끊긴 뒤의 추적 정확도 및 지연 시간은 별도의 closed-loop 평가가
필요하다. 위 비교는 MLX로 실행했다. CUDA 결과는 아래에 따로 기록했다.

[비교 결과](../outputs/tapnextpp_surface_tracking/ed1b96295de7459e8d993c71a7a75f82/adaptive_comparison.json) ·
[이동/관측 끊김 검증](../outputs/tapnextpp_surface_tracking/ed1b96295de7459e8d993c71a7a75f82/check_adaptive.log) ·
[프레임별 영역](../outputs/tapnextpp_surface_tracking/ed1b96295de7459e8d993c71a7a75f82/tracks.npz)

#### CUDA 이동 영역과 Graph 재사용

CUDA 기본 설정은 MLX와 같은 관측 기반
영역 선택을 CUDA FP16 autocast에 적용한다. 영역이 움직이면 절대 패치
좌표로 이어지는 상태를 찾고, 새 패치는 upstream의 초기 recurrent 연산으로
처리한다. 144패치 영역이 그대로인 연속 프레임에서는 CUDA Graph를
재사용한다. 초기 프레임, 영역 이동, 검색 범위 확대는 일반 CUDA 호출로
처리한다. Graph 이전에 반환한 상태도 이후 호출에서 계속 사용할 수 있다.
`tap_cuda_graph: false`로 설정하면 같은 이동 영역을 일반 CUDA 호출로 실행한다.

```bash
PYTHONPATH=visual/third_party/tapnet:visual/third_party/co_tracker WANDB_MODE=disabled \
  python main.py --config visual/configs/tapnextpp_surface_tracking_cuda.yaml
```

2026-11-15 RTX 3090, PyTorch 2.14.1+cu130에서 `bb01149` 검출기와 같은
149프레임으로 비교했다. 전체 화면 → 이동 영역 → 이동 영역 → 전체 화면
순으로 실행한 뒤 Graph를 추가한 경로를 두 번 재생했다. 조건마다 첫 프레임을
제외한 두 실행의 시간을 합산했다. Graph 경로의 워밍업은 약 290ms였다.

| 경로 | 추적 평균 / P95 (ms) | 삽입 추적 평균 / P95 (ms) | 전체 평균 / P95 (ms) | 가림 ≤4px | 삽입 ≤4px |
| --- | ---: | ---: | ---: | ---: | ---: |
| 전체 화면, CUDA FP16 캐시 | 22.01 / 22.51 | 22.04 / 22.39 | 86.64 / 89.12 | 199/240 | 25/66 |
| 이동 영역, 일반 CUDA 호출 | 21.94 / 22.18 | 21.65 / 21.92 | 87.28 / 90.57 | 236/240 | 62/66 |
| 이동 영역, CUDA Graph | 9.43 / 8.95 | 8.85 / 9.11 | 75.51 / 78.42 | 236/240 | 62/66 |

Graph와 일반 호출에서 시간 외 결과 배열이 일치했다. 기존 4px 성공은
모두 유지했고, 삽입 37건이 새로 성공했다. 이 영상에서는 144패치를 계속
사용했고, 영역이 실제 이동한 세 프레임은 Graph 설정에서도 평균 38.11ms,
최대 39.37ms였다. 별도 관측 이동 및 끊김 테스트에서 CUDA 상태의 재배치,
전체 화면 확대, 재관측 후 복귀, Graph를 반복 호출한 뒤에도 앞선 상태를
다시 사용할 수 있음을 확인했다. 영역 선택을 공유하도록 정리한 뒤 MLX
재생의 시간 외 결과도 기존 기록과 일치했다. 제어 정책을 실행하는
closed-loop의 지연 시간과 성공률은 이번 측정에 포함되지 않았다.

기존 고정 영역 비교를 재현하려면 설정 복사본에서 MLX는
`tap_adaptive_crop: false`와 `tap_native_patch_crop_xyxy: [0, 7, 24, 31]`을,
CUDA는 `tap_adaptive_crop: false`와 `tap_cuda_graph: false`를 사용한다.

[CUDA 비교 결과](../outputs/tapnextpp_surface_tracking/f67610fc6aea4e8285d7ecc214485ce8/cuda_adaptive_comparison.json) ·
[CUDA 상태 검증](../outputs/tapnextpp_surface_tracking/f67610fc6aea4e8285d7ecc214485ce8/check_cuda_graph.log) ·
[Graph 프레임별 기록](../outputs/tapnextpp_surface_tracking/f67610fc6aea4e8285d7ecc214485ce8/tracks.npz)

#### 통합 리플레이의 MPS 검출기 입력

TAPNext++ 표면 추적 리플레이는 메인 검출기의 장치가 MPS이면 RGB, 깊이,
카메라 내·외부 보정값을 프레임마다 MPS에 올려 검출기에 넘긴다. 추적과
3D 융합에는 원래 호스트 프레임을 사용한다. 설정의 `detector_inputs`는
기본값 `auto`와 비교용 `numpy`, 명시적 `mps`와 `cuda`를 지원한다.

Apple M4 Pro에서 MLX 0.32.4 이동 영역 추적과 같은 Expert 에피소드
`12615147`의 149프레임을 사용했다. NumPy → MPS → MPS → NumPy 순서로
실행하고, 각 실행의 초기 프레임을 뺀 조건당 296프레임을 집계했다.
검출 시간에는 프레임별 MPS 전송, Peg 출력 회수와 추적용 BGR 변환이
포함된다. 전체 시간은 검출·추적·융합을 포함하며 파일 읽기와 모델
로드, 워밍업 및 채점은 계측 구간 밖이다.

| 구간 | NumPy 입력 평균 / p95 (ms) | MPS 입력 평균 / p95 (ms) |
| --- | ---: | ---: |
| 검출 및 입력 준비 | 36.04 / 43.95 | 33.30 / 40.31 |
| TAPNext++ 추적 | 19.66 / 27.50 | 17.55 / 24.87 |
| 3D 융합 | 1.16 / 1.63 | 1.12 / 1.75 |
| 전체 | 56.85 / 70.19 | 51.96 / 63.37 |

평균 전체 지연 기준 처리율은 **17.59 → 19.24 FPS**였다. 추적기와
융합 설정은 두 조건에서 같았고, 표의 구간별 시간에는 실행 간 변동이 포함된다.
추적 결과 좌표와 가시성, 강체 정합 결과, 융합 좌표와 영역 선택은 전체
149프레임에서 두 조건과 모든 반복이 일치했다.
메인 검출기의 물체 ID와 출력 순서, 픽셀·카메라·월드 좌표, 마스크,
메타데이터와 진단 정보도 149프레임에서 정확히 일치했다.
[통합 비교 원본](../outputs/tapnextpp_mps_detector_inputs/comparison.json) ·
[검출기 출력 비교](../outputs/tapnextpp_mps_detector_inputs/detector_exact.json)

#### 통합 리플레이의 CUDA 검출기 입력

`detector_inputs: auto`는 CUDA 검출기에도 RGB-D와 보정값을 해당 장치의
텐서로 넘긴다. TAPNext++의 BGR 입력과 3D 융합은 호스트 프레임을
사용한다. 이전 입력 전달 방식을 재현하려면 `detector_inputs: numpy`를
지정한다. CUDA 컴파일 워밍업에 사용할 에피소드 프레임 수는
`detector_warmup_frames`로 지정할 수 있다.

2026-11-15 Vast.ai RTX 3090에서 Expert 에피소드 `12615147`의 메인 카메라
149프레임으로 비교했다. 양쪽 모두 2픽셀 격자와 확대 컴파일을 사용하는
메인 CUDA 검출기 및 이동 영역 TAPNext++ CUDA Graph를 실행했다. 조건별로
40프레임을 검출 워밍업에 사용했다. NumPy → CUDA → CUDA → NumPy 순서로
실행했으며, 각 실행의 0번 프레임을 빼고 조건당 296프레임을 집계했다.
검출 시간은 프레임 전송, Peg 출력 회수와 BGR 변환을 포함한다.

| 구간 | NumPy 입력 평균 / p95 (ms) | CUDA 입력 평균 / p95 (ms) |
| --- | ---: | ---: |
| 검출 및 입력 준비 | 34.16 / 38.18 | 30.53 / 34.15 |
| TAPNext++ 추적 | 9.42 / 8.92 | 9.44 / 8.96 |
| 3D 융합 | 1.85 / 2.80 | 1.82 / 2.70 |
| 전체 | 45.43 / 48.82 | 41.80 / 45.48 |

전체 처리율은 **22.01 → 23.93 FPS**였다. 0번 프레임까지 포함하면
21.96 → 23.87 FPS였다. 저장된 추적 좌표·가시성, 강체 정합과 융합 결과,
추적 영역은 전체 149프레임에서 두 조건과 모든 반복이 일치했다.
검출기의 물체 ID, 출력 순서, 픽셀·카메라·월드 좌표, 마스크, 메타데이터 및
진단 정보도 149프레임에서 정확히 일치했다.
[프레임별 검증과 통합 계측](../outputs/tapnextpp_cuda_detector_inputs/comparison.json)

#### RTX 3090에서 꼭짓점 검출과 추적 중첩

2026-11-15, RTX 3090에서 `af4dca2` 메인 검출기를 사용해 Expert 에피소드
`12615147` 149프레임을 재생했다. 두 중첩 조건 모두 검출에 사용한 CUDA RGB
텐서를 추적에도 사용했다. 현재 프레임의 Peg 마스크로 영역을 고른 뒤 같은
CPU 스레드에서 별도 CUDA 스트림에 추적을 제출했다. 검출을 완료한 후 두
결과로 융합했다. 조건별로 첫 프레임을 제외한 예열 후 두 반복을 집계했다.
전체 시간은 입력 준비부터 융합 완료까지이며 GNN 실행은 측정 범위 밖이다.

| 실행 순서 | 전체 평균 / P95 / 최대 (ms) | 가림 복원 ≤4px | 삽입 ≤4px |
| --- | ---: | ---: | ---: |
| 순차 실행 | 41.99 / 45.53 / 73.36 | 236/240 | 62/66 |
| Peg 마스크 준비 후 중첩 | 35.19 / 38.26 / 71.53 | 236/240 | 62/66 |
| 평면 피팅 완료 후 중첩 | 34.62 / 37.79 / 70.72 | 236/240 | 62/66 |

동일 프로세스의 교차 재생에서는 전체 149프레임의 물체 ID, 출력 순서,
픽셀·카메라·월드 좌표, 마스크, 메타데이터 및 진단 정보가 세 조건에서
일치했다. 가시 Peg의 검출은 802/806건이 4px 이내였다. 앞선 별도 프로세스
비교에서 마스크 준비 후 중첩 조건의 67번 프레임 검출 해시가 한 번 달랐고,
교차 재생에서 67번 프레임의 모든 검출 필드가 일치했다. 별도 CPU 작업자
스레드에서 추적을 호출한 조건은 평균 47.03ms로 늘었다. 이 결과는 CUDA
리플레이 실험이며, 메인 실행은 순차 경로다. 검출 최적화의 품질 기준은
[2,073프레임 홀드아웃](corner_detection.md#새-궤적에서-cuda와-mps-평가)이다.

[교차 재생 결과와 검출 해시](../outputs/tapnextpp_cuda_overlap/schedules.json) ·
[교차 재생 설정](../outputs/tapnextpp_cuda_overlap/schedules.yaml) ·
[별도 프로세스의 해시 비교](../outputs/tapnextpp_surface_tracking/566261ca81cf4487913f167d8a831853/ab_comparison.json) ·
[별도 CPU 작업자 스레드 비교](../outputs/tapnextpp_surface_tracking/58488d1ff4604912ad284e55eeba129e/ab_comparison.json)

#### CUDA 평면·윤곽 결과 전송 묶기

2026-11-16, 같은 RTX 3090과 Expert 에피소드 `12615147`에서
[검출 결과 전송 묶기](corner_detection.md#cuda-평면윤곽-결과의-일괄-전송)를
순차 실행과 평면 피팅 후 중첩에 각각 적용했다. 149프레임 전체를 채점하고,
지연은 첫 프레임을 제외한 각 조건의 두 예열 후 반복을 합산했다. 조건의
실행 순서를 두 번째 반복에서 뒤집었다. 삽입 시간은 동결 평가 자료에서
삽입으로 표시된 21프레임에 대해 집계했다.

| 실행 순서 | 전체 평균 / P95 (ms) | 삽입 평균 / P95 (ms) | 삽입 복원 ≤4px |
| --- | ---: | ---: | ---: |
| 순차 | 41.40 / 44.93 | 40.20 / 44.91 | 62/66 |
| 순차와 전송 묶기 | 40.97 / 44.46 | 39.75 / 44.15 | 62/66 |
| 평면 피팅 후 검출·추적 중첩 | 34.27 / 38.31 | 33.09 / 37.31 | 62/66 |
| 평면 피팅 후 중첩과 전송 묶기 | 33.56 / 36.58 | 32.32 / 36.61 | 62/66 |

전체 가림 복원은 네 조건에서 236/240건이었다. 149프레임의 검출 좌표,
물체 ID, 마스크, 순서, 메타데이터와 진단 정보의 해시가 일치했다.
강체 투영 좌표와 융합 좌표도 일치했다. 중첩에서 전송을 묶으면 융합까지
평균 0.71 ms가 줄었고, 30 ms 목표까지 3.56 ms가 남았다.

[교차 실행 결과](../outputs/corner_cuda_readbacks/integrated.json) ·
[재생 설정](../outputs/corner_cuda_readbacks/integrated.yaml)
