# TAPNext++ point tracking

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
