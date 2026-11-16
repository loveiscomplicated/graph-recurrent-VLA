# GNN 리더보드 · 11월 3주차

2026-11-16 기준. 고정 Hole의 보류 테스트셋 30개 궤적, 6,773프레임에서 비교했습니다. `Expert→Random`, `Expert→Random→Expert`, `Random`을 각각 10개씩 포함합니다. 두 모델은 같은 검출 좌표, 관측 Robot9, 30 mm 추적 게이트로 추론했습니다. 오차는 같은 Object21 GT로 계산했습니다.

위치 오차는 프레임별 3D 거리, 회전 오차는 GT와의 최단 회전각입니다. `P95`는 95번째 백분위수입니다. 기존 1위인 **자동 대응 Transformer + GNN · 조기 종료 재개**는 seed 0·1·2의 오차를 각 프레임에서 먼저 평균한 뒤 집계했습니다. **Fixed-Hole Expert v1**은 고정 Hole Expert 데이터에서 처음부터 학습한 Transformer와 GNN의 첫 버전입니다.

| 모델 | 테스트 loss ↓ | Peg 중심 위치 (mm) ↓ | Hole 중심 위치 (mm) ↓ | 끝점–Hole 평균 (mm) ↓ | Peg 회전 (°) ↓ | Hole 회전 (°) ↓ | Peg 반크기 X (mm) ↓ | Peg 반크기 Y (mm) ↓ | Peg 반크기 Z (mm) ↓ | Hole 반폭 (mm) ↓ | 크기 MAE (mm) ↓ | 끝점–Hole P95 (mm) ↓ | 끝점–Hole ≤5 mm ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [자동 대응 Transformer + GNN · 조기 종료 재개](experiment_leaderboard.md#detected-resume-transformer) | 6.865 | 53.12 | 35.72 | 39.21 | 5.17 | 4.21 | 6.92 | 3.53 | 3.57 | 2.89 | 4.23 | 122.55 | 7.1% |
| Fixed-Hole Expert v1 | 4.724 | 43.57 | 30.59 | 24.69 | 2.39 | 3.45 | 2.88 | 1.70 | 1.71 | 1.92 | 2.05 | 82.64 | 20.9% |

## 세부 성능

각 셀은 **Fixed-Hole Expert v1 / 자동 대응 Transformer + GNN · 조기 종료 재개** 순서입니다. 위치와 회전은 해당 정책의 전체 프레임 평균 오차입니다.

| 정책 | 프레임 | Loss | Peg 위치 (mm) | Hole 위치 (mm) | 끝점–Hole (mm) | Peg 회전 (°) | Hole 회전 (°) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 전체 | 6,773 | 4.724 / 6.865 | 43.57 / 53.12 | 30.59 / 35.72 | **24.69 / 39.21** | 2.39 / 5.17 | 3.45 / 4.21 |
| Expert→Random | 2,510 | 4.676 / 4.980 | 43.87 / 39.62 | 26.45 / 32.57 | 27.28 / 26.77 | 2.95 / 4.84 | 3.62 / 4.02 |
| Expert→Random→Expert | 1,753 | 2.715 / 2.085 | 16.03 / 13.75 | 11.44 / 13.07 | 24.35 / 20.00 | 3.01 / 3.62 | 2.41 / 3.03 |
| 완전 Random | 2,510 | 6.177 / 12.089 | 62.51 / 94.13 | 48.10 / 54.69 | **22.33 / 65.06** | 1.41 / 6.58 | 4.00 / 5.22 |

[정책별 분포와 30개 에피소드 상세 비교](../outputs/fixed_hole_policy_holdout/report.md) · [비교 원본 JSON](../outputs/fixed_hole_policy_holdout/comparison.json)

## Condition 1 테스트 · VertexEdgeGatedMPNN

2026-11-17 KST에 확장 학습 모델의 재개 체크포인트를 기존 `condition_1/test` 2,573프레임에서 로컬 CPU와 배치 크기 128로 평가했습니다. 위 고정 Hole 보류 테스트는 6,773프레임을 사용합니다. 두 확장 모델은 같은 41,256개 학습 프레임과 2,144개 검증 프레임을 사용했고, 테스트에서 프레임 순서와 프레임별 검출 꼭짓점이 일치했습니다. 표면점 모델은 TAPNext++가 추적한 비꼭짓점 Peg 표면점을 추가합니다. 두 확장 모델 모두 40에포크에서 재개했고, 전체 학습 중 검증 loss가 가장 낮은 체크포인트를 평가했습니다.

| 모델 | 학습 프레임 | Best 에포크 | 테스트 loss ↓ | Peg 중심 (mm) ↓ | Hole 중심 (mm) ↓ | Peg 끝점의 Hole 상대 위치 (mm) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 기존 검출기 단독 GNN | 17,380 | 26 | 0.33475 | 2.67 | 7.54 | 6.87 |
| `expanded_direct_corner_gnn` · 검출 꼭짓점 단독 | 41,256 | 38 | 0.06659 | **1.02** | 3.38 | 3.15 |
| `expanded_surface_corner_gnn` · 검출 꼭짓점 + Peg 표면점 | 41,256 | 44 | **0.05770** | 1.66 | **2.66** | **2.77** |

- 기존 검출기 단독 GNN의 [26에포크 체크포인트](../outputs/direct_corner_gnn/28aa7201e4f044dc82582b2694548312/best.pt).
- 확장 검출 꼭짓점 단독 모델은 54에포크에서 조기 종료했고, 기존 38에포크 가중치가 유지됐습니다. [체크포인트](../outputs/expanded_direct_corner_gnn/17fb97384a564d17ba5b9126a6a17d24/best.pt) · [평가 JSON](../outputs/expanded_direct_corner_gnn/17fb97384a564d17ba5b9126a6a17d24/condition_1_test_evaluation.json)
- 확장 Peg 표면점 추가 모델은 52에포크에서 조기 종료했고, 44에포크에서 검증 최저값을 기록했습니다. [체크포인트](../outputs/expanded_surface_corner_gnn/a02a4fdc46ee4e2f822e507978dd7839/best.pt) · [평가 JSON](../outputs/expanded_surface_corner_gnn/a02a4fdc46ee4e2f822e507978dd7839/condition_1_test_evaluation.json)

### 검출 꼭짓점 노이즈 강건성

2026-11-17 KST에 동일한 `condition_1/test` 2,573프레임에서 추가 학습 전후를 비교했습니다. 기존 best 체크포인트(검출 단독 38에포크, 표면점 추가 44에포크)에서 시작해 학습 시 검출 꼭짓점에 축별 표준편차 1 mm의 가우시안 노이즈를 적용했습니다. 두 모델 모두 추가 학습 10에포크 중 원본 검증 세트에서 loss가 가장 낮았던 3에포크 체크포인트를 평가했습니다.

테스트의 1 mm와 2 mm는 좌표축별 노이즈 표준편차입니다. 같은 프레임의 같은 검출 꼭짓점에 두 모델과 추가 학습 전후 모두 동일한 난수(seed 20261117)를 사용했습니다. 표면점, TCP, GT는 원본을 유지했습니다. 각 셀은 **추가 학습 전 → 후** 평균 위치 오차입니다.

| 모델 | 테스트 노이즈 | Peg 중심 (mm) ↓ | Hole 중심 (mm) ↓ | Peg 끝점의 Hole 상대 위치 (mm) ↓ |
| --- | --- | ---: | ---: | ---: |
| 검출 꼭짓점 단독 | 원본 | 1.017 → 0.968 | 3.381 → 3.177 | 3.150 → 2.991 |
| 검출 꼭짓점 + Peg 표면점 | 원본 | 1.657 → 1.865 | 2.657 → 2.741 | 2.771 → 2.772 |
| 검출 꼭짓점 단독 | 1 mm | 1.281 → 1.249 | 3.593 → 3.341 | 3.446 → 3.267 |
| 검출 꼭짓점 + Peg 표면점 | 1 mm | 1.763 → 1.941 | 3.062 → 3.006 | 3.248 → 3.095 |
| 검출 꼭짓점 단독 | 2 mm | 1.783 → 1.749 | 4.162 → 3.789 | 4.255 → 3.995 |
| 검출 꼭짓점 + Peg 표면점 | 2 mm | 2.019 → 2.126 | 3.981 → 3.646 | 4.309 → 3.853 |

원본 대비 2 mm 노이즈에서 늘어난 끝점 상대 위치 오차는 검출 단독에서 1.106 → 1.004 mm, 표면점 추가에서 1.538 → 1.081 mm로 줄었습니다. 표면점 모델의 Peg 중심 절대 오차는 세 조건 모두에서 늘었습니다.

- 검출 꼭짓점 단독. [추가 학습 체크포인트](../outputs/expanded_direct_corner_gnn_jitter/4495468003974a3289cf5f2e8ba9b203/best.pt) · [강건성 평가 JSON](../outputs/expanded_direct_corner_gnn_jitter/4495468003974a3289cf5f2e8ba9b203/condition_1_test_jitter_evaluation.json)
- 검출 꼭짓점 + Peg 표면점. [추가 학습 체크포인트](../outputs/expanded_surface_corner_gnn_jitter/a7b48ca74df14bc3b7fefe45ef279d56/best.pt) · [강건성 평가 JSON](../outputs/expanded_surface_corner_gnn_jitter/a7b48ca74df14bc3b7fefe45ef279d56/condition_1_test_jitter_evaluation.json)

## Hugging Face 체크포인트

아래 고정 Hole 모델의 체크포인트는 같은 비공개 모델 저장소에 있습니다. 각 행의 Transformer와 GNN을 한 쌍으로 사용합니다. 링크는 고정된 revision의 `best.pt`를 가리킵니다. [모델 저장소](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models)

| 모델 | Transformer | GNN |
| --- | --- | --- |
| Fixed-Hole Expert v1 | [epoch 208](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/resolve/4445f4a9d6b67bb337942cab64ebcc44c8464382/graph/58cb23209b44485588e6dce05c72552c/artifacts/checkpoint/c57a93c17a913fd9fd514625a8cb25572f9f287e7a79fbecf07a0c33bc6c4893/best.pt) | [epoch 111](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/resolve/4445f4a9d6b67bb337942cab64ebcc44c8464382/graph/5b9dd788675b4bc1bae2cd76101e80e9/artifacts/checkpoint/443b2c114af7cd0fe1038ddf3171038874e1741b80782493123c157ffab257cb/best.pt) |
| 자동 대응 Transformer + GNN · 조기 종료 재개 · seed 0 | [Transformer](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/resolve/ffbc89d3c25e3127749491ef46a0abfb07b5e1a3/graph/5755e14cbb2b4bdc8323f18ed23c27d0/artifacts/checkpoint/fb555e95493a64b3e0a8d1e3138fc21adcc7d07d6cd9ec62ff1efcb79b88c076/best.pt) | [epoch 48](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/resolve/ffbc89d3c25e3127749491ef46a0abfb07b5e1a3/graph/a1fd52b4ab53496693a8737596f92baf/artifacts/checkpoint/3c642e0d4e977ec33d3431d1af9926d65c1945f0b0d989aff9c892cf35f3a8cf/best.pt) |
| 자동 대응 Transformer + GNN · 조기 종료 재개 · seed 1 | [Transformer](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/resolve/ffbc89d3c25e3127749491ef46a0abfb07b5e1a3/graph/20016101c0e94fa3991c7c8befc1040e/artifacts/checkpoint/d5a726f80d35827f6994e391f36fef3d351be219e817c7ce5c412295ccf3cdc7/best.pt) | [epoch 36](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/resolve/ffbc89d3c25e3127749491ef46a0abfb07b5e1a3/graph/acfe559cf95b429fb1c0e03e15d9e7b8/artifacts/checkpoint/fff05dc37d87c02386a9a735767816af464bb95de936b22105cbfe497e3d6065/best.pt) |
| 자동 대응 Transformer + GNN · 조기 종료 재개 · seed 2 | [Transformer](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/resolve/ffbc89d3c25e3127749491ef46a0abfb07b5e1a3/graph/5439879feb664125a8350e100f9269cf/artifacts/checkpoint/ea47af588d99fece019f6729534da7a14c236e6fc783488eb05bea0b29b6a4f9/best.pt) | [epoch 64](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/resolve/ffbc89d3c25e3127749491ef46a0abfb07b5e1a3/graph/68adb1f4baa74facbb26b05337f60b0a/artifacts/checkpoint/963898ce5506a5faddfe15aa9313c3fa273777e3601f915c486ce98765f8c2cb/best.pt) |

## 구동 방법

검출 캐시와 Robot9가 준비된 CUDA 작업 디렉터리에서 실행합니다. 기존 Vast.ai 환경에서는 아래 명령으로 접속합니다. Hugging Face 다운로드에는 해당 비공개 저장소를 읽을 수 있는 로그인이 필요합니다. 이 환경에 로그인 기록이 없으면 `hf auth login`을 실행합니다.

```bash
ssh -p 44216 root@171.5.178.57
cd /workspace/fixed_hole_expert_training.C7lccNHr
source /venv/main/bin/activate
export PYTHONPATH="$PWD:$PWD/deps"
```

같은 셸에서 아래 다운로드 함수와 실행할 모델의 `hf_pair` 명령을 실행합니다.

```bash
repo=Graph-Recurrent-VLA/graph-recurrent-models
hf_pair() {
  transformer_pt=$(hf download "$repo" "graph/$2/artifacts/checkpoint/$3/best.pt" --revision "$1" --quiet) || return
  gnn_pt=$(hf download "$repo" "graph/$4/artifacts/checkpoint/$5/best.pt" --revision "$1" --quiet)
}
```

**Fixed-Hole Expert v1**

```bash
hf_pair 4445f4a9d6b67bb337942cab64ebcc44c8464382 \
  58cb23209b44485588e6dce05c72552c c57a93c17a913fd9fd514625a8cb25572f9f287e7a79fbecf07a0c33bc6c4893 \
  5b9dd788675b4bc1bae2cd76101e80e9 443b2c114af7cd0fe1038ddf3171038874e1741b80782493123c157ffab257cb
```

**자동 대응 Transformer + GNN · 조기 종료 재개 · seed 0**

```bash
hf_pair ffbc89d3c25e3127749491ef46a0abfb07b5e1a3 \
  5755e14cbb2b4bdc8323f18ed23c27d0 fb555e95493a64b3e0a8d1e3138fc21adcc7d07d6cd9ec62ff1efcb79b88c076 \
  a1fd52b4ab53496693a8737596f92baf 3c642e0d4e977ec33d3431d1af9926d65c1945f0b0d989aff9c892cf35f3a8cf
```

자동 대응 Transformer + GNN · 조기 종료 재개의 seed 1과 2는 위 표의 해당 쌍을 사용합니다. 성능 표의 세 seed 집계를 재현할 때는 세 쌍을 각각 평가합니다.

아래는 준비된 Expert 검증 에피소드의 첫 32프레임으로 추론을 확인하는 예제입니다. GNN에 기록된 Transformer 해시를 대조한 뒤, 검출 좌표와 Robot9로 추적하고 Object21 예측과 물리 오차를 출력합니다.

```bash
python - "$transformer_pt" "$gnn_pt" <<'PY'
import sys
from pathlib import Path

import torch

from data_collection.episode_dataset import file_sha256
from data_collection.expert_corner_curriculum import load_expert
from evaluation.expert_corner_curriculum import rollout_tensors
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.fixed_id_corner import FixedIdCornerPredictor
from main import load_config
from training.detected_corner_gnn import graph_batch
from training.fixed_id_gnn import evaluate_batch

transformer_path, gnn_path = map(Path, sys.argv[1:])
tracker_state = torch.load(transformer_path, map_location='cpu', weights_only=False)
gnn_state = torch.load(gnn_path, map_location='cpu', weights_only=False)
expected_tracker = gnn_state.get('temporal_sha256') or gnn_state['config']['temporal']['sha256']
if file_sha256(transformer_path) != expected_tracker:
    raise ValueError('Transformer and GNN checkpoints are not a matched pair')

config = load_config('graph/configs/temporal/fixed_hole_expert_curriculum.yaml')
device = torch.device(config['device'])
torch.backends.cuda.matmul.allow_tf32 = True

def frozen(model, state):
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)

transformer = frozen(FixedIdCornerPredictor(**tracker_state['config']['model']), tracker_state)
gnn_settings = gnn_state['config']['gnn']
gnn = frozen(VertexEdgeGatedMPNN(**gnn_settings.get('model', gnn_settings)), gnn_state)
episode = next(item for item in load_expert(config['data']) if item['split'] == 'validation')
keys = ('xyz', 'kind', 'valid', 'robot', 'target')
example = dict(episode, **{key: episode[key][:32] for key in keys})
arrays, _ = rollout_tensors(transformer, [example], config['association_gate_m'], device)
with torch.inference_mode():
    graph, features = graph_batch(*(arrays[key] for key in ('xyz', 'kinds', 'valid', 'observed', 'tcp')))
    output = gnn(graph, features)
    count = len(arrays['target'])
    print('Transformer epoch', tracker_state['epoch'], '| GNN epoch', gnn_state['epoch'])
    print('Object21', tuple(output['object21'].shape))
    print({name: round(float(total) / count, 3) for name, total in evaluate_batch(output, arrays['target']).items()})
PY
```
