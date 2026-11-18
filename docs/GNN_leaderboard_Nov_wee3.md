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

이 섹션의 프레임 선정은 당시 TAPNext++ 추적 꼭짓점과 GT를 대응시킨
[기존 인덱스](data_collection.md#기존-추적-기반-프레임-선택)를 사용합니다.
현재 Five-mode의 꼭짓점 좌표와 가시성 라벨은 모두 해당 프레임의 RGB-D
검출 결과에서 가져옵니다. TAPNext++는 선택적인 비꼭짓점 표면점만 제공합니다.
[Five-mode 입력 규칙](data_collection.md#five-mode-graph-input)

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

### 검출 꼭짓점 노이즈 강건성 · 5 mm 추가 학습

2026-11-17 KST에 동일한 `condition_1/test` 2,573프레임에서 비교했습니다. 각 모델의 1 mm 추가 학습 best(3에포크)를 시작점으로 삼아, 검출 꼭짓점에 축별 표준편차 5 mm의 가우시안 노이즈를 적용해 10에포크 더 학습했습니다. 5 mm 학습 후에는 원본 검증 loss가 가장 낮았던 체크포인트를 사용했습니다. 검출 단독은 10에포크, 표면점 추가는 4에포크입니다.

테스트 노이즈는 좌표축별 표준편차입니다. 두 모델과 학습 전후에 같은 프레임의 같은 검출 꼭짓점에 동일한 난수(seed 20261117)를 사용했습니다. 표면점, TCP, GT는 원본을 유지했습니다. 각 셀은 **5 mm 추가 학습 전 → 후**입니다.

| 모델 | 테스트 노이즈 | Peg 중심 (mm) ↓ | Hole 중심 (mm) ↓ | Peg 끝점의 Hole 상대 위치 (mm) ↓ | Object21 loss ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| 검출 꼭짓점 단독 | 원본 | 0.968 → 1.150 | 3.177 → 5.111 | 2.991 → 4.726 | 0.06485 → 0.15513 |
| 검출 꼭짓점 + Peg 표면점 | 원본 | 1.865 → 1.890 | 2.741 → 4.443 | 2.772 → 4.742 | 0.06254 → 0.15031 |
| 검출 꼭짓점 단독 | 4 mm | 2.917 → 2.863 | 5.236 → 6.036 | 6.083 → 6.442 | 0.22579 → 0.26404 |
| 검출 꼭짓점 + Peg 표면점 | 4 mm | 2.684 → 2.121 | 5.440 → 5.482 | 5.906 → 5.917 | 0.21898 → 0.22184 |
| 검출 꼭짓점 단독 | 5 mm | 3.534 → 3.436 | 6.100 → 6.506 | 7.262 → 7.219 | 0.31494 → 0.32422 |
| 검출 꼭짓점 + Peg 표면점 | 5 mm | 3.017 → 2.241 | 6.464 → 6.019 | 7.065 → 6.486 | 0.30568 → 0.26182 |
| 검출 꼭짓점 단독 | 6 mm | 4.161 → 4.023 | 7.014 → 7.041 | 8.482 → 8.065 | 0.42142 → 0.39651 |
| 검출 꼭짓점 + Peg 표면점 | 6 mm | 3.371 → 2.381 | 7.538 → 6.625 | 8.276 → 7.122 | 0.41000 → 0.31039 |

표면점 추가 모델은 5·6 mm에서 세 위치 오차와 Object21 loss가 모두 줄었습니다. 검출 단독 모델의 Object21 loss는 6 mm에서 줄었습니다. 원본 입력의 loss는 두 모델 모두 약 2.4배로 증가했습니다.

- 검출 꼭짓점 단독 · 추가 학습 10에포크 best. [체크포인트](../outputs/expanded_direct_corner_gnn_strong_jitter/7236d4a3f008467bbae8dd09f8d22a9e/best.pt) · [평가 JSON](../outputs/expanded_direct_corner_gnn_strong_jitter/7236d4a3f008467bbae8dd09f8d22a9e/condition_1_test_jitter_evaluation.json)
- 검출 꼭짓점 + Peg 표면점 · 추가 학습 4에포크 best. [체크포인트](../outputs/expanded_surface_corner_gnn_strong_jitter/10fa325e42df4e7c8e48230056aa2991/best.pt) · [평가 JSON](../outputs/expanded_surface_corner_gnn_strong_jitter/10fa325e42df4e7c8e48230056aa2991/condition_1_test_jitter_evaluation.json)

## Five-mode 테스트 · Peg 표면점 모델 추가 학습

2026-11-17 KST에 전체 Five-mode 릴리스의 검출기 기준 `condition_1/test` 62,309프레임과 `condition_2/test` 5,181프레임을 Vast.ai 61028523의 RTX 3090에서 평가했습니다. 평가 전후의 프레임 ID, 검출 꼭짓점, Peg 표면점, TCP와 Object21 GT를 맞췄습니다. 두 실험은 `condition_1`에서 학습한 표면점 모델의 동일한 57에포크 `best.pt`에서 별도로 시작했습니다.

50:50 혼합은 에포크당 `condition_1` 40,000프레임과 `condition_2` 40,000프레임을 사용했습니다. `condition_2` 단독은 매 에포크 `condition_2/train` 43,383프레임을 사용했습니다. 두 실험 모두 검증에서 `condition_1` 평균 끝점 오차의 시작값 2.773 mm 대비 10% 이내 악화(3.050 mm 이하)를 허용하고, 이 기준을 충족하는 모델 중 `condition_2` 평균 끝점 오차가 가장 낮은 가중치를 선택했습니다. 두 실험 모두 허용치를 충족한 새 가중치가 없어 5에포크에서 조기 종료했고, 시작 체크포인트가 최종 선택됐습니다. 아래 표의 **추가 학습 후** 수치는 각 실험의 5에포크 `last.pt` 평가입니다. 변화율은 반올림 전 값으로 계산했습니다.

### 50:50 혼합

| 테스트 지표 | `condition_1` 전 → 후 | 변화 | `condition_2` 전 → 후 | 변화 |
| --- | ---: | ---: | ---: | ---: |
| Hole 기준 Peg 끝점 | 2.03 → 2.71 mm | +33.5% | 21.31 → 15.72 mm | −26.2% |
| Peg 위치 | 1.34 → 1.91 mm | +42.6% | 10.69 → 7.99 mm | −25.3% |
| Hole 위치 | 1.45 → 2.03 mm | +40.3% | 7.67 → 7.05 mm | −8.1% |
| Peg 회전 | 1.14° → 1.26° | +10.5% | 5.29° → 4.21° | −20.4% |
| Hole 회전 | 0.65° → 1.22° | +85.9% | 1.53° → 1.90° | +24.0% |
| Hole 반지름 | 0.66 → 0.87 mm | +33.1% | 1.73 → 1.69 mm | −2.0% |
| 테스트 손실 | 0.04211 → 0.06495 | +54.2% | 2.00578 → 1.34486 | −33.0% |

### `condition_2` 단독

| 테스트 지표 | `condition_1` 전 → 후 | 변화 | `condition_2` 전 → 후 | 변화 |
| --- | ---: | ---: | ---: | ---: |
| Hole 기준 Peg 끝점 | 2.03 → 4.13 mm | +103.7% | 21.31 → 16.52 mm | −22.5% |
| Peg 위치 | 1.34 → 2.74 mm | +105.0% | 10.69 → 8.66 mm | −18.9% |
| Hole 위치 | 1.45 → 2.54 mm | +75.3% | 7.67 → 7.23 mm | −5.7% |
| Peg 회전 | 1.14° → 1.71° | +50.4% | 5.29° → 4.41° | −16.8% |
| Hole 회전 | 0.65° → 1.45° | +122.0% | 1.53° → 2.26° | +47.4% |
| Hole 반지름 | 0.66 → 0.99 mm | +51.4% | 1.73 → 1.81 mm | +4.9% |
| 테스트 손실 | 0.04211 → 0.12156 | +188.7% | 2.00578 → 1.43267 | −28.6% |

5에포크 가중치의 `condition_1`·`condition_2` 검증 끝점 오차는 50:50 혼합에서 3.423 mm·15.877 mm, 단독 학습에서 4.780 mm·17.176 mm였습니다. 테스트에서는 50:50 혼합의 오차가 단독 학습보다 표의 모든 지표에서 낮았습니다.

- 추가 학습 전 기준. [57에포크 체크포인트](../outputs/five_mode_condition_1_peg_surface/adddc72d35f94deb995385af2e3f89e7/best.pt) · [Condition 1 평가 JSON](../outputs/five_mode_condition_1_test_comparison.json) · [Condition 2 평가 JSON](../outputs/five_mode_condition_2_before_finetuning_test_comparison.json)
- 평가한 5에포크 체크포인트와 검증 기록은 61028523의 `/workspace/five_mode_finetune_comparison`에 있습니다. 50:50 혼합은 `outputs/finetune_peg_surface_balanced/mix_1_1/4180338f5737449eb9c98cbd274128d5/`, 단독 학습은 `outputs/finetune_peg_surface_condition_2_only/mix_0_1/bc2c700b6d8a4305bb509a92bcfdab96/`입니다. 각 경로에서 `last.pt`, `history.json`, `selection.json`을 확인할 수 있습니다.

## Five-mode 전체 릴리스 · 검출기 기준 테스트

2026-11-17 KST에 900개 시드, 4,500개 에피소드의 Five-mode 전체 릴리스에서 `VertexEdgeGatedMPNN`을 평가했습니다. 이 비교는 RGB-D 검출 꼭짓점으로 분류한 `condition_1/test` 62,309프레임과 `condition_2/test` 5,181프레임을 사용합니다. 각 조건에서 네 체크포인트의 테스트 프레임 ID와 순서가 일치했습니다. 평가 장비는 RTX 3090이며 배치 크기는 1,792입니다. [릴리스 매니페스트](../outputs/five_mode_recovery/release/manifest.json) · [검출기 기준 프레임 인덱스](../outputs/five_mode_recovery/release/condition_index.json)

두 기준 모델은 같은 `condition_1` 학습 502,598프레임과 검증 60,011프레임으로 처음부터 학습했습니다. 검출 단독 모델은 101에포크에서 조기 종료했고 98에포크가 검증 loss 기준 best입니다. Peg 표면점 모델은 57에포크에서 조기 종료했고 해당 가중치가 best입니다. 표면점 모델은 매 프레임의 검출 꼭짓점에 TAPNext++로 추적한 비꼭짓점 Peg 표면점을 더합니다.

### `condition_1/test` · 62,309프레임

| 모델 | Best 에포크 | 테스트 loss ↓ | Peg 중심 (mm) ↓ | Hole 중심 (mm) ↓ | Peg 끝점의 Hole 상대 위치 (mm) ↓ | Peg 회전 (°) ↓ | Hole 회전 (°) ↓ | Hole 반지름 (mm) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 검출 꼭짓점 단독 | 98 | 0.13382 | 2.61 | 2.63 | 3.59 | 1.97 | 1.68 | 2.27 |
| 검출 꼭짓점 + Peg 표면점 · 추가 학습 전 | 57 | **0.04211** | **1.34** | **1.45** | **2.03** | **1.14** | **0.65** | **0.66** |
| 검출 꼭짓점 + Peg 표면점 · 80:20 | 추가 1 | 0.05211 | 1.48 | 1.72 | 2.33 | 1.21 | 0.79 | 0.68 |
| 검출 꼭짓점 + Peg 표면점 · 90:10 | 추가 7 | 0.04597 | 1.42 | 1.59 | 2.18 | 1.16 | 0.80 | 0.68 |

### `condition_2/test` · 5,181프레임

| 모델 | Best 에포크 | 테스트 loss ↓ | Peg 중심 (mm) ↓ | Hole 중심 (mm) ↓ | Peg 끝점의 Hole 상대 위치 (mm) ↓ | Peg 회전 (°) ↓ | Hole 회전 (°) ↓ | Hole 반지름 (mm) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 검출 꼭짓점 단독 | 98 | 6.76143 | 32.24 | 13.64 | 60.97 | 19.27 | 2.53 | 2.20 |
| 검출 꼭짓점 + Peg 표면점 · 추가 학습 전 | 57 | 2.00578 | 10.69 | 7.67 | 21.31 | 5.29 | **1.53** | 1.73 |
| 검출 꼭짓점 + Peg 표면점 · 80:20 | 추가 1 | 1.60959 | 9.71 | **6.75** | 18.00 | **4.63** | 1.69 | 1.83 |
| 검출 꼭짓점 + Peg 표면점 · 90:10 | 추가 7 | **1.55591** | **9.38** | 6.76 | **17.31** | 4.83 | 1.72 | **1.57** |

### 혼합 추가 학습과 체크포인트 선택

80:20과 90:10 실험은 각각 원래 Peg 표면점 모델의 57에포크 best에서 시작했습니다. 혼합 비율은 `condition_1 : condition_2`의 프레임 비율이며, 매 에포크 80,000프레임을 중복 없이 샘플링했습니다. 초기 모델의 `condition_1` 검증 끝점 오차는 2.773 mm였습니다. 이 값을 기준으로 정한 상한을 지킨 후보 중 `condition_2` 검증 끝점 오차가 가장 낮은 체크포인트를 선택했습니다. 최소 5에포크를 실행하고, 적격 후보의 오차가 0.1 mm 이상 개선되지 않은 상태가 5에포크 연속 이어지면 종료했습니다.

| 혼합 비율 | 시작 가중치 | 초기 학습률 | `condition_1` 검증 상한 | 조기 종료 에포크 | 선택 | `condition_1` 검증 끝점 (mm) | `condition_2` 검증 끝점 (mm) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 80:20 | 57에포크 best | `1e-4` | +10% (3.050 mm) | 6 | 80:20의 1에포크 | 3.049 | 19.415 |
| 60:40 | 80:20의 1에포크 | `1e-4` | +10% (3.050 mm) | 5 | 80:20의 1에포크 유지 | 3.049 | 19.415 |
| 50:50 | 80:20의 1에포크 | `1e-4` | +10% (3.050 mm) | 5 | 80:20의 1에포크 유지 | 3.049 | 19.415 |
| 90:10 | 57에포크 best | `3e-5` | +5% (2.911 mm) | 12 | 90:10의 7에포크 | 2.903 | 19.000 |

60:40과 50:50에서 학습한 모든 에포크가 `condition_1` 검증 상한을 넘었습니다. 두 단계가 넘긴 체크포인트의 SHA-256은 80:20 best와 일치했습니다. 90:10은 추가 학습 전 대비 `condition_1/test`의 평균 끝점 오차가 7.4% 증가했고, `condition_2/test`에서는 18.8% 감소했습니다. +5% 상한은 체크포인트를 선택하는 검증셋에 적용했습니다.

- 검출 단독 98에포크 · [체크포인트](../outputs/five_mode_condition_1_detector_corners/09e84d04d29c4a278ad02bc0035f5be7/best.pt)
- Peg 표면점 57에포크 · [체크포인트](../outputs/five_mode_condition_1_peg_surface/adddc72d35f94deb995385af2e3f89e7/best.pt)
- 80:20의 1에포크 · [체크포인트](../outputs/finetune_five_mode_peg_surface_curriculum/mix_4_1/a16d77e4d55d4b09be4863359f539368/best.pt) · [80:20 → 60:40 → 50:50 선택 및 테스트 JSON](../outputs/finetune_five_mode_peg_surface_curriculum/test_before_after.json)
- 90:10의 7에포크 · [체크포인트](../outputs/finetune_five_mode_peg_surface_90_10/mix_9_1/0dab2f4b69d440adb24123cf2004da91/best.pt) · [선택 및 테스트 JSON](../outputs/finetune_five_mode_peg_surface_90_10/test_comparison.json)
- 기준 모델 평가 · [Condition 1 JSON](../outputs/five_mode_condition_1_test_comparison.json) · [Condition 2 JSON](../outputs/five_mode_condition_2_before_finetuning_test_comparison.json)

### Five-mode 가시성 조건별 모델 선택

2026-11-17 KST에 Vast.ai 61028523의 RTX 3090에서 같은 릴리스의
`condition_2/validation` 6,744프레임을 배치 크기 1,600으로 평가했습니다.
`condition_1`에는 Peg 표면점 모델의 57에포크 `best.pt`를 고정했습니다.
검증 평균 끝점 오차가 가장 낮은 50:50 혼합의 5에포크 `last.pt`를
`condition_2`에 선택했습니다. 50:50은 57에포크에서 별도로 시작한 혼합 실험입니다.

| `condition_2` 검증 후보 | Hole 기준 Peg 끝점 평균 (mm) | P95 (mm) |
| --- | ---: | ---: |
| 90:10 추가 학습 7에포크 | 19.00 | 67.43 |
| 50:50 추가 학습 5에포크 `last.pt` | **15.88** | **60.89** |

같은 테스트 프레임 ID와 순서에서 배치 크기 1,792로 평가했습니다.
각 셀은 57에포크 단일 모델에서 조건별 모델 선택으로의 변화입니다.

| 테스트 조건 | 프레임 | Hole 기준 Peg 끝점 평균 (mm) | P95 (mm) |
| --- | ---: | ---: | ---: |
| `condition_1` | 62,309 | 2.03 → 2.03 | 4.17 → 4.17 |
| `condition_2` | 5,181 | 21.31 → **15.72** | 65.01 → **55.09** |
| 두 조건 합계 | 67,490 | 3.51 → **3.08** | 10.71 → **8.34** |

전환 오차는 같은 에피소드의 연속 두 프레임에서 `‖Δ예측 − ΔGT‖`를
계산했습니다. 벡터는 Hole 기준 Peg 끝점입니다. `condition_3`를 사이에 둔
두 프레임은 연속 전환에 포함하지 않았습니다.

| 테스트 전환 | 연속 프레임 쌍 | 전환 평균 (mm) | 전환 P95 (mm) |
| --- | ---: | ---: | ---: |
| `condition_1 → condition_2` | 728 | 9.50 → 9.29 | 39.72 → 35.99 |
| `condition_2 → condition_1` | 668 | 9.83 → 9.26 | 39.35 → 30.41 |
| 두 방향 합계 | 1,396 | 9.66 → **9.27** | 39.74 → **32.15** |

전환 오차가 10 mm를 초과한 비율은 26.4%에서 **25.4%**로 줄었습니다.
`condition_1` 안에서 연속한 프레임의 전환 오차 P95는 2.69 mm였습니다.
[평가 설정](../graph/configs/evaluate_five_mode_routed.yaml) ·
[선택 기록](../outputs/five_mode_routed_evaluation/2902e0d2718a442383a7f9cf396faa0d/selection.json) ·
[전체 지표](../outputs/five_mode_routed_evaluation/2902e0d2718a442383a7f9cf396faa0d/results.json) ·
[프레임별 예측](../outputs/five_mode_routed_evaluation/2902e0d2718a442383a7f9cf396faa0d/test_endpoint_predictions.npz)

### Five-mode `condition_2` 끝점 오차 진단

2026-11-17 KST에 선택한 50:50 혼합 5에포크 `last.pt`를 RTX 3090에서
다시 평가했습니다. 앞선 조건별 모델 선택 실험과 같은 5,181개
`condition_2/test` 프레임과 Object21 GT를 사용했습니다. 재추론한 끝점의
좌표 차이는 앞선 실행 대비 축별 최대 0.033 mm였고, 평균과 P95는 소수점
둘째 자리까지 일치했습니다.

그룹은 해당 프레임의 검출 꼭짓점으로 판정했습니다. Peg 관측 부족은
꼭짓점 2–3개 또는 꼭짓점 4개 이상이면서 평면 적합 RMS 1 mm 이하입니다.
Hole 림 부족은 검출 림 꼭짓점 1–2개입니다.

| 관측 상태 | 프레임 | Hole 기준 Peg 끝점 평균 (mm) | P95 (mm) |
| --- | ---: | ---: | ---: |
| Peg 점 부족·공면만 | 4,242 | 11.22 | 26.90 |
| Hole 림 부족만 | 743 | 30.03 | 75.86 |
| 두 문제 모두 | 196 | 58.76 | 115.13 |

Peg만 관측이 부족한 그룹에서 꼭짓점 2–3개는 1,119프레임에 평균 19.35 mm,
P95 79.45 mm였습니다. 4개 이상 공면은 3,123프레임에 평균 8.31 mm,
P95 21.11 mm였습니다. 전체 P95를 초과하는 259프레임 중 168개는 Hole 림이
부족했습니다.

모델 출력의 기하 항 `Peg 중심 + R(q)·(+X)·Peg X 반길이 − Hole 중심`에서
한 항만 GT 값으로 바꿨습니다. 나머지 예측값과 모델의 끝점 보정량을 고정하고
동일한 GT 끝점과 비교했습니다. 셀은 평균 / P95이며 단위는 mm입니다.

| 관측 상태 | 원래 예측 | Peg 중심 GT | Peg 회전 GT | Peg X 반길이 GT | Hole 중심 GT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 전체 | 15.72 / 55.09 | **11.01** / 51.65 | 13.00 / 53.53 | 15.45 / 53.34 | 11.75 / **32.27** |
| Peg 점 부족·공면만 | 11.22 / 26.90 | **6.36 / 15.01** | 8.72 / 17.98 | 10.87 / 26.20 | 11.06 / 26.76 |
| Hole 림 부족만 | 30.03 / 75.86 | 29.61 / 75.23 | 29.89 / 72.09 | 30.12 / 76.05 | **6.07 / 13.35** |
| 두 문제 모두 | 58.76 / 115.13 | **41.13 / 64.61** | 41.53 / 65.05 | 58.84 / 115.49 | 48.20 / 118.96 |

전체 평균은 Peg 중심을 바꿨을 때 4.71 mm, 전체 P95는 Hole 중심을 바꿨을
때 22.81 mm 줄어 가장 큰 개선을 보였습니다. 개별 치환의 변화량에는
다른 항과 끝점 보정량이 서로 상쇄하던 효과도 포함됩니다.
[평가 설정](../graph/configs/evaluate_five_mode_endpoint_diagnostics.yaml) ·
[전체 지표](../outputs/five_mode_endpoint_diagnostics/36a7446e73c740eb9ac1751b3d94736c/results.json) ·
[프레임별 예측과 검출 통계](../outputs/five_mode_endpoint_diagnostics/36a7446e73c740eb9ac1751b3d94736c/test_endpoint_diagnostics.npz)

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
