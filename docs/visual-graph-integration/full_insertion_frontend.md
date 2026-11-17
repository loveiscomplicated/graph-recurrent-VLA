# 전체 삽입 구간으로 Transformer와 GNN 재학습

## 수집과 분할

[기존 954에폭 Transformer](spatiotemporal_corner_tracker_convergence.md)와 [54에폭 GNN](hybrid_input_gnn_finetune.md)은 삽입 전 프레임으로 학습했다. [삽입 실패 분석](recurrent_moving_box_failure_analysis.md)의 후속 실험으로, 성공하는 Expert 데모의 정렬, 진입, 깊은 삽입 및 성공 직후 관측을 캡처했다. 512×512 Main RGB-D, 동일한 RANSAC 검출과 5 mm 이내 GT 꼭짓점 ID 매칭을 사용했다. 검출 좌표와 검출 마스크, 시뮬레이터 GT 20점, TCP, GT Object21, GT 입구 기준 깊이를 저장했다. 짝수 시드는 Expert가 파지 후 정렬을 시작할 때 Box에 수평 평행 이동을 적용하고, 홀수 시드는 고정했다.

3100000–3100119를 학습, 3100120–3100175를 검증, 새 시드 3100176–3100207을 테스트에 사용한다. 기존 체크포인트의 평가에만 쓰인 시드를 현재 검증에 포함했다. 학습과 검증의 일부 초기 구간은 같은 시드와 프레임으로 재생했을 때 GT Object21의 성분별 최대 차이가 0.000002 이하인 연속 프레임에 한해 기존 RANSAC 기록을 재사용했다. 재생이 달라지는 첫 프레임부터 새 검출을 실행했다. 새 테스트는 모든 프레임에서 새 검출을 실행했다.

## 모델 학습

Transformer를 기존 954에폭 가중치에서 시작했다. 각 에피소드의 실제 검출 시퀀스를 순서대로 입력하고, 현재 프레임에 검출되지 않은 점의 3D 좌표에만 기존 Smooth L1 손실을 적용했다. 실측이 있는 슬롯에는 검출기의 좌표를 정확히 유지한다. 현재·과거 프레임만 보는 시간 → 같은 강체 내 공간 → 시간 attention 구조 및 기존 슬롯 기준점은 유지했다. AdamW로 학습했으며 검증 선택은 입구 전과 입구 이후의 미검출점 평균 오차를 동일 비중으로 합산했다.

Transformer 검증 최저 가중치를 동결하고 기존 Box·Hole 강체 보정을 적용해 새 20점 그래프 입력을 만들었다. GNN은 기존 54에폭에서 초기화해, 이 좌표와 TCP 및 검출 플래그로 GT Object21을 맞히도록 기존 그래프 구성과 손실을 사용했다. Mamba는 이 실험에 사용하지 않았다.

첫 Transformer 실행은 추가 220에폭 한도로 끝났다. 최저 검증 점수를 낸 추가 206에폭 가중치에서 옵티마이저를 새로 초기화해 후속 학습을 실행했다. 검증 점수가 이전 기준보다 0.05 mm 이상 개선되지 않은 상태가 55에폭 지속되면 학습률을 0.00005 → 0.00002 → 0.000007로 내리고, 최저 학습률에서도 같은 기준으로 정체되면 종료하도록 했다. 첫 GNN을 초기값 후보로 사용하고 후속 Transformer가 만든 좌표로 모든 GNN 업데이트를 수행했다. 원래 954에폭 Transformer 및 54에폭 GNN은 동일 테스트 장면에서 계속 비교 기준으로 사용했다.

## 평가

같은 새 테스트 프레임에서 기존 앞단, 새 Transformer와 기존 GNN의 조합, 새 Transformer와 새 GNN의 조합을 대조한다. 깊이는 GT Peg 끝점 중심을 Hole의 삽입축으로 투영한 뒤 입구 면을 기준으로 계산했다. 입구 전은 0 mm 이하, 얕은 진입은 0–20 mm에서 0을 제외한 구간, 깊은 진입은 20 mm 초과다. 꼭짓점 오차는 해당 프레임의 미검출 슬롯 3D 거리로 계산하며, Object21 횡방향은 GT Hole 축에 수직인 상대 위치 벡터 오차다.

### 데이터 및 종료 조건

| 분할 | 시도 | 성공 데모 | 고정 / 이동 | 프레임 | 기존 검출 재사용 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 학습 | 120 | 116 | 59 / 57 | 14,258 | 5,858 |
| 검증 | 56 | 53 | 27 / 26 | 6,462 | 921 |
| 테스트 | 32 | 32 | 16 / 16 | 3,874 | 0 |

후속 Transformer는 추가 431에폭에서 최저 학습률 정체 기준으로 종료했다. 후속 학습의 추가 430에폭 가중치를 선택했다. 검증 미검출점 평균 3D 오차는 기존 → 첫 학습 → 후속 학습 순으로 입구 전 4.51 → 4.33 → 3.99 mm, 입구 후 15.39 → 5.82 → 4.96 mm였다. 최종 GNN은 첫 GNN 가중치로 초기화한 뒤 52에폭에서 최저 학습률 정체 기준으로 종료했다. 선택된 28에폭 가중치의 검증 손실은 0.06191이며, 최종 Transformer 좌표에서 초기 검증 손실은 0.06338이었다.

### 테스트 오차

미검출 Peg 점의 수치는 Transformer 3D 좌표 오차다. Box 강체 보정은 Peg 점을 변경하지 않는다.

| 깊이 | 프레임 | 미검출 Peg 점 P90 기존 → 새 Transformer | Object21 횡방향 P90 기존 앞단 | 새 Transformer + 기존 GNN | 새 Transformer + 새 GNN |
| --- | ---: | ---: | ---: | ---: | ---: |
| 입구 전 | 3,437 | 9.79 → 9.26 mm | 4.16 mm | 3.84 mm | 3.60 mm |
| 얕은 진입 | 98 | 19.66 → 13.40 mm | 8.46 mm | 8.23 mm | 5.05 mm |
| 깊은 진입 | 339 | 59.12 → 16.11 mm | 17.49 mm | 8.28 mm | 4.87 mm |

깊은 진입의 최종 횡방향 오차 P90은 고정 Box 172프레임에서 17.84 → 4.87 mm, 이동 Box 167프레임에서 16.64 → 4.71 mm였다. 이동 Box 분류는 해당 에피소드에서 Box가 움직였는지를 따르며 깊은 진입 프레임의 순간 이동 여부와는 구분한다.

첫 학습의 깊은 진입 Object21 횡방향 P90은 4.83 mm였고, 후속 학습에서는 4.87 mm였다. 최종 가중치 선택은 검증 손실로 했다. 이 테스트는 성공한 Expert 제어 궤적에서 수집한 입력 좌표를 사용했으며, Mamba가 실패했던 장면을 재실행하거나 폐루프 성공률을 평가하지 않았다. 프레임 수가 적은 얕은 구간은 같은 32개 장면의 프레임을 합친 수치다. 꼭짓점 ID 대응에는 여전히 GT를 사용했다.

## 검증과 산출물

Vast.ai RTX 3090에서 실험을 실행했다. 모든 테스트 프레임에서 Transformer와 Box 보정 전후의 직접 검출 좌표가 보존되는 것을 확인했다. 고정·이동 테스트 장면 한 개씩을 대상으로 프레임 단위 Transformer와 배치 결과를 비교했고, 최대 3D 좌표 차이는 0.00023 mm였다. 데이터·체크포인트 해시, 성공 종료 프레임 및 저장된 테스트 예측을 확인했다.

- [최종 Transformer](../../outputs/full_insertion_frontend/source_runs/trained_until_plateau/tracker/best.pt) · [최종 GNN](../../outputs/full_insertion_frontend/source_runs/trained_until_plateau/gnn/best.pt)
- [최종 검증 집계](../../outputs/full_insertion_frontend/source_runs/verified_plateau.json) · [테스트 예측](../../outputs/full_insertion_frontend/source_runs/trained_until_plateau/gnn/heldout.npz)
- [검출 데이터 수집 코드](../../outputs/full_insertion_frontend/collect.py) · [첫 학습 코드](../../outputs/full_insertion_frontend/train.py) · [후속 학습 코드](../../outputs/full_insertion_frontend/train_until_plateau.py)
