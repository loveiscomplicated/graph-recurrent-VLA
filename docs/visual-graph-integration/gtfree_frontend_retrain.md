# GT 없는 번호 배정 입력으로 Transformer와 GNN 파인튜닝

## 데이터와 기준

[기존 GT 없는 번호 배정](gtfree_corner_ids.md)을 모든 비교에 사용했다. [검출점을 보존하는 비교](gtfree_corner_recall.md)에서 사용했던 기존 매칭 규칙을 유지했다. 입력은 RANSAC의 원시 후보 좌표와 물체/Hole 후보 표식, TCP 위치다. 시뮬레이터 GT는 꼭짓점 3D 정답, Object21 학습 정답과 채점에 사용했다. 관측 슬롯 좌표는 직접 검출 좌표다.

고정 Box는 홀수 시드, 이동 Box는 짝수 시드이며 Expert가 파지 후 정렬을 시작할 때 이동한다. 120개 학습 시드(3100000–3100119)와 24개 검증 시드(3100120–3100143)에서는 성공 종료까지 수집한 데모를 사용했다. 테스트는 앞단 모델 선택이나 매칭 조건 선택에 쓰지 않은 시드 24개(3100208–3100231)다. 모든 테스트 장면에서 같은 RGB-D 후보와 같은 관측 마스크를 기존 및 새 가중치에 제공했다.

## 학습 및 비교

Transformer는 [이전 전체 삽입 학습](full_insertion_frontend.md)에서 선택한 430 후속 에폭 가중치로 시작했다. 현 검출 여부와 슬롯 순서는 유지하고 미검출 슬롯에만 기존 Smooth L1을 적용했다. 입구 전과 입구 후의 검증 미검출 평균 3D 오차에 같은 비중을 둬 체크포인트를 선택했다. 최저 학습률에서 55에폭 동안 기준 오차가 0.05 mm 이상 개선되지 않으면 종료하는 기존 정체 기준을 적용했다.

새 Transformer를 고정한 상태에서 기존 강체 보정을 거친 20점으로 GNN 학습 입력을 다시 만들었다. GNN은 이전 28 후속 에폭 가중치에 기존 Object21 손실로 추가 학습했다. 기존 GNN과 재학습 GNN의 대조에는 같은 보정 좌표·TCP·검출 여부를 전달했다. GNN은 검증 손실을 기준으로 선택했다.

| 비교 | 번호 배정 | Transformer | GNN |
| --- | --- | --- | --- |
| 현재 기준 | 동일한 GT 없는 배정 | 기존 | 기존 |
| Transformer만 교체 | 동일 | 재학습 | 기존 |
| 앞단 전체 | 동일 | 재학습 | 재학습 |

Tracker 비교에서는 강체 보정 직전의 같은 미검출 슬롯의 3D 거리를 사용했다. Object21 비교에서는 GT Hole 축에 수직인 Peg 끝점 상대 위치 오차를 계산했다. 삽입 깊이는 입구 면 대비 GT Peg 끝점 중심의 축 방향 좌표이며 0 mm 이하, 0~20 mm, 20 mm 초과로 나눴다. Recurrent는 이 오프라인 비교에 연결하지 않았다.

## 평가 결과

RTX 3090에서 총 168개 궤적을 수집했다. 학습에는 성공한 116개 궤적(14,258프레임)을, 검증에는 성공한 21개 궤적(2,572프레임)을 사용했다. 테스트는 스크립트 Expert가 실패한 궤적 1개를 포함해 24개(2,873프레임)를 모두 평가했다. 고정 Box와 이동 Box의 결과를 합산했다.

Transformer는 추가 342에폭에서 최저 학습률 정체 조건을 충족했고, 검증 성능이 가장 좋은 추가 333에폭 가중치를 선택했다. 검증 미검출점 평균 3D 오차는 삽입 전 3.95 → 3.44 mm, 삽입 후 5.45 → 4.78 mm로 줄었다. GNN은 추가 38에폭에서 최저 학습률 정체 조건을 충족했고, 추가 2에폭에서 검증 손실이 초기 0.07707에서 0.07361로 줄어 해당 가중치를 선택했다.

| 구간 | 프레임 | 공통 미검출 슬롯 수 | 미검출점 3D P90 기존 → 새 Transformer (mm) | 기존 Transformer + 기존 GNN Object21 P90 (mm) | 새 Transformer + 기존 GNN Object21 P90 (mm) | 새 Transformer + 새 GNN Object21 P90 (mm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 전체 | 2,873 | 15,719 | 8.49 → 7.73 | 3.24 | 3.19 | 2.92 |
| 삽입 전 | 2,479 | 11,327 | 6.30 → 5.84 | 3.04 | 3.03 | 2.63 |
| 얕은 진입 | 94 | 925 | 11.48 → 11.12 | 4.55 | 4.06 | 4.55 |
| 깊은 삽입 | 300 | 3,467 | 13.47 → 11.44 | 5.54 | 5.93 | 5.89 |

새 Transformer는 모든 구간에서 같은 미검출 슬롯의 P90을 낮췄고, 깊은 삽입에서 감소 폭이 2.03 mm로 가장 컸다. 새 Transformer의 보정 좌표를 두 GNN에 동일하게 주었을 때, 재학습 GNN은 전체 Object21 횡방향 P90을 3.19 → 2.92 mm, 삽입 전은 3.03 → 2.63 mm로 줄였다. 얕은 진입에서는 4.06 → 4.55 mm로 늘었고, 깊은 삽입에서는 5.93 → 5.89 mm로 소폭 줄었다. 기존 앞단 전체와 비교하면 깊은 삽입에서는 5.54 → 5.89 mm로 늘었다. 이 표는 저장된 Expert 궤적의 오프라인 오차를 나타낸다.

검증 스크립트에서 체크포인트 식별값, 테스트 시드 분리, 관측 슬롯의 검출 좌표 보존, 체크포인트 선택과 P90 재계산을 확인했다.

- [Transformer 체크포인트](../../outputs/gtfree_frontend_retrain/source_runs/finetuned/tracker/best.pt)
- [GNN 체크포인트](../../outputs/gtfree_frontend_retrain/source_runs/finetuned/gnn/best.pt)
- [학습 및 평가 수치](../../outputs/gtfree_frontend_retrain/source_runs/finetuned/gnn/summary.json)
- [테스트 프레임별 배열](../../outputs/gtfree_frontend_retrain/source_runs/finetuned/gnn/heldout.npz)
- [수집 코드](../../outputs/gtfree_frontend_retrain/collect_raw.py), [학습 코드](../../outputs/gtfree_frontend_retrain/train_gtfree.py), [검증 코드](../../outputs/gtfree_frontend_retrain/verify.py)
