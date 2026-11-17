# 공간·시간 Transformer 학습 36→80에폭

## 실험 조건

[기존 36에폭 실험](spatiotemporal_corner_tracker.md)에서 공간·시간 attention을 사용하고 강체 거리 손실을 쓰지 않은 모델을 선택해 최대 80에폭으로 반복 학습했다. 이전 체크포인트에는 AdamW 상태가 없으므로 동일한 초기화와 배치 난수 시드 0에서 다시 시작했다. 모델 구조, 학습률 0.0003, 배치 크기 6, 학습 120개 장면(13,448프레임), 검증 24개 장면(2,706프레임), 인내 에폭 10, 검증 미검출점 평균 3D 오차 기준을 유지했다. 추적기 입력의 검출 좌표와 꼭짓점 ID 배정은 이전과 같다.

새 실행의 첫 36에폭에서 이전 실행과 검증 평균 오차 최대 차이는 0.0024 mm, 검증 P90 최대 차이는 0.0248 mm였다. 36에폭 검증 평균은 새 실행 21.9887 mm, 기존 실행 21.9904 mm였다. 아래 표의 36에폭 수치는 비교 기준인 기존 체크포인트 결과다.

## 학습과 검증

| 에폭 | 검증 미검출점 평균 3D 오차 (mm) | 검증 미검출점 P90 (mm) |
| ---: | ---: | ---: |
| 36 | 21.99 | 46.09 |
| 50 | 19.55 | 40.15 |
| 65 | 17.02 | 36.02 |
| 75 | 15.54 | 33.64 |
| 80 | 15.28 | 31.39 |

80에폭에서 검증 평균과 P90이 모두 이번 실행의 최저였다. 사전에 정한 검증 평균 기준으로 80에폭을 선택했다. 마지막 에폭까지 검증 오차가 개선됐으며, 이 실행으로 수렴 시점은 확인할 수 없다.

## 동일 테스트에서의 비교

동일한 미학습 32개 장면, 3,352프레임, 동일한 관측 플래그를 사용했다. 꼭짓점 오차는 현재 검출이 끊긴 슬롯만 포함한다. Object21은 추적기 출력을 기존 GT 학습 GNN에 제공하고, 가중치는 동결했다. 횡방향 오차는 예측 Peg 끝점–Hole 중심 상대 벡터의 오차 중 GT 삽입축에 수직인 성분이다.

| 지표 | 36에폭 P90 | 80에폭 P90 |
| --- | ---: | ---: |
| 전체 미검출점 3D 오차 | 33.74 mm | 20.01 mm |
| Box 이동 중 미검출 Box 외곽점 3D 오차 | 24.28 mm | 17.09 mm |
| Box 이동 중 미검출 Hole 입구 3D 오차 | 30.22 mm | 26.92 mm |
| 전 구간 미검출 Hole 입구 3D 오차 | 28.30 mm | 33.04 mm |
| 삽입 직전 미검출 Hole 입구 3D 오차 | 29.33 mm | 18.50 mm |
| Box 이동 중 Object21 횡방향 오차 | 18.21 mm | 11.80 mm |
| 삽입 직전 Object21 횡방향 오차 | 17.12 mm | 12.92 mm |
| 삽입 직전 Object21 Hole 중심 오차 | 16.86 mm | 12.14 mm |
| 삽입 직전 Object21 Hole 삽입축 오차 | 12.72° | 9.74° |

추가 데이터 없이 학습 예산을 늘려 전체 미검출점과 삽입 직전 Object21 오차를 낮췄다. Hole 입구는 이동 중과 삽입 직전의 P90이 줄었으며 전 구간 P90은 커졌다. 평가는 GT로 검출 후보의 꼭짓점 ID를 확인한 상태에서 진행했다. 카메라 관측만 사용하는 ID 배정과 실제 삽입 성공률은 이 실험의 측정 범위 밖이다.

## 검증 및 산출물

RTX 3090 CUDA에서 학습과 추론을 실행했다. 새 실행은 80에폭 학습 및 검증에 45.2초가 걸렸다. 공통 ExperimentLogger에 기록된 완료 상태와 커밋, 스크립트 및 체크포인트 해시, 36에폭 실행과 같은 데이터·분할·GNN 가중치, 관측 슬롯 좌표 보존을 확인했다. 테스트 3D 예측과 Object21로 전체 미검출점 P90과 삽입 직전 횡방향 P90을 NumPy에서 다시 계산했다.

- [80에폭 체크포인트](../../outputs/spatiotemporal_corner_tracker_extended/source_runs/train/spatial_temporal/best.pt) · [에폭별 학습 이력](../../outputs/spatiotemporal_corner_tracker_extended/source_runs/train/spatial_temporal/summary.json)
- [테스트 지표](../../outputs/spatiotemporal_corner_tracker_extended/source_runs/eval/summary.json) · [꼭짓점 예측](../../outputs/spatiotemporal_corner_tracker_extended/source_runs/train/test_predictions.npz) · [Object21 예측](../../outputs/spatiotemporal_corner_tracker_extended/source_runs/eval/object21_predictions.npz)
- [실행 스크립트](../../outputs/spatiotemporal_corner_tracker_extended/run_experiment.sh) · [독립 검증](../../outputs/spatiotemporal_corner_tracker_extended/verify.py)
