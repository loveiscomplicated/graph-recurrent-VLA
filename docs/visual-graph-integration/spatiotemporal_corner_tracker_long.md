# 공간·시간 Transformer 추가 학습과 수렴 확인

## 조건

[80에폭 실험](spatiotemporal_corner_tracker_extended.md)과 같은 공간·시간 Transformer를 최대 240에폭까지 학습했다. 80에폭 체크포인트에는 옵티마이저 상태가 없으므로 동일한 시드 0에서 초기화하고 같은 배치 순서로 다시 실행했다. 학습 데이터 120개 장면(13,448프레임), 검증 24개 장면(2,706프레임), AdamW, 학습률 0.0003, 배치 크기 6, 손실 및 모델 구조는 유지했다. 검증 미검출점 평균 3D 오차를 체크포인트 선택에 사용하고, 그 값이 25에폭 연속 개선되지 않으면 종료하도록 설정했다. 평가 체크포인트와 별도로 마지막 에폭의 가중치, AdamW, 배치 순서 및 PyTorch 난수 상태를 저장했다.

테스트는 기존과 같은 32개 장면과 3,352프레임이다. 동일한 검출 좌표, 관측 플래그, GT 기반 꼭짓점 ID 배정을 재사용했다. Object21 평가는 같은 GNN의 가중치를 고정하고 입력 좌표만 바꿨다.

## 검증 오차와 체크포인트 선택

| 에폭 | 미검출점 평균 3D 오차 (mm) | 미검출점 P90 (mm) |
| ---: | ---: | ---: |
| 80 (이전 실행) | 15.28 | 31.39 |
| 120 | 13.16 | 27.71 |
| 160 | 11.62 | 24.00 |
| 200 | 10.77 | 22.09 |
| 230 | 10.08 | 21.31 |
| 236 | 10.11 | 20.81 |
| 237 (평균 기준 선택) | 10.03 | 21.39 |
| 240 (마지막) | 10.31 | 22.00 |

237에폭이 평균 오차 최저였고, P90 최저는 236에폭이었다. 이 실험은 25에폭 조기 종료 조건에 도달하기 전에 240에폭 한도에 도달했다. 이전 실행과 새 실행 첫 80에폭의 최대 차이는 평균 0.081 mm, P90 0.114 mm였다. 새 실행의 80에폭 값은 평균 15.2831 mm, P90 31.3979 mm였다.

## 같은 테스트 장면에서의 결과

미검출점 3D 오차의 표본은 총 20,961개이며, Object21의 삽입 직전 구간은 시드별 마지막 정렬 5프레임씩 총 160프레임이다. 횡방향 오차는 GT의 Hole 삽입축에 수직인 Peg 끝점–Hole 중심 상대 위치 오차다.

| 테스트 지표 | 80에폭 P90 | 237에폭 P90 |
| --- | ---: | ---: |
| 전체 미검출 꼭짓점 3D 오차 | 20.01 mm | 12.40 mm |
| 이동 중 미검출 Box 외곽점 3D 오차 | 17.09 mm | 10.99 mm |
| 이동 중 미검출 Hole 입구 3D 오차 | 26.92 mm | 19.53 mm |
| 삽입 직전 미검출 Hole 입구 3D 오차 | 18.50 mm | 14.02 mm |
| 이동 중 Object21 횡방향 오차 | 11.80 mm | 9.36 mm |
| 삽입 직전 Object21 횡방향 오차 | 12.92 mm | 10.01 mm |
| 삽입 직전 Object21 Hole 중심 위치 오차 | 12.14 mm | 9.11 mm |
| 삽입 직전 Object21 Hole 삽입축 각도 오차 | 9.74° | 10.77° |

데이터를 추가하지 않고 학습을 늘린 결과, 미검출점과 Object21의 전체 위치 오차가 줄었다. 삽입 직전 Hole 삽입축의 각도 오차 P90은 커졌다. 선택 기준은 Object21이나 삽입축 오차가 아닌 검증 미검출점 평균이다. 해당 테스트는 GT로 꼭짓점 ID를 부여한 조건이다. 실제 카메라 입력에서의 ID 배정과 삽입 성공률은 별도로 검증해야 한다.

## 검증과 산출물

RTX 3090의 CUDA에서 240에폭 학습 및 검증에 129.4초가 걸렸다. 공통 ExperimentLogger 완료 상태 및 데이터 분할과 스크립트·체크포인트 해시를 확인했다. 원본 실험과 테스트 입력 및 GNN이 일치했고, 관측 슬롯의 입력 좌표가 출력에 그대로 보존됐다. 저장된 예측으로 전체 미검출점 3D P90과 삽입 직전 Object21 횡방향 P90을 다시 계산했다.

- [237에폭 검증 최저 체크포인트](../../outputs/spatiotemporal_corner_tracker_long/source_runs/train/spatial_temporal/best.pt) · [240에폭 재개용 상태](../../outputs/spatiotemporal_corner_tracker_long/source_runs/train/spatial_temporal/resume.pt)
- [전체 학습 이력](../../outputs/spatiotemporal_corner_tracker_long/source_runs/train/spatial_temporal/summary.json) · [테스트 지표](../../outputs/spatiotemporal_corner_tracker_long/source_runs/eval/summary.json)
- [꼭짓점 테스트 예측](../../outputs/spatiotemporal_corner_tracker_long/source_runs/train/test_predictions.npz) · [Object21 테스트 예측](../../outputs/spatiotemporal_corner_tracker_long/source_runs/eval/object21_predictions.npz) · [독립 검증](../../outputs/spatiotemporal_corner_tracker_long/verify.py)
