# 공간·시간 Transformer 수렴 확인

## 재개 조건

[240에폭 실험](spatiotemporal_corner_tracker_long.md)의 마지막 가중치, AdamW 옵티마이저, 배치 순서와 PyTorch 난수 상태를 불러와 241에폭부터 학습했다. 학습 120개 장면, 검증 24개 장면, 모델, 손실 및 배치 크기 6을 유지했다. 기존 검증 평균 최저인 237에폭을 초기 비교 기준으로 사용했다. 테스트 결과는 체크포인트 혹은 학습률 선택에 사용하지 않았다.

검증 미검출점 평균 3D 오차가 이전 기준에서 0.10 mm 이상 줄 때마다 정체 구간 계산을 다시 시작했다. 60에폭 동안 이 기준을 충족하지 않으면 학습률을 0.0003에서 0.0001로, 이후 0.00003으로 낮췄다. 마지막 학습률에서 같은 기준으로 정체되면 종료하도록 정했다. 최대 1,600에폭 한도도 설정했다. 학습 중 25에폭마다, 학습률이 바뀔 때, 종료할 때 재개 상태를 저장했다. 평가용 최저 검증 평균 체크포인트를 별도로 보관했다.

테스트는 기존과 동일한 32개 장면(3,352프레임)을 사용했다. 검출 좌표, 관측 플래그, GT 기반 꼭짓점 ID를 재사용하고, Object21 비교에서는 같은 GNN을 동결했다.

## 학습률 변경과 종료

| 구간 | 학습률 | 구간 마지막까지의 검증 평균 최저 (mm) |
| --- | ---: | ---: |
| 재개 전 | 0.0003 | 10.028 |
| 241–800에폭 | 0.0003 | 6.291 |
| 801–897에폭 | 0.0001 | 5.943 |
| 898–957에폭 | 0.00003 | 5.860 |

957에폭에서 정의한 정체 조건으로 종료했다. 최저 학습률에서 60에폭 동안 최저 검증 평균은 5.943 → 5.860 mm로 0.083 mm 개선되어 0.10 mm 기준에 못 미쳤다. 전체 실행에서 검증 평균이 최소인 954에폭 체크포인트를 선택했다. 검증 P90 최저는 898에폭의 13.47 mm이며, 선택된 954에폭의 P90은 13.55 mm다. 240에폭에서 시작한 추가 717에폭에 400.8초가 걸렸다.

## 같은 테스트 장면에서의 결과

| 지표 | 기존 237에폭 | 새 954에폭 |
| --- | ---: | ---: |
| 검증 미검출점 평균 3D 오차 | 10.03 mm | 5.86 mm |
| 검증 미검출점 3D 오차 P90 | 21.39 mm | 13.55 mm |
| 테스트 전체 미검출점 3D 오차 P90 | 12.40 mm | 7.03 mm |
| 테스트 이동 중 미검출 Box 외곽 3D 오차 P90 | 10.99 mm | 6.77 mm |
| 테스트 이동 중 미검출 Hole 입구 3D 오차 P90 | 19.53 mm | 18.58 mm |
| 테스트 삽입 직전 미검출 Hole 입구 3D 오차 P90 | 14.02 mm | 6.93 mm |
| 테스트 이동 중 Object21 횡방향 오차 P90 | 9.36 mm | 8.31 mm |
| 테스트 삽입 직전 Object21 횡방향 오차 P90 | 10.01 mm | 8.70 mm |
| 테스트 삽입 직전 Object21 Hole 중심 오차 P90 | 9.11 mm | 8.35 mm |
| 테스트 삽입 직전 Object21 Hole 축 각도 오차 P90 | 10.77° | 10.34° |

추적점 복원 P90의 개선 폭은 Object21 횡방향 P90보다 컸다. 이동 중 Hole 입구의 미검출점 오차 P90은 18.58 mm가 남았다. Object21 평가는 실제 검출 좌표에 GT로 꼭짓점 ID를 연결한 조건에서 같은 GNN으로 진행했으며 삽입 직전 표본은 160프레임이다. 실제 카메라 입력만 사용하는 ID 배정과 삽입 성공률은 별도 평가 범위다.

### 재학습 전 Transformer와 동일 미검출점 비교

기존 Transformer에 과거 GT 신원이 확인된 추적 후보가 있고, 현재 RANSAC 검출은 대응되지 않은 점만 골랐다. 기존 Transformer는 자체 track ID를 사용하고, 954에폭 모델은 평가에서 GT로 배정한 물리 꼭짓점 슬롯을 사용한다. 두 모델에 같은 프레임·물리 꼭짓점 표본을 적용했으며 기존 Transformer의 강체 보정은 별도로 표기했다.

| 구간 및 지점 | 점-프레임 수 | 기존 Transformer P90 (mm) | 954에폭 P90 (mm) | 기존 + 강체 보정 P90 (mm) |
| --- | ---: | ---: | ---: | ---: |
| 전 구간 전체 | 4,985 | 41.96 | 9.09 | 35.67 |
| 삽입 직전 Peg | 166 | 85.09 | 12.14 | 85.09 |
| 삽입 직전 Box 외곽 | 252 | 47.26 | 10.59 | 3.60 |
| 삽입 직전 Hole 입구 | 101 | 33.63 | 6.99 | 5.27 |
| 이동 중 Hole 입구 | 106 | 32.82 | 18.63 | 4.45 |

앞 절의 삽입 직전 RANSAC 미검출 Hole 오차 6.93 mm는 새 모델의 해당 미검출점 122개 전체에서 계산됐다. 위 직접 비교 표의 6.99 mm는 기존 추적 후보와도 신원을 확인할 수 있는 공통 표본 101개에 대한 수치다.

## 검증과 산출물

Vast.ai RTX 3090의 CUDA에서 학습 및 평가를 실행하고 공통 ExperimentLogger를 사용했다. 재개 파일의 해시와 직전 240에폭 이력, 학습률 변경 에폭 및 종료 조건, 체크포인트 선택, 동일 테스트 입력과 GNN 해시를 확인했다. 관측 슬롯 좌표가 그대로 유지됐고, 저장된 예측으로 전체 미검출점 P90과 삽입 직전 Object21 횡방향 P90을 NumPy에서 다시 계산했다. 957에폭의 재개 상태와 954에폭 평가용 체크포인트를 각각 저장했다.

- [954에폭 체크포인트](../../outputs/spatiotemporal_corner_tracker_convergence/source_runs/train/spatial_temporal/best.pt) · [957에폭 재개 상태](../../outputs/spatiotemporal_corner_tracker_convergence/source_runs/train/spatial_temporal/resume.pt)
- [전체 학습 이력](../../outputs/spatiotemporal_corner_tracker_convergence/source_runs/train/spatial_temporal/summary.json) · [테스트 지표](../../outputs/spatiotemporal_corner_tracker_convergence/source_runs/eval/summary.json)
- [꼭짓점 예측](../../outputs/spatiotemporal_corner_tracker_convergence/source_runs/train/test_predictions.npz) · [Object21 예측](../../outputs/spatiotemporal_corner_tracker_convergence/source_runs/eval/object21_predictions.npz) · [독립 검증](../../outputs/spatiotemporal_corner_tracker_convergence/verify.py)
