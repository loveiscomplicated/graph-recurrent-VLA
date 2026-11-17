# 이동 Box에서 Recurrent 필터 제거 실험

## 조건

[이동 Box 폐루프 실험](closed_loop_moving_box_mamba.md)의 같은 32개 장면에서 새 앞단만 사용했다. RANSAC, GT 기반 꼭짓점 번호 배정, 954에폭 Transformer, Box 강체 보정, 54에폭 GNN, Mamba 5개, TCP 6D와 44차원 특징 변환 및 기존 정규화 통계는 유지했다. 250스텝 제어 한도와 60스텝에 시작하는 이동 스케줄, 80 mm 접근·경로 가드를 유지했다. 제어가 달라지면 가드에 도달하는 시점과 실제 변위는 달라질 수 있다.

Recurrent의 KinFilter 전체를 끄고 GNN의 원본 Object21을 디코더 특징 변환에 제공했다. KinFilter는 상태 추정용 학습 모델이 아니라 Hole 위치·자세의 누적 평균, 파지 후 손 기준 Peg 평균, 형상 값 누적 평균을 계산하는 규칙이다. 기존 Mamba 가중치는 필터를 적용한 입력으로 학습됐다. 이번 비교에서는 그 가중치와 정규화 통계를 사용해 실행 시점의 필터 효과를 측정했다.

이전과 같은 영상 시드 700488, 700627, 700866, 700901만 녹화했다. 프레임 번호, Box 변위, 결과를 보존했으며 내보낸 영상의 제목에 Recurrent filter OFF를 표시했다.

## 결과

| 새 앞단 + 기존 Mamba 5개 | 기존 Recurrent 필터 | Recurrent 필터 제거 |
| --- | ---: | ---: |
| 폐루프 성공 | 2/32 (6.25%) | 14/32 (43.75%) |
| 실제로 이동한 장면 | 32/32 | 32/32 |
| 이동 중 Mamba 입력의 Hole 중심 오차 P90 | 79.74 mm | 2.91 mm |
| 이동 중 Mamba 입력의 횡방향 오차 P90 | 64.08 mm | 5.73 mm |

기존 필터의 두 성공 장면은 필터 제거 후에도 성공했고, 12개 장면이 추가로 성공했다. 두 조건 모두 실패한 장면은 18개, 기존 필터만 성공한 장면은 0개였다. 성공률 차이는 37.5%p이고, 정확 양측 McNemar 검정의 p값은 0.000488이다. 양쪽 모두 모든 장면에서 Box가 실제 이동했으며, 최종 변위 범위는 40.4–120.0 mm였다. 새 조건의 성공 14개는 모두 Box 이동이 멈춘 뒤 성공 판정을 받았다.

이동 중 필터 전 GNN의 Hole 중심 오차 P90은 기존 조건 2.98 mm(699프레임), 필터 제거 조건 2.91 mm(693프레임)였다. Mamba는 자기 행동으로 만든 관측으로 계속 제어하므로 조건 간 오차는 별도 궤적에서 집계했다. 이전 조건에서 같은 프레임의 필터 전·후 위치 오차가 크게 달랐고, 이번 조건에서는 GNN의 Object21이 그대로 입력됐다. 이 실험은 누적 Hole 평균만 껐을 때와 전체 필터를 없앴을 때의 기여를 분리하지 않는다.

## 영상

- [시드 700627, 120.0 mm 이동 뒤 성공](../../outputs/closed_loop_moving_box_no_filter/source_runs/videos/700627_no_filter_success.mp4)
- [시드 700866, 92.7 mm 이동 뒤 실패](../../outputs/closed_loop_moving_box_no_filter/source_runs/videos/700866_no_filter_fail.mp4)
- [시드 700488, 70.2 mm 이동 뒤 성공](../../outputs/closed_loop_moving_box_no_filter/source_runs/videos/700488_no_filter_success.mp4)
- [시드 700901, 40.4 mm 이동 뒤 성공](../../outputs/closed_loop_moving_box_no_filter/source_runs/videos/700901_no_filter_success.mp4)

## 검증

Vast RTX 3090의 CUDA 렌더링과 CPU PhysX로 실행했다. 공통 ExperimentLogger 완료 상태 및 기존 실험과 카메라·시드·이동 계획·초기 관측·모델 해시가 같음을 검증했다. 전 프레임에서 GNN의 Object21과 필터 제거 조건에서 디코더가 받은 Object21이 일치함을 확인했다. 네 영상의 해상도, 프레임 수, 20 fps와 디코딩을 검증했다. 앞단의 물리 ID는 GT 기반이고 Box 회전은 이번 실험의 대상이 아니다.

- [성공률과 실행 정보](../../outputs/closed_loop_moving_box_no_filter/source_runs/summary.json) · [실행 코드](../../outputs/closed_loop_moving_box_no_filter/run_no_filter.py) · [독립 집계](../../outputs/closed_loop_moving_box_no_filter/summarize.py)
