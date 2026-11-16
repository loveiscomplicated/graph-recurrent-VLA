# 기존 Mamba로 Box 이동 중 폐루프 성공률 평가

## 실험 조건

[고정 Hole 폐루프 평가](closed_loop_hybrid_mamba.md)의 시드 32개, 최신 Main 카메라, 개선된 CUDA RANSAC 검출기, 최대 250스텝을 유지했다. 두 앞단은 기존 가변 track Transformer·GNN과 새 954에폭 Transformer·강체 보정·54에폭 GNN이다. 두 조건 모두 같은 Mamba 5개 앙상블, 필터, TCP 6D와 Hole 기준 전처리를 사용했다. Mamba를 추가 학습하지 않았다. 새 앞단은 시뮬레이터 GT로 물리 꼭짓점 ID를 연결하고 실제 검출·추적·보정 좌표로 GNN을 실행한다. Mamba의 물체 상태 입력은 매 프레임 해당 GNN이 예측한 Object21이다.

장면마다 두 조건의 초기 상태와 이동 스케줄을 맞췄다. 20 Hz에서 60번째 스텝(3초)부터 Box를 수평으로 옮기는 계획을 적용했다. 계획 방향은 시드별로 동일하고 무작위이며, 목표 변위는 60–120 mm, 이동 시간은 0.8–1.4초다. 기존 이동 수집의 80 mm 접근 및 경로 가드를 재사용했다. Peg 끝점이 Hole 가까이에 오면 이동을 멈추므로 동일 시드라도 실제 이동량은 정책의 궤적에 따라 다를 수 있다. 움직임 발생은 실제 누적 변위 1 mm 초과로 판정했다. 삽입 성공은 시뮬레이터 판정이며 이동 종료 이후 성공도 포함한다. Box 회전은 가하지 않았다.

MP4에는 Main 카메라, 앞단 종류, 프레임 번호, 현재 Box 변위가 표시된다. 마지막에는 성공 또는 종료 이유가 표시된다. 64개 롤아웃의 결과로 성공률을 집계하고, 이동이 발생한 새 앞단 롤아웃을 성공과 실패로 나누어 각 그룹의 중간 이동량과 큰 이동량 사례만 로컬에 보관한다.

## 폐루프 결과

| 앞단 + 기존 Mamba 5개 | 같은 장면의 고정 Box 성공 | 이동 Box 성공 | 이동 Box 성공률 95% Wilson |
| --- | ---: | ---: | ---: |
| 기존 Tracker·GNN | 9/32 (28.1%) | 2/32 (6.25%) | 1.7–20.1% |
| 새 Tracker·강체 보정·GNN | 22/32 (68.8%) | 2/32 (6.25%) | 1.7–20.1% |

32개 장면 모두 양쪽 구성에서 실제로 이동했다. 새 앞단의 실제 최대 변위는 장면별 40.4–120.0 mm, 중앙값 89.9 mm였다. 양쪽 모두 성공은 1개, 새 앞단만 성공은 1개, 기존 앞단만 성공은 1개, 양쪽 모두 실패는 29개였다. 양쪽의 네 성공 롤아웃에서 삽입 판정은 모두 실제 Box 이동이 멈춘 뒤에 발생했다.

## 실제로 Box가 움직인 프레임의 추정 오차

| 앞단 | 이동 중 프레임 | 필터 전 Hole 중심 P90 | 필터 후 Hole 중심 P90 | 필터 전 횡방향 P90 | 필터 후 횡방향 P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 | 707 | 32.20 mm | 80.96 mm | 33.66 mm | 70.90 mm |
| 새 앞단 | 699 | 2.98 mm | 79.74 mm | 5.93 mm | 64.08 mm |

새 앞단의 원본 Hole 중심은 이동을 따라갔으나 고정 Hole용 Mamba의 누적 평균 필터를 거치면서 과거 위치의 영향이 남았다. 횡방향 오차는 GT 삽입축에 수직한 Peg 끝점과 Hole 중심의 상대 벡터 오차다. 각 행은 해당 제어기가 방문한 프레임을 사용한다. 같은 행 안의 필터 전·후는 같은 프레임 집합이다. 필터를 변경해 제어하는 별도 실험은 수행하지 않았다.

## 내려받은 영상

| 결과 | 환경 시드 | Box 실제 이동량 | Main 카메라 MP4 |
| --- | ---: | ---: | --- |
| 성공 | 700901 | 40.4 mm | [영상](../../outputs/closed_loop_moving_box_mamba/source_runs/shard_24_28/videos/700901_hybrid_success.mp4) |
| 성공 | 700488 | 70.2 mm | [영상](../../outputs/closed_loop_moving_box_mamba/source_runs/shard_0_4/videos/700488_hybrid_success.mp4) |
| 실패 | 700866 | 92.7 mm | [영상](../../outputs/closed_loop_moving_box_mamba/source_runs/shard_24_28/videos/700866_hybrid_fail.mp4) |
| 실패 | 700627 | 120.0 mm | [영상](../../outputs/closed_loop_moving_box_mamba/source_runs/shard_4_8/videos/700627_hybrid_fail.mp4) |

## 검증

RTX 3090 CUDA 렌더링과 CPU PhysX에서 실행했고, 공통 ExperimentLogger로 기록했다. 모델 해시와 카메라, 시드, 평가 한도를 고정 Box 실험과 대조했다. 장면별 이동 계획이 두 조건에서 같은지, 실제 이동 여부·거리, 64개 성공 판정의 집계, 내려받은 네 MP4의 H.264 형식, 512×512 해상도, 20 fps, 프레임 수 및 전체 디코딩을 검증했다.

- [집계된 성공률과 실행 정보](../../outputs/closed_loop_moving_box_mamba/source_runs/summary.json) · [롤아웃 기록 디렉터리](../../outputs/closed_loop_moving_box_mamba/source_runs/)
- [이동·녹화 실행 코드](../../outputs/closed_loop_moving_box_mamba/run_rollouts.py) · [검증·집계 코드](../../outputs/closed_loop_moving_box_mamba/summarize.py)
