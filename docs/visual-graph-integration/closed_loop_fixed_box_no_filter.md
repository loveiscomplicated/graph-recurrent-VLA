# 고정 Box에서 Recurrent 필터 제거 실험

## 조건

[고정 Box 폐루프 평가](closed_loop_hybrid_mamba.md)의 새 앞단을 대상으로 같은 시드 32개에서 폐루프를 반복했다. 두 조건 모두 동일한 CUDA RANSAC 검출기, GT 기반 꼭짓점 ID 배정, 954에폭 Transformer, 강체 보정, 54에폭 GNN, 기존 Mamba 5개와 저장된 정규화 통계를 사용한다. 기존 Mamba는 규칙 기반 필터를 거친 입력으로 학습됐다. 이번 실행에서는 Recurrent 쪽 KinFilter 전체만 우회해 GNN의 Object21을 그대로 전처리에 넘겼다. 앞단의 Box 강체 보정은 같이 실행한다.

최신 Main 카메라와 최대 250스텝을 유지했다. Box 이동을 가하지 않았으며, 시드별 초기 상태와 첫 프레임 Graph 추정값을 기존 실험과 대조했다. 두 조건 모두 각자 Mamba 행동으로 다음 관측을 만들고 ManiSkill 삽입 성공 판정을 사용했다.

## 폐루프 성공률

| Box 조건 | Recurrent 필터 켬 | 필터 끔 | 변화 |
| --- | ---: | ---: | ---: |
| 고정 | 22/32 (68.8%) | 25/32 (78.1%) | +9.4%p |
| [이동](closed_loop_moving_box_no_filter.md) | 2/32 (6.25%) | 14/32 (43.75%) | +37.5%p |

고정 Box에서 둘 다 성공 20개, 필터를 끈 조건만 성공 5개, 필터를 켠 조건만 성공 2개, 둘 다 실패 5개였다. 동일 장면의 정확 양측 McNemar 검정은 p=0.453125다. 32개 표본에서 필터를 끄면 성공 수는 3개 늘었으나 차이는 불확실하다. 성공한 장면의 걸린 스텝 수 중앙값은 필터를 켰을 때 102, 껐을 때 116이었다. 두 조건 모두 32개 장면에서 적어도 한 번 Peg를 잡았다.

## 고정 Box의 3D 입력 오차

필터를 켠 롤아웃의 동일 4,726프레임에서 직접 비교한 수치다.

| 3D 입력 지표 P90 | GNN 출력 | Recurrent 필터 적용 후 |
| --- | ---: | ---: |
| Hole 중심 | 7.98 mm | 4.84 mm |
| Peg 중심 | 8.01 mm | 2.84 mm |
| 삽입축에 수직한 Peg 끝점–Hole 중심 상대 위치 오차 | 9.62 mm | 9.56 mm |

필터는 각 물체의 중심 위치 오차를 줄였지만 횡방향 상대 위치 오차의 변화는 작았다. 필터를 끈 롤아웃 4,689프레임의 Mamba 입력 횡방향 P90은 10.83 mm였다. 각 제어 정책이 방문한 상태를 집계해 조건 간 프레임 수준 직접 비교와 구별한다. 고정 Box에서 개별 추정의 안정화에는 이점이 있었으나 폐루프 성공이 필터에 의존한다는 근거는 확보되지 않았다. 이동 Box에서는 누적 평균이 과거 위치를 포함하면서 오차가 크게 증가했고, 필터 제거 실험에서 성공률이 크게 개선됐다. 전체 필터를 우회한 실험이므로 Hole 평균, 파지 Peg 평균, 형상 평균의 기여는 분리하지 않았다.

## 검증

Vast RTX 3090의 CUDA 렌더링과 CPU PhysX에서 실행하고 공통 ExperimentLogger로 기록했다. 원본 롤아웃과 시드·초기 GT·첫 프레임 GNN 출력, 카메라, 최대 스텝과 모델 해시가 같은지 대조했다. 필터를 끈 모든 프레임에서 GNN Object21과 Mamba에 제공한 물체 상태가 일치함을 확인했다. 32개 쌍의 성공 판정과 오류 통계는 별도 코드로 검증했다. 새 앞단의 물리 ID는 GT로 배정했다.

- [성공률·오차 집계와 실행 ID](../../outputs/closed_loop_fixed_box_no_filter/source_runs/summary.json) · [장면별 기록](../../outputs/closed_loop_fixed_box_no_filter/source_runs/)
- [실행 코드](../../outputs/closed_loop_fixed_box_no_filter/run_no_filter.py) · [독립 검증 및 집계](../../outputs/closed_loop_fixed_box_no_filter/summarize.py)
