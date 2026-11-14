# 실험 결과 리더보드

대상: **2026-11-08 KST**에 정리한 Graph 평가와 **2026-11-09 KST**에 추가한 자동 코너 대응 + GNN 평가 및 조기 종료 재개 결과. 원격 학습의 실제 시작·종료 날짜를 로컬 파일 수정 시각으로 단정하지 않는다.

## 읽는 방법

- 모든 값은 **오차이며 낮을수록 좋다**. 각 모델의 상세 표는 요청한 9개 항목 / 총 21D 형식을 따른다.
- `3-seed 평균`: 각 프레임에서 seed 0·1·2의 오차를 평균하고 프레임 전체에 대해 평균한 값. `중앙값`, `90백분위`도 **프레임별 3-seed 평균 오차 분포**에서 구한다. 예측 ensemble의 오차, seed별 중앙값 평균, 3N개 오차를 합친 분포와 다르다.
- Seed 0·1·2 열은 해당 seed의 프레임 평균이며, **굵게 표시한 seed는 그 행의 최소 오차**다. 반올림 전 값으로 비교한다.
- `—`는 로컬 원본에 해당 세부값이 없다는 뜻이다. 집계 요약만 있는 GNN epoch 90과 Transformer 60 결과의 seed 평균은 생성하지 않았다.
- 위치/크기는 mm, 회전은 quaternion 4D 표현에 대응하는 각도 오차(°)다. 상대 위치는 `relative_3d_error_mm`이며 Peg 끝점 자체의 위치 오차와 구분한다. 서로 단위가 달라 21D 합산 점수는 만들지 않는다.
- 각 cohort 안의 요약 순위는 **Hole 위치 3-seed 평균만** 기준으로 한다. 21D 전 항목의 종합 순위가 아니다. 학습 데이터·정보 입력·체크포인트 선택 예산이 달라 인과적 ablation 결론으로 읽지 않는다.
- 모든 집합은 개발 검증 결과다. 물리 조건 판정은 GT 보조 인증이며, 해당 151프레임은 새 GNN 체크포인트 선택에도 사용됐다. 독립 최종 테스트나 삽입 성공률을 측정한 리더보드가 아니다.

[실험별 체크포인트 매핑으로 이동](#실험별-체크포인트-매핑)

자동 코너 대응 + GNN의 20 epoch·40 epoch·조기 종료 재개 여섯 결과는 코드 정리 시
정확한 수치 재현 대상으로 유지한다. 관련 구현과 의존성을 바꾸기 전 원본 checkpoint,
프레임별 배열과 전체 정밀도 지표를 보존·대조하는
[검증 기준](detected_corner_gnn.md#evaluation-preservation)을 따른다.

## 전체 4,102프레임 종합 순위

9개 항목의 **3-seed 평균 오차를 각 기준값으로 나눈 뒤 동일 가중치로 평균**한 종합 오차다. 낮을수록 좋으며 무차원 수치다. 각 행은 학습 loss가 아니라 물리 오차로 계산했다. 값 1은 평균적으로 각 항목의 기준 오차 수준이라는 뜻이며 정확도 100점이나 합격선은 아니다.

`종합 오차 = (Σ 항목별 평균 오차 / 항목별 기준 오차) / 9`

| 항목 | Peg 위치 | Peg 회전 | 끝점−Hole 상대 위치 | Hole 위치 | Hole 회전 | Peg 반쪽 크기 X | Peg 반쪽 크기 Y | Peg 반쪽 크기 Z | Hole 반지름 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 기준 오차 | 10mm | 0.1rad (약 5.730°) | 10mm | 10mm | 0.1rad (약 5.730°) | 40mm | 10mm | 10mm | 20mm |

기준은 기존 학습의 위치·회전·크기 스케일을 참고한 비교용 선택이며 성능 허용 기준은 아니다. 9개 항목을 같은 비중으로 보므로 21개 스칼라 차원을 각각 같은 비중으로 합산한 값과 다르다. 기준이나 가중치를 바꾸면 순위도 바뀔 수 있다.

같은 4,102프레임 결과의 참고 순위다. 기존 시계열과 자동 대응의 프레임 ID·GT는 일치하지만, vertex-family와 나머지 계열 사이의 일치는 별도로 검증하지 않았다. 입력 정보와 학습 조건도 달라 구조의 인과적 비교가 아니며, 모두 개발 검증 결과다. 40 epoch와 조기 종료 재개는 독립된 학습이 아니다.

| 순위 | 실험 / 체크포인트 | 계열 | 종합 오차 ↓ | Hole 위치 오차 |
| ---: | --- | --- | ---: | ---: |
| 1 | [자동 대응 Transformer + GNN · 재개 · 조기 종료](#detected-resume-transformer) | 자동 코너 대응 | 0.1162 | 1.831mm |
| 2 | [자동 대응 Mamba + GNN · 재개 · 조기 종료](#detected-resume-mamba) | 자동 코너 대응 | 0.1196 | 1.749mm |
| 3 | [자동 대응 Transformer + GNN · 40 epoch](../outputs/detected_corner_gnn_fresh/comparison.json) | 자동 코너 대응 | 0.1297 | 1.923mm |
| 4 | [자동 대응 Mamba + GNN · 40 epoch](../outputs/detected_corner_gnn_fresh/comparison.json) | 자동 코너 대응 | 0.1433 | 1.970mm |
| 5 | [자동 대응 Transformer + GNN · 20 epoch](#checkpoint-detected-transformer) | 자동 코너 대응 | 0.1694 | 2.304mm |
| 6 | [자동 대응 Mamba + GNN · 20 epoch](#checkpoint-detected-mamba) | 자동 코너 대응 | 0.1847 | 2.549mm |
| 7 | [GNN 조기 종료 + Mamba 조기 종료](#checkpoint-21) | 선택 M0·시계열 | 0.6585 | 14.922mm |
| 8 | [Mamba · 120 epoch 한도 best](#checkpoint-11) | 선택 M0·시계열 | 0.6723 | 14.828mm |
| 9 | [M2 Oracle family · GT 입력 참고](#checkpoint-9) | vertex-family | 0.7953 | 24.308mm |
| 10 | [용량 통제](#checkpoint-10) | vertex-family | 0.7956 | 24.111mm |
| 11 | [M1 Detector family](#checkpoint-8) | vertex-family | 0.7989 | 24.551mm |
| 12 | [GNN 조기 종료 + Transformer 조기 종료](#checkpoint-22) | 선택 M0·시계열 | 0.8030 | 18.350mm |
| 13 | [M0 XYZ/type](#checkpoint-7) | vertex-family | 0.8119 | 24.468mm |
| 14 | [Transformer · 120 epoch 한도 best](#checkpoint-12) | 선택 M0·시계열 | 0.8630 | 20.663mm |
| 15 | [GNN · 물리151 검증 조기 종료](#checkpoint-5) | 선택 M0·시계열 | 4.6509 | 132.023mm |
| 16 | [Selected M0 · epoch 60 last](#checkpoint-3) | 선택 M0·시계열 | 4.6617 | 132.317mm |

각 원본의 9개 지표와 체크포인트는 아래 평가 집합별 표를 참고한다.

## 당시 진행 중이던 코너 추적 모델 중간 검증

기록 시각: **2026-11-08 21:54 KST**. 검증 40개 에피소드, 4,102프레임에서 seed별 가림 계획을 사용한 결과다. 같은 seed의 두 모델에는 같은 방식으로 가림 계획을 생성한다. 학습은 진행 중이며 Mamba는 5에폭, Transformer는 3에폭까지 저장됐다. 모든 수치는 완료된 에폭의 검증 결과다.

**아직 최종 성능 비교나 기존 리더보드 순위에 넣을 수 없다.** 1~5에폭은 코너 움직임 사전학습으로 GNN이 고정돼 있어 21D 출력 성능은 학습된 GNN의 성능이 아니다. 서로 다른 에폭의 최신 결과를 우열 비교하지 않는다. 각 seed의 검증 가림 계획이 다르며, 평균은 모두 개별 seed의 평균이다. 최종 3-seed frame-paired 집계와는 다르다.

### 동일 3에폭의 코너 추적 검증

위치 오차는 검증 로그의 코너 예측별 유클리드 거리 분포이며 mm 단위다. 해당 검증 통계에는 ID 불일치로 loss 감독에서 제외된 예측도 포함된다. 따라서 학습 진행 로그의 `position_error_mm`과 모집단이 다르다. 가림 코너 커버리지는 실제 누락 코너 프레임 중 채택 예측으로 덮인 비율, 오연결은 잘못 연결된 재등장 횟수 / 전체 재등장 횟수다.

| 모델 | seed | 평균 오차 | 중앙값 | P90 | 예측 수 | 가림 커버리지 | 오연결 / 재등장 | 업데이트 / 531 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mamba | 0 | 10.40 mm | 0.93 mm | 24.57 mm | 40,797 | 15.3% | 71 / 642 | 531 |
| Mamba | 1 | 9.85 mm | 0.86 mm | 21.73 mm | 40,926 | 15.5% | 76 / 649 | 531 |
| Mamba | 2 | 9.20 mm | 0.95 mm | 13.45 mm | 40,593 | 15.9% | 66 / 600 | 531 |
| Transformer | 0 | 12.13 mm | 1.06 mm | 36.50 mm | 41,004 | 15.5% | 76 / 789 | 531 |
| Transformer | 1 | 11.20 mm | 0.85 mm | 30.43 mm | 40,898 | 15.4% | 99 / 736 | 531 |
| Transformer | 2 | 11.21 mm | 1.16 mm | 30.78 mm | 41,110 | 15.8% | 146 / 872 | 531 |

### 3에폭 미완성 모델의 21D 검증 점수

학습 과정에서 실제 수행한 3에폭 검증을 세 seed의 프레임 평균으로 집계했다. seed별 검증 프레임 수는 4,102개다. 가림 계획은 seed마다 다르며 기존의 가림 없는 4,102프레임 표와 평가 조건이 다르므로, 순위는 그 표의 수치에 단순 대입한 참고 순위다. GNN이 고정된 미완성 모델이다. 새로운 추론을 실행한 결과는 아니다.

| 모델 | Peg 위치 | Peg 회전 | 끝점−Hole 상대 위치 | Hole 위치 | Hole 회전 | 기존 표에 단순 대입 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Transformer | 65.03 mm | 84.83° | 130.71 mm | 140.78 mm | 82.97° | 7개 중 7위 |
| Mamba | 64.96 mm | 84.83° | 130.68 mm | 140.92 mm | 82.97° | 7개 중 7위 |

두 모델을 함께 넣으면 Transformer 7위, Mamba 8위다. 기존 6위의 Hole 위치 오차는 132.32 mm다. 조건 차이와 학습 진행 중인 점 때문에 이를 공정한 최종 순위로 해석하지 않는다.

### 최신 저장 에폭의 21D 검증 참고

각 셀은 검증 프레임 평균이다. 위치·크기·상대 위치는 mm, 회전은 도(°). Mamba 5에폭과 Transformer 3에폭은 학습량이 다르고, GNN은 고정 상태이므로 순위를 매기지 않는다.

| 모델 | seed | epoch | Peg 위치 | Peg 회전 | 끝점−Hole 상대 위치 | Hole 위치 | Hole 회전 | Peg 크기 x | Peg 크기 y | Peg 크기 z | Hole 반지름 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mamba | 0 | 5 | 62.49 | 97.80 | 133.57 | 136.65 | 79.41 | 78.85 | 13.05 | 11.83 | 10.23 |
| Mamba | 1 | 5 | 62.43 | 90.67 | 133.40 | 134.93 | 90.65 | 74.06 | 12.97 | 12.83 | 7.69 |
| Mamba | 2 | 5 | 69.87 | 66.02 | 124.74 | 150.91 | 78.87 | 74.75 | 13.17 | 11.60 | 9.68 |
| Transformer | 0 | 3 | 62.63 | 97.80 | 133.80 | 136.51 | 79.41 | 78.85 | 13.05 | 11.83 | 10.23 |
| Transformer | 1 | 3 | 62.46 | 90.67 | 133.40 | 134.89 | 90.65 | 74.06 | 12.97 | 12.83 | 7.69 |
| Transformer | 2 | 3 | 70.00 | 66.03 | 124.92 | 150.96 | 78.87 | 74.75 | 13.17 | 11.60 | 9.68 |

검증 원본의 에폭별 값, 원격 경로와 SHA-256은 [저장된 검증 스냅샷](../outputs/corner_tracking_leaderboard/history_snapshot.json)에 보존했다. 실행 중인 학습을 중단하거나 추가 평가하지 않았다. 최종 체크포인트와 동일 조건의 별도 평가가 완료되면 정식 순위에 반영한다.

## 평가 집합별 리더보드

### 기존 검증 표본 전체 · 960프레임

40개 부모 에피소드의 원래 검증 표본. 학습 데이터·학습량·선택 기준은 모델마다 다릅니다.

| 순위 | 실험 / 체크포인트 | Peg 위치 | Peg 회전 | 끝점−Hole 상대 위치 | Hole 위치 | Hole 회전 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | [조기 종료 GNN 출발 · 노드 토큰 Mamba 공동 학습](#checkpoint-24) | 2.332 mm | 0.419° | 5.446 mm | 5.413 mm | 0.804° |
| 2 | [60-epoch GNN 출발 · 노드 토큰 Mamba 공동 학습](#checkpoint-25) | 2.486 mm | 0.427° | 5.777 mm | 5.817 mm | 0.811° |
| 3 | [GNN 조기 종료 + 노드 토큰 Mamba 조기 종료](#checkpoint-23) | 9.735 mm | 0.937° | 13.969 mm | 14.679 mm | 1.583° |
| 4 | [GNN 조기 종료 + Mamba 조기 종료](#checkpoint-21) | 9.678 mm | 0.956° | 14.100 mm | 14.930 mm | 1.689° |
| 5 | [Mamba · 120 epoch 한도 best](#checkpoint-11) | 9.159 mm | 1.121° | 15.241 mm | 14.935 mm | 1.821° |
| 6 | [Mamba · 프레임별 노드 토큰 · 120 epoch 한도 best](#checkpoint-20) | 9.851 mm | 1.083° | 14.674 mm | 15.148 mm | 1.739° |
| 7 | [Mamba · 90 epoch 한도 best](#checkpoint-17) | 9.933 mm | 1.139° | 14.824 mm | 15.164 mm | 1.951° |
| 8 | [Mamba · 60 epoch 한도 best](#checkpoint-15) | 9.653 mm | 1.302° | 15.858 mm | 16.078 mm | 2.105° |
| 9 | [GNN 조기 종료 + Transformer 조기 종료](#checkpoint-22) | 10.839 mm | 1.972° | 17.968 mm | 18.432 mm | 3.236° |
| 10 | [Transformer · 90 epoch 한도 best](#checkpoint-18) | 11.102 mm | 2.237° | 21.015 mm | 20.489 mm | 3.604° |
| 11 | [Transformer · 120 epoch 한도 best](#checkpoint-12) | 11.915 mm | 2.103° | 19.234 mm | 20.713 mm | 3.579° |
| 12 | [Transformer · 60 epoch 한도 best](#checkpoint-16) | 11.297 mm | 2.393° | 21.324 mm | 21.690 mm | 4.117° |
| 13 | [M0 원래 개발셋 · best](#checkpoint-13) | 8.285 mm | 18.714° | 30.246 mm | 45.745 mm | 9.114° |
| 14 | [M0 · detector Rim-only](#checkpoint-6) | 10.463 mm | 19.134° | 67.266 mm | 82.788 mm | 10.200° |
| 15 | [Selected M0 · 60 epoch 한도 best](#checkpoint-4) | 42.583 mm | 15.811° | 120.431 mm | 120.178 mm | 28.477° |
| 16 | [Selected M0 · 30 epoch 한도 best](#checkpoint-1) | 42.695 mm | 15.626° | 121.466 mm | 120.581 mm | 28.299° |
| 17 | [GNN · 물리151 검증 조기 종료](#checkpoint-5) | 45.414 mm | 17.783° | 115.206 mm | 129.254 mm | 43.651° |
| 18 | [Selected M0 · epoch 90 last](#checkpoint-14) | 45.460 mm | 17.752° | 115.204 mm | 129.276 mm | 43.659° |
| 19 | [Selected M0 · epoch 60 last](#checkpoint-3) | 45.640 mm | 17.774° | 115.321 mm | 129.561 mm | 43.781° |
| 20 | [Selected M0 · epoch 30 last](#checkpoint-2) | 45.950 mm | 17.391° | 116.971 mm | 131.016 mm | 43.900° |

### 기존 검증 표본 중 두 물리 조건 충족 · 151프레임

39개 부모 에피소드. GT 보조 사후 인증 집합이며 GNN 조기 종료 모델은 이 집합을 체크포인트 선택에도 사용했습니다.

| 순위 | 실험 / 체크포인트 | Peg 위치 | Peg 회전 | 끝점−Hole 상대 위치 | Hole 위치 | Hole 회전 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | [GNN · 물리151 검증 조기 종료](#checkpoint-5) | 1.030 mm | 0.737° | 1.503 mm | 0.993 mm | 0.716° |
| 2 | [Selected M0 · epoch 90 last](#checkpoint-14) | 1.039 mm | 0.742° | 1.533 mm | 0.999 mm | 0.718° |
| 3 | [Selected M0 · epoch 60 last](#checkpoint-3) | 1.077 mm | 0.774° | 1.600 mm | 1.049 mm | 0.755° |
| 4 | [조기 종료 GNN 출발 · 노드 토큰 Mamba 공동 학습](#checkpoint-24) | 1.255 mm | 0.495° | 1.874 mm | 1.376 mm | 0.460° |
| 5 | [GNN 조기 종료 + Transformer 조기 종료](#checkpoint-22) | 1.362 mm | 0.783° | 2.266 mm | 1.473 mm | 0.792° |
| 6 | [60-epoch GNN 출발 · 노드 토큰 Mamba 공동 학습](#checkpoint-25) | 1.320 mm | 0.507° | 1.984 mm | 1.479 mm | 0.453° |
| 7 | [Selected M0 · epoch 30 last](#checkpoint-2) | 1.399 mm | 0.981° | 2.362 mm | 1.524 mm | 0.909° |
| 8 | [GNN 조기 종료 + Mamba 조기 종료](#checkpoint-21) | 1.329 mm | 0.700° | 2.174 mm | 1.582 mm | 0.898° |
| 9 | [Transformer · 120 epoch 한도 best](#checkpoint-12) | 1.371 mm | 0.740° | 2.264 mm | 1.591 mm | 0.825° |
| 10 | [GNN 조기 종료 + 노드 토큰 Mamba 조기 종료](#checkpoint-23) | 1.288 mm | 0.687° | 2.181 mm | 1.592 mm | 0.782° |
| 11 | [Mamba · 90 epoch 한도 best](#checkpoint-17) | 1.331 mm | 0.727° | 2.173 mm | 1.608 mm | 0.974° |
| 12 | [Mamba · 프레임별 노드 토큰 · 120 epoch 한도 best](#checkpoint-20) | 1.364 mm | 0.793° | 2.427 mm | 1.679 mm | 0.842° |
| 13 | [Mamba · 120 epoch 한도 best](#checkpoint-11) | 1.344 mm | 0.717° | 2.283 mm | 1.714 mm | 0.987° |
| 14 | [Transformer · 90 epoch 한도 best](#checkpoint-18) | 1.407 mm | 0.807° | 2.391 mm | 1.744 mm | 0.861° |
| 15 | [Mamba · 60 epoch 한도 best](#checkpoint-15) | 1.389 mm | 0.788° | 2.439 mm | 1.775 mm | 0.932° |
| 16 | [Transformer · 60 epoch 한도 best](#checkpoint-16) | 1.495 mm | 0.906° | 2.531 mm | 1.809 mm | 0.861° |
| 17 | [Selected M0 · 60 epoch 한도 best](#checkpoint-4) | 3.062 mm | 2.360° | 6.158 mm | 5.244 mm | 3.005° |
| 18 | [Selected M0 · 30 epoch 한도 best](#checkpoint-1) | 3.137 mm | 2.416° | 6.415 mm | 5.502 mm | 3.041° |
| 19 | [M0 원래 개발셋 · best](#checkpoint-13) | 5.072 mm | 4.354° | 10.630 mm | 13.018 mm | 6.895° |
| 20 | [M0 · detector Rim-only](#checkpoint-6) | 5.829 mm | 7.257° | 15.152 mm | 18.372 mm | 6.460° |

### 기존 검증 에피소드 전체 — 선택 M0·시계열 계열 · 4,102프레임

40개 부모 에피소드 전체를 평가했습니다. 시계열은 완전한 에피소드를 causal하게 재생합니다.

| 순위 | 실험 / 체크포인트 | Peg 위치 | Peg 회전 | 끝점−Hole 상대 위치 | Hole 위치 | Hole 회전 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | [Mamba · 120 epoch 한도 best](#checkpoint-11) | 9.149 mm | 1.141° | 15.191 mm | 14.828 mm | 1.823° |
| 2 | [GNN 조기 종료 + Mamba 조기 종료](#checkpoint-21) | 9.671 mm | 0.979° | 13.979 mm | 14.922 mm | 1.706° |
| 3 | [GNN 조기 종료 + Transformer 조기 종료](#checkpoint-22) | 10.840 mm | 2.008° | 17.954 mm | 18.350 mm | 3.216° |
| 4 | [Transformer · 120 epoch 한도 best](#checkpoint-12) | 11.740 mm | 2.136° | 19.164 mm | 20.663 mm | 3.557° |
| 5 | [GNN · 물리151 검증 조기 종료](#checkpoint-5) | 44.965 mm | 17.956° | 118.374 mm | 132.023 mm | 43.467° |
| 6 | [Selected M0 · epoch 60 last](#checkpoint-3) | 45.183 mm | 17.949° | 118.474 mm | 132.317 mm | 43.593° |

### 자동 코너 대응 + GNN · 검증 전체 4,102프레임

기존 4,102프레임 시계열 표와 프레임 ID·순서·21D GT가 일치한다. 입력 및 학습 조건이 기존 선택 M0 계열과 달라 별도 cohort로 정렬한다. 코너 예측기는 고정하고 GNN만 학습했다. 40에폭 결과는 처음부터 학습한 것이며, 재개 결과는 그 마지막 GNN과 옵티마이저 상태에서 이어서 검증 loss 개선이 10에폭 없으면 종료, 최대 100에폭까지 학습했다. 재개 시 patience는 0으로 시작했다. 20에폭은 별도 최초 학습이다. 모든 결과는 개발 검증 loss 최소 체크포인트이며 독립 테스트 결과가 아니다.

| 순위 | 실험 / 체크포인트 | Peg 위치 | Peg 회전 | 끝점−Hole 상대 위치 | Hole 위치 | Hole 회전 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | [자동 대응 Mamba + GNN · 재개 · 조기 종료](#detected-resume-mamba) | 1.644mm | 0.862° | 2.930mm | 1.749mm | 0.925° |
| 2 | [자동 대응 Transformer + GNN · 재개 · 조기 종료](#detected-resume-transformer) | 1.425mm | 0.766° | 2.587mm | 1.831mm | 1.000° |
| 3 | [자동 대응 Transformer + GNN · 40 epoch](../outputs/detected_corner_gnn_fresh/comparison.json) | 1.563mm | 0.900° | 2.857mm | 1.923mm | 1.084° |
| 4 | [자동 대응 Mamba + GNN · 40 epoch](../outputs/detected_corner_gnn_fresh/comparison.json) | 1.921mm | 0.993° | 3.313mm | 1.970mm | 1.115° |
| 5 | [자동 대응 Transformer + GNN · 20 epoch](#checkpoint-detected-transformer) | 2.028mm | 1.106° | 3.461mm | 2.304mm | 1.561° |
| 6 | [자동 대응 Mamba + GNN · 20 epoch](#checkpoint-detected-mamba) | 2.376mm | 1.252° | 3.766mm | 2.549mm | 1.454° |

추론 시 GT 코너 ID를 사용하지 않는다. 30mm 거리 기준은 별도 calibration 에피소드에서 두 모델에 공통으로 선택했고, 한 번도 검출되지 않은 점은 생성하지 않는다. 검출점은 실제 좌표를 그대로 쓰고, 미검출 코너는 예측 좌표를 쓴다. 검출기의 PEG/BOX 구분과 검출·예측 여부만 GNN에 넣으며 HOLE의 GT 가족 라벨이나 내부 track ID는 입력하지 않는다. 다만 코너별 시계열 모델은 앞선 학습에서 GT 보조 고정 ID로 이력을 구성했다. 자연스러운 검출 누락을 사용하고 인위적 가림은 추가하지 않았다.

[20 epoch 비교](../outputs/detected_corner_gnn_training/comparison/70e8de0c5baa480ead2deacfd4bd95c4/comparison.json) · [40 epoch 비교](../outputs/detected_corner_gnn_fresh/comparison.json) · [재개 비교](../outputs/detected_corner_gnn_resume/comparison.json) · [전체 학습 곡선](../outputs/detected_corner_gnn_resume/learning_curves.png)

<a id="detected-resume-mamba"></a>
#### 자동 대응 Mamba + GNN · 조기 종료 재개

| 21D 항목 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 1.644mm | 1.174mm | 3.146mm | 1.617mm | 1.615mm | 1.699mm |
| Peg 회전 | 0.862° | 0.669° | 1.488° | 0.836° | 0.852° | 0.898° |
| Peg 끝점−Hole 상대 위치 | 2.930mm | 2.310mm | 5.107mm | 2.855mm | 2.828mm | 3.107mm |
| Hole 위치 | 1.749mm | 1.570mm | 2.815mm | 1.577mm | 1.768mm | 1.902mm |
| Hole 회전 | 0.925° | 0.789° | 1.716° | 0.671° | 1.145° | 0.957° |
| Peg 반쪽 크기 X | 0.937mm | 0.742mm | 1.670mm | 0.954mm | 1.008mm | 0.848mm |
| Peg 반쪽 크기 Y | 0.371mm | 0.280mm | 0.753mm | 0.371mm | 0.432mm | 0.309mm |
| Peg 반쪽 크기 Z | 0.371mm | 0.282mm | 0.756mm | 0.361mm | 0.435mm | 0.315mm |
| Hole 반지름 | 0.697mm | 0.515mm | 1.320mm | 0.561mm | 0.840mm | 0.690mm |

| Seed | 종료 epoch | 최고 epoch | 최고 검증 loss | 최고 체크포인트 / SHA-256 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 79 | 69 | 0.062545 | [best.pt](../outputs/detected_corner_gnn_resume/runs/mamba/seed_0/ae783b56839d48cd971c398c16c9417a/best.pt) `30d1c24c4904d70fab321ce190f989283002797b1a045526f490871c8ed56035` |
| 1 | 56 | 46 | 0.062052 | [best.pt](../outputs/detected_corner_gnn_resume/runs/mamba/seed_1/524879d1388c493ea76df7caa7dea206/best.pt) `04ea9fae5135ba0a6f3a4b41825aab174ede36fb2a7d62e96b1eb0fa9c51b7f4` |
| 2 | 100 (상한) | 99 | 0.072026 | [best.pt](../outputs/detected_corner_gnn_resume/runs/mamba/seed_2/b2557b2c997246adbb5c722006c149be/best.pt) `310d30e124f771b97ccb2ff828620c5c4eb72b329da35cde57d08a8eb6d5af60` |

<a id="detected-resume-transformer"></a>
#### 자동 대응 Transformer + GNN · 조기 종료 재개

| 21D 항목 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 1.425mm | 1.080mm | 2.755mm | 1.398mm | 1.502mm | 1.374mm |
| Peg 회전 | 0.766° | 0.651° | 1.309° | 0.708° | 0.830° | 0.759° |
| Peg 끝점−Hole 상대 위치 | 2.587mm | 2.242mm | 4.451mm | 2.552mm | 2.684mm | 2.525mm |
| Hole 위치 | 1.831mm | 1.597mm | 3.053mm | 1.640mm | 1.867mm | 1.987mm |
| Hole 회전 | 1.000° | 0.811° | 1.757° | 0.716° | 1.140° | 1.145° |
| Peg 반쪽 크기 X | 0.961mm | 0.787mm | 1.711mm | 0.944mm | 1.032mm | 0.906mm |
| Peg 반쪽 크기 Y | 0.434mm | 0.318mm | 0.777mm | 0.454mm | 0.531mm | 0.316mm |
| Peg 반쪽 크기 Z | 0.431mm | 0.320mm | 0.775mm | 0.448mm | 0.528mm | 0.316mm |
| Hole 반지름 | 0.852mm | 0.646mm | 1.585mm | 0.720mm | 1.138mm | 0.698mm |

| Seed | 종료 epoch | 최고 epoch | 최고 검증 loss | 최고 체크포인트 / SHA-256 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 58 | 48 | 0.047962 | [best.pt](../outputs/detected_corner_gnn_resume/runs/transformer/seed_0/a1fd52b4ab53496693a8737596f92baf/best.pt) `3c642e0d4e977ec33d3431d1af9926d65c1945f0b0d989aff9c892cf35f3a8cf` |
| 1 | 50 | 36 | 0.053788 | [best.pt](../outputs/detected_corner_gnn_resume/runs/transformer/seed_1/acfe559cf95b429fb1c0e03e15d9e7b8/best.pt) `fff05dc37d87c02386a9a735767816af464bb95de936b22105cbfe497e3d6065` |
| 2 | 74 | 64 | 0.048101 | [best.pt](../outputs/detected_corner_gnn_resume/runs/transformer/seed_2/68adb1f4baa74facbb26b05337f60b0a/best.pt) `963898ce5506a5faddfe15aa9313c3fa273777e3601f915c486ce98765f8c2cb` |

### 전체 프레임 학습 vertex-family 비교 — 검증 전체 · 4,102프레임

같은 cohort 안에서 4개 조건의 프레임/GT parity가 검증됐습니다. 입력과 학습 프로토콜이 달라 위 시계열 계열과 별도 순위를 사용합니다.

| 순위 | 실험 / 체크포인트 | Peg 위치 | Peg 회전 | 끝점−Hole 상대 위치 | Hole 위치 | Hole 회전 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | [용량 통제](#checkpoint-10) | 4.518 mm | 2.612° | 24.534 mm | 24.111 mm | 5.580° |
| 2 | [M2 Oracle family · GT 입력 참고](#checkpoint-9) | 4.092 mm | 2.418° | 24.419 mm | 24.308 mm | 5.834° |
| 3 | [M0 XYZ/type](#checkpoint-7) | 4.679 mm | 2.706° | 24.625 mm | 24.468 mm | 5.843° |
| 4 | [M1 Detector family](#checkpoint-8) | 4.304 mm | 2.451° | 24.846 mm | 24.551 mm | 5.505° |

### 전체 프레임 학습 vertex-family 비교 — Target-present · 3,645프레임

BOX/Rim 관측이 있는 프레임만 사용합니다. 4,102개 전체 평가와 순위를 섞지 않습니다.

| 순위 | 실험 / 체크포인트 | Peg 위치 | Peg 회전 | 끝점−Hole 상대 위치 | Hole 위치 | Hole 회전 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | [용량 통제](#checkpoint-10) | 3.994 mm | 2.001° | 18.703 mm | 18.434 mm | 4.813° |
| 2 | [M2 Oracle family · GT 입력 참고](#checkpoint-9) | 3.605 mm | 1.759° | 18.624 mm | 18.651 mm | 5.098° |
| 3 | [M0 XYZ/type](#checkpoint-7) | 4.178 mm | 2.107° | 18.703 mm | 18.836 mm | 5.105° |
| 4 | [M1 Detector family](#checkpoint-8) | 3.802 mm | 1.820° | 18.943 mm | 18.930 mm | 4.726° |

## 실험별 21D 상세 표

### 기존 검증 표본 전체 · 960프레임


#### Mamba · 120 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-11)

60-epoch M0 고정. 90→120 추가 학습 후 개발 검증 best 선택. best epoch 119 / 115 / 106.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 9.159 mm | 10.225 mm | 18.304 mm | 9.066 mm | 10.680 mm | **7.730 mm** |
| Peg 회전 | 4D | 1.121° | 0.986° | 1.871° | 1.234° | **1.047°** | 1.081° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 15.241 mm | 15.439 mm | 26.783 mm | 17.417 mm | **14.091 mm** | 14.217 mm |
| Hole 위치 | 3D | 14.935 mm | 13.226 mm | 31.280 mm | 15.064 mm | 16.718 mm | **13.023 mm** |
| Hole 회전 | 4D | 1.821° | 1.644° | 2.981° | 1.964° | **1.727°** | 1.771° |
| Peg 반쪽 크기 X | 1D | 30.605 mm | 39.797 mm | 65.939 mm | 32.597 mm | 33.086 mm | **26.131 mm** |
| Peg 반쪽 크기 Y | 1D | 2.333 mm | 1.936 mm | 5.017 mm | 2.628 mm | 2.813 mm | **1.559 mm** |
| Peg 반쪽 크기 Z | 1D | 2.299 mm | 1.690 mm | 5.034 mm | 2.973 mm | 2.251 mm | **1.674 mm** |
| Hole 반지름 | 1D | 8.107 mm | 8.375 mm | 14.477 mm | **7.158 mm** | 7.659 mm | 9.502 mm |
| **합계** | **21D** | | | | | | |

근거: [mamba_summary_21d.json](../outputs/temporal_continuation_120/temporal_continuation_120/mamba_summary_21d.json) · [seed_0 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/mamba/seed_0/e333b69eb92f44788fe67903b8a31872/predictions.npz) · [seed_1 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/mamba/seed_1/10145c66268d4a1b8d5aa2ebdc840812/predictions.npz) · [seed_2 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/mamba/seed_2/deddfb3bbf954dd4a7857c6ff7bd7799/predictions.npz)

#### Mamba · 프레임별 노드 토큰 · 120 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-20)

60-epoch M0 고정. 기존 64D 관측에 정렬된 pooling 전 노드 토큰을 추가하고 처음부터 120 epoch 학습. 개발 검증 best epoch 108 / 112 / 89. 노드 순서와 새 projection이 추가되므로 기존 Mamba와 매개변수 수 및 연산량이 다르다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 9.851 mm | 11.530 mm | 19.573 mm | 10.183 mm | 10.848 mm | **8.522 mm** |
| Peg 회전 | 4D | 1.083° | 0.943° | 1.807° | 1.063° | **1.040°** | 1.146° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 14.674 mm | 14.687 mm | 26.479 mm | 15.616 mm | 14.480 mm | **13.927 mm** |
| Hole 위치 | 3D | 15.148 mm | 14.306 mm | 30.164 mm | 15.800 mm | 16.547 mm | **13.097 mm** |
| Hole 회전 | 4D | 1.739° | 1.539° | 2.986° | 1.806° | 1.759° | **1.652°** |
| Peg 반쪽 크기 X | 1D | 30.605 mm | 39.797 mm | 65.939 mm | 32.597 mm | 33.086 mm | **26.131 mm** |
| Peg 반쪽 크기 Y | 1D | 2.333 mm | 1.936 mm | 5.017 mm | 2.628 mm | 2.813 mm | **1.559 mm** |
| Peg 반쪽 크기 Z | 1D | 2.299 mm | 1.690 mm | 5.034 mm | 2.973 mm | 2.251 mm | **1.674 mm** |
| Hole 반지름 | 1D | 8.107 mm | 8.375 mm | 14.477 mm | **7.158 mm** | 7.659 mm | 9.502 mm |
| **합계** | **21D** | | | | | | |

근거: [3-seed 집계](../outputs/node_token_mamba/summary_21d.json) · [평가 표본](../outputs/node_token_mamba/validation_frames.jsonl) · [seed 0 프레임별 오차](../outputs/node_token_mamba/training/seed_0/5ec76e979b37428f898b7b0efca9a8ca/selected_errors.npz) · [seed 1 프레임별 오차](../outputs/node_token_mamba/training/seed_1/1d54e275b02a4a0fa0609084b0a07d09/selected_errors.npz) · [seed 2 프레임별 오차](../outputs/node_token_mamba/training/seed_2/20b1a0a1b42141e9a1139ea43242bb6f/selected_errors.npz)

#### Mamba · 90 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-17)

60-epoch M0 고정. 저장된 960-frame 오차 순서를 사용하며 두 집합의 모든 집계값을 요약과 대조했습니다. best epoch 82 / 88 / 83.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 9.933 mm | 11.541 mm | 19.440 mm | 10.059 mm | 11.011 mm | **8.729 mm** |
| Peg 회전 | 4D | 1.139° | 1.002° | 2.025° | **1.055°** | 1.293° | 1.067° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 14.824 mm | 14.150 mm | 27.156 mm | 15.190 mm | 14.765 mm | **14.516 mm** |
| Hole 위치 | 3D | 15.164 mm | 13.616 mm | 30.120 mm | 16.644 mm | 16.694 mm | **12.152 mm** |
| Hole 회전 | 4D | 1.951° | 1.791° | 3.381° | 2.029° | **1.907°** | 1.917° |
| Peg 반쪽 크기 X | 1D | 30.605 mm | 39.797 mm | 65.939 mm | 32.597 mm | 33.086 mm | **26.131 mm** |
| Peg 반쪽 크기 Y | 1D | 2.333 mm | 1.936 mm | 5.017 mm | 2.628 mm | 2.813 mm | **1.559 mm** |
| Peg 반쪽 크기 Z | 1D | 2.299 mm | 1.690 mm | 5.034 mm | 2.973 mm | 2.251 mm | **1.674 mm** |
| Hole 반지름 | 1D | 8.107 mm | 8.375 mm | 14.477 mm | **7.158 mm** | 7.659 mm | 9.502 mm |
| **합계** | **21D** | | | | | | |

근거: [mamba_summary_21d.json](../outputs/temporal_continuation_90/mamba_summary_21d.json) · [mamba_0.npz](../outputs/temporal_continuation_90/seed_errors/mamba_0.npz) · [mamba_1.npz](../outputs/temporal_continuation_90/seed_errors/mamba_1.npz) · [mamba_2.npz](../outputs/temporal_continuation_90/seed_errors/mamba_2.npz)

#### Mamba · 60 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-15)

60-epoch M0 고정. 시계열 개발 검증 59부모로 best 선택; epoch 59 / 58 / 53.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 9.653 mm | 10.803 mm | 18.665 mm | 9.961 mm | 10.867 mm | **8.130 mm** |
| Peg 회전 | 4D | 1.302° | 1.169° | 2.180° | **1.237°** | 1.319° | 1.350° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 15.858 mm | 15.876 mm | 28.086 mm | 18.545 mm | 14.866 mm | **14.164 mm** |
| Hole 위치 | 3D | 16.078 mm | 14.614 mm | 32.695 mm | 16.603 mm | 17.917 mm | **13.715 mm** |
| Hole 회전 | 4D | 2.105° | 1.983° | 3.535° | 2.311° | 2.153° | **1.850°** |
| Peg 반쪽 크기 X | 1D | 30.605 mm | 39.797 mm | 65.939 mm | 32.597 mm | 33.086 mm | **26.131 mm** |
| Peg 반쪽 크기 Y | 1D | 2.333 mm | 1.936 mm | 5.017 mm | 2.628 mm | 2.813 mm | **1.559 mm** |
| Peg 반쪽 크기 Z | 1D | 2.299 mm | 1.690 mm | 5.034 mm | 2.973 mm | 2.251 mm | **1.674 mm** |
| Hole 반지름 | 1D | 8.107 mm | 8.375 mm | 14.477 mm | **7.158 mm** | 7.659 mm | 9.502 mm |
| **합계** | **21D** | | | | | | |

근거: [summary_21d.json](../outputs/selected_m0_mamba/fresh_60/summary_21d.json) · [seed_0 predictions.npz](../outputs/selected_m0_mamba/fresh_60/seed_0/predictions.npz) · [seed_1 predictions.npz](../outputs/selected_m0_mamba/fresh_60/seed_1/predictions.npz) · [seed_2 predictions.npz](../outputs/selected_m0_mamba/fresh_60/seed_2/predictions.npz)

#### Transformer · 90 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-18)

60-epoch M0 고정. 저장된 960-frame 오차 순서를 사용하며 두 집합의 모든 집계값을 요약과 대조했습니다. best epoch 89 / 82 / 80.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 11.102 mm | 12.201 mm | 21.979 mm | 11.253 mm | 12.574 mm | **9.479 mm** |
| Peg 회전 | 4D | 2.237° | 1.840° | 4.617° | **2.174°** | 2.265° | 2.271° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 21.015 mm | 19.553 mm | 41.824 mm | 20.335 mm | 23.180 mm | **19.531 mm** |
| Hole 위치 | 3D | 20.489 mm | 19.664 mm | 40.743 mm | 22.172 mm | 21.195 mm | **18.101 mm** |
| Hole 회전 | 4D | 3.604° | 2.438° | 8.308° | 4.713° | 3.260° | **2.840°** |
| Peg 반쪽 크기 X | 1D | 30.605 mm | 39.797 mm | 65.939 mm | 32.597 mm | 33.086 mm | **26.131 mm** |
| Peg 반쪽 크기 Y | 1D | 2.333 mm | 1.936 mm | 5.017 mm | 2.628 mm | 2.813 mm | **1.559 mm** |
| Peg 반쪽 크기 Z | 1D | 2.299 mm | 1.690 mm | 5.034 mm | 2.973 mm | 2.251 mm | **1.674 mm** |
| Hole 반지름 | 1D | 8.107 mm | 8.375 mm | 14.477 mm | **7.158 mm** | 7.659 mm | 9.502 mm |
| **합계** | **21D** | | | | | | |

근거: [transformer_summary_21d.json](../outputs/temporal_continuation_90/transformer_summary_21d.json) · [transformer_0.npz](../outputs/temporal_continuation_90/seed_errors/transformer_0.npz) · [transformer_1.npz](../outputs/temporal_continuation_90/seed_errors/transformer_1.npz) · [transformer_2.npz](../outputs/temporal_continuation_90/seed_errors/transformer_2.npz)

#### Transformer · 120 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-12)

60-epoch M0 고정. 90→120 추가 학습 후 개발 검증 best 선택. best epoch 111 / 101 / 118.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 11.915 mm | 13.227 mm | 23.132 mm | 13.264 mm | 13.390 mm | **9.091 mm** |
| Peg 회전 | 4D | 2.103° | 1.529° | 4.374° | **2.003°** | 2.258° | 2.047° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 19.234 mm | 17.084 mm | 41.323 mm | **17.144 mm** | 22.793 mm | 17.766 mm |
| Hole 위치 | 3D | 20.713 mm | 20.597 mm | 40.191 mm | 24.099 mm | 20.566 mm | **17.473 mm** |
| Hole 회전 | 4D | 3.579° | 2.393° | 8.279° | 4.787° | 3.263° | **2.687°** |
| Peg 반쪽 크기 X | 1D | 30.605 mm | 39.797 mm | 65.939 mm | 32.597 mm | 33.086 mm | **26.131 mm** |
| Peg 반쪽 크기 Y | 1D | 2.333 mm | 1.936 mm | 5.017 mm | 2.628 mm | 2.813 mm | **1.559 mm** |
| Peg 반쪽 크기 Z | 1D | 2.299 mm | 1.690 mm | 5.034 mm | 2.973 mm | 2.251 mm | **1.674 mm** |
| Hole 반지름 | 1D | 8.107 mm | 8.375 mm | 14.477 mm | **7.158 mm** | 7.659 mm | 9.502 mm |
| **합계** | **21D** | | | | | | |

근거: [transformer_summary_21d.json](../outputs/temporal_continuation_120/temporal_continuation_120/transformer_summary_21d.json) · [seed_0 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/transformer/seed_0/299e0511de81487faf51d9393f982de2/predictions.npz) · [seed_1 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/transformer/seed_1/a5530784df2a4a7b84a292a9395880b1/predictions.npz) · [seed_2 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/transformer/seed_2/780027874899493d8a3391eeed93fee8/predictions.npz)

#### Transformer · 60 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-16)

60-epoch M0 고정. best epoch 57 / 58 / 53. 로컬에는 집계 요약만 있어 seed별 평균은 —입니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 11.297 mm | 12.989 mm | 21.800 mm | — | — | — |
| Peg 회전 | 4D | 2.393° | 1.911° | 4.951° | — | — | — |
| Peg 끝점과 Hole의 상대 위치 | 3D | 21.324 mm | 19.022 mm | 44.370 mm | — | — | — |
| Hole 위치 | 3D | 21.690 mm | 20.957 mm | 42.839 mm | — | — | — |
| Hole 회전 | 4D | 4.117° | 2.716° | 9.181° | — | — | — |
| Peg 반쪽 크기 X | 1D | 30.605 mm | 39.797 mm | 65.939 mm | — | — | — |
| Peg 반쪽 크기 Y | 1D | 2.333 mm | 1.936 mm | 5.017 mm | — | — | — |
| Peg 반쪽 크기 Z | 1D | 2.299 mm | 1.690 mm | 5.034 mm | — | — | — |
| Hole 반지름 | 1D | 8.107 mm | 8.375 mm | 14.477 mm | — | — | — |
| **합계** | **21D** | | | | | | |

근거: [summary_21d.json](../outputs/selected_m0_transformer/summary_21d.json) · [source_checkpoints.json](../outputs/temporal_continuation_90/source_checkpoints.json)

#### M0 원래 개발셋 · best

[체크포인트 매핑으로 이동](#checkpoint-13)

원래 개발셋에서 학습한 기준 모델. Selected M0와 학습 데이터가 다릅니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 8.285 mm | 7.031 mm | 15.277 mm | 8.352 mm | 8.765 mm | **7.739 mm** |
| Peg 회전 | 4D | 18.714° | 3.925° | 109.979° | **18.029°** | 19.433° | 18.682° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 30.246 mm | 21.334 mm | 66.328 mm | 30.554 mm | 30.132 mm | **30.052 mm** |
| Hole 위치 | 3D | 45.745 mm | 22.177 mm | 133.343 mm | 50.428 mm | 44.878 mm | **41.930 mm** |
| Hole 회전 | 4D | 9.114° | 7.445° | 18.324° | 9.056° | **8.611°** | 9.676° |
| Peg 반쪽 크기 X | 1D | 13.411 mm | 11.472 mm | 22.780 mm | 14.422 mm | 13.111 mm | **12.700 mm** |
| Peg 반쪽 크기 Y | 1D | 2.296 mm | 2.385 mm | 4.098 mm | **2.240 mm** | 2.278 mm | 2.369 mm |
| Peg 반쪽 크기 Z | 1D | 2.304 mm | 2.409 mm | 3.998 mm | **2.265 mm** | 2.283 mm | 2.365 mm |
| Hole 반지름 | 1D | 2.363 mm | 2.498 mm | 4.012 mm | 2.375 mm | 2.412 mm | **2.303 mm** |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/summary.json) · [reference_seed_0.npz](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/reference_seed_0.npz) · [reference_seed_1.npz](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/reference_seed_1.npz) · [reference_seed_2.npz](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/reference_seed_2.npz)

#### M0 · detector Rim-only

[체크포인트 매핑으로 이동](#checkpoint-6)

검출기 Rim 후보만 BOX 입력으로 유지. 입력 필터가 다른 실험입니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 10.463 mm | 9.384 mm | 19.871 mm | 10.731 mm | **10.289 mm** | 10.368 mm |
| Peg 회전 | 4D | 19.134° | 15.992° | 32.554° | **18.665°** | 19.363° | 19.374° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 67.266 mm | 61.096 mm | 132.332 mm | 67.498 mm | 67.461 mm | **66.837 mm** |
| Hole 위치 | 3D | 82.788 mm | 82.414 mm | 155.009 mm | 82.770 mm | 82.876 mm | **82.719 mm** |
| Hole 회전 | 4D | 10.200° | 10.596° | 19.460° | 10.185° | 10.296° | **10.120°** |
| Peg 반쪽 크기 X | 1D | 25.819 mm | 21.009 mm | 50.433 mm | 27.361 mm | 26.635 mm | **23.460 mm** |
| Peg 반쪽 크기 Y | 1D | 2.097 mm | 1.886 mm | 3.711 mm | **2.033 mm** | 2.177 mm | 2.081 mm |
| Peg 반쪽 크기 Z | 1D | 2.072 mm | 1.970 mm | 3.668 mm | **2.016 mm** | 2.133 mm | 2.066 mm |
| Hole 반지름 | 1D | 2.310 mm | 2.492 mm | 4.127 mm | 2.317 mm | **2.300 mm** | 2.312 mm |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/rim_only_experiment/evaluation/ef864771775547c0bce9e1ddaf39838a/summary.json) · [filtered_seed_0.npz](../outputs/rim_only_experiment/evaluation/ef864771775547c0bce9e1ddaf39838a/filtered_seed_0.npz) · [filtered_seed_1.npz](../outputs/rim_only_experiment/evaluation/ef864771775547c0bce9e1ddaf39838a/filtered_seed_1.npz) · [filtered_seed_2.npz](../outputs/rim_only_experiment/evaluation/ef864771775547c0bce9e1ddaf39838a/filtered_seed_2.npz)

#### Selected M0 · 60 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-4)

전체 검증 손실 기준 best epoch: seed 순서 3 / 5 / 50.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 42.583 mm | 57.680 mm | 85.907 mm | 48.781 mm | **36.384 mm** | 42.584 mm |
| Peg 회전 | 4D | 15.811° | 10.871° | 35.169° | **13.662°** | 18.697° | 15.075° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 120.431 mm | 111.561 mm | 271.687 mm | 126.279 mm | 126.063 mm | **108.950 mm** |
| Hole 위치 | 3D | 120.178 mm | 98.664 mm | 327.338 mm | **119.338 mm** | 121.566 mm | 119.630 mm |
| Hole 회전 | 4D | 28.477° | 28.519° | 64.415° | **27.195°** | 30.820° | 27.415° |
| Peg 반쪽 크기 X | 1D | 27.772 mm | 32.171 mm | 58.488 mm | 26.595 mm | 31.067 mm | **25.654 mm** |
| Peg 반쪽 크기 Y | 1D | 3.704 mm | 3.533 mm | 6.796 mm | 4.189 mm | 5.360 mm | **1.562 mm** |
| Peg 반쪽 크기 Z | 1D | 3.446 mm | 3.279 mm | 6.197 mm | 3.757 mm | 4.885 mm | **1.697 mm** |
| Hole 반지름 | 1D | 5.853 mm | 5.851 mm | 10.368 mm | **3.986 mm** | 3.986 mm | 9.586 mm |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/selected_vertex_continuation/evaluation_best/fadf9805ef1d45f1bc6cf6f6a0542f76/summary.json) · [selected_seed_0.npz](../outputs/selected_vertex_continuation/evaluation_best/fadf9805ef1d45f1bc6cf6f6a0542f76/selected_seed_0.npz) · [selected_seed_1.npz](../outputs/selected_vertex_continuation/evaluation_best/fadf9805ef1d45f1bc6cf6f6a0542f76/selected_seed_1.npz) · [selected_seed_2.npz](../outputs/selected_vertex_continuation/evaluation_best/fadf9805ef1d45f1bc6cf6f6a0542f76/selected_seed_2.npz)

#### Selected M0 · 30 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-1)

원래 960프레임의 전체 검증 손실로 best 선택.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 42.695 mm | 57.663 mm | 86.717 mm | 48.781 mm | **36.384 mm** | 42.922 mm |
| Peg 회전 | 4D | 15.626° | 10.185° | 35.260° | **13.662°** | 18.697° | 14.518° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 121.466 mm | 112.412 mm | 275.877 mm | 126.279 mm | 126.063 mm | **112.057 mm** |
| Hole 위치 | 3D | 120.581 mm | 99.344 mm | 327.346 mm | **119.338 mm** | 121.566 mm | 120.839 mm |
| Hole 회전 | 4D | 28.299° | 28.513° | 64.929° | 27.195° | 30.820° | **26.883°** |
| Peg 반쪽 크기 X | 1D | 27.466 mm | 31.519 mm | 57.746 mm | 26.595 mm | 31.067 mm | **24.736 mm** |
| Peg 반쪽 크기 Y | 1D | 3.797 mm | 3.466 mm | 6.993 mm | 4.189 mm | 5.360 mm | **1.841 mm** |
| Peg 반쪽 크기 Z | 1D | 3.527 mm | 3.295 mm | 6.426 mm | 3.757 mm | 4.885 mm | **1.941 mm** |
| Hole 반지름 | 1D | 5.907 mm | 5.697 mm | 10.498 mm | **3.986 mm** | 3.986 mm | 9.749 mm |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/summary.json) · [selected_seed_0.npz](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/selected_seed_0.npz) · [selected_seed_1.npz](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/selected_seed_1.npz) · [selected_seed_2.npz](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/selected_seed_2.npz)

#### GNN · 물리151 검증 조기 종료

[체크포인트 매핑으로 이동](#checkpoint-5)

사용자가 제시한 예시 모델. best epoch 106 / 91 / 105. 평가의 151프레임을 선택에도 사용했습니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 45.414 mm | 63.366 mm | 92.683 mm | 43.630 mm | 49.247 mm | **43.365 mm** |
| Peg 회전 | 4D | 17.783° | 14.317° | 46.543° | 20.111° | 16.917° | **16.322°** |
| Peg 끝점과 Hole의 상대 위치 | 3D | 115.206 mm | 106.292 mm | 260.457 mm | 115.085 mm | 121.507 mm | **109.027 mm** |
| Hole 위치 | 3D | 129.254 mm | 115.987 mm | 324.723 mm | 139.341 mm | 129.033 mm | **119.387 mm** |
| Hole 회전 | 4D | 43.651° | 52.566° | 86.977° | 68.193° | 35.183° | **27.576°** |
| Peg 반쪽 크기 X | 1D | 30.486 mm | 39.571 mm | 65.874 mm | 32.768 mm | 32.424 mm | **26.266 mm** |
| Peg 반쪽 크기 Y | 1D | 2.285 mm | 1.819 mm | 4.918 mm | 2.689 mm | 2.676 mm | **1.490 mm** |
| Peg 반쪽 크기 Z | 1D | 2.260 mm | 1.695 mm | 4.942 mm | 3.053 mm | 2.131 mm | **1.595 mm** |
| Hole 반지름 | 1D | 8.009 mm | 8.138 mm | 14.431 mm | **6.970 mm** | 7.607 mm | 9.450 mm |
| **합계** | **21D** | | | | | | |

근거: [detailed_summary.json](../outputs/temporal_continuation_120/gnn_selected_validation/evaluation_input/detailed_summary.json)

#### Selected M0 · epoch 90 last

[체크포인트 매핑으로 이동](#checkpoint-14)

최종 epoch 90 가중치. 로컬에는 집계 요약만 있어 seed별 평균은 —로 표기합니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 45.460 mm | 63.478 mm | 92.727 mm | — | — | — |
| Peg 회전 | 4D | 17.752° | 14.258° | 46.442° | — | — | — |
| Peg 끝점과 Hole의 상대 위치 | 3D | 115.204 mm | 106.395 mm | 260.792 mm | — | — | — |
| Hole 위치 | 3D | 129.276 mm | 116.055 mm | 324.723 mm | — | — | — |
| Hole 회전 | 4D | 43.659° | 52.564° | 87.001° | — | — | — |
| Peg 반쪽 크기 X | 1D | 30.516 mm | 39.626 mm | 65.919 mm | — | — | — |
| Peg 반쪽 크기 Y | 1D | 2.280 mm | 1.807 mm | 4.905 mm | — | — | — |
| Peg 반쪽 크기 Z | 1D | 2.255 mm | 1.680 mm | 4.929 mm | — | — | — |
| Hole 반지름 | 1D | 8.017 mm | 8.140 mm | 14.441 mm | — | — | — |
| **합계** | **21D** | | | | | | |

근거: [summary_final_vs_60.json](../outputs/gnn_continuation_90/summary_final_vs_60.json)

#### Selected M0 · epoch 60 last

[체크포인트 매핑으로 이동](#checkpoint-3)

최종 epoch 60 가중치. 시계열 모델이 고정해 사용한 M0입니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 45.640 mm | 64.183 mm | 92.992 mm | **43.352 mm** | 49.907 mm | 43.661 mm |
| Peg 회전 | 4D | 17.774° | 14.087° | 46.262° | 20.139° | 17.210° | **15.972°** |
| Peg 끝점과 Hole의 상대 위치 | 3D | 115.321 mm | 106.775 mm | 262.405 mm | 115.212 mm | 121.716 mm | **109.035 mm** |
| Hole 위치 | 3D | 129.561 mm | 116.784 mm | 324.724 mm | 140.024 mm | 129.093 mm | **119.566 mm** |
| Hole 회전 | 4D | 43.781° | 53.188° | 86.337° | 69.096° | 34.726° | **27.523°** |
| Peg 반쪽 크기 X | 1D | 30.605 mm | 39.797 mm | 65.939 mm | 32.597 mm | 33.086 mm | **26.131 mm** |
| Peg 반쪽 크기 Y | 1D | 2.333 mm | 1.936 mm | 5.017 mm | 2.628 mm | 2.813 mm | **1.559 mm** |
| Peg 반쪽 크기 Z | 1D | 2.299 mm | 1.690 mm | 5.034 mm | 2.973 mm | 2.251 mm | **1.674 mm** |
| Hole 반지름 | 1D | 8.107 mm | 8.375 mm | 14.477 mm | **7.158 mm** | 7.659 mm | 9.502 mm |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/summary.json) · [selected_seed_0.npz](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/selected_seed_0.npz) · [selected_seed_1.npz](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/selected_seed_1.npz) · [selected_seed_2.npz](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/selected_seed_2.npz)

#### Selected M0 · epoch 30 last

[체크포인트 매핑으로 이동](#checkpoint-2)

최종 epoch 30 가중치. best와 구분합니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 45.950 mm | 64.805 mm | 94.060 mm | 44.547 mm | 50.382 mm | **42.922 mm** |
| Peg 회전 | 4D | 17.391° | 14.166° | 44.450° | 18.133° | 19.521° | **14.518°** |
| Peg 끝점과 Hole의 상대 위치 | 3D | 116.971 mm | 107.247 mm | 266.587 mm | 115.627 mm | 123.230 mm | **112.057 mm** |
| Hole 위치 | 3D | 131.016 mm | 120.251 mm | 324.818 mm | 142.868 mm | 129.339 mm | **120.839 mm** |
| Hole 회전 | 4D | 43.900° | 54.754° | 84.859° | 71.866° | 32.949° | **26.883°** |
| Peg 반쪽 크기 X | 1D | 31.157 mm | 40.643 mm | 67.130 mm | 32.652 mm | 36.082 mm | **24.736 mm** |
| Peg 반쪽 크기 Y | 1D | 2.293 mm | 2.280 mm | 4.376 mm | 2.246 mm | 2.793 mm | **1.841 mm** |
| Peg 반쪽 크기 Z | 1D | 2.168 mm | 1.865 mm | 4.348 mm | 2.431 mm | 2.131 mm | **1.941 mm** |
| Hole 반지름 | 1D | 8.493 mm | 8.809 mm | 14.864 mm | 7.929 mm | **7.802 mm** | 9.749 mm |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/summary.json) · [reference_seed_0.npz](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/reference_seed_0.npz) · [reference_seed_1.npz](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/reference_seed_1.npz) · [reference_seed_2.npz](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/reference_seed_2.npz)

#### 초기 Mamba · seed 0만 완료

[체크포인트 매핑으로 이동](#checkpoint-19)

3-seed 순위에서 제외. 단일 seed 결과이며 나머지 seed를 추정하지 않습니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | — | — | — | 11.829 mm | — | — |
| Peg 회전 | 4D | — | — | — | 1.951° | — | — |
| Peg 끝점과 Hole의 상대 위치 | 3D | — | — | — | 21.557 mm | — | — |
| Hole 위치 | 3D | — | — | — | 20.388 mm | — | — |
| Hole 회전 | 4D | — | — | — | 4.198° | — | — |
| Peg 반쪽 크기 X | 1D | — | — | — | 32.597 mm | — | — |
| Peg 반쪽 크기 Y | 1D | — | — | — | 2.628 mm | — | — |
| Peg 반쪽 크기 Z | 1D | — | — | — | 2.973 mm | — | — |
| Hole 반지름 | 1D | — | — | — | 7.158 mm | — | — |
| **합계** | **21D** | | | | | | |

근거: [original_validation_21d.json](../outputs/selected_m0_mamba/seed_0/original_validation_21d.json)

#### GNN 조기 종료 + Mamba 조기 종료

[체크포인트 매핑으로 이동](#checkpoint-21)

물리151 검증으로 선택한 GNN을 고정. Mamba best epoch 264 / 213 / 85, 조기 종료 epoch 294 / 243 / 115. 검증 loss 기준으로 best 선택. 해당 151프레임은 GNN 선택에도 사용됐다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 9.678 mm | 11.290 mm | 18.980 mm | 10.660 mm | 10.418 mm | **7.955 mm** |
| Peg 회전 | 4D | 0.956° | 0.816° | 1.603° | **0.787°** | 0.944° | 1.138° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 14.100 mm | 13.953 mm | 26.177 mm | 14.432 mm | **12.954 mm** | 14.913 mm |
| Hole 위치 | 3D | 14.930 mm | 13.783 mm | 30.907 mm | 15.269 mm | 16.538 mm | **12.984 mm** |
| Hole 회전 | 4D | 1.689° | 1.511° | 2.803° | 1.612° | **1.520°** | 1.936° |
| Peg 반쪽 크기 X | 1D | 30.486 mm | 39.571 mm | 65.874 mm | 32.768 mm | 32.424 mm | **26.266 mm** |
| Peg 반쪽 크기 Y | 1D | 2.285 mm | 1.819 mm | 4.918 mm | 2.689 mm | 2.676 mm | **1.490 mm** |
| Peg 반쪽 크기 Z | 1D | 2.260 mm | 1.695 mm | 4.942 mm | 3.053 mm | 2.131 mm | **1.595 mm** |
| Hole 반지름 | 1D | 8.009 mm | 8.138 mm | 14.431 mm | **6.970 mm** | 7.607 mm | 9.450 mm |
| **합계** | **21D** | | | | | | |

근거: [3-seed 집계](../outputs/temporal_early_stopping/mamba_summary_21d.json) · [seed 0 예측](../outputs/temporal_early_stopping/evaluation/mamba/seed_0/183b8a11f3cd40669bc13a44a2fff2f2/predictions.npz) · [seed 1 예측](../outputs/temporal_early_stopping/evaluation/mamba/seed_1/29c516ee7cd344cd86c1f9827f2bac81/predictions.npz) · [seed 2 예측](../outputs/temporal_early_stopping/evaluation/mamba/seed_2/48b201634afc4967b49c14a5818d4b93/predictions.npz)


#### GNN 조기 종료 + Transformer 조기 종료

[체크포인트 매핑으로 이동](#checkpoint-22)

물리151 검증으로 선택한 GNN을 고정. Transformer best epoch 165 / 151 / 183, 조기 종료 epoch 195 / 181 / 213. 검증 loss 기준으로 best 선택. 해당 151프레임은 GNN 선택에도 사용됐다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 10.839 mm | 12.735 mm | 20.296 mm | 11.378 mm | 12.121 mm | **9.018 mm** |
| Peg 회전 | 4D | 1.972° | 1.574° | 3.804° | **1.824°** | 2.030° | 2.062° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 17.968 mm | 17.716 mm | 34.745 mm | 17.607 mm | 19.990 mm | **16.306 mm** |
| Hole 위치 | 3D | 18.432 mm | 18.224 mm | 35.946 mm | 20.566 mm | 19.537 mm | **15.194 mm** |
| Hole 회전 | 4D | 3.236° | 2.138° | 7.654° | 4.345° | 2.934° | **2.430°** |
| Peg 반쪽 크기 X | 1D | 30.486 mm | 39.571 mm | 65.874 mm | 32.768 mm | 32.424 mm | **26.266 mm** |
| Peg 반쪽 크기 Y | 1D | 2.285 mm | 1.819 mm | 4.918 mm | 2.689 mm | 2.676 mm | **1.490 mm** |
| Peg 반쪽 크기 Z | 1D | 2.260 mm | 1.695 mm | 4.942 mm | 3.053 mm | 2.131 mm | **1.595 mm** |
| Hole 반지름 | 1D | 8.009 mm | 8.138 mm | 14.431 mm | **6.970 mm** | 7.607 mm | 9.450 mm |
| **합계** | **21D** | | | | | | |

근거: [3-seed 집계](../outputs/temporal_early_stopping/transformer_summary_21d.json) · [seed 0 예측](../outputs/temporal_early_stopping/evaluation/transformer/seed_0/9058a3586f0445dfafec0b93e9bc5875/predictions.npz) · [seed 1 예측](../outputs/temporal_early_stopping/evaluation/transformer/seed_1/69c8a27f7d014b6db300e2cd9c68c7dd/predictions.npz) · [seed 2 예측](../outputs/temporal_early_stopping/evaluation/transformer/seed_2/ec956145e49b491cba71702ff1df0c01/predictions.npz)

#### GNN 조기 종료 + 노드 토큰 Mamba 조기 종료

[체크포인트 매핑으로 이동](#checkpoint-23)

물리151 검증으로 선택한 GNN best를 seed별로 고정. 노드 토큰 Mamba best epoch 97 / 285 / 198, 조기 종료 epoch 127 / 315 / 228. 검증 loss 기준 best 선택. 151프레임은 GNN 선택에도 사용됐다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 9.735 mm | 10.870 mm | 19.540 mm | 10.581 mm | 10.270 mm | **8.352 mm** |
| Peg 회전 | 4D | 0.937° | 0.832° | 1.617° | 0.951° | 0.994° | **0.866°** |
| Peg 끝점과 Hole의 상대 위치 | 3D | 13.969 mm | 14.573 mm | 24.745 mm | 14.835 mm | **13.362 mm** | 13.710 mm |
| Hole 위치 | 3D | 14.679 mm | 13.163 mm | 29.577 mm | 15.733 mm | 15.935 mm | **12.369 mm** |
| Hole 회전 | 4D | 1.583° | 1.412° | 2.681° | 1.821° | **1.320°** | 1.609° |
| Peg 반쪽 크기 X | 1D | 30.486 mm | 39.571 mm | 65.874 mm | 32.768 mm | 32.424 mm | **26.266 mm** |
| Peg 반쪽 크기 Y | 1D | 2.285 mm | 1.819 mm | 4.918 mm | 2.689 mm | 2.676 mm | **1.490 mm** |
| Peg 반쪽 크기 Z | 1D | 2.260 mm | 1.695 mm | 4.942 mm | 3.053 mm | 2.131 mm | **1.595 mm** |
| Hole 반지름 | 1D | 8.009 mm | 8.138 mm | 14.431 mm | **6.970 mm** | 7.607 mm | 9.450 mm |
| **합계** | **21D** | | | | | | |

근거: [3-seed 비교](../outputs/node_token_early_stopping/comparison/785a048f899144e086fb48ed25b79b14/comparison.json) · [평가 표본](../outputs/node_token_mamba/validation_frames.jsonl) · [seed 0 프레임별 오차](../outputs/node_token_early_stopping/training/seed_0/6d16da7ceac54470ad6bc4f7858fef62/selected_errors.npz) · [seed 1 프레임별 오차](../outputs/node_token_early_stopping/training/seed_1/70ace57f59f54409bc75d140268036f1/selected_errors.npz) · [seed 2 프레임별 오차](../outputs/node_token_early_stopping/training/seed_2/ab6c964532824f13b9acc382b5901fb5/selected_errors.npz)


### 기존 검증 표본 중 두 물리 조건 충족 · 151프레임


#### GNN · 물리151 검증 조기 종료

[체크포인트 매핑으로 이동](#checkpoint-5)

사용자가 제시한 예시 모델. best epoch 106 / 91 / 105. 평가의 151프레임을 선택에도 사용했습니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 1.030 mm | 0.568 mm | 2.218 mm | 1.081 mm | 1.023 mm | **0.987 mm** |
| Peg 회전 | 4D | 0.737° | 0.531° | 1.095° | **0.624°** | 0.772° | 0.816° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 1.503 mm | 1.042 mm | 2.840 mm | **1.480 mm** | 1.492 mm | 1.539 mm |
| Hole 위치 | 3D | 0.993 mm | 0.765 mm | 1.810 mm | **0.902 mm** | 0.917 mm | 1.160 mm |
| Hole 회전 | 4D | 0.716° | 0.624° | 1.293° | **0.606°** | 0.679° | 0.862° |
| Peg 반쪽 크기 X | 1D | 0.507 mm | 0.357 mm | 0.718 mm | **0.438 mm** | 0.504 mm | 0.579 mm |
| Peg 반쪽 크기 Y | 1D | 0.333 mm | 0.266 mm | 0.566 mm | **0.315 mm** | 0.326 mm | 0.358 mm |
| Peg 반쪽 크기 Z | 1D | 0.321 mm | 0.268 mm | 0.530 mm | **0.290 mm** | 0.332 mm | 0.342 mm |
| Hole 반지름 | 1D | 0.615 mm | 0.511 mm | 1.127 mm | **0.538 mm** | 0.716 mm | 0.590 mm |
| **합계** | **21D** | | | | | | |

근거: [detailed_summary.json](../outputs/temporal_continuation_120/gnn_selected_validation/evaluation_input/detailed_summary.json)

#### Selected M0 · epoch 90 last

[체크포인트 매핑으로 이동](#checkpoint-14)

최종 epoch 90 가중치. 로컬에는 집계 요약만 있어 seed별 평균은 —로 표기합니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 1.039 mm | 0.579 mm | 2.203 mm | — | — | — |
| Peg 회전 | 4D | 0.742° | 0.532° | 1.108° | — | — | — |
| Peg 끝점과 Hole의 상대 위치 | 3D | 1.533 mm | 1.051 mm | 2.963 mm | — | — | — |
| Hole 위치 | 3D | 0.999 mm | 0.763 mm | 1.847 mm | — | — | — |
| Hole 회전 | 4D | 0.718° | 0.638° | 1.264° | — | — | — |
| Peg 반쪽 크기 X | 1D | 0.505 mm | 0.351 mm | 0.689 mm | — | — | — |
| Peg 반쪽 크기 Y | 1D | 0.336 mm | 0.270 mm | 0.569 mm | — | — | — |
| Peg 반쪽 크기 Z | 1D | 0.324 mm | 0.267 mm | 0.534 mm | — | — | — |
| Hole 반지름 | 1D | 0.617 mm | 0.513 mm | 1.137 mm | — | — | — |
| **합계** | **21D** | | | | | | |

근거: [summary_final_vs_60.json](../outputs/gnn_continuation_90/summary_final_vs_60.json)

#### Selected M0 · epoch 60 last

[체크포인트 매핑으로 이동](#checkpoint-3)

최종 epoch 60 가중치. 시계열 모델이 고정해 사용한 M0입니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 1.077 mm | 0.581 mm | 2.282 mm | 1.124 mm | 1.073 mm | **1.034 mm** |
| Peg 회전 | 4D | 0.774° | 0.549° | 1.146° | **0.649°** | 0.815° | 0.858° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 1.600 mm | 1.150 mm | 2.922 mm | **1.550 mm** | 1.631 mm | 1.620 mm |
| Hole 위치 | 3D | 1.049 mm | 0.771 mm | 1.861 mm | 0.957 mm | **0.951 mm** | 1.240 mm |
| Hole 회전 | 4D | 0.755° | 0.642° | 1.355° | **0.657°** | 0.714° | 0.894° |
| Peg 반쪽 크기 X | 1D | 0.557 mm | 0.397 mm | 0.795 mm | **0.472 mm** | 0.524 mm | 0.674 mm |
| Peg 반쪽 크기 Y | 1D | 0.349 mm | 0.281 mm | 0.596 mm | **0.317 mm** | 0.346 mm | 0.382 mm |
| Peg 반쪽 크기 Z | 1D | 0.340 mm | 0.259 mm | 0.566 mm | **0.303 mm** | 0.348 mm | 0.368 mm |
| Hole 반지름 | 1D | 0.644 mm | 0.550 mm | 1.180 mm | **0.564 mm** | 0.760 mm | 0.609 mm |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/summary.json) · [selected_seed_0.npz](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/selected_seed_0.npz) · [selected_seed_1.npz](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/selected_seed_1.npz) · [selected_seed_2.npz](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/selected_seed_2.npz)

#### Selected M0 · epoch 30 last

[체크포인트 매핑으로 이동](#checkpoint-2)

최종 epoch 30 가중치. best와 구분합니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 1.399 mm | 0.887 mm | 2.979 mm | 1.463 mm | 1.397 mm | **1.337 mm** |
| Peg 회전 | 4D | 0.981° | 0.729° | 1.636° | **0.783°** | 1.097° | 1.062° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 2.362 mm | 1.797 mm | 3.769 mm | **2.195 mm** | 2.293 mm | 2.597 mm |
| Hole 위치 | 3D | 1.524 mm | 1.264 mm | 2.732 mm | 1.272 mm | **1.267 mm** | 2.033 mm |
| Hole 회전 | 4D | 0.909° | 0.842° | 1.502° | **0.838°** | 0.881° | 1.008° |
| Peg 반쪽 크기 X | 1D | 0.883 mm | 0.739 mm | 1.214 mm | **0.572 mm** | 0.792 mm | 1.287 mm |
| Peg 반쪽 크기 Y | 1D | 0.456 mm | 0.357 mm | 0.921 mm | 0.445 mm | **0.442 mm** | 0.481 mm |
| Peg 반쪽 크기 Z | 1D | 0.446 mm | 0.367 mm | 0.905 mm | **0.441 mm** | 0.444 mm | 0.453 mm |
| Hole 반지름 | 1D | 0.769 mm | 0.617 mm | 1.498 mm | **0.647 mm** | 0.995 mm | 0.664 mm |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/summary.json) · [reference_seed_0.npz](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/reference_seed_0.npz) · [reference_seed_1.npz](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/reference_seed_1.npz) · [reference_seed_2.npz](../outputs/selected_vertex_continuation/evaluation_final_vs_final/657659bf8dca47b6a86e9ac66dfba844/reference_seed_2.npz)

#### Transformer · 120 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-12)

60-epoch M0 고정. 90→120 추가 학습 후 개발 검증 best 선택. best epoch 111 / 101 / 118.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 1.371 mm | 0.942 mm | 2.763 mm | 1.382 mm | **1.302 mm** | 1.428 mm |
| Peg 회전 | 4D | 0.740° | 0.524° | 1.308° | 0.727° | **0.708°** | 0.785° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 2.264 mm | 1.797 mm | 3.727 mm | **2.143 mm** | 2.229 mm | 2.419 mm |
| Hole 위치 | 3D | 1.591 mm | 1.346 mm | 2.453 mm | **1.371 mm** | 1.536 mm | 1.865 mm |
| Hole 회전 | 4D | 0.825° | 0.699° | 1.327° | **0.689°** | 0.801° | 0.986° |
| Peg 반쪽 크기 X | 1D | 0.557 mm | 0.397 mm | 0.795 mm | **0.472 mm** | 0.524 mm | 0.674 mm |
| Peg 반쪽 크기 Y | 1D | 0.349 mm | 0.281 mm | 0.596 mm | **0.317 mm** | 0.346 mm | 0.382 mm |
| Peg 반쪽 크기 Z | 1D | 0.340 mm | 0.259 mm | 0.566 mm | **0.303 mm** | 0.348 mm | 0.368 mm |
| Hole 반지름 | 1D | 0.644 mm | 0.550 mm | 1.180 mm | **0.564 mm** | 0.760 mm | 0.609 mm |
| **합계** | **21D** | | | | | | |

근거: [transformer_summary_21d.json](../outputs/temporal_continuation_120/temporal_continuation_120/transformer_summary_21d.json) · [seed_0 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/transformer/seed_0/299e0511de81487faf51d9393f982de2/predictions.npz) · [seed_1 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/transformer/seed_1/a5530784df2a4a7b84a292a9395880b1/predictions.npz) · [seed_2 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/transformer/seed_2/780027874899493d8a3391eeed93fee8/predictions.npz)

#### Mamba · 90 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-17)

60-epoch M0 고정. 저장된 960-frame 오차 순서를 사용하며 두 집합의 모든 집계값을 요약과 대조했습니다. best epoch 82 / 88 / 83.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 1.331 mm | 0.919 mm | 2.262 mm | 1.394 mm | 1.343 mm | **1.256 mm** |
| Peg 회전 | 4D | 0.727° | 0.533° | 1.123° | **0.629°** | 0.803° | 0.748° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 2.173 mm | 1.818 mm | 3.254 mm | **2.011 mm** | 2.334 mm | 2.174 mm |
| Hole 위치 | 3D | 1.608 mm | 1.458 mm | 2.441 mm | **1.537 mm** | 1.577 mm | 1.711 mm |
| Hole 회전 | 4D | 0.974° | 0.867° | 1.687° | 0.926° | **0.875°** | 1.120° |
| Peg 반쪽 크기 X | 1D | 0.557 mm | 0.397 mm | 0.795 mm | **0.472 mm** | 0.524 mm | 0.674 mm |
| Peg 반쪽 크기 Y | 1D | 0.349 mm | 0.281 mm | 0.596 mm | **0.317 mm** | 0.346 mm | 0.382 mm |
| Peg 반쪽 크기 Z | 1D | 0.340 mm | 0.259 mm | 0.566 mm | **0.303 mm** | 0.348 mm | 0.368 mm |
| Hole 반지름 | 1D | 0.644 mm | 0.550 mm | 1.180 mm | **0.564 mm** | 0.760 mm | 0.609 mm |
| **합계** | **21D** | | | | | | |

근거: [mamba_summary_21d.json](../outputs/temporal_continuation_90/mamba_summary_21d.json) · [mamba_0.npz](../outputs/temporal_continuation_90/seed_errors/mamba_0.npz) · [mamba_1.npz](../outputs/temporal_continuation_90/seed_errors/mamba_1.npz) · [mamba_2.npz](../outputs/temporal_continuation_90/seed_errors/mamba_2.npz)

#### Mamba · 프레임별 노드 토큰 · 120 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-20)

60-epoch M0 고정. 기존 64D 관측에 정렬된 pooling 전 노드 토큰을 추가하고 처음부터 120 epoch 학습. 개발 검증 best epoch 108 / 112 / 89. 노드 순서와 새 projection이 추가되므로 기존 Mamba와 매개변수 수 및 연산량이 다르다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 1.364 mm | 0.984 mm | 2.335 mm | **1.308 mm** | 1.444 mm | 1.341 mm |
| Peg 회전 | 4D | 0.793° | 0.657° | 1.252° | **0.611°** | 0.936° | 0.832° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 2.427 mm | 2.131 mm | 3.560 mm | **2.033 mm** | 2.762 mm | 2.487 mm |
| Hole 위치 | 3D | 1.679 mm | 1.505 mm | 2.551 mm | **1.438 mm** | 1.597 mm | 2.002 mm |
| Hole 회전 | 4D | 0.842° | 0.787° | 1.337° | 0.806° | **0.805°** | 0.914° |
| Peg 반쪽 크기 X | 1D | 0.557 mm | 0.397 mm | 0.795 mm | **0.472 mm** | 0.524 mm | 0.674 mm |
| Peg 반쪽 크기 Y | 1D | 0.349 mm | 0.281 mm | 0.596 mm | **0.317 mm** | 0.346 mm | 0.382 mm |
| Peg 반쪽 크기 Z | 1D | 0.340 mm | 0.259 mm | 0.566 mm | **0.303 mm** | 0.348 mm | 0.368 mm |
| Hole 반지름 | 1D | 0.644 mm | 0.550 mm | 1.180 mm | **0.564 mm** | 0.760 mm | 0.609 mm |
| **합계** | **21D** | | | | | | |

근거: [3-seed 집계](../outputs/node_token_mamba/summary_21d.json) · [평가 표본](../outputs/node_token_mamba/validation_frames.jsonl) · [seed 0 프레임별 오차](../outputs/node_token_mamba/training/seed_0/5ec76e979b37428f898b7b0efca9a8ca/selected_errors.npz) · [seed 1 프레임별 오차](../outputs/node_token_mamba/training/seed_1/1d54e275b02a4a0fa0609084b0a07d09/selected_errors.npz) · [seed 2 프레임별 오차](../outputs/node_token_mamba/training/seed_2/20b1a0a1b42141e9a1139ea43242bb6f/selected_errors.npz)

#### Mamba · 120 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-11)

60-epoch M0 고정. 90→120 추가 학습 후 개발 검증 best 선택. best epoch 119 / 115 / 106.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 1.344 mm | 0.974 mm | 2.203 mm | 1.407 mm | 1.327 mm | **1.297 mm** |
| Peg 회전 | 4D | 0.717° | 0.573° | 1.146° | **0.660°** | 0.728° | 0.764° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 2.283 mm | 1.934 mm | 3.557 mm | **2.099 mm** | 2.472 mm | 2.279 mm |
| Hole 위치 | 3D | 1.714 mm | 1.523 mm | 2.585 mm | **1.416 mm** | 1.862 mm | 1.864 mm |
| Hole 회전 | 4D | 0.987° | 0.904° | 1.490° | 1.085° | **0.862°** | 1.014° |
| Peg 반쪽 크기 X | 1D | 0.557 mm | 0.397 mm | 0.795 mm | **0.472 mm** | 0.524 mm | 0.674 mm |
| Peg 반쪽 크기 Y | 1D | 0.349 mm | 0.281 mm | 0.596 mm | **0.317 mm** | 0.346 mm | 0.382 mm |
| Peg 반쪽 크기 Z | 1D | 0.340 mm | 0.259 mm | 0.566 mm | **0.303 mm** | 0.348 mm | 0.368 mm |
| Hole 반지름 | 1D | 0.644 mm | 0.550 mm | 1.180 mm | **0.564 mm** | 0.760 mm | 0.609 mm |
| **합계** | **21D** | | | | | | |

근거: [mamba_summary_21d.json](../outputs/temporal_continuation_120/temporal_continuation_120/mamba_summary_21d.json) · [seed_0 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/mamba/seed_0/e333b69eb92f44788fe67903b8a31872/predictions.npz) · [seed_1 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/mamba/seed_1/10145c66268d4a1b8d5aa2ebdc840812/predictions.npz) · [seed_2 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/mamba/seed_2/deddfb3bbf954dd4a7857c6ff7bd7799/predictions.npz)

#### Transformer · 90 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-18)

60-epoch M0 고정. 저장된 960-frame 오차 순서를 사용하며 두 집합의 모든 집계값을 요약과 대조했습니다. best epoch 89 / 82 / 80.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 1.407 mm | 0.959 mm | 2.441 mm | **1.305 mm** | 1.446 mm | 1.470 mm |
| Peg 회전 | 4D | 0.807° | 0.616° | 1.231° | **0.730°** | 0.812° | 0.880° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 2.391 mm | 1.997 mm | 3.554 mm | **2.195 mm** | 2.234 mm | 2.745 mm |
| Hole 위치 | 3D | 1.744 mm | 1.554 mm | 2.638 mm | **1.474 mm** | 1.608 mm | 2.150 mm |
| Hole 회전 | 4D | 0.861° | 0.756° | 1.484° | 0.754° | **0.714°** | 1.113° |
| Peg 반쪽 크기 X | 1D | 0.557 mm | 0.397 mm | 0.795 mm | **0.472 mm** | 0.524 mm | 0.674 mm |
| Peg 반쪽 크기 Y | 1D | 0.349 mm | 0.281 mm | 0.596 mm | **0.317 mm** | 0.346 mm | 0.382 mm |
| Peg 반쪽 크기 Z | 1D | 0.340 mm | 0.259 mm | 0.566 mm | **0.303 mm** | 0.348 mm | 0.368 mm |
| Hole 반지름 | 1D | 0.644 mm | 0.550 mm | 1.180 mm | **0.564 mm** | 0.760 mm | 0.609 mm |
| **합계** | **21D** | | | | | | |

근거: [transformer_summary_21d.json](../outputs/temporal_continuation_90/transformer_summary_21d.json) · [transformer_0.npz](../outputs/temporal_continuation_90/seed_errors/transformer_0.npz) · [transformer_1.npz](../outputs/temporal_continuation_90/seed_errors/transformer_1.npz) · [transformer_2.npz](../outputs/temporal_continuation_90/seed_errors/transformer_2.npz)

#### Mamba · 60 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-15)

60-epoch M0 고정. 시계열 개발 검증 59부모로 best 선택; epoch 59 / 58 / 53.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 1.389 mm | 0.957 mm | 2.427 mm | 1.542 mm | 1.360 mm | **1.265 mm** |
| Peg 회전 | 4D | 0.788° | 0.606° | 1.216° | 0.801° | **0.777°** | 0.785° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 2.439 mm | 2.176 mm | 3.581 mm | 2.570 mm | 2.599 mm | **2.149 mm** |
| Hole 위치 | 3D | 1.775 mm | 1.677 mm | 2.818 mm | **1.608 mm** | 2.062 mm | 1.656 mm |
| Hole 회전 | 4D | 0.932° | 0.857° | 1.431° | 0.932° | **0.819°** | 1.045° |
| Peg 반쪽 크기 X | 1D | 0.557 mm | 0.397 mm | 0.795 mm | **0.472 mm** | 0.524 mm | 0.674 mm |
| Peg 반쪽 크기 Y | 1D | 0.349 mm | 0.281 mm | 0.596 mm | **0.317 mm** | 0.346 mm | 0.382 mm |
| Peg 반쪽 크기 Z | 1D | 0.340 mm | 0.259 mm | 0.566 mm | **0.303 mm** | 0.348 mm | 0.368 mm |
| Hole 반지름 | 1D | 0.644 mm | 0.550 mm | 1.180 mm | **0.564 mm** | 0.760 mm | 0.609 mm |
| **합계** | **21D** | | | | | | |

근거: [summary_21d.json](../outputs/selected_m0_mamba/fresh_60/summary_21d.json) · [seed_0 predictions.npz](../outputs/selected_m0_mamba/fresh_60/seed_0/predictions.npz) · [seed_1 predictions.npz](../outputs/selected_m0_mamba/fresh_60/seed_1/predictions.npz) · [seed_2 predictions.npz](../outputs/selected_m0_mamba/fresh_60/seed_2/predictions.npz)

#### Transformer · 60 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-16)

60-epoch M0 고정. best epoch 57 / 58 / 53. 로컬에는 집계 요약만 있어 seed별 평균은 —입니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 1.495 mm | 1.121 mm | 2.587 mm | — | — | — |
| Peg 회전 | 4D | 0.906° | 0.749° | 1.396° | — | — | — |
| Peg 끝점과 Hole의 상대 위치 | 3D | 2.531 mm | 2.280 mm | 3.422 mm | — | — | — |
| Hole 위치 | 3D | 1.809 mm | 1.684 mm | 2.619 mm | — | — | — |
| Hole 회전 | 4D | 0.861° | 0.742° | 1.457° | — | — | — |
| Peg 반쪽 크기 X | 1D | 0.557 mm | 0.397 mm | 0.795 mm | — | — | — |
| Peg 반쪽 크기 Y | 1D | 0.349 mm | 0.281 mm | 0.596 mm | — | — | — |
| Peg 반쪽 크기 Z | 1D | 0.340 mm | 0.259 mm | 0.566 mm | — | — | — |
| Hole 반지름 | 1D | 0.644 mm | 0.550 mm | 1.180 mm | — | — | — |
| **합계** | **21D** | | | | | | |

근거: [summary_21d.json](../outputs/selected_m0_transformer/summary_21d.json) · [source_checkpoints.json](../outputs/temporal_continuation_90/source_checkpoints.json)

#### Selected M0 · 60 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-4)

전체 검증 손실 기준 best epoch: seed 순서 3 / 5 / 50.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 3.062 mm | 2.520 mm | 5.125 mm | 4.366 mm | 3.708 mm | **1.113 mm** |
| Peg 회전 | 4D | 2.360° | 2.151° | 3.009° | 3.549° | 2.637° | **0.893°** |
| Peg 끝점과 Hole의 상대 위치 | 3D | 6.158 mm | 5.730 mm | 9.701 mm | 10.108 mm | 6.540 mm | **1.827 mm** |
| Hole 위치 | 3D | 5.244 mm | 4.837 mm | 7.722 mm | 10.236 mm | 4.236 mm | **1.260 mm** |
| Hole 회전 | 4D | 3.005° | 2.396° | 5.375° | 4.596° | 3.518° | **0.901°** |
| Peg 반쪽 크기 X | 1D | 4.797 mm | 3.954 mm | 10.088 mm | 9.586 mm | 4.159 mm | **0.645 mm** |
| Peg 반쪽 크기 Y | 1D | 1.816 mm | 1.737 mm | 2.961 mm | 2.417 mm | 2.647 mm | **0.383 mm** |
| Peg 반쪽 크기 Z | 1D | 1.819 mm | 1.828 mm | 2.769 mm | 2.520 mm | 2.568 mm | **0.370 mm** |
| Hole 반지름 | 1D | 1.773 mm | 1.790 mm | 2.878 mm | 2.431 mm | 2.252 mm | **0.637 mm** |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/selected_vertex_continuation/evaluation_best/fadf9805ef1d45f1bc6cf6f6a0542f76/summary.json) · [selected_seed_0.npz](../outputs/selected_vertex_continuation/evaluation_best/fadf9805ef1d45f1bc6cf6f6a0542f76/selected_seed_0.npz) · [selected_seed_1.npz](../outputs/selected_vertex_continuation/evaluation_best/fadf9805ef1d45f1bc6cf6f6a0542f76/selected_seed_1.npz) · [selected_seed_2.npz](../outputs/selected_vertex_continuation/evaluation_best/fadf9805ef1d45f1bc6cf6f6a0542f76/selected_seed_2.npz)

#### Selected M0 · 30 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-1)

원래 960프레임의 전체 검증 손실로 best 선택.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 3.137 mm | 2.534 mm | 5.279 mm | 4.366 mm | 3.708 mm | **1.337 mm** |
| Peg 회전 | 4D | 2.416° | 2.261° | 2.993° | 3.549° | 2.637° | **1.062°** |
| Peg 끝점과 Hole의 상대 위치 | 3D | 6.415 mm | 5.836 mm | 9.957 mm | 10.108 mm | 6.540 mm | **2.597 mm** |
| Hole 위치 | 3D | 5.502 mm | 5.031 mm | 8.129 mm | 10.236 mm | 4.236 mm | **2.033 mm** |
| Hole 회전 | 4D | 3.041° | 2.547° | 5.312° | 4.596° | 3.518° | **1.008°** |
| Peg 반쪽 크기 X | 1D | 5.011 mm | 4.114 mm | 9.691 mm | 9.586 mm | 4.159 mm | **1.287 mm** |
| Peg 반쪽 크기 Y | 1D | 1.848 mm | 1.792 mm | 3.086 mm | 2.417 mm | 2.647 mm | **0.481 mm** |
| Peg 반쪽 크기 Z | 1D | 1.847 mm | 1.824 mm | 2.813 mm | 2.520 mm | 2.568 mm | **0.453 mm** |
| Hole 반지름 | 1D | 1.782 mm | 1.710 mm | 2.909 mm | 2.431 mm | 2.252 mm | **0.664 mm** |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/summary.json) · [selected_seed_0.npz](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/selected_seed_0.npz) · [selected_seed_1.npz](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/selected_seed_1.npz) · [selected_seed_2.npz](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/selected_seed_2.npz)

#### M0 원래 개발셋 · best

[체크포인트 매핑으로 이동](#checkpoint-13)

원래 개발셋에서 학습한 기준 모델. Selected M0와 학습 데이터가 다릅니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 5.072 mm | 4.406 mm | 7.326 mm | **4.830 mm** | 4.986 mm | 5.399 mm |
| Peg 회전 | 4D | 4.354° | 4.218° | 6.306° | **3.730°** | 5.198° | 4.134° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 10.630 mm | 8.862 mm | 18.829 mm | 10.844 mm | 10.546 mm | **10.501 mm** |
| Hole 위치 | 3D | 13.018 mm | 12.980 mm | 21.253 mm | 13.805 mm | **12.294 mm** | 12.955 mm |
| Hole 회전 | 4D | 6.895° | 6.167° | 10.789° | **6.602°** | 6.635° | 7.448° |
| Peg 반쪽 크기 X | 1D | 10.913 mm | 11.214 mm | 18.378 mm | **10.280 mm** | 11.481 mm | 10.979 mm |
| Peg 반쪽 크기 Y | 1D | 2.561 mm | 2.513 mm | 4.353 mm | 2.541 mm | **2.475 mm** | 2.667 mm |
| Peg 반쪽 크기 Z | 1D | 2.583 mm | 2.480 mm | 4.126 mm | 2.645 mm | **2.391 mm** | 2.714 mm |
| Hole 반지름 | 1D | 2.510 mm | 2.607 mm | 4.124 mm | 2.520 mm | 2.591 mm | **2.418 mm** |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/summary.json) · [reference_seed_0.npz](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/reference_seed_0.npz) · [reference_seed_1.npz](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/reference_seed_1.npz) · [reference_seed_2.npz](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/reference_seed_2.npz)

#### M0 · detector Rim-only

[체크포인트 매핑으로 이동](#checkpoint-6)

검출기 Rim 후보만 BOX 입력으로 유지. 입력 필터가 다른 실험입니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 5.829 mm | 4.765 mm | 8.531 mm | 5.737 mm | 6.015 mm | **5.735 mm** |
| Peg 회전 | 4D | 7.257° | 5.973° | 9.716° | 6.858° | **6.268°** | 8.645° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 15.152 mm | 10.050 mm | 17.411 mm | 14.981 mm | 15.817 mm | **14.657 mm** |
| Hole 위치 | 3D | 18.372 mm | 14.685 mm | 24.409 mm | 18.347 mm | 18.808 mm | **17.960 mm** |
| Hole 회전 | 4D | 6.460° | 3.370° | 15.634° | 6.379° | 6.962° | **6.038°** |
| Peg 반쪽 크기 X | 1D | 12.383 mm | 11.701 mm | 23.317 mm | **11.550 mm** | 12.822 mm | 12.778 mm |
| Peg 반쪽 크기 Y | 1D | 2.550 mm | 2.640 mm | 4.335 mm | **2.396 mm** | 2.686 mm | 2.568 mm |
| Peg 반쪽 크기 Z | 1D | 2.535 mm | 2.667 mm | 4.159 mm | **2.457 mm** | 2.591 mm | 2.557 mm |
| Hole 반지름 | 1D | 2.438 mm | 2.698 mm | 4.056 mm | 2.469 mm | **2.407 mm** | 2.440 mm |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/rim_only_experiment/evaluation/ef864771775547c0bce9e1ddaf39838a/summary.json) · [filtered_seed_0.npz](../outputs/rim_only_experiment/evaluation/ef864771775547c0bce9e1ddaf39838a/filtered_seed_0.npz) · [filtered_seed_1.npz](../outputs/rim_only_experiment/evaluation/ef864771775547c0bce9e1ddaf39838a/filtered_seed_1.npz) · [filtered_seed_2.npz](../outputs/rim_only_experiment/evaluation/ef864771775547c0bce9e1ddaf39838a/filtered_seed_2.npz)

#### 초기 Mamba · seed 0만 완료

[체크포인트 매핑으로 이동](#checkpoint-19)

3-seed 순위에서 제외. 단일 seed 결과이며 나머지 seed를 추정하지 않습니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | — | — | — | 2.314 mm | — | — |
| Peg 회전 | 4D | — | — | — | 0.880° | — | — |
| Peg 끝점과 Hole의 상대 위치 | 3D | — | — | — | 4.138 mm | — | — |
| Hole 위치 | 3D | — | — | — | 3.393 mm | — | — |
| Hole 회전 | 4D | — | — | — | 3.306° | — | — |
| Peg 반쪽 크기 X | 1D | — | — | — | 0.472 mm | — | — |
| Peg 반쪽 크기 Y | 1D | — | — | — | 0.317 mm | — | — |
| Peg 반쪽 크기 Z | 1D | — | — | — | 0.303 mm | — | — |
| Hole 반지름 | 1D | — | — | — | 0.564 mm | — | — |
| **합계** | **21D** | | | | | | |

근거: [original_validation_21d.json](../outputs/selected_m0_mamba/seed_0/original_validation_21d.json)

#### GNN 조기 종료 + Mamba 조기 종료

[체크포인트 매핑으로 이동](#checkpoint-21)

물리151 검증으로 선택한 GNN을 고정. Mamba best epoch 264 / 213 / 85, 조기 종료 epoch 294 / 243 / 115. 검증 loss 기준으로 best 선택. 해당 151프레임은 GNN 선택에도 사용됐다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 1.329 mm | 0.966 mm | 2.204 mm | **1.290 mm** | 1.311 mm | 1.385 mm |
| Peg 회전 | 4D | 0.700° | 0.547° | 1.085° | **0.609°** | 0.675° | 0.817° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 2.174 mm | 1.786 mm | 3.549 mm | **2.009 mm** | 2.238 mm | 2.276 mm |
| Hole 위치 | 3D | 1.582 mm | 1.307 mm | 2.490 mm | 1.516 mm | **1.491 mm** | 1.739 mm |
| Hole 회전 | 4D | 0.898° | 0.788° | 1.425° | **0.819°** | 0.840° | 1.036° |
| Peg 반쪽 크기 X | 1D | 0.507 mm | 0.357 mm | 0.718 mm | **0.438 mm** | 0.504 mm | 0.579 mm |
| Peg 반쪽 크기 Y | 1D | 0.333 mm | 0.266 mm | 0.566 mm | **0.315 mm** | 0.326 mm | 0.358 mm |
| Peg 반쪽 크기 Z | 1D | 0.321 mm | 0.268 mm | 0.530 mm | **0.290 mm** | 0.332 mm | 0.342 mm |
| Hole 반지름 | 1D | 0.615 mm | 0.511 mm | 1.127 mm | **0.538 mm** | 0.716 mm | 0.590 mm |
| **합계** | **21D** | | | | | | |

근거: [3-seed 집계](../outputs/temporal_early_stopping/mamba_summary_21d.json) · [seed 0 예측](../outputs/temporal_early_stopping/evaluation/mamba/seed_0/183b8a11f3cd40669bc13a44a2fff2f2/predictions.npz) · [seed 1 예측](../outputs/temporal_early_stopping/evaluation/mamba/seed_1/29c516ee7cd344cd86c1f9827f2bac81/predictions.npz) · [seed 2 예측](../outputs/temporal_early_stopping/evaluation/mamba/seed_2/48b201634afc4967b49c14a5818d4b93/predictions.npz)


#### GNN 조기 종료 + Transformer 조기 종료

[체크포인트 매핑으로 이동](#checkpoint-22)

물리151 검증으로 선택한 GNN을 고정. Transformer best epoch 165 / 151 / 183, 조기 종료 epoch 195 / 181 / 213. 검증 loss 기준으로 best 선택. 해당 151프레임은 GNN 선택에도 사용됐다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 1.362 mm | 0.923 mm | 2.337 mm | 1.326 mm | 1.465 mm | **1.294 mm** |
| Peg 회전 | 4D | 0.783° | 0.589° | 1.325° | **0.646°** | 0.881° | 0.822° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 2.266 mm | 1.914 mm | 3.354 mm | **2.036 mm** | 2.574 mm | 2.188 mm |
| Hole 위치 | 3D | 1.473 mm | 1.276 mm | 2.275 mm | **1.302 mm** | 1.386 mm | 1.731 mm |
| Hole 회전 | 4D | 0.792° | 0.712° | 1.335° | 0.717° | **0.708°** | 0.951° |
| Peg 반쪽 크기 X | 1D | 0.507 mm | 0.357 mm | 0.718 mm | **0.438 mm** | 0.504 mm | 0.579 mm |
| Peg 반쪽 크기 Y | 1D | 0.333 mm | 0.266 mm | 0.566 mm | **0.315 mm** | 0.326 mm | 0.358 mm |
| Peg 반쪽 크기 Z | 1D | 0.321 mm | 0.268 mm | 0.530 mm | **0.290 mm** | 0.332 mm | 0.342 mm |
| Hole 반지름 | 1D | 0.615 mm | 0.511 mm | 1.127 mm | **0.538 mm** | 0.716 mm | 0.590 mm |
| **합계** | **21D** | | | | | | |

근거: [3-seed 집계](../outputs/temporal_early_stopping/transformer_summary_21d.json) · [seed 0 예측](../outputs/temporal_early_stopping/evaluation/transformer/seed_0/9058a3586f0445dfafec0b93e9bc5875/predictions.npz) · [seed 1 예측](../outputs/temporal_early_stopping/evaluation/transformer/seed_1/69c8a27f7d014b6db300e2cd9c68c7dd/predictions.npz) · [seed 2 예측](../outputs/temporal_early_stopping/evaluation/transformer/seed_2/ec956145e49b491cba71702ff1df0c01/predictions.npz)

#### GNN 조기 종료 + 노드 토큰 Mamba 조기 종료

[체크포인트 매핑으로 이동](#checkpoint-23)

물리151 검증으로 선택한 GNN best를 seed별로 고정. 노드 토큰 Mamba best epoch 97 / 285 / 198, 조기 종료 epoch 127 / 315 / 228. 검증 loss 기준 best 선택. 151프레임은 GNN 선택에도 사용됐다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 1.288 mm | 0.933 mm | 2.175 mm | 1.302 mm | **1.267 mm** | 1.296 mm |
| Peg 회전 | 4D | 0.687° | 0.524° | 1.115° | **0.667°** | 0.690° | 0.704° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 2.181 mm | 1.934 mm | 3.411 mm | 2.290 mm | **2.013 mm** | 2.241 mm |
| Hole 위치 | 3D | 1.592 mm | 1.478 mm | 2.422 mm | 1.680 mm | **1.391 mm** | 1.706 mm |
| Hole 회전 | 4D | 0.782° | 0.747° | 1.096° | 0.818° | **0.663°** | 0.866° |
| Peg 반쪽 크기 X | 1D | 0.507 mm | 0.357 mm | 0.718 mm | **0.438 mm** | 0.504 mm | 0.579 mm |
| Peg 반쪽 크기 Y | 1D | 0.333 mm | 0.266 mm | 0.566 mm | **0.315 mm** | 0.326 mm | 0.358 mm |
| Peg 반쪽 크기 Z | 1D | 0.321 mm | 0.268 mm | 0.530 mm | **0.290 mm** | 0.332 mm | 0.342 mm |
| Hole 반지름 | 1D | 0.615 mm | 0.511 mm | 1.127 mm | **0.538 mm** | 0.716 mm | 0.590 mm |
| **합계** | **21D** | | | | | | |

근거: [3-seed 비교](../outputs/node_token_early_stopping/comparison/785a048f899144e086fb48ed25b79b14/comparison.json) · [평가 표본](../outputs/node_token_mamba/validation_frames.jsonl) · [seed 0 프레임별 오차](../outputs/node_token_early_stopping/training/seed_0/6d16da7ceac54470ad6bc4f7858fef62/selected_errors.npz) · [seed 1 프레임별 오차](../outputs/node_token_early_stopping/training/seed_1/70ace57f59f54409bc75d140268036f1/selected_errors.npz) · [seed 2 프레임별 오차](../outputs/node_token_early_stopping/training/seed_2/ab6c964532824f13b9acc382b5901fb5/selected_errors.npz)


### 기존 검증 에피소드 전체 — 선택 M0·시계열 계열 · 4,102프레임

#### Mamba · 120 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-11)

완전한 에피소드 causal 재생. 고정 M0의 크기·반지름을 그대로 사용.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 9.149 mm | 10.162 mm | 18.314 mm | 9.023 mm | 10.719 mm | **7.705 mm** |
| Peg 회전 | 4D | 1.141° | 0.988° | 1.896° | 1.249° | **1.064°** | 1.109° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 15.191 mm | 15.183 mm | 26.977 mm | 17.455 mm | **13.948 mm** | 14.170 mm |
| Hole 위치 | 3D | 14.828 mm | 13.327 mm | 30.654 mm | 14.928 mm | 16.588 mm | **12.966 mm** |
| Hole 회전 | 4D | 1.823° | 1.655° | 2.974° | 1.966° | **1.729°** | 1.775° |
| Peg 반쪽 크기 X | 1D | 30.123 mm | 39.405 mm | 65.788 mm | 32.095 mm | 32.494 mm | **25.780 mm** |
| Peg 반쪽 크기 Y | 1D | 2.315 mm | 1.899 mm | 5.001 mm | 2.618 mm | 2.792 mm | **1.536 mm** |
| Peg 반쪽 크기 Z | 1D | 2.281 mm | 1.626 mm | 5.033 mm | 2.961 mm | 2.235 mm | **1.649 mm** |
| Hole 반지름 | 1D | 8.084 mm | 8.362 mm | 14.392 mm | **7.270 mm** | 7.691 mm | 9.291 mm |
| **합계** | **21D** | | | | | | |

근거: [summary_21d.json](../outputs/temporal_continuation_120/summary_21d.json) · [seed_0 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/mamba/seed_0/e333b69eb92f44788fe67903b8a31872/predictions.npz) · [seed_1 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/mamba/seed_1/10145c66268d4a1b8d5aa2ebdc840812/predictions.npz) · [seed_2 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/mamba/seed_2/deddfb3bbf954dd4a7857c6ff7bd7799/predictions.npz)

#### Transformer · 120 epoch 한도 best

[체크포인트 매핑으로 이동](#checkpoint-12)

완전한 에피소드 causal 재생. 고정 M0의 크기·반지름을 그대로 사용.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 11.740 mm | 12.834 mm | 22.921 mm | 12.984 mm | 13.303 mm | **8.934 mm** |
| Peg 회전 | 4D | 2.136° | 1.588° | 4.396° | **2.040°** | 2.295° | 2.073° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 19.164 mm | 17.072 mm | 40.425 mm | **17.084 mm** | 22.691 mm | 17.716 mm |
| Hole 위치 | 3D | 20.663 mm | 20.722 mm | 39.738 mm | 24.065 mm | 20.573 mm | **17.350 mm** |
| Hole 회전 | 4D | 3.557° | 2.403° | 8.236° | 4.713° | 3.294° | **2.664°** |
| Peg 반쪽 크기 X | 1D | 30.123 mm | 39.405 mm | 65.788 mm | 32.095 mm | 32.494 mm | **25.780 mm** |
| Peg 반쪽 크기 Y | 1D | 2.315 mm | 1.899 mm | 5.001 mm | 2.618 mm | 2.792 mm | **1.536 mm** |
| Peg 반쪽 크기 Z | 1D | 2.281 mm | 1.626 mm | 5.033 mm | 2.961 mm | 2.235 mm | **1.649 mm** |
| Hole 반지름 | 1D | 8.084 mm | 8.362 mm | 14.392 mm | **7.270 mm** | 7.691 mm | 9.291 mm |
| **합계** | **21D** | | | | | | |

근거: [summary_21d.json](../outputs/temporal_continuation_120/summary_21d.json) · [seed_0 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/transformer/seed_0/299e0511de81487faf51d9393f982de2/predictions.npz) · [seed_1 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/transformer/seed_1/a5530784df2a4a7b84a292a9395880b1/predictions.npz) · [seed_2 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/transformer/seed_2/780027874899493d8a3391eeed93fee8/predictions.npz)

#### GNN · 물리151 검증 조기 종료

[체크포인트 매핑으로 이동](#checkpoint-5)

물리151로 선택한 체크포인트를 전체 에피소드에 적용.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 44.965 mm | 63.082 mm | 92.657 mm | 43.168 mm | 48.668 mm | **43.060 mm** |
| Peg 회전 | 4D | 17.956° | 14.340° | 46.547° | 20.325° | 17.018° | **16.526°** |
| Peg 끝점과 Hole의 상대 위치 | 3D | 118.374 mm | 107.259 mm | 273.266 mm | 118.503 mm | 124.687 mm | **111.932 mm** |
| Hole 위치 | 3D | 132.023 mm | 117.717 mm | 336.085 mm | 141.880 mm | 131.881 mm | **122.307 mm** |
| Hole 회전 | 4D | 43.467° | 52.181° | 86.861° | 67.289° | 35.443° | **27.670°** |
| Peg 반쪽 크기 X | 1D | 30.007 mm | 39.250 mm | 65.615 mm | 32.263 mm | 31.838 mm | **25.920 mm** |
| Peg 반쪽 크기 Y | 1D | 2.269 mm | 1.724 mm | 4.912 mm | 2.680 mm | 2.656 mm | **1.471 mm** |
| Peg 반쪽 크기 Z | 1D | 2.244 mm | 1.639 mm | 4.941 mm | 3.041 mm | 2.117 mm | **1.573 mm** |
| Hole 반지름 | 1D | 7.992 mm | 8.172 mm | 14.269 mm | **7.089 mm** | 7.642 mm | 9.246 mm |
| **합계** | **21D** | | | | | | |

근거: [summary_21d.json](../outputs/temporal_continuation_120/summary_21d.json) · [gnn_seed_0.npz](../outputs/temporal_continuation_120/report_data/gnn_seed_0.npz) · [gnn_seed_1.npz](../outputs/temporal_continuation_120/report_data/gnn_seed_1.npz) · [gnn_seed_2.npz](../outputs/temporal_continuation_120/report_data/gnn_seed_2.npz)

#### Selected M0 · epoch 60 last

[체크포인트 매핑으로 이동](#checkpoint-3)

시계열 모델과 동일 프레임에서 평가한 고정 입력 M0.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 45.183 mm | 63.883 mm | 92.851 mm | **42.897 mm** | 49.308 mm | 43.344 mm |
| Peg 회전 | 4D | 17.949° | 14.082° | 46.263° | 20.348° | 17.307° | **16.193°** |
| Peg 끝점과 Hole의 상대 위치 | 3D | 118.474 mm | 107.353 mm | 272.927 mm | 118.626 mm | 124.846 mm | **111.949 mm** |
| Hole 위치 | 3D | 132.317 mm | 117.797 mm | 336.086 mm | 142.532 mm | 131.938 mm | **122.480 mm** |
| Hole 회전 | 4D | 43.593° | 52.288° | 86.410° | 68.185° | 34.982° | **27.611°** |
| Peg 반쪽 크기 X | 1D | 30.123 mm | 39.405 mm | 65.788 mm | 32.095 mm | 32.494 mm | **25.780 mm** |
| Peg 반쪽 크기 Y | 1D | 2.315 mm | 1.899 mm | 5.001 mm | 2.618 mm | 2.792 mm | **1.536 mm** |
| Peg 반쪽 크기 Z | 1D | 2.281 mm | 1.626 mm | 5.033 mm | 2.961 mm | 2.235 mm | **1.649 mm** |
| Hole 반지름 | 1D | 8.084 mm | 8.362 mm | 14.392 mm | **7.270 mm** | 7.691 mm | 9.291 mm |
| **합계** | **21D** | | | | | | |

근거: [summary_21d.json](../outputs/temporal_continuation_120/summary_21d.json) · [seed_0 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/mamba/seed_0/e333b69eb92f44788fe67903b8a31872/predictions.npz) · [seed_1 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/mamba/seed_1/10145c66268d4a1b8d5aa2ebdc840812/predictions.npz) · [seed_2 predictions.npz](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/mamba/seed_2/deddfb3bbf954dd4a7857c6ff7bd7799/predictions.npz)

#### GNN 조기 종료 + Mamba 조기 종료

[체크포인트 매핑으로 이동](#checkpoint-21)

물리151 검증으로 선택한 GNN을 고정. Mamba best epoch 264 / 213 / 85, 조기 종료 epoch 294 / 243 / 115. 검증 loss 기준으로 best 선택. 해당 151프레임은 GNN 선택에도 사용됐다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 9.671 mm | 11.245 mm | 19.003 mm | 10.641 mm | 10.428 mm | **7.943 mm** |
| Peg 회전 | 4D | 0.979° | 0.842° | 1.625° | **0.804°** | 0.958° | 1.176° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 13.979 mm | 13.967 mm | 25.946 mm | 14.249 mm | **12.795 mm** | 14.891 mm |
| Hole 위치 | 3D | 14.922 mm | 13.888 mm | 30.538 mm | 15.294 mm | 16.447 mm | **13.025 mm** |
| Hole 회전 | 4D | 1.706° | 1.506° | 2.851° | 1.633° | **1.535°** | 1.950° |
| Peg 반쪽 크기 X | 1D | 30.007 mm | 39.250 mm | 65.615 mm | 32.263 mm | 31.838 mm | **25.920 mm** |
| Peg 반쪽 크기 Y | 1D | 2.269 mm | 1.724 mm | 4.912 mm | 2.680 mm | 2.656 mm | **1.471 mm** |
| Peg 반쪽 크기 Z | 1D | 2.244 mm | 1.639 mm | 4.941 mm | 3.041 mm | 2.117 mm | **1.573 mm** |
| Hole 반지름 | 1D | 7.992 mm | 8.172 mm | 14.269 mm | **7.089 mm** | 7.642 mm | 9.246 mm |
| **합계** | **21D** | | | | | | |

근거: [3-seed 집계](../outputs/temporal_early_stopping/full_episode_summary_21d.json) · [seed 0 예측](../outputs/temporal_early_stopping/evaluation/mamba/seed_0/183b8a11f3cd40669bc13a44a2fff2f2/predictions.npz) · [seed 1 예측](../outputs/temporal_early_stopping/evaluation/mamba/seed_1/29c516ee7cd344cd86c1f9827f2bac81/predictions.npz) · [seed 2 예측](../outputs/temporal_early_stopping/evaluation/mamba/seed_2/48b201634afc4967b49c14a5818d4b93/predictions.npz)


#### GNN 조기 종료 + Transformer 조기 종료

[체크포인트 매핑으로 이동](#checkpoint-22)

물리151 검증으로 선택한 GNN을 고정. Transformer best epoch 165 / 151 / 183, 조기 종료 epoch 195 / 181 / 213. 검증 loss 기준으로 best 선택. 해당 151프레임은 GNN 선택에도 사용됐다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 10.840 mm | 12.559 mm | 20.162 mm | 11.440 mm | 12.132 mm | **8.948 mm** |
| Peg 회전 | 4D | 2.008° | 1.675° | 3.853° | **1.886°** | 2.063° | 2.075° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 17.954 mm | 17.275 mm | 34.423 mm | 17.623 mm | 20.086 mm | **16.154 mm** |
| Hole 위치 | 3D | 18.350 mm | 18.063 mm | 35.797 mm | 20.546 mm | 19.517 mm | **14.987 mm** |
| Hole 회전 | 4D | 3.216° | 2.144° | 7.517° | 4.336° | 2.892° | **2.418°** |
| Peg 반쪽 크기 X | 1D | 30.007 mm | 39.250 mm | 65.615 mm | 32.263 mm | 31.838 mm | **25.920 mm** |
| Peg 반쪽 크기 Y | 1D | 2.269 mm | 1.724 mm | 4.912 mm | 2.680 mm | 2.656 mm | **1.471 mm** |
| Peg 반쪽 크기 Z | 1D | 2.244 mm | 1.639 mm | 4.941 mm | 3.041 mm | 2.117 mm | **1.573 mm** |
| Hole 반지름 | 1D | 7.992 mm | 8.172 mm | 14.269 mm | **7.089 mm** | 7.642 mm | 9.246 mm |
| **합계** | **21D** | | | | | | |

근거: [3-seed 집계](../outputs/temporal_early_stopping/full_episode_summary_21d.json) · [seed 0 예측](../outputs/temporal_early_stopping/evaluation/transformer/seed_0/9058a3586f0445dfafec0b93e9bc5875/predictions.npz) · [seed 1 예측](../outputs/temporal_early_stopping/evaluation/transformer/seed_1/69c8a27f7d014b6db300e2cd9c68c7dd/predictions.npz) · [seed 2 예측](../outputs/temporal_early_stopping/evaluation/transformer/seed_2/ec956145e49b491cba71702ff1df0c01/predictions.npz)

### 자동 코너 대응 + GNN · 20 epoch · 검증 전체 4,102프레임

#### 자동 대응 Transformer + GNN

[체크포인트 매핑](#checkpoint-detected-transformer). GNN best epoch 20 / 16 / 20.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 2.028 mm | 1.570 mm | 3.755 mm | **1.894 mm** | 2.132 mm | 2.058 mm |
| Peg 회전 | 4D | 1.106° | 0.942° | 1.766° | 1.088° | **1.083°** | 1.147° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 3.461 mm | 2.990 mm | 5.694 mm | 3.415 mm | **3.382 mm** | 3.585 mm |
| Hole 위치 | 3D | 2.304 mm | 1.910 mm | 4.034 mm | **2.158 mm** | 2.460 mm | 2.294 mm |
| Hole 회전 | 4D | 1.561° | 1.339° | 2.719° | **0.915°** | 1.815° | 1.952° |
| Peg 반쪽 크기 X | 1D | 1.485 mm | 1.168 mm | 2.758 mm | 1.528 mm | **1.341 mm** | 1.587 mm |
| Peg 반쪽 크기 Y | 1D | 0.768 mm | 0.657 mm | 1.360 mm | **0.715 mm** | 0.729 mm | 0.858 mm |
| Peg 반쪽 크기 Z | 1D | 0.756 mm | 0.643 mm | 1.339 mm | 0.717 mm | **0.700 mm** | 0.852 mm |
| Hole 반지름 | 1D | 1.799 mm | 1.814 mm | 3.071 mm | **0.922 mm** | 2.390 mm | 2.085 mm |
| **합계** | **21D** | | | | | | |

근거: [3-seed 비교](../outputs/detected_corner_gnn_training/comparison/70e8de0c5baa480ead2deacfd4bd95c4/comparison.json) · [seed 0 예측](../outputs/detected_corner_gnn_training/comparison/70e8de0c5baa480ead2deacfd4bd95c4/transformer_seed_0_predictions.npz) · [seed 1 예측](../outputs/detected_corner_gnn_training/comparison/70e8de0c5baa480ead2deacfd4bd95c4/transformer_seed_1_predictions.npz) · [seed 2 예측](../outputs/detected_corner_gnn_training/comparison/70e8de0c5baa480ead2deacfd4bd95c4/transformer_seed_2_predictions.npz)

#### 자동 대응 Mamba + GNN

[체크포인트 매핑](#checkpoint-detected-mamba). GNN best epoch 20 / 19 / 20.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 2.376 mm | 1.656 mm | 4.741 mm | 2.435 mm | **2.129 mm** | 2.565 mm |
| Peg 회전 | 4D | 1.252° | 1.006° | 1.968° | **1.166°** | 1.184° | 1.407° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 3.766 mm | 3.169 mm | 6.143 mm | 3.767 mm | **3.333 mm** | 4.198 mm |
| Hole 위치 | 3D | 2.549 mm | 2.250 mm | 4.104 mm | **2.372 mm** | 2.436 mm | 2.839 mm |
| Hole 회전 | 4D | 1.454° | 1.293° | 2.461° | **1.051°** | 1.587° | 1.725° |
| Peg 반쪽 크기 X | 1D | 1.660 mm | 1.286 mm | 3.328 mm | **1.525 mm** | 1.532 mm | 1.923 mm |
| Peg 반쪽 크기 Y | 1D | 0.942 mm | 0.758 mm | 1.911 mm | 1.046 mm | **0.713 mm** | 1.068 mm |
| Peg 반쪽 크기 Z | 1D | 0.940 mm | 0.752 mm | 1.911 mm | 1.038 mm | **0.730 mm** | 1.051 mm |
| Hole 반지름 | 1D | 1.820 mm | 1.775 mm | 3.473 mm | **1.001 mm** | 2.287 mm | 2.173 mm |
| **합계** | **21D** | | | | | | |

근거: [3-seed 비교](../outputs/detected_corner_gnn_training/comparison/70e8de0c5baa480ead2deacfd4bd95c4/comparison.json) · [seed 0 예측](../outputs/detected_corner_gnn_training/comparison/70e8de0c5baa480ead2deacfd4bd95c4/mamba_seed_0_predictions.npz) · [seed 1 예측](../outputs/detected_corner_gnn_training/comparison/70e8de0c5baa480ead2deacfd4bd95c4/mamba_seed_1_predictions.npz) · [seed 2 예측](../outputs/detected_corner_gnn_training/comparison/70e8de0c5baa480ead2deacfd4bd95c4/mamba_seed_2_predictions.npz)


### 전체 프레임 학습 vertex-family 비교 — 검증 전체 · 4,102프레임

#### 용량 통제

[체크포인트 매핑으로 이동](#checkpoint-10)

학습 590부모 / 60,600프레임, 검증 40부모. best epoch 12 / 6 / 12. 동일 cohort의 total validation loss로 best 선택.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 4.518 mm | 3.076 mm | 9.882 mm | 4.353 mm | 4.900 mm | **4.300 mm** |
| Peg 회전 | 4D | 2.612° | 1.838° | 4.937° | **2.163°** | 3.093° | 2.581° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 24.534 mm | 12.562 mm | 57.701 mm | 24.513 mm | 24.835 mm | **24.255 mm** |
| Hole 위치 | 3D | 24.111 mm | 12.266 mm | 59.816 mm | 24.260 mm | 24.168 mm | **23.906 mm** |
| Hole 회전 | 4D | 5.580° | 3.192° | 15.598° | 5.606° | 5.685° | **5.448°** |
| Peg 반쪽 크기 X | 1D | 5.168 mm | 3.540 mm | 11.996 mm | 5.221 mm | 5.422 mm | **4.859 mm** |
| Peg 반쪽 크기 Y | 1D | 0.838 mm | 0.707 mm | 1.528 mm | 0.837 mm | 0.966 mm | **0.710 mm** |
| Peg 반쪽 크기 Z | 1D | 0.829 mm | 0.692 mm | 1.506 mm | 0.835 mm | 0.938 mm | **0.714 mm** |
| Hole 반지름 | 1D | 2.372 mm | 2.542 mm | 4.226 mm | 2.421 mm | 2.370 mm | **2.324 mm** |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/summary.json) · [type_control_seed_0_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/type_control_seed_0_natural.npz) · [type_control_seed_1_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/type_control_seed_1_natural.npz) · [type_control_seed_2_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/type_control_seed_2_natural.npz)

#### M2 Oracle family · GT 입력 참고

[체크포인트 매핑으로 이동](#checkpoint-9)

학습 590부모 / 60,600프레임, 검증 40부모. best epoch 15 / 12 / 13. GT-certified family token을 입력하므로 deployable 모델과 정보 조건이 다릅니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 4.092 mm | 2.564 mm | 9.170 mm | 4.164 mm | 4.062 mm | **4.051 mm** |
| Peg 회전 | 4D | 2.418° | 1.641° | 4.317° | **2.270°** | 2.511° | 2.474° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 24.419 mm | 12.153 mm | 63.096 mm | 24.453 mm | **24.177 mm** | 24.628 mm |
| Hole 위치 | 3D | 24.308 mm | 11.534 mm | 64.376 mm | 24.397 mm | **24.139 mm** | 24.388 mm |
| Hole 회전 | 4D | 5.834° | 3.520° | 15.660° | 5.860° | 5.958° | **5.685°** |
| Peg 반쪽 크기 X | 1D | 5.297 mm | 3.020 mm | 12.288 mm | **5.175 mm** | 5.222 mm | 5.494 mm |
| Peg 반쪽 크기 Y | 1D | 0.921 mm | 0.763 mm | 1.724 mm | 0.994 mm | 0.979 mm | **0.790 mm** |
| Peg 반쪽 크기 Z | 1D | 0.924 mm | 0.764 mm | 1.734 mm | 1.006 mm | 0.978 mm | **0.787 mm** |
| Hole 반지름 | 1D | 2.379 mm | 2.507 mm | 4.288 mm | 2.368 mm | 2.422 mm | **2.346 mm** |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/summary.json) · [oracle_seed_0_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/oracle_seed_0_natural.npz) · [oracle_seed_1_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/oracle_seed_1_natural.npz) · [oracle_seed_2_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/oracle_seed_2_natural.npz)

#### M0 XYZ/type

[체크포인트 매핑으로 이동](#checkpoint-7)

학습 590부모 / 60,600프레임, 검증 40부모. best epoch 14 / 6 / 15. 동일 cohort의 total validation loss로 best 선택.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 4.679 mm | 3.281 mm | 9.445 mm | **4.112 mm** | 5.266 mm | 4.659 mm |
| Peg 회전 | 4D | 2.706° | 2.037° | 4.484° | **2.494°** | 3.030° | 2.594° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 24.625 mm | 12.778 mm | 58.001 mm | **24.466 mm** | 24.815 mm | 24.594 mm |
| Hole 위치 | 3D | 24.468 mm | 12.741 mm | 59.743 mm | 24.347 mm | 24.783 mm | **24.275 mm** |
| Hole 회전 | 4D | 5.843° | 3.517° | 15.758° | 5.960° | **5.651°** | 5.918° |
| Peg 반쪽 크기 X | 1D | 5.472 mm | 3.730 mm | 12.161 mm | 5.428 mm | 5.997 mm | **4.992 mm** |
| Peg 반쪽 크기 Y | 1D | 0.910 mm | 0.746 mm | 1.660 mm | **0.659 mm** | 1.066 mm | 1.006 mm |
| Peg 반쪽 크기 Z | 1D | 0.917 mm | 0.750 mm | 1.658 mm | **0.659 mm** | 1.081 mm | 1.009 mm |
| Hole 반지름 | 1D | 2.359 mm | 2.509 mm | 4.306 mm | 2.361 mm | **2.342 mm** | 2.375 mm |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/summary.json) · [baseline_seed_0_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/baseline_seed_0_natural.npz) · [baseline_seed_1_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/baseline_seed_1_natural.npz) · [baseline_seed_2_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/baseline_seed_2_natural.npz)

#### M1 Detector family

[체크포인트 매핑으로 이동](#checkpoint-8)

학습 590부모 / 60,600프레임, 검증 40부모. best epoch 16 / 11 / 14. 동일 cohort의 total validation loss로 best 선택.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 4.304 mm | 2.881 mm | 8.959 mm | **4.257 mm** | 4.279 mm | 4.376 mm |
| Peg 회전 | 4D | 2.451° | 1.786° | 3.986° | **2.333°** | 2.540° | 2.481° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 24.846 mm | 12.129 mm | 60.458 mm | **24.429 mm** | 25.170 mm | 24.938 mm |
| Hole 위치 | 3D | 24.551 mm | 11.682 mm | 60.408 mm | **24.265 mm** | 24.989 mm | 24.398 mm |
| Hole 회전 | 4D | 5.505° | 3.297° | 15.543° | 5.521° | **5.425°** | 5.568° |
| Peg 반쪽 크기 X | 1D | 5.491 mm | 3.181 mm | 13.410 mm | **5.256 mm** | 5.846 mm | 5.371 mm |
| Peg 반쪽 크기 Y | 1D | 0.877 mm | 0.760 mm | 1.490 mm | 0.809 mm | **0.765 mm** | 1.057 mm |
| Peg 반쪽 크기 Z | 1D | 0.880 mm | 0.764 mm | 1.497 mm | 0.812 mm | **0.762 mm** | 1.066 mm |
| Hole 반지름 | 1D | 2.371 mm | 2.518 mm | 4.227 mm | **2.355 mm** | 2.378 mm | 2.380 mm |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/summary.json) · [detector_seed_0_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/detector_seed_0_natural.npz) · [detector_seed_1_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/detector_seed_1_natural.npz) · [detector_seed_2_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/detector_seed_2_natural.npz)

### 전체 프레임 학습 vertex-family 비교 — Target-present · 3,645프레임

#### 용량 통제

[체크포인트 매핑으로 이동](#checkpoint-10)

학습 590부모 / 60,600프레임, 검증 40부모. best epoch 12 / 6 / 12. 동일 cohort의 total validation loss로 best 선택.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 3.994 mm | 2.833 mm | 6.986 mm | 3.793 mm | 4.412 mm | **3.778 mm** |
| Peg 회전 | 4D | 2.001° | 1.749° | 3.312° | **1.549°** | 2.641° | 1.811° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 18.703 mm | 9.996 mm | 43.837 mm | 18.647 mm | 19.049 mm | **18.414 mm** |
| Hole 위치 | 3D | 18.434 mm | 9.653 mm | 44.036 mm | 18.627 mm | 18.517 mm | **18.159 mm** |
| Hole 회전 | 4D | 4.813° | 2.898° | 11.678° | 4.820° | 4.967° | **4.654°** |
| Peg 반쪽 크기 X | 1D | 4.216 mm | 3.161 mm | 8.491 mm | 4.252 mm | 4.676 mm | **3.720 mm** |
| Peg 반쪽 크기 Y | 1D | 0.834 mm | 0.717 mm | 1.523 mm | 0.854 mm | 0.954 mm | **0.694 mm** |
| Peg 반쪽 크기 Z | 1D | 0.823 mm | 0.707 mm | 1.497 mm | 0.851 mm | 0.926 mm | **0.694 mm** |
| Hole 반지름 | 1D | 2.392 mm | 2.541 mm | 4.213 mm | 2.447 mm | 2.387 mm | **2.343 mm** |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/summary.json) · [type_control_seed_0_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/type_control_seed_0_natural.npz) · [type_control_seed_1_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/type_control_seed_1_natural.npz) · [type_control_seed_2_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/type_control_seed_2_natural.npz)

#### M2 Oracle family · GT 입력 참고

[체크포인트 매핑으로 이동](#checkpoint-9)

학습 590부모 / 60,600프레임, 검증 40부모. best epoch 15 / 12 / 13. GT-certified family token을 입력하므로 deployable 모델과 정보 조건이 다릅니다.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 3.605 mm | 2.424 mm | 7.249 mm | 3.739 mm | 3.607 mm | **3.470 mm** |
| Peg 회전 | 4D | 1.759° | 1.529° | 2.741° | **1.567°** | 1.915° | 1.795° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 18.624 mm | 9.564 mm | 43.889 mm | 18.895 mm | **18.226 mm** | 18.752 mm |
| Hole 위치 | 3D | 18.651 mm | 9.373 mm | 44.721 mm | 18.740 mm | **18.443 mm** | 18.771 mm |
| Hole 회전 | 4D | 5.098° | 3.307° | 11.180° | 5.169° | 5.208° | **4.918°** |
| Peg 반쪽 크기 X | 1D | 3.968 mm | 2.777 mm | 9.121 mm | **3.747 mm** | 3.928 mm | 4.229 mm |
| Peg 반쪽 크기 Y | 1D | 0.915 mm | 0.765 mm | 1.726 mm | 0.962 mm | 0.982 mm | **0.801 mm** |
| Peg 반쪽 크기 Z | 1D | 0.918 mm | 0.764 mm | 1.735 mm | 0.976 mm | 0.978 mm | **0.799 mm** |
| Hole 반지름 | 1D | 2.407 mm | 2.527 mm | 4.290 mm | 2.397 mm | 2.458 mm | **2.367 mm** |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/summary.json) · [oracle_seed_0_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/oracle_seed_0_natural.npz) · [oracle_seed_1_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/oracle_seed_1_natural.npz) · [oracle_seed_2_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/oracle_seed_2_natural.npz)

#### M0 XYZ/type

[체크포인트 매핑으로 이동](#checkpoint-7)

학습 590부모 / 60,600프레임, 검증 40부모. best epoch 14 / 6 / 15. 동일 cohort의 total validation loss로 best 선택.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 4.178 mm | 3.039 mm | 7.162 mm | **3.520 mm** | 4.852 mm | 4.161 mm |
| Peg 회전 | 4D | 2.107° | 1.895° | 3.356° | **1.739°** | 2.570° | 2.013° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 18.703 mm | 10.345 mm | 44.312 mm | 18.534 mm | 19.057 mm | **18.519 mm** |
| Hole 위치 | 3D | 18.836 mm | 10.620 mm | 45.701 mm | 18.701 mm | 19.204 mm | **18.604 mm** |
| Hole 회전 | 4D | 5.105° | 3.231° | 11.701° | 5.207° | **4.943°** | 5.166° |
| Peg 반쪽 크기 X | 1D | 4.334 mm | 3.379 mm | 8.594 mm | 4.116 mm | 5.240 mm | **3.646 mm** |
| Peg 반쪽 크기 Y | 1D | 0.932 mm | 0.774 mm | 1.686 mm | **0.677 mm** | 1.096 mm | 1.022 mm |
| Peg 반쪽 크기 Z | 1D | 0.937 mm | 0.773 mm | 1.667 mm | **0.675 mm** | 1.111 mm | 1.024 mm |
| Hole 반지름 | 1D | 2.381 mm | 2.496 mm | 4.302 mm | 2.386 mm | **2.356 mm** | 2.401 mm |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/summary.json) · [baseline_seed_0_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/baseline_seed_0_natural.npz) · [baseline_seed_1_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/baseline_seed_1_natural.npz) · [baseline_seed_2_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/baseline_seed_2_natural.npz)

#### M1 Detector family

[체크포인트 매핑으로 이동](#checkpoint-8)

학습 590부모 / 60,600프레임, 검증 40부모. best epoch 16 / 11 / 14. 동일 cohort의 total validation loss로 best 선택.

| 예측 항목 | 차원 | 3-seed 평균 | 중앙값 | 90백분위 | Seed 0 평균 | Seed 1 평균 | Seed 2 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Peg 위치 | 3D | 3.802 mm | 2.733 mm | 6.922 mm | **3.721 mm** | 3.787 mm | 3.899 mm |
| Peg 회전 | 4D | 1.820° | 1.645° | 2.819° | **1.731°** | 1.885° | 1.842° |
| Peg 끝점과 Hole의 상대 위치 | 3D | 18.943 mm | 10.034 mm | 45.145 mm | **18.472 mm** | 19.466 mm | 18.892 mm |
| Hole 위치 | 3D | 18.930 mm | 9.626 mm | 44.561 mm | **18.602 mm** | 19.410 mm | 18.778 mm |
| Hole 회전 | 4D | 4.726° | 3.063° | 10.853° | 4.714° | **4.638°** | 4.825° |
| Peg 반쪽 크기 X | 1D | 4.021 mm | 2.857 mm | 8.443 mm | **3.816 mm** | 4.408 mm | 3.840 mm |
| Peg 반쪽 크기 Y | 1D | 0.870 mm | 0.771 mm | 1.474 mm | 0.805 mm | **0.771 mm** | 1.033 mm |
| Peg 반쪽 크기 Z | 1D | 0.875 mm | 0.773 mm | 1.470 mm | 0.813 mm | **0.769 mm** | 1.042 mm |
| Hole 반지름 | 1D | 2.395 mm | 2.488 mm | 4.237 mm | **2.382 mm** | 2.400 mm | 2.404 mm |
| **합계** | **21D** | | | | | | |

근거: [summary.json](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/summary.json) · [detector_seed_0_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/detector_seed_0_natural.npz) · [detector_seed_1_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/detector_seed_1_natural.npz) · [detector_seed_2_natural.npz](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/detector_seed_2_natural.npz)

## 실험별 체크포인트 매핑

상세 표의 실험명과 아래 항목은 같은 이름이다. 원격 경로는 기록된 값이며 현재 원격 파일의 존재 여부는 확인하지 않았다. 로컬 링크는 해당 파일의 SHA-256이 기록값과 일치한 경우에만 표시한다. `best.pt`, `last.pt`, `final_epoch.pt`는 서로 다른 선택 기준이므로 구별한다. 시계열 모델은 모델 자체와 고정 M0 체크포인트가 모두 필요하다.

| 실험 | 매핑 상태 |
| --- | --- |
| [자동 대응 Transformer + GNN](#checkpoint-detected-transformer) | frozen Transformer·GNN seed별 경로·SHA-256 확인 |
| [자동 대응 Mamba + GNN](#checkpoint-detected-mamba) | frozen Mamba·GNN seed별 경로·SHA-256 확인 |
| [Selected M0 · 30 epoch 한도 best](#checkpoint-1) | seed별 경로·SHA-256 확인 |
| [Selected M0 · epoch 30 last](#checkpoint-2) | seed별 경로·SHA-256 확인 |
| [Selected M0 · epoch 60 last](#checkpoint-3) | seed별 경로·SHA-256 확인 |
| [Selected M0 · 60 epoch 한도 best](#checkpoint-4) | seed별 경로·SHA-256 확인 |
| [GNN · 물리151 검증 조기 종료](#checkpoint-5) | seed별 경로·SHA-256 확인 |
| [M0 · detector Rim-only](#checkpoint-6) | seed별 경로·SHA-256 확인 |
| [M0 XYZ/type](#checkpoint-7) | seed별 경로·SHA-256 확인 |
| [M1 Detector family](#checkpoint-8) | seed별 경로·SHA-256 확인 |
| [M2 Oracle family · GT 입력 참고](#checkpoint-9) | seed별 경로·SHA-256 확인 |
| [용량 통제](#checkpoint-10) | seed별 경로·SHA-256 확인 |
| [60-epoch GNN 출발 · 노드 토큰 Mamba 공동 학습](#checkpoint-25) | seed별 원격 체크포인트·SHA-256 확인 |
| [조기 종료 GNN 출발 · 노드 토큰 Mamba 공동 학습](#checkpoint-24) | seed별 원격 체크포인트·SHA-256 확인 |
| [GNN 조기 종료 + 노드 토큰 Mamba 조기 종료](#checkpoint-23) | seed별 경로·SHA-256 확인 |
| [GNN 조기 종료 + Mamba 조기 종료](#checkpoint-21) | seed별 경로·SHA-256 확인 |
| [GNN 조기 종료 + Transformer 조기 종료](#checkpoint-22) | seed별 경로·SHA-256 확인 |
| [Mamba · 프레임별 노드 토큰 · 120 epoch 한도 best](#checkpoint-20) | seed별 경로·SHA-256 확인 |
| [Mamba · 120 epoch 한도 best](#checkpoint-11) | seed별 경로·SHA-256 확인 |
| [Transformer · 120 epoch 한도 best](#checkpoint-12) | seed별 경로·SHA-256 확인 |
| [M0 원래 개발셋 · best](#checkpoint-13) | 정확한 seed별 체크포인트 미확인 |
| [Selected M0 · epoch 90 last](#checkpoint-14) | 정확한 seed별 체크포인트 미확인 |
| [Mamba · 60 epoch 한도 best](#checkpoint-15) | 정확한 seed별 체크포인트 미확인 |
| [Transformer · 60 epoch 한도 best](#checkpoint-16) | 정확한 seed별 체크포인트 미확인 |
| [Mamba · 90 epoch 한도 best](#checkpoint-17) | 정확한 seed별 체크포인트 미확인 |
| [Transformer · 90 epoch 한도 best](#checkpoint-18) | 정확한 seed별 체크포인트 미확인 |
| [초기 Mamba · seed 0만 완료](#checkpoint-19) | 정확한 seed별 체크포인트 미확인 |

<a id="checkpoint-detected-transformer"></a>
### 자동 대응 Transformer + GNN

시계열 모델은 고정하고 GNN만 20에폭 학습했다. 두 체크포인트가 함께 필요하며 해시는 모두 로컬 사본과 대조했다.

| Seed | GNN best epoch | GNN 체크포인트 / SHA-256 | Frozen Transformer / SHA-256 |
| ---: | ---: | --- | --- |
| 0 | 20 | [best.pt](../outputs/detected_corner_gnn_training/runs/transformer/seed_0/e0f05dd13a4343a6b66fb879db46e6cc/best.pt) `93455ad755cdee4e95dab9a35c85cd2122c5d8ff7f33dba4e22d17b024133a18` | [best.pt](../outputs/fixed_id_corner_early_stopping/runs/transformer/seed_0/5755e14cbb2b4bdc8323f18ed23c27d0/best.pt) `fb555e95493a64b3e0a8d1e3138fc21adcc7d07d6cd9ec62ff1efcb79b88c076` |
| 1 | 16 | [best.pt](../outputs/detected_corner_gnn_training/runs/transformer/seed_1/884f8e1c25564a3aa666b84e951ebff6/best.pt) `5aed9f4cabb09f4c420fce1fc3f5bc9efc7251dadfb62f0275cf720c834a9c2a` | [best.pt](../outputs/fixed_id_corner_early_stopping/runs/transformer/seed_1/20016101c0e94fa3991c7c8befc1040e/best.pt) `d5a726f80d35827f6994e391f36fef3d351be219e817c7ce5c412295ccf3cdc7` |
| 2 | 20 | [best.pt](../outputs/detected_corner_gnn_training/runs/transformer/seed_2/e3bbe5da8767496697c189ebcd122861/best.pt) `b59a6777821327b563aeeb0200d0a1649e23b3843c77b690e266e4119378edad` | [best.pt](../outputs/fixed_id_corner_early_stopping/runs/transformer/seed_2/5439879feb664125a8350e100f9269cf/best.pt) `ea47af588d99fece019f6729534da7a14c236e6fc783488eb05bea0b29b6a4f9` |

<a id="checkpoint-detected-mamba"></a>
### 자동 대응 Mamba + GNN

시계열 모델은 고정하고 GNN만 20에폭 학습했다. 두 체크포인트가 함께 필요하며 해시는 모두 로컬 사본과 대조했다.

| Seed | GNN best epoch | GNN 체크포인트 / SHA-256 | Frozen Mamba / SHA-256 |
| ---: | ---: | --- | --- |
| 0 | 20 | [best.pt](../outputs/detected_corner_gnn_training/runs/mamba/seed_0/4ff48150508b4197a290837d62f5fd26/best.pt) `1090b4fe901ab09a149b612556c3abff6204192190d99bdfb35ecc52862326ef` | [best.pt](../outputs/fixed_id_corner_early_stopping/runs/mamba/seed_0/49f52444978e4e0398e52d83c22dbe02/best.pt) `b005edb5c295a2fd1a0e1db87f2ac4190d81f8f5b2d0cfb8735b7d1698b3c5bc` |
| 1 | 19 | [best.pt](../outputs/detected_corner_gnn_training/runs/mamba/seed_1/5697cc56b6f34a829ec815636040464f/best.pt) `687458303c049d5703bf50fad8efdf083dbf5a2fc55090acdaaae4f9bf846fbd` | [best.pt](../outputs/fixed_id_corner_early_stopping/runs/mamba/seed_1/ab40eddc784345a7ab59e0ba5c778225/best.pt) `69098ae75508a66d55c0885ef76d6b59cdce45cf502cdabf8b5053282c0be28a` |
| 2 | 20 | [best.pt](../outputs/detected_corner_gnn_training/runs/mamba/seed_2/d47032993ecb47b1b607f267c4c261b1/best.pt) `bcfcd56513da0d172f9c53f3f6809642446ef0d58bc22dea44ece08b8dbb749e` | [best.pt](../outputs/fixed_id_corner_early_stopping/runs/mamba/seed_2/6461d15c80544230acad7adf19f30a62/best.pt) `102b8f23b0eaeb7011f2e3c948b1acf4a05d622d590236df08aee96a5bf49cbb` |

<a id="checkpoint-1"></a>
### Selected M0 · 30 epoch 한도 best

기록 근거: [원본 기록](../outputs/selected_vertex_m0/combined_matrix/matrix.json)

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | 3 | `/workspace/selected_vertex_m0/training_seed_0/7857b4fa28d142b4bc056064364288a6/baseline/seed_0/fa47f2350764476085fe64bbbe933432/best.pt` | `9842651839128015d628ae3e81c216875bc89a5ec78a12b374957de4e58b188d` | [SHA-256 일치](../outputs/selected_vertex_m0/training_seed_0/7857b4fa28d142b4bc056064364288a6/baseline/seed_0/fa47f2350764476085fe64bbbe933432/best.pt) |
| 1 | 5 | `/workspace/selected_vertex_m0/training_seed_1/c22089fff8d04829b21b7c53d3a71908/baseline/seed_1/d5c5145108454938b85c8e0f5e7518da/best.pt` | `6e37a0021a86417b5373169e2e550446056377ed8f794578f79aeee3a587b59b` | [SHA-256 일치](../outputs/selected_vertex_m0/training_seed_1/c22089fff8d04829b21b7c53d3a71908/baseline/seed_1/d5c5145108454938b85c8e0f5e7518da/best.pt) |
| 2 | 30 | `/workspace/selected_vertex_m0/training_seed_2/b32c8cde881d46e2abebdfd5ff794252/baseline/seed_2/fd44c6e08bc04ecca56acdd4f1e13c7e/best.pt` | `f017b2266af73f9817f5518d450b72142dcbacd98d76e8c823394537596b3701` | [SHA-256 일치](../outputs/selected_vertex_m0/training_seed_2/b32c8cde881d46e2abebdfd5ff794252/baseline/seed_2/fd44c6e08bc04ecca56acdd4f1e13c7e/best.pt) |

<a id="checkpoint-2"></a>
### Selected M0 · epoch 30 last

기록 근거: [원본 기록](../outputs/selected_vertex_continuation/final_30_matrix/matrix.json)

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | 30 | `/workspace/selected_vertex_continuation_matched/final_30_matrix/seed_0.pt` | `61247ad31d437179e9c0a26331cd38cce723acc5c03f461686b857b95e127be8` | [SHA-256 일치](../outputs/selected_vertex_continuation/final_30_matrix/seed_0.pt) |
| 1 | 30 | `/workspace/selected_vertex_continuation_matched/final_30_matrix/seed_1.pt` | `fb95bb64d84aeff6e541e5293c01743e5f0ceed13a3c4ac7860bf0c853c60f3a` | [SHA-256 일치](../outputs/selected_vertex_continuation/final_30_matrix/seed_1.pt) |
| 2 | 30 | `/workspace/selected_vertex_continuation_matched/final_30_matrix/seed_2.pt` | `f5dbf1228551bb36e5e9e57447e8d4d8c4de0ab58fb0a40f9c929cf48e5ebbd7` | [SHA-256 일치](../outputs/selected_vertex_continuation/final_30_matrix/seed_2.pt) |

<a id="checkpoint-3"></a>
### Selected M0 · epoch 60 last

기록 근거: [원본 기록](../outputs/selected_vertex_continuation/final_matrix/matrix.json)

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | 60 | `/workspace/selected_vertex_continuation_matched/training_seed_0/ba9daea778694023845cb88ece0b7624/training/695438c8b53f4bbb81269b056af6d133/final_epoch.pt` | `a621ad0de1524377be9af1f23e9d6d5ebff62d6357fdfe443811887e551e36a6` | [SHA-256 일치](../outputs/selected_vertex_continuation/training_seed_0/ba9daea778694023845cb88ece0b7624/training/695438c8b53f4bbb81269b056af6d133/final_epoch.pt) |
| 1 | 60 | `/workspace/selected_vertex_continuation_matched/training_seed_1/03b6af139fec43a89ab150e2f472eba1/training/e8694a2f53e1401bafd3cf1e41e42c4c/final_epoch.pt` | `b17b37b4dd0de175fa132a96d14386ef672a4fcbe7dd6085b5cb29eb17279f2b` | [SHA-256 일치](../outputs/selected_vertex_continuation/training_seed_1/03b6af139fec43a89ab150e2f472eba1/training/e8694a2f53e1401bafd3cf1e41e42c4c/final_epoch.pt) |
| 2 | 60 | `/workspace/selected_vertex_continuation_matched/training_seed_2/fc92d6f27d93470a99cbb190331d9ad1/training/a0c146fa0b5b45d985bb6fa93165f3a3/final_epoch.pt` | `d6118f3850dfe39f7bd4f903fe85baa587f0e699d3fdd746ebb1b020daa5fa50` | [SHA-256 일치](../outputs/selected_vertex_continuation/training_seed_2/fc92d6f27d93470a99cbb190331d9ad1/training/a0c146fa0b5b45d985bb6fa93165f3a3/final_epoch.pt) |

<a id="checkpoint-4"></a>
### Selected M0 · 60 epoch 한도 best

기록 근거: [원본 기록](../outputs/selected_vertex_continuation/best_matrix/matrix.json)

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | 3 | `/workspace/selected_vertex_continuation_matched/training_seed_0/ba9daea778694023845cb88ece0b7624/training/695438c8b53f4bbb81269b056af6d133/best.pt` | `b86662bf66e568a024dfe5d2cceb3314bbd65a20505edbd928422cc3af99575e` | [SHA-256 일치](../outputs/selected_vertex_continuation/training_seed_0/ba9daea778694023845cb88ece0b7624/training/695438c8b53f4bbb81269b056af6d133/best.pt) |
| 1 | 5 | `/workspace/selected_vertex_continuation_matched/training_seed_1/03b6af139fec43a89ab150e2f472eba1/training/e8694a2f53e1401bafd3cf1e41e42c4c/best.pt` | `a0fac11274a4f2f615a38e54efa2fa416efec9fc9de70b974f5a545bda09d67f` | [SHA-256 일치](../outputs/selected_vertex_continuation/training_seed_1/03b6af139fec43a89ab150e2f472eba1/training/e8694a2f53e1401bafd3cf1e41e42c4c/best.pt) |
| 2 | 50 | `/workspace/selected_vertex_continuation_matched/training_seed_2/fc92d6f27d93470a99cbb190331d9ad1/training/a0c146fa0b5b45d985bb6fa93165f3a3/best.pt` | `bfbb6d86ef1f77c0924fa25ecb79a1cfb134d161bcf9f4960b9a1828c335838f` | [SHA-256 일치](../outputs/selected_vertex_continuation/training_seed_2/fc92d6f27d93470a99cbb190331d9ad1/training/a0c146fa0b5b45d985bb6fa93165f3a3/best.pt) |

<a id="checkpoint-5"></a>
### GNN · 물리151 검증 조기 종료

기록 근거: [원본 기록](../outputs/temporal_continuation_120/gnn_selected_validation/evaluation_input/detailed_summary.json)

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | 106 | `/workspace/gnn_selected_validation/training_seed_0/80ba420034254d7e8ed7201acfd79787/training/b18af96c7aed4029bd582a9c55abc9c1/best.pt` | `95fe325991d899b5e86f61748f26e16f1fdf8c79e2b663a0751d4a83474e8568` | [SHA-256 일치](../outputs/temporal_continuation_120/gnn_selected_validation/training_seed_0/80ba420034254d7e8ed7201acfd79787/training/b18af96c7aed4029bd582a9c55abc9c1/best.pt) |
| 1 | 91 | `/workspace/gnn_selected_validation/training_seed_1/26617c574cd244ed95dc0f05bdfb0bbf/training/e0e56b4eef8f41a3b863e74dc96ae12a/best.pt` | `5e4584ec5fa2a885612f70960344580d08685715ccdbc8f4af43dd8bb775e762` | [SHA-256 일치](../outputs/temporal_continuation_120/gnn_selected_validation/training_seed_1/26617c574cd244ed95dc0f05bdfb0bbf/training/e0e56b4eef8f41a3b863e74dc96ae12a/best.pt) |
| 2 | 105 | `/workspace/gnn_selected_validation/training_seed_2/ab6e086f27b44e0ab593a7d141aa91af/training/090925b56bda417da1993bf27c859cec/best.pt` | `bd0774f42d656194fdc539485d7376034f58413e6424f3271f99c4b170398fe9` | [SHA-256 일치](../outputs/temporal_continuation_120/gnn_selected_validation/training_seed_2/ab6e086f27b44e0ab593a7d141aa91af/training/090925b56bda417da1993bf27c859cec/best.pt) |

<a id="checkpoint-6"></a>
### M0 · detector Rim-only

기록 근거: [원본 기록](../outputs/rim_only_experiment/evaluation/ef864771775547c0bce9e1ddaf39838a/summary.json)

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | — | `/workspace/rim_only_experiment/training/1794b7d5bcb841d8a1abbbe489b6f495/baseline/seed_0/edb8235500964dd59f8f64ae50b3e49a/best.pt` | `0d8eb877e382ec2ec27bd597e6ee245cf301ca7a7f0ab59586ce8026a2173c20` | [SHA-256 일치](../outputs/rim_only_experiment/training/1794b7d5bcb841d8a1abbbe489b6f495/baseline/seed_0/edb8235500964dd59f8f64ae50b3e49a/best.pt) |
| 1 | — | `/workspace/rim_only_experiment/training/1794b7d5bcb841d8a1abbbe489b6f495/baseline/seed_1/361957cfde7046c69b1125268c406b7a/best.pt` | `0c3aa589c7527f0ec5366802cd4c8b26062141b28209fd97483197ee9c50d999` | [SHA-256 일치](../outputs/rim_only_experiment/training/1794b7d5bcb841d8a1abbbe489b6f495/baseline/seed_1/361957cfde7046c69b1125268c406b7a/best.pt) |
| 2 | — | `/workspace/rim_only_experiment/training/1794b7d5bcb841d8a1abbbe489b6f495/baseline/seed_2/227a6dc264fa42eea91118268feb0442/best.pt` | `ca8ca217fe91ca6bc0a1524056065b7f39495deeb1b8d53600fd484200b8ae74` | [SHA-256 일치](../outputs/rim_only_experiment/training/1794b7d5bcb841d8a1abbbe489b6f495/baseline/seed_2/227a6dc264fa42eea91118268feb0442/best.pt) |

<a id="checkpoint-7"></a>
### M0 XYZ/type

기록 근거: [원본 기록](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/summary.json)

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | 14 | `/workspace/vertex_family_full_training/training_seed_0/fe5915d9cee7424bbf636c5d11b7bb24/baseline/seed_0/aec4af3bf5374faeae935c374af458af/best.pt` | `e1349ac77ed69e42e83a753ee899da5f0c2bbfdff8d976c3b0dba824a39fd6b0` | 일치 사본 미확인 |
| 1 | 6 | `/workspace/vertex_family_full_training/training_seed_1/07d60d00d9a94739819623269d60c697/baseline/seed_1/108cba80633e4eaa80ca1864e10e3251/best.pt` | `fe2d44e82c034653e7d2e605eb3eeafe33ab348e024d1f86a52a83b426b0cb47` | 일치 사본 미확인 |
| 2 | 15 | `/workspace/vertex_family_full_training/training_seed_2/a2f480bf76564558801b1efc0feb90bc/baseline/seed_2/1c89f1b994a94318af89f42227d41562/best.pt` | `da1c72ba4a52d63d0adc8da84d7e8481e4086fc5e8d504982062c5f014b6b28e` | 일치 사본 미확인 |

<a id="checkpoint-8"></a>
### M1 Detector family

기록 근거: [원본 기록](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/summary.json)

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | 16 | `/workspace/vertex_family_full_training/training_seed_0/fe5915d9cee7424bbf636c5d11b7bb24/detector/seed_0/09b65643eb1b4d5dbce514ecf90b75b8/best.pt` | `b679afb783c14633d4f2aff6280699656b4dd04aa6fa263f726c9ae3ce2d23a1` | 일치 사본 미확인 |
| 1 | 11 | `/workspace/vertex_family_full_training/training_seed_1/07d60d00d9a94739819623269d60c697/detector/seed_1/730f70355b3f4367844f3e077b7b9656/best.pt` | `d71c3cf52f6c6fb182eea8565d91f1affac04b0f64f7622b544621b732c5219a` | 일치 사본 미확인 |
| 2 | 14 | `/workspace/vertex_family_full_training/training_seed_2/a2f480bf76564558801b1efc0feb90bc/detector/seed_2/533dea5338cd4f4e9132ccf2aca851a0/best.pt` | `0588022e2ea647dfb15e8061d19e71ca13b32fccba00357b42c7e708c7abc2ae` | 일치 사본 미확인 |

<a id="checkpoint-9"></a>
### M2 Oracle family · GT 입력 참고

기록 근거: [원본 기록](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/summary.json)

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | 15 | `/workspace/vertex_family_full_training/training_seed_0/fe5915d9cee7424bbf636c5d11b7bb24/oracle/seed_0/798b504f3ed7442ebd42150817525af2/best.pt` | `2bdb2e369e747958fa064a6f45432484484abe1a700fa4602e8e3605a12f7a0b` | 일치 사본 미확인 |
| 1 | 12 | `/workspace/vertex_family_full_training/training_seed_1/07d60d00d9a94739819623269d60c697/oracle/seed_1/03f93f9b1b804c978e7947c7ef980edb/best.pt` | `7efcad713cf81e7ea027c07c03f7e94d2d1b3013aa3a8e277806a74620b54d2e` | 일치 사본 미확인 |
| 2 | 13 | `/workspace/vertex_family_full_training/training_seed_2/a2f480bf76564558801b1efc0feb90bc/oracle/seed_2/dc42363c830f4077aa5481a56b5413a4/best.pt` | `d575ff134bc510f583243dc7cc46e9cd841988e3e29ca0fdd05ab77b31a03d69` | 일치 사본 미확인 |

<a id="checkpoint-10"></a>
### 용량 통제

기록 근거: [원본 기록](../outputs/vertex_family_full_training/evaluation/15b2cfa92d7a474ca68ff178e14f8fe6/summary.json)

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | 12 | `/workspace/vertex_family_full_training/training_seed_0/fe5915d9cee7424bbf636c5d11b7bb24/type_control/seed_0/34756542340c4b9ebd2b60fccdb8c7fa/best.pt` | `a9ab60fa8eafff86da55a17c7a86597e13c087af8c9210872d42504b6dde9270` | 일치 사본 미확인 |
| 1 | 6 | `/workspace/vertex_family_full_training/training_seed_1/07d60d00d9a94739819623269d60c697/type_control/seed_1/d3c88be49b834b3ba5f423aa9a9e5a24/best.pt` | `ee55fa842a34b88ba2f36df3b43bce18172788a2b34b1f8a93800dd763b8675a` | 일치 사본 미확인 |
| 2 | 12 | `/workspace/vertex_family_full_training/training_seed_2/a2f480bf76564558801b1efc0feb90bc/type_control/seed_2/2519eb933f8147da8eab3dd86f8c6b11/best.pt` | `9d61b52160fd4d436e16375c07ca543cdba390ad359b4e483bc2924e1fad0bdd` | 일치 사본 미확인 |

<a id="checkpoint-25"></a>
### 60-epoch GNN 출발 · 노드 토큰 Mamba 공동 학습

60 epoch GNN과 노드 토큰 Mamba의 기존 공동 학습 best를 이어서 두 모델을 함께 학습했다. seed 0·1은 조기 종료했고 seed 2는 500 epoch 한도에 도달했다. 시작 공동 학습 epoch는 20 / 18 / 20이다.

| Seed | Best epoch | 종료 epoch | 평가 기록 | 체크포인트 SHA-256 |
| ---: | ---: | ---: | --- | --- |
| 0 | 404 | 434 | [평가 JSON](../outputs/joint_node_token_continuation/evaluation/seed_0/0aacd636cb094f0688e5f19cf57b1710/original_validation_21d.json) | `b15ab24f70213ee0a495bcc9f1a8cd9453a3516bc154dc57c2e0df5277edab03` |
| 1 | 190 | 220 | [평가 JSON](../outputs/joint_node_token_continuation/evaluation/seed_1/d4c50cbec5124c61a770356d2f58513a/original_validation_21d.json) | `9ca8200c05776cbb79073ceaa33080ab08415e60916dd8d0b7adaa24d1a56ad5` |
| 2 | 488 | 500 | [평가 JSON](../outputs/joint_node_token_continuation/evaluation/seed_2/e243222b2e8e4530b3f3ef0da1b9bb23/original_validation_21d.json) | `4de1b29f371e9edcb56242a0a979aefe5f0a68e884ba71e4309f37a64eb22af8` |

원격 체크포인트 경로는 seed 순서대로 다음과 같다.

- `/workspace/joint_node_token_continuation/training/seed_0/0e8f638f6760402e917b96e5ad034f86/best.pt`
- `/workspace/joint_node_token_continuation/training/seed_1/bdab5a89b1f54d088937dcdf655cf06a/best.pt`
- `/workspace/joint_node_token_continuation/training/seed_2/4ebd4382b1dd4dacbe6ce9e716f6f171/best.pt`
 평가 JSON과 함께 보존한 raw 오류 배열에서 seed별 프레임과 GT 일치를 확인했다.

<a id="checkpoint-24"></a>
### 조기 종료 GNN 출발 · 노드 토큰 Mamba 공동 학습

물리151 검증으로 선택한 GNN과 조기 종료된 노드 토큰 Mamba에서 출발해 두 모델을 함께 학습했다. 세 seed 모두 조기 종료했다. 물리151은 출발 GNN 선택에도 사용됐다.

| Seed | Best epoch | 종료 epoch | 평가 기록 | 체크포인트 SHA-256 |
| ---: | ---: | ---: | --- | --- |
| 0 | 355 | 385 | [평가 JSON](../outputs/joint_node_token_early_stopping/evaluation/seed_0/109ceb7b7f854ae0ad586325cbd2ded9/original_validation_21d.json) | `3c39b2e762d2dccf10aa5f0cfff6ea073524e9e553cf336a4b86a6733233df73` |
| 1 | 373 | 403 | [평가 JSON](../outputs/joint_node_token_early_stopping/evaluation/seed_1/921142ff4f2f40c4a28a77230b61d882/original_validation_21d.json) | `e5e99c69f27efa1692ecb423bb3bebf886cb025ed067009ac70ea66a489af023` |
| 2 | 412 | 442 | [평가 JSON](../outputs/joint_node_token_early_stopping/evaluation/seed_2/9b6ff101518147e2a77fe4be4d595092/original_validation_21d.json) | `ca3c0ef217b661b3f133055d21a48fb31801816501b09ad8c182e528b74af021` |

원격 체크포인트 경로는 seed 순서대로 다음과 같다.

- `/workspace/joint_node_token_early_stopping/training/seed_0/c24b5c810c4e47a6b5deb30a9bda5fca/best.pt`
- `/workspace/joint_node_token_early_stopping/training/seed_1/fb8ceb9f4c564a778a95961fe197bffe/best.pt`
- `/workspace/joint_node_token_early_stopping/training/seed_2/8a34475119314a9dbdc1ae91f81fb5fb/best.pt`
 평가 JSON과 함께 보존한 raw 오류 배열에서 seed별 프레임과 GT 일치를 확인했다.

<a id="checkpoint-23"></a>
### GNN 조기 종료 + 노드 토큰 Mamba 조기 종료

물리151 검증으로 선택한 GNN best를 고정했다. 입력 GNN 체크포인트는 [GNN · 물리151 검증 조기 종료](#checkpoint-5)와 같다.

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | 97 | `/workspace/node_token_early_stopping/training/seed_0/6d16da7ceac54470ad6bc4f7858fef62/best.pt` | `56754ee621fcfe03c23c77f75bc639e3a24cecc0a06b35b2bcb1b7daa813f811` | [SHA-256 일치](../outputs/node_token_early_stopping/training/seed_0/6d16da7ceac54470ad6bc4f7858fef62/best.pt) |
| 1 | 285 | `/workspace/node_token_early_stopping/training/seed_1/70ace57f59f54409bc75d140268036f1/best.pt` | `7a5bbd76bde7054e565434cb264366d8465d4711db6b17628eabc527ac94da1e` | [SHA-256 일치](../outputs/node_token_early_stopping/training/seed_1/70ace57f59f54409bc75d140268036f1/best.pt) |
| 2 | 198 | `/workspace/node_token_early_stopping/training/seed_2/ab6c964532824f13b9acc382b5901fb5/best.pt` | `a7e7e09cb9a67b96665ef1832aeb226e129835568aa027a1b67c93bcdde49aff` | [SHA-256 일치](../outputs/node_token_early_stopping/training/seed_2/ab6c964532824f13b9acc382b5901fb5/best.pt) |

<a id="checkpoint-21"></a>
### GNN 조기 종료 + Mamba 조기 종료

물리151 검증으로 선택한 GNN best를 고정하고, 시계열 모델을 별도로 조기 종료했다. 고정 GNN은 [GNN · 물리151 검증 조기 종료](#checkpoint-5) 참조.

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | 264 | `/workspace/temporal_early_stopping/training/mamba/seed_0/3242292d88104d45ae9d3bffe9297087/best.pt` | `680fdac167297aa683f857754bfac6a86efb0a184ce9d9838cf53e82e5850eee` | [SHA-256 일치](../outputs/temporal_early_stopping/training/mamba/seed_0/3242292d88104d45ae9d3bffe9297087/best.pt) |
| 1 | 213 | `/workspace/temporal_early_stopping/training/mamba/seed_1/6d2c1405f8a54b17a597f0f1dfe50feb/best.pt` | `ec9acf3c585d5b6bf54852bc6090556d302dadaf28ff453994bad00e2f0e5fb6` | [SHA-256 일치](../outputs/temporal_early_stopping/training/mamba/seed_1/6d2c1405f8a54b17a597f0f1dfe50feb/best.pt) |
| 2 | 85 | `/workspace/temporal_early_stopping/training/mamba/seed_2/eedaac5de1304fc7ac9781c9a398f6dc/best.pt` | `3b2ed304ac865015c7c8db2fd73bfc3dbfcdb6abeb99b08009f7b553a2339545` | [SHA-256 일치](../outputs/temporal_early_stopping/training/mamba/seed_2/eedaac5de1304fc7ac9781c9a398f6dc/best.pt) |

<a id="checkpoint-22"></a>
### GNN 조기 종료 + Transformer 조기 종료

물리151 검증으로 선택한 GNN best를 고정하고, 시계열 모델을 별도로 조기 종료했다. 고정 GNN은 [GNN · 물리151 검증 조기 종료](#checkpoint-5) 참조.

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | 165 | `/workspace/temporal_early_stopping/training/transformer/seed_0/f412b6c2517940e5b1487765dda1086a/best.pt` | `3f70688bfcda7f140167a1b5bb646f38561fcd62893ade6e9573742281aa007a` | [SHA-256 일치](../outputs/temporal_early_stopping/training/transformer/seed_0/f412b6c2517940e5b1487765dda1086a/best.pt) |
| 1 | 151 | `/workspace/temporal_early_stopping/training/transformer/seed_1/5913ceb14d254c42835b842e76441e7b/best.pt` | `b747018303e8c253361426c9cd6e060f3bbed0ef869c3427f9dcf299f01dd14f` | [SHA-256 일치](../outputs/temporal_early_stopping/training/transformer/seed_1/5913ceb14d254c42835b842e76441e7b/best.pt) |
| 2 | 183 | `/workspace/temporal_early_stopping/training/transformer/seed_2/e0795f37b6b845729497f479f80653ae/best.pt` | `a0ab0711e90804309dc264cdc197dd4122af4fc63dabe4ac666b32590aa55740` | [SHA-256 일치](../outputs/temporal_early_stopping/training/transformer/seed_2/e0795f37b6b845729497f479f80653ae/best.pt) |

<a id="checkpoint-20"></a>
### Mamba · 프레임별 노드 토큰 · 120 epoch 한도 best

시계열 체크포인트. 세 seed 모두 「Selected M0 · epoch 60 last」를 고정했으며, 노드 토큰과 기존 Object21 관측을 함께 사용한다. 로컬 체크포인트는 원격 SHA-256과 일치한다.

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | 108 | `/workspace/node_token_mamba/training/seed_0/5ec76e979b37428f898b7b0efca9a8ca/best.pt` | `24fce85a684e803ec4788847d6e91c48aa35ac638e745086788ea7f3a9f4ca21` | [SHA-256 일치](../outputs/node_token_mamba/training/seed_0/5ec76e979b37428f898b7b0efca9a8ca/best.pt) |
| 1 | 112 | `/workspace/node_token_mamba/training/seed_1/1d54e275b02a4a0fa0609084b0a07d09/best.pt` | `ea70fbc8ccbddf7e06c7bd2dee89d6ffd94223e5fa966a50ca8d42cfec439dca` | [SHA-256 일치](../outputs/node_token_mamba/training/seed_1/1d54e275b02a4a0fa0609084b0a07d09/best.pt) |
| 2 | 89 | `/workspace/node_token_mamba/training/seed_2/20b1a0a1b42141e9a1139ea43242bb6f/best.pt` | `fb2dab8fcd10e04e15a6c81be60da1161fc62aa0d224f9d5500ff65b61df0aa0` | [SHA-256 일치](../outputs/node_token_mamba/training/seed_2/20b1a0a1b42141e9a1139ea43242bb6f/best.pt) |

<a id="checkpoint-11"></a>
### Mamba · 120 epoch 한도 best

기록 근거: [원본 기록](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/mamba/seed_0/e333b69eb92f44788fe67903b8a31872/provenance.json) · [원본 기록](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/mamba/seed_1/10145c66268d4a1b8d5aa2ebdc840812/provenance.json) · [원본 기록](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/mamba/seed_2/deddfb3bbf954dd4a7857c6ff7bd7799/provenance.json)

시계열 체크포인트다. 입력으로 고정한 M0는 별도 「Selected M0 · epoch 60 last」에 기록했다.

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | — | `/workspace/temporal_continuation_120/training/mamba/seed_0/04d86ec2714f4369bbfc20e1c6e2faff/best.pt` | `d4c944e14c87c80b608ee8198baaf8a8edb7f865c736600f2b5adfd80aee2318` | [SHA-256 일치](../outputs/temporal_continuation_120/temporal_continuation_120/training/mamba/seed_0/04d86ec2714f4369bbfc20e1c6e2faff/best.pt) |
| 1 | — | `/workspace/temporal_continuation_120/training/mamba/seed_1/313e2998c1ec41b4a8d34f7d0b05e73c/best.pt` | `3d757e43ad0d2876fa6376a67f52354b7bcbe379663ab54256cf392be82c4ad0` | [SHA-256 일치](../outputs/temporal_continuation_120/temporal_continuation_120/training/mamba/seed_1/313e2998c1ec41b4a8d34f7d0b05e73c/best.pt) |
| 2 | — | `/workspace/temporal_continuation_120/training/mamba/seed_2/fabc97689df249678616ff86d2eda9e9/best.pt` | `1c2e675e9203f417f01d352d1c7467b6348277227ecd7d360f2c2f4ead9cfd11` | [SHA-256 일치](../outputs/temporal_continuation_120/temporal_continuation_120/training/mamba/seed_2/fabc97689df249678616ff86d2eda9e9/best.pt) |

<a id="checkpoint-12"></a>
### Transformer · 120 epoch 한도 best

기록 근거: [원본 기록](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/transformer/seed_0/299e0511de81487faf51d9393f982de2/provenance.json) · [원본 기록](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/transformer/seed_1/a5530784df2a4a7b84a292a9395880b1/provenance.json) · [원본 기록](../outputs/temporal_continuation_120/temporal_continuation_120/evaluation/transformer/seed_2/780027874899493d8a3391eeed93fee8/provenance.json)

시계열 체크포인트다. 입력으로 고정한 M0는 별도 「Selected M0 · epoch 60 last」에 기록했다.

| Seed | Epoch | 기록된 원격 체크포인트 | SHA-256 | 로컬 확인 |
| ---: | ---: | --- | --- | --- |
| 0 | — | `/workspace/temporal_continuation_120/training/transformer/seed_0/361dbb04995e43c2b0e5a804d53e649e/best.pt` | `453e3115a7e90c1f2195888acf3e96153d0543f7e6dc6af633f6d7529900d561` | [SHA-256 일치](../outputs/temporal_continuation_120/temporal_continuation_120/training/transformer/seed_0/361dbb04995e43c2b0e5a804d53e649e/best.pt) |
| 1 | — | `/workspace/temporal_continuation_120/training/transformer/seed_1/01235cc0fc9e46c0925ebb8a01725105/best.pt` | `652446a2d5363d3ec34bc9de91b990d522d6407ea243f36ad41e2d123a1bad36` | [SHA-256 일치](../outputs/temporal_continuation_120/temporal_continuation_120/training/transformer/seed_1/01235cc0fc9e46c0925ebb8a01725105/best.pt) |
| 2 | — | `/workspace/temporal_continuation_120/training/transformer/seed_2/8b231e3c75164278b3a02fcc1e3ca73e/best.pt` | `7a60e54d5992fee846a7469b173420fd9316821d5ca79a04d123537443900369` | [SHA-256 일치](../outputs/temporal_continuation_120/temporal_continuation_120/training/transformer/seed_2/8b231e3c75164278b3a02fcc1e3ca73e/best.pt) |

<a id="checkpoint-13"></a>
### M0 원래 개발셋 · best

기록 근거: [원본 기록](../outputs/selected_vertex_m0/evaluation/840b377c9d584bdaac4d43771e58f6f7/effective_config.json)

평가 설정에는 reference matrix 위치가 기록되어 있지만, 해당 평가의 각 seed 체크포인트 경로와 해시를 로컬 근거에서 확인하지 못했다. 별도 Rim-only 실험의 기준 모델과 동일하다고 가정하지 않았다.

<a id="checkpoint-14"></a>
### Selected M0 · epoch 90 last

기록 근거: [원본 기록](../outputs/temporal_continuation_90/comparison_report.md)

seed별 정확한 평가 체크포인트 경로와 해시가 요약에 없다. 보고서에 기록된 서버 실행 상위 경로만 확인된다.

<a id="checkpoint-15"></a>
### Mamba · 60 epoch 한도 best

기록 근거: [원본 기록](../outputs/temporal_continuation_90/source_checkpoints.json)

seed별 60-epoch last.pt의 경로와 해시는 기록되어 있으나, 이 표가 사용한 best.pt의 경로·해시는 확인되지 않았다. last.pt를 best.pt와 동일시하지 않는다.

<a id="checkpoint-16"></a>
### Transformer · 60 epoch 한도 best

기록 근거: [원본 기록](../outputs/temporal_continuation_90/source_checkpoints.json)

seed별 60-epoch last.pt의 경로와 해시는 기록되어 있으나, 이 표가 사용한 best.pt의 경로·해시는 확인되지 않았다. last.pt를 best.pt와 동일시하지 않는다.

<a id="checkpoint-17"></a>
### Mamba · 90 epoch 한도 best

기록 근거: [원본 기록](../outputs/temporal_continuation_90/mamba_summary_21d.json)

평가 요약과 seed별 오차 배열은 있으나 사용한 best.pt의 경로·해시는 로컬 기록에서 확인하지 못했다.

<a id="checkpoint-18"></a>
### Transformer · 90 epoch 한도 best

기록 근거: [원본 기록](../outputs/temporal_continuation_90/transformer_summary_21d.json)

평가 요약과 seed별 오차 배열은 있으나 사용한 best.pt의 경로·해시는 로컬 기록에서 확인하지 못했다.

<a id="checkpoint-19"></a>
### 초기 Mamba · seed 0만 완료

기록 근거: [원본 기록](../outputs/selected_m0_mamba/seed_0/original_validation_21d.json)

seed 0의 training_output 디렉터리는 기록되었으나 평가에 사용한 체크포인트 파일의 경로와 해시는 확인하지 못했다.

## 범위와 검증

- 동일 실험의 중복 복사본과 대조군은 중복 등록하지 않았다. seed가 1개뿐인 초기 Mamba는 상세 참고표로만 남겼다.
- 물리 선택 자체, 데이터 복원·컴파일 벤치마크, random-rollout 수집, loss만 있는 기록은 3-seed 21D 예측 결과가 아니므로 순위에서 제외했다.
- raw per-frame archive가 있는 결과는 순서/프레임 집합을 확인하고 3-seed 통계량을 재계산했다. 90-epoch temporal 오차 파일에는 sample ID가 없으므로 원래 960개 순서로 해석하고 전체/151개 집계가 기존 요약과 일치하는지 교차 검증했다.
- 집계 요약만 있는 GNN epoch 90·Transformer 60은 그 요약을 인용했다. 전체 4,102프레임 시계열 비교와 vertex-family 비교는 별도 cohort로 유지했다.
- 출처·SHA-256·정규화된 표 데이터와 검증 목록: [verification.json](../outputs/experiment_leaderboard/verification.json). 원본 `outputs/`는 Git 제외 대상이므로 저장소를 옮기면 원본 링크가 없을 수 있다. 이 Markdown 자체에는 표의 모든 수치를 포함했다.

총 40개 3-seed 상세 표와 2개 단일-seed 참고 표. 기존 산출물만 읽었으며 모델 추론·학습·원격 업로드는 실행하지 않았다.
