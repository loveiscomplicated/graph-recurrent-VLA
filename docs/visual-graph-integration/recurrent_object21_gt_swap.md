# 이동 Box 진입 시 Object21 GT 교체

[재학습 Mamba의 실패 분석](recurrent_moving_box_failure_analysis.md)에서 얕은 진입 후 정체한 `700674`와 깊게 삽입한 뒤 정체한 `700882`를 골랐다. Vast.ai RTX 3090에서 동일한 카메라, Box 이동 계획, RANSAC, Transformer, 강체 보정, GNN 및 재학습 Mamba 5개를 사용했다. Recurrent 필터는 우회했다. GT를 활용한 꼭짓점 번호 매칭은 두 조건에서 동일하다.

- **Graph 유지.** 매 스텝 GNN이 예측한 Object21을 Mamba에 제공한다.
- **진입 후 GT.** Peg 끝점 중심이 Hole의 사각 개구부 범위 안에서 입구 면을 0.1 mm 넘는 첫 스텝까지 기존 제어로 접근한다. 해당 스텝의 행동을 계산하기 직전부터 Mamba에 들어가는 Object21의 21개 값을 전부 시뮬레이터 GT로 교체한다. 이후 되돌리지 않는다. 로봇 상태와 Mamba의 이전 관측 이력은 계속 사용하고 GNN 출력도 비교용으로 계속 기록한다.

같은 초기 시드에서 두 롤아웃을 독립 재생했다. 진입 직전까지의 GT 상대 위치, 행동, Box 변위를 대조했다. 첫 번째 GT 주입의 기준은 Peg 단면 전체의 통과나 접촉 판정과 다르다. 깊이 표의 최대값은 행동 직전 기록에서 입구 개구부에 들어온 끝점 중심의 최대 깊이이며 최종 성공 여부는 환경의 동작 후 판정이다.

## 결과

| 시드 | 조건 | 입구 기준 통과 스텝 | 최대 삽입 깊이 | 성공 기준 깊이 | 종료 |
| --- | --- | ---: | ---: | ---: | --- |
| 700674 | Graph 유지 | 89 | 0.29 mm | 81.20 mm | 250스텝 시간 초과 |
| 700674 | 진입 후 GT | 90 | 0.23 mm | 81.20 mm | 250스텝 시간 초과 |
| 700882 | Graph 유지 | 64 | 93.18 mm | 98.91 mm | 250스텝 시간 초과 |
| 700882 | 진입 후 GT | 64 | 32.54 mm | 98.91 mm | 250스텝 시간 초과 |

700674는 GT 이후에도 입구 부근에서 정체했다. 같은 90스텝에서 Graph의 횡방향 추정 오차는 2.56 mm였고 GT를 사용한 궤적에서 처음 교체되기 직전 Graph의 횡방향 추정 오차는 2.94 mm였다. 700882는 GT로 바꾼 뒤 32.54 mm까지 진행하고 개구부 범위를 벗어났다. GT 조건의 마지막 Peg 끝점 중심은 구멍 축에서 횡방향으로 214.33 mm 떨어졌으며 파지는 마지막까지 유지됐다. Graph 유지 시 마지막 이탈은 3.04 mm였다. 700882가 GT로 전환한 64스텝에는 Graph 횡방향 오차가 약 3.0 mm였으며, Graph 유지 조건에서 오차가 60 mm대로 커지는 시점은 이보다 뒤다.

현재 가중치의 Mamba에서 얕은 진입 직후 Object21 전체를 GT로 교체해도 두 장면 모두 복구하지 못했다. 이 실험은 Graph가 큰 오차를 내기 시작한 뒤에만 교체하는 조건이나, 삽입 후 데이터로 GNN 또는 Mamba를 추가 학습하는 조건의 결과를 알려주지 않는다.

## 검증 및 영상

두 롤아웃에서 최초 환경 상태가 일치하며 공통 스텝까지 Box 변위가 같았다. GT 전환 스텝에서 실제 Peg–Hole 상대 위치 차이는 700674에서 0.080 mm, 700882에서 0.012 mm였다. 접근 전체에서의 최대 차이는 각각 0.826 mm와 0.012 mm로, 독립 재생의 물리·관측 이력이 특히 700674에서 소폭 달랐다. 각 스텝에서 디코더가 받은 벡터가 교체 타이밍에 맞춰 정확히 Graph 또는 GT와 일치하는 것을 확인했다. 공통 ExperimentLogger 완료 상태, 기존 가중치 파일 해시 7개, 영상 4개의 디코딩, 20 fps, 512×512 및 프레임 수를 확인했다.

- 700674 [Graph 유지](../../outputs/recurrent_object21_gt_swap/source_runs/700674/graph/700674_hybrid_fail.mp4) · [진입 후 GT](../../outputs/recurrent_object21_gt_swap/source_runs/700674/gt_after_entry/700674_hybrid_fail.mp4)
- 700882 [Graph 유지](../../outputs/recurrent_object21_gt_swap/source_runs/700882/graph/700882_hybrid_fail.mp4) · [진입 후 GT](../../outputs/recurrent_object21_gt_swap/source_runs/700882/gt_after_entry/700882_hybrid_fail.mp4)
- [검증 집계](../../outputs/recurrent_object21_gt_swap/source_runs/verified.json) · [실행 스크립트](../../outputs/recurrent_object21_gt_swap/compare.py) · [검증 스크립트](../../outputs/recurrent_object21_gt_swap/verify.py)
