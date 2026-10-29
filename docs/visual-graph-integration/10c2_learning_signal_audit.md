# Task 10-C2 — Geometry-conditioned Learning Signal Audit

ExperimentLogger UUID `87ac9525933c4562b8930018fb165b0a`. Parent BC `3788798a719f4ec4a66c224ca42f5d63`.

**주 결과: 고정한 Robot·과거 이력·Peg 품질·Box 조건을 모두 만족하는 cross-episode Pair는 Train96/Validation12 모두 0개다. H1과 H2의 상대적 설명력을 분리할 근거는 INCONCLUSIVE다.** Robot-only 매칭의 제한된 사례는 보존하지만 Peg의 독립적인 인과 효과로 해석하지 않는다.

## 실행 계약과 기준 동결

Train96 11,975 Graph, Validation12 1,477 Graph. Test12를 열지 않았다. 원본 HDF/캐시/체크포인트 변경, 새 BC·Probe 학습, 모델 forward, DINO/SAM2, simulator 재생성은 없다. 공통 ExperimentLogger와 기존 ActionCodec(Method B)/controller_commands를 재사용했다. 저장된 Native prediction에 codec을 다시 decode하지 않았다.

`train_matching_freeze.json`은 Action 파일과 Validation을 읽기 전에 Train matching CSV/protocol SHA를 고정했다. 기하 preflight만 읽어 support/offset 기준을 정했고 모든 threshold 결과를 보고한다. Train의 인접 중심 변화 중앙값 3.67mm, p95 14.29mm, p99 168.62mm는 실제 이동·회전·가시성 변화가 섞여 **센서 noise 추정치가 아니다**. 정확한 GT pose/관측 source purity가 없어 20mm도 실제 Peg 이동의 보증이 아니다.

관측 Graph XYZ는 이미 TCP inverse pose로 변환된 좌표다. 동일 Graph의 Peg/Box mean/min/max/bbox/span 정의는 10-C1 geometry_summary와 같다. mean은 표면 중심이며 actor origin이 아니다. 결측은 NaN, 지지점 32 미만·span 대각선 250mm 초과·mean–bbox center 거리 20mm 초과는 heuristic quality 실패로 별도 기록한다. 이 기준을 통과해도 geometry가 정확하다는 의미는 아니다. Box도 같은 품질 gate를 적용했다. 원자료에는 매우 큰 Box span이 있어 많은 쌍이 탈락한다. 이를 실제 Box 위치 차이라고 해석하지 않는다.

Robot76 전체 Euclidean distance는 사용하지 않았다. 관절7 qpos RMS(rad), qvel RMS(rad/s), TCP 위치 norm(mm)·quaternion geodesic(deg), opening(mm), opening velocity(m/s), TCP 선속도·각속도 RMS를 각각 gate했다. 현재·과거1/3/5 step에서 최악 거리를 사용하고 t<lag는 첫 frame padding이다. quaternion q/-q 동치, 중복 finger 상태를 중복 가중하지 않았다. phase는 관측 opening>=65mm/open, <=45mm/small, 사이/transition이고 open t<30만 early_open Reach proxy다. 정확한 Expert phase/grasp label은 없다. 절대 t 차이만 사용하며 episode 최종 길이/미래 state/action/GT를 matching에 쓰지 않았다.

기준표(모든 차원이 동시 통과; Box 품질/거리까지 통과해야 primary eligible):

```text
      level  tcp_mm  q_rad  angle_deg  qvel  opening_mm  gripvel  linear  angular   time  box_mm
     strict   3.000  0.020      3.000 0.100       1.000    0.010   0.030    0.150  3.000   5.000
    primary   5.000  0.040      5.000 0.200       2.000    0.020   0.050    0.250  5.000  10.000
sensitivity  10.000  0.080     10.000 0.400       4.000    0.040   0.100    0.500 10.000  20.000
```

Control <5mm, 5–10/10–20mm 불확실, Different >=20mm(20–40/40+ 별도). 같은 Episode pair는 제외하고 모든 후보를 중복 없는 unordered frame pair로 저장했다. Selected는 robot score 오름차순으로 10-step episode block 및 unordered episode pair를 각 1회만 쓴다. 추가 episode-disjoint subset도 저장했다. 모든 쌍은 episode cluster 종속성을 유지하며 독립 표본 신뢰구간/p-value를 만들지 않는다.

## A — 조건부 다양성/관측 가능성

Train96:

```text
      level  robot_pairs  episodes  peg_quality  control  uncertain  different  box_controlled  selected  exploratory_blocks
     strict            3         2            0        0          0          0               0         0                   0
    primary           77        51           10        2          3          5               0         0                   4
sensitivity         2341        95          359       16         27        316               0         0                  33
```

Validation12 (기준 동결 후 내부 validation끼리; Train에 합산 안 함):

```text
      level  robot_pairs  episodes  peg_quality  control  uncertain  different  box_controlled  selected  exploratory_blocks
     strict            0         0            0        0          0          0               0         0                   0
    primary            0         0            0        0          0          0               0         0                   0
sensitivity           43         6            0        0          0          0               0         0                   0
```

원 frame 품질:

```text
     split  frames  episodes  peg_quality  box_quality
     train   11975        96         5454          576
validation    1477        12          652           58
```

Train 주 기준 Robot 유사도·geometry 분포(단위는 protocol):

```text
index  q_rad   qvel  tcp_mm  angle_deg  opening_mm  gripvel  linear  angular   box_mm  peg_offset_mm
count 77.000 77.000  77.000     77.000      77.000   77.000  77.000   77.000   75.000         58.000
 mean  0.018  0.028   3.930      2.245       0.017    0.000   0.008    0.022  213.522         52.750
  50%  0.018  0.017   4.046      2.381       0.000    0.000   0.005    0.006  142.147         45.919
  max  0.036  0.137   4.993      4.873       1.284    0.002   0.049    0.214 1131.126        229.627
```

`geometry_pair_coverage.csv`에 모든 threshold/offset 구간/유효·불량·Box 미통제 수/고유 frame·episode/coverage/최대 재사용을, `episode_coverage.csv`에는 **0쌍인 episode까지** 보존했다. 전체 후보 수가 유효 독립 사례 수는 아니다. Missing과 span 불량 등을 `frame_geometry_quality.csv`에서 재구분할 수 있다. t 분포와 phase는 전체 Pair CSV에 있다. 엄격 매칭의 scarcity는 데이터의 조건부 support가 제한적이라는 근거지만 “데이터에서 학습 불가능”의 증명은 아니다.

## B/C — Expert와 E0/E1 조건부 Action

Box-controlled primary에서 Expert/모델 비교 가능한 표본은 0이다. 따라서 primary의 평균·방향·gain은 정의하지 않았다. 필수 CSV에는 **모든 robot 후보의** 결과/quality/Box flags를 남겨 실패를 숨기지 않았다.

아래는 **사후 탐색적 Box-uncontrolled** 기술 통계다. 기준을 완화한 새로운 primary로 취급하지 않는다. 기존 Robot gate와 Peg quality를 유지하고 Box gate만 빼서 같은 score/block/episode-pair 제한을 적용했다. 일부 episode를 반복 사용하므로 n개 독립 실험이 아니다. `exploratory_episode_macro.csv`에 cluster별 평균, `exploratory_robot_only_pairs.csv`에 재사용 frame 정보가 있다.

```text
      level pair_class model  n  episodes  small_expert  expert_delta_mm  model_delta_mm  delta_error_mm  direction_cosine  signed_gain  zero_improvement_mm
    primary  different    E0  3         6             0            9.605           1.943           8.103             0.748        0.166                1.502
    primary  different    E1  3         6             0            9.605           1.335           8.936             0.591        0.075                0.670
    primary  uncertain    E0  1         2             1            0.271           1.065           0.884               NaN          NaN               -0.613
    primary  uncertain    E1  1         2             1            0.271           0.324           0.327               NaN          NaN               -0.056
sensitivity    control    E0  3         6             1            9.761           6.562           3.638             0.995        0.745                6.123
sensitivity    control    E1  3         6             1            9.761           7.301           3.263             0.966        0.755                6.498
sensitivity  different    E0 29        38             2            5.306           3.931           2.868             0.717        0.719                2.439
sensitivity  different    E1 29        38             2            5.306           3.537           2.917             0.697        0.699                2.389
sensitivity  uncertain    E0  1         2             1            0.271           1.065           0.884               NaN          NaN               -0.613
sensitivity  uncertain    E1  1         2             1            0.271           0.324           0.327               NaN          NaN               -0.056
```

원시 주 기준 different 5쌍의 평균은 Expert 8.420mm, E0 3.918mm, E1 4.054mm, Δerror 6.584/7.113mm다. control 2쌍은 Expert 0.270mm라 방향 판정에서 제외했다. 이 비교는 표본이 적고 episode/구간/Box 및 관측 상태가 달라 Peg 조건부 신호의 일반적 크기를 추정하지 못한다. 완화된 sensitivity의 상대적으로 좋은 재현도 함께 보존했으며 주 기준을 대체하지 않는다.

모든 5개 Different 후보는 early_open 구간이고, 고유 frame 9개(Train의 0.0752%), 고유 Episode 7개다. 시간·episode pair 재사용을 제거한 탐색 subset은 3쌍/6 Episode이며 Expert 9.605mm 대비 E0/E1 1.943/1.335mm, Δerror 8.103/8.936mm다. 이 subset에는 control pair가 남지 않아 공정한 control 대비 추정도 불가능하다.

주 기준 Different 전체 실제 사례(반복 시점 포함; 사후 실패 사례 선택 없음):

```text
 episode_a  t_a  episode_b  t_b model  peg_offset_mm  box_mm  expert_delta_mm  model_delta_mm  delta_error_mm  direction_cosine  signed_gain  robot_only_blocks
     30071    0      30107    0    E0         46.828  35.167           10.434           2.026           9.210             0.666        0.129               True
     30071    1      30107    1    E0         43.434  64.299           10.276          10.269           2.124             0.979        0.978              False
     30097    0      30070    0    E0         83.744  22.892            9.421           1.009           8.855             0.598        0.064               True
     30026    0      30098    0    E0        129.752  17.259            8.960           2.794           6.244             0.980        0.306               True
     30026    0      30065    0    E0        167.139 148.526            3.011           3.492           6.489            -0.992       -1.150              False
     30071    0      30107    0    E1         46.828  35.167           10.434           1.268           9.380             0.850        0.103               True
     30071    1      30107    1    E1         43.434  64.299           10.276          12.594           2.636             0.994        1.218              False
     30097    0      30070    0    E1         83.744  22.892            9.421           1.095           8.704             0.688        0.080               True
     30026    0      30098    0    E1        129.752  17.259            8.960           1.641           8.723             0.234        0.043               True
     30026    0      30065    0    E1        167.139 148.526            3.011           3.670           6.124            -0.678       -0.826              False
```

명령은 설치된 ManiSkill PDEEPoseController의 clip/scale을 거친 Cartesian translation이다. 각 frame의 root command를 해당 robot base world rotation, TCP inverse rotation으로 바꾼 뒤 homologous TCP axes에서 b−a를 계산했다. achieved TCP 이동/IK 결과는 측정하지 않았다. Δvector error=||modelΔ−expertΔ||; signed gain=dot(modelΔ,expertΔ)/||expertΔ||²; magnitude ratio=||modelΔ||/||expertΔ||; zero improvement=||expertΔ||−error. ExpertΔ<1mm는 small로 기록하고 cosine/gain/ratio를 NaN 처리했다. 1mm는 방향 안정성용 사전 고정 분석 gate이며 controller 정확도 주장과 다르다. Zero modelΔ의 cosine도 NaN이다. Peg-action gain은 offset>=20mm와 ExpertΔ>=1mm에서만 보고한다. Rotation component norm과 gripper physical target 차이는 보조 지표이며 rotation의 finite geodesic 차이와 동일하지 않다.

이 비교는 로봇 상태가 완전히 같지 않고 이전 GRU/vision history 전체가 같은 것도 아니다. 동작 진행/Box/가시성/과거 물체 상태가 남는다. 특히 early_open Reach proxy 외 phase의 Peg-action cosine에 같은 방향을 요구하지 않는다. `expert_conditional_action.csv`의 phase별 원자료를 보존한다.

## Task 10-B/10-C1과의 연결

10-B는 동일 robot·previous hidden·vision prefix의 실제 Peg world XY ±5/10mm 개입이며 primary22에서 E0/E1 cosine 0.254/0.308, gain 0.185/0.116이었다. 이번 분석은 다른 episode·history의 관찰 비교다. 두 증거를 하나의 표본으로 합산하지 않는다. 이번 offset 5–10mm 후보는 원시 분포로 보존했지만 표면 중심 편향 때문에 실제 10-B의 5/10mm pose 개입과 동등하지 않다. 주 기준에서 유효 Box 통제 쌍 0이므로 **10-B의 exact object–robot 조합이 Train96에 충분히 있었다/없었다는 판정은 INCONCLUSIVE**다. Original HDF와 current-runtime 개입 observation도 byte-identical하지 않다. Intervention을 Probe 학습/threshold/모델 선택에 쓰지 않았다.

10-C1은 raw summary의 이동 정보와 downstream의 불안정성을 지지했으나, 이번 Train96 관찰 매칭은 H1/H2를 분리할 만큼의 controlled support를 제공하지 못했다. 그 사실로 10-B의 동일상태 반응 부족을 부정하지도, 데이터 다양성만으로 설명하지도 않는다.

## Q1–Q6 판정

- **Q1 CONFIRMED:** Train96 Robot 후보는 strict 3 / primary 77 / sensitivity 2,341쌍. primary Peg-quality 10쌍 중 control2/불확실3/different5다. Box까지 통제한 유효 pair0. 모든 offset 수·episode 수·coverage는 CSV/위 표에 있다. **H1 SUPPORTED(좁은 범위):** 이 관측 기반의 엄격 조건에서 비교 가능한 사례가 부족하다. 실제 GT 기반 데이터 다양성 부족은 INCONCLUSIVE다.
- **Q2 INCONCLUSIVE:** Robot-only 탐색에서는 Different의 ExpertΔ가 control보다 크지만 controlled pair가0이고 공변량·episode 집중이 남는다. Peg에 따른 체계적 correction이라 일반화할 수 없다.
- **Q3 INCONCLUSIVE:** 특정 관찰쌍에서 E0/E1 Δ 재현 부족은 수치로 확인되나 주 비교 표본이0이다. 작은 ExpertΔ의 control을 실패로 부르지 않는다. E0/E1 우열도 고르지 않는다.
- **Q4 SUPPORTED / INCONCLUSIVE:** 조건부 사례 scarcity와 관측 품질 문제는 제한적 설명 후보로 지지된다. 이것이 closed-loop geometry correction 실패의 주원인이라는 판단은 불가능하다. H1/H2는 함께 존재할 수 있다.
- **Q5 INCONCLUSIVE:** Box 미통제 실제 사례의 8.420mm ExpertΔ 대비 model 3.918/4.054mm 및 6.584/7.113mm Δerror는 재현 공백의 사례다. 하지만 충분한 독립 Peg 신호가 있었다는 전제를 충족하지 않아 이를 H2의 확증으로 제시하지 않는다.
- **Q6 SUPPORTED(다음 검증 우선순위):** 먼저 데이터의 조건부 support와 관측 geometry 신뢰도를 점검할 근거가 가장 강하다. BC objective와 representation 중 무엇을 먼저 변경할지는 INCONCLUSIVE다. 새 학습/구조 변경은 NOT_TESTED이고 실행하지 않았다.

최소 추가 검증은 소수의 사전 선택 robot/구간 anchor에서 **서로 다른 기존 episode**의 실제 Peg/Box 위치 또는 가시성 provenance를 수작업/저장자료로 검증하는 것이다. 5/10mm pose 변화와 centroid 변화의 대응·Box 교란을 분리하고, 동일한 고정 기준에 controlled pair가 남는지 확인한다. 그 뒤에도 pair가 거의 없다면 제한된 matched-state demonstration 다양성 보완을 별도 설계한다. 신호가 충분한 controlled 사례가 확인될 때에만 frozen 모델의 Δ 재현을 다시 검증한다. 이번에 추가 수집/학습은 실행하지 않았다.

## 검증과 한계

CONFIRMED: 108 episode cache receipt SHA, saved prediction target/action alignment, timestep, 10-C1 feature manifest/checkpoint/cache hash 연결 및 3e−5 prediction tolerance, 입력 전후 SHA 동일. 원본 Train96의 actor GT pose는 사용하지 않았다. Checkpoint weight update0, policy inference0. 저장 prediction–checkpoint 연결은 이전10-C1의 frozen forward 재현과 이번 hash/readback으로 검증했으며 새 forward를 수행하지 않았다.

C2의 3개 단위 계약 테스트: cross-episode exclusion/group gate/block reuse, controller Δ/zero response/small Δ, TCP rotation. 기존 Plan A 계약 테스트 9개도 통과했다(합성 forward/backward 계약이며 optimizer step은 없다). 원본 checkpoint의 forward/update는 실행하지 않았다. 실제 dataset 결과는 `verification.json`/CSV readback으로 별도 확인한다. CPU NumPy/SciPy + 설치 controller preprocessing만 실행했다. SAPIEN import의 Vulkan fallback warning은 physics 실행을 뜻하지 않는다. Simulator, CUDA/MPS, 새 vision, Test12, closed-loop insertion은 NOT_TESTED다. 공통 infrastructure 수정·외부 게시·git commit 없음. 실행 환경에 pandas3.1.0을 추가했으며 모델/캐시는 변경하지 않았다.

![Audit figures](../../outputs/task10c2/audit_figures.png)

[Verification](../../outputs/task10c2/verification.json) · [Matching](../../outputs/task10c2/robot_state_matching.csv) · [Coverage](../../outputs/task10c2/geometry_pair_coverage.csv) · [Expert](../../outputs/task10c2/expert_conditional_action.csv) · [Models](../../outputs/task10c2/model_conditional_action.csv) · [Comparison](../../outputs/task10c2/conditional_response_comparison.csv)
