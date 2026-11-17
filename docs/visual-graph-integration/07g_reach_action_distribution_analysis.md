# Task07-G — Expert-State vs Autonomous Reach–Descend Action Analysis

2026-10-26 KST. **정상 Expert 관측에서도 Reach의 방향 전환·작은 위치 보정에 모방 잔차가 나타났다. 자율 접근에서는 t19 이후 목표 반대 radial 명령과 실제 거리 증가가 누적됐다.**
Expert 상태에서 정확하게 모방하다가 자율 상태에서만 실패한다는 설명은 이번 Seed의 기록과 맞지 않는다.
H1과 H2가 함께 지지되지만 BC Loss·Graph 정보·GRU History의 인과 기여율은 **INCONCLUSIVE**다.

ExperimentLogger UUID: **`de3e7e82202045238aa4c2526fd82d0a`**. 단일 Seed41005의 저장 자료 진단이며 새로운 성공률 평가가 아니다.

[Static HTML](../../experiments/de3e7e82202045238aa4c2526fd82d0a/reach_action_distribution_analysis.html) · [정량 JSON](../../experiments/de3e7e82202045238aa4c2526fd82d0a/analysis.json) · [검증](../../experiments/de3e7e82202045238aa4c2526fd82d0a/verification.json) · [Source/Input SHA-256](../../experiments/de3e7e82202045238aa4c2526fd82d0a/source_hashes.json)

## 자료·시간·측정 계약

- Expert: 07-E `c76d3cf8f8354c7199e13f3068d2d1c3` / A. t0..25 Reach, t26..39 Descend. 실제 Expert Action40개와 같은 관측/실제 Expert History의 Student shadow Action40개를 비교했다. Shadow 첫 음수 gripper는 t40이며 t0..39에는 없다.
- Autonomous: 07-F `a878851dd4ee4b7a853502915115454c` / A, `PREDICTED`, 전체250 Action. 집중 접근 구간 t0..34와 이후 t35..249를 별도 집계했다. B/C Oracle은 기존 결과만 참고했다.
- 07-C `268e3b8cc21943fe97d554d69249a495`와 07-D `d046e3a679a3430aab5e69f279c3f0a7`의 결과/manifest를 확인했다. 07-D의 정상 접근 뒤 Model Close 성공과 07-E A의 성공은 Close 자체보다 접근 상태를 먼저 볼 근거다. 새로운 intervention을 실행하지 않았다.
- Checkpoint: training pipeline `2be5e62357a44b14a5651a1a346edc7a`, execution `002abd72baf84c4da185fd7134c3146f`, epoch39/global step8959, SHA-256 `3f4ded2101c91b103760e07cc100fe480866778d0309fddac4a0efb83116d776`.
- 로컬 read-only bundle의 checkpoint 및 config/split/dataset/vision manifest identity를 기존 `load_bundle`로 검증했다. Train96 normalizer는 07-C 저장 mean/std와 exact이며 fit/refit하지 않았다. 원본 학습 run 디렉터리 대신 검증된 bundle과 training metrics 복사본을 사용했다.
- 계약은 **Observation_t → Action_t → Observation_(t+1)**. 그림의 거리 e_t는 적용 전 관측, Δe_t는 Action_t를 실제 적용한 뒤의 거리 변화다. t19의 Action으로 생긴 증가는 t20에서 관측된다.
- 서로 다른 두 궤적의 같은 t Action을 빼지 않았다. Action prediction error는 Expert 관측 안에서만 계산한다. Autonomous Expert 정답, Stage, Counterfactual Expert Action을 생성하지 않았다.
- Controller는 설치 ManiSkill3.0.1 `PDEEPoseController._clip_and_scale_action`을 config-only stub에 직접 호출했다. Translation component clip ±1 후 ±0.1m, rotation **벡터 norm clip1 후 −0.1 곱**, intrinsic XYZ Euler matrix로 회전한다. Raw rotation Z를 실제 TCP 각도 오차로 해석하지 않는다. Finger joint target은 clip(gripper) ±1→−0.01..0.04m다.
- Applied-space는 controller 전처리 명령 공간이며 실제 TCP 변위가 아니다. 회전 command geodesic은 두 Euler 회전행렬의 상대각이며 실제 달성 orientation 차이와 구분한다.
- Robot76 base world quaternion offset66..69로 root translation을 world로 회전한 뒤 GT target 방향과 비교했다. Peg-relative target은 저장된 초기 grasp-relative transform을 현재 Peg pose에 합성한다. GT pose/target은 metric에만 사용했다.
- 평균 MSE는 프레임×해당 성분 수를 분모로 사용했다. 방향 cosine은 translation norm0에서 미정의로 기록하며 임의로1을 주지 않는다. 이번40개는 모두 정의됐다. 성분별 MAE/MSE/p95/max와 벡터 오차 분포는 JSON/CSV/NPZ에 모두 보존했다. 결과로 임의 PASS/FAIL 경계를 만들지 않았다.

## A. 정상 Expert 상태에서 Action 모방

| 지표 | Reach t0..25 (26) | Descend t26..39 (14) |
| --- | ---: | ---: |
| Action7 MSE | 0.0115518 | 0.0009738 |
| Action7 MAE | 0.0482608 | 0.0160522 |
| Translation XYZ MSE | 0.0014255 | 0.0002261 |
| Translation XYZ MAE | 0.0251003 | 0.0094314 |
| Rotation native XYZ MSE | 0.0127748 | 0.0004760 |
| Rotation native XYZ MAE | 0.0536525 | 0.0147081 |
| Gripper MSE | 0.0382617 | 0.0047107 |
| Gripper MAE | 0.1015673 | 0.0399469 |
| Translation direction cosine 평균 | 0.500 | 0.974 |
| Controller translation vector error 평균 mm | 5.193 | 1.939 |
| Controller rotation command geodesic 평균 ° | 0.191 | 0.188 |
| Controller gripper joint target error 평균 mm/finger | 2.521 | 0.501 |

Reach translation 벡터 명령 오차는 평균 **5.193mm**, p95 **10.456mm**, 최대 **17.945mm(t13)**다.
Descend는 평균 **1.939mm**, 최대 **6.924mm(t26)**다.
Expert가 실제로 도달한 t40 target 잔차1.760mm와 비교하면 mm 단위 명령 잔차는 무시하기 어렵지만, **명령 잔차를 그대로 누적된 TCP 오차나 실패 확률로 환산할 수 없다**.

Reach의 cosine 중앙값은0.936, 평균은0.500다. 다수 초기 명령은 잘 정렬되지만, 보정 전환의 반대 방향 때문에 평균이 낮아졌다.
양의 dot는 Reach20/26, Descend14/14이다. 정확한 모방의 PASS 판정이 아닌 기술 통계다.

- 초기 t0에서 Expert Z−17.525mm 대비 shadow−9.394mm로 하강량이 작다. t1..11은 주 이동 방향을 잘 재현한다.
- **t13에서 Expert가 Y/Z 양의 보정으로 전환했지만 Student는 Y/Z 음의 이동을 유지했다.** 같은 관측/History의 벡터 오차17.945mm, cosine−0.946이다. t14..15에도 크기·방향 잔차가 남는다.
- t18..25의 작은 Expert 보정에서도 Student가 큰 음의 Y/Z 명령을 내는 프레임이 있다. 예를 들어 t23 Expert norm0.788mm에 Student norm7.722mm, t24 Expert norm0.616mm에 Student norm7.294mm다. 정확한 정지0 명령은 t0..39에 없으므로 이 구간을 `정지 실패`로 단정하지 않고 **작은 보정/감속의 모방 잔차**로 기록한다.
- Reach→Descend의 t26은 Expert 약−10mm 하강에 shadow−3.491mm이며 Y−2.162mm가 남는다. 이후 t30..38의 주 하강 방향은 대체로 일치한다. 수평 보정은 초기 Descend에서 더 크다.
- t39 Expert Z−2.626mm로 감속할 때 shadow−5.836mm와 Y+1.739mm가 남는다. 모델이 접근 완료 직전 감속을 완전히 재현하지 못했다.
- 모든 Expert Reach/Descend gripper 정답은+1이다. Shadow도 t0..39에 양수이며 **조기 음수 Close는 없다**. 다만 t15에는 raw+0.261326으로 낮아져 가상 목표가21.533mm/finger다(Expert40mm). 즉 음수 전환이 없다는 사실과 조기 부분 닫기 명령 경향은 구분해야 한다. t39는+0.77118로 덜 열린 가상 목표34.279mm/finger를 낸다. 실제 Expert는40mm/finger를 유지했으므로 해당 virtual command로 환경이 움직였다고 주장하지 않는다.

| Expert 관측 t / Stage | Expert XYZ 명령 mm | Student 가상 XYZ 명령 mm | 벡터 오차 mm | 방향 cosine |
| --- | --- | --- | ---: | ---: |
| 0 / reach | 2.365/-9.342/-17.525 | 3.993/-9.657/-9.394 | 8.298 | 0.940 |
| 12 / reach | 0.994/-6.699/-12.578 | 0.914/-6.927/-15.747 | 3.178 | 0.997 |
| 13 / reach | 1.540/3.136/5.177 | 0.672/-4.834/-10.877 | 17.945 | -0.946 |
| 14 / reach | 6.463/6.194/9.374 | 0.233/-1.787/7.994 | 10.219 | 0.615 |
| 15 / reach | 1.576/2.558/2.050 | 4.799/-1.934/11.017 | 10.534 | 0.569 |
| 18 / reach | 0.428/-0.942/0.691 | -3.946/-2.216/-7.205 | 9.117 | -0.433 |
| 19 / reach | 1.588/1.357/2.190 | -1.338/-2.252/-4.921 | 8.494 | -0.946 |
| 22 / reach | -0.579/-0.831/-0.780 | -1.473/-3.870/-6.328 | 6.388 | 0.931 |
| 23 / reach | 0.011/-0.785/0.070 | -1.696/-4.602/-5.964 | 7.341 | 0.522 |
| 24 / reach | 0.347/-0.019/0.508 | -1.406/-4.754/-5.351 | 7.734 | -0.694 |
| 25 / reach | 0.259/0.739/0.314 | -1.595/-1.957/-3.549 | 5.062 | -0.809 |
| 26 / descend | 0.003/0.056/-10.000 | -0.809/-2.162/-3.491 | 6.924 | 0.831 |
| 27 / descend | -0.239/0.066/-9.997 | 0.357/-3.448/-9.593 | 3.587 | 0.937 |
| 28 / descend | -0.265/0.070/-9.996 | 1.247/-2.168/-10.114 | 2.704 | 0.966 |
| 36 / descend | -0.267/0.121/-9.996 | -0.069/0.695/-9.822 | 0.632 | 0.998 |
| 39 / descend | -0.065/0.053/-2.626 | 0.576/1.739/-5.836 | 3.683 | 0.957 |

| 실제 전처리 발생 | Expert Reach | Shadow Reach | Expert Descend | Shadow Descend |
| --- | ---: | ---: | ---: | ---: |
| Translation component clip | 0 | 0 | 0 | 0 |
| Rotation norm clip | 24 | 24 | 0 | 0 |
| Gripper clip | 0 | 2 | 0 | 8 |

Rotation native MSE는 Reach0.012775, Descend0.000476지만, 전처리 후 command geodesic 평균은 각각0.191°/0.188°다.
초기 Z축 native magnitude 차이는 norm clipping 때문에 상당 부분 포화된다. t25 회전 command 차이1.239°도 있어 전환부 잔차가0은 아니다. 이것은 실제 TCP 회전 실패의 독립 인과 증거가 아니다.

**분류: `ON_EXPERT_STATE_IMITATION_ERROR` / `CONFIRMED_OBSERVATION`.** 원인이 BC Loss인지, 관측/Graph 정보인지, GRU인지 이 진단만으로 분리하지 못한다.

![Expert native Action 비교](../../experiments/de3e7e82202045238aa4c2526fd82d0a/expert_native_actions.png)

![Expert controller command 비교](../../experiments/de3e7e82202045238aa4c2526fd82d0a/expert_controller_commands.png)

## B. 자율 접근 진행률과 t19 이후

| 적용 Action t (마지막 결과 관측) | e 시작→끝 mm | 거리 증가 step 수 | 명령 cosine 평균 | 명령 norm 평균 mm | 회전 오차 시작→끝 ° |
| --- | --- | ---: | ---: | ---: | --- |
| 0_10 (t11) | 141.453→75.145 | 0/11 | 0.879 | 18.486 | 58.302/33.356 |
| 11_18 (t19) | 75.145→56.400 | 0/8 | 0.422 | 13.719 | 33.356/15.464 |
| 19_26 (t27) | 56.400→84.967 | 8/8 | -0.608 | 15.136 | 15.464/2.576 |
| 27_34 (t35) | 84.967→107.129 | 6/8 | -0.292 | 17.970 | 2.576/2.255 |
| 35_249 (t250) | 107.129→254.002 | 121/215 | -0.418 | 12.758 | 2.255/64.192 |

표의 끝 값은 마지막 Action 후 관측이다. 예를 들어 t11..18은 e11→e19, t19..26은 e19→e27이다.
자율 최소 e19=**56.400mm** 뒤 Action19..26 **8/8 step**에서 실제 거리가 증가해 e27=**84.967mm**가 된다.
같은8개 명령의 목표방향 cosine은 모두 음수(평균−0.608)이며, radial 명령은 평균−9.329mm/step이다.
실제 TCP 순이동은 World **+38.025/−21.612/−9.520mm**다. 목표에 필요한 수평 방향과 어긋나는 이동이 누적돼, 추가 하강에도 전체 위치 거리는+28.567mm 증가했다.

| 자율 관측 t | World XYZ 명령 mm | 목표방향 cosine | e_t mm | 실제 Δe_t mm | 회전 오차 ° | Raw gripper |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.993/-9.657/-9.394 | 0.872 | 141.453 | -2.448 | 58.302 | 0.961 |
| 10 | 3.399/-10.559/-14.206 | 0.800 | 80.871 | -5.726 | 35.595 | 0.991 |
| 13 | 4.061/-8.663/-9.502 | 0.568 | 65.736 | -3.280 | 28.894 | 1.015 |
| 18 | 8.830/-7.253/-8.443 | 0.069 | 56.549 | -0.149 | 17.722 | 1.036 |
| 19 | 9.752/-7.659/-8.265 | -0.075 | 56.400 | 0.649 | 15.464 | 1.037 |
| 20 | 10.510/-7.944/-6.959 | -0.272 | 57.049 | 1.572 | 13.204 | 1.023 |
| 23 | 11.848/-7.202/-0.862 | -0.791 | 64.525 | 4.079 | 6.458 | 0.961 |
| 24 | 12.926/-6.500/0.461 | -0.850 | 68.604 | 4.736 | 4.236 | 0.917 |
| 25 | 14.742/-5.158/2.066 | -0.869 | 73.340 | 5.361 | 2.059 | 0.826 |
| 26 | 17.925/-3.214/5.082 | -0.859 | 78.701 | 6.265 | 0.737 | 0.549 |
| 27 | 21.011/2.084/8.859 | -0.754 | 84.967 | 7.097 | 2.576 | -0.335 |
| 28 | 16.889/5.243/2.927 | -0.603 | 92.064 | 5.983 | 3.819 | -0.807 |
| 34 | 0.853/18.919/-5.106 | 0.345 | 109.669 | -2.540 | 0.376 | -0.960 |

회전 오차는 t19 **15.464°**에서 t26 **0.737°**로 감소하고, t27에는 **2.576°**다.
따라서 이 구간에 **실제 회전 정렬의 진행과 위치 접근의 후퇴가 동시에 관측**됐다. 거리 증가만 보고 모든 Action이 잘못됐다고 판정하지 않는다.
그러나 e19에서 이미56.4mm 남은 상태에서 반대 radial 이동이8step 지속되고, t26 Z 명령까지+5.082mm로 바뀌며, t27에는 Z+8.859mm와 함께 닫는다. 실제 손가락 배치도 불리하므로 **회전 정렬을 하면서 파지 위치를 완성하지 못했다**는 설명이 지지된다. 자율 행동에 Expert Descend/Lift Stage를 붙이지 않았다. Translation norm 변화와 연속 명령 사이의 방향 변화각도 `reach_progress.csv/npz`에 기록했다. 별도 사전 임계값이 없는 `유의미한 최초 크기 변화` 시점의 단일 판정은 INCONCLUSIVE로 남긴다.

최초 전환은 다음처럼 구분한다.

1. **행동 변화**: t11..18에 목표방향 cosine이0.753→0.069로 감소한다. t19는 radial 명령의 첫 부호 반전이다. 이는 사전 정의한 물리적 방향 지표이며 임의의 `유의미한 Action 차이` 임계값을 추가하지 않았다. XYZ 및 step별 명령 변화 원자료로 전환 구간을 제시한다.
2. **최초 거리 증가**: Action t19 후, 즉 관측 t20에서+0.649mm. t19 자체가 최초 원인이라는 뜻은 아니다.
3. **실제 파지가 어렵다고 확인된 상태**: 첫 Close 직전 t27. 07-E의 같은 자율 prefix 뒤 Full Close 실패라는 기존 물리 증거가 있다. 더 이른 시점의 인과적 실패 확정은 **INCONCLUSIVE**다.

Peg의 Action19..26 순변위는0.000205mm 미만이고, world/현재 Peg-relative target 거리의 Close 이전 최대 차이는 **0.000347mm**다.
따라서 이 구간의28.567mm 거리 증가는 Peg 이동에 의해 움직인 목표를 정적으로 평가한 착시로 설명되지 않는다. 현재 Peg-relative metric도 같은 추세를 보인다.

t27의 위치 차이는 World +60.157/−49.120/+34.462mm, Peg-frame −77.543/+4.351/+34.462mm다.
Opening은68.317mm, fingertip 중심→Peg 표면은33.205/40.499mm이며, pad 중심 segment가 Peg와 교차하지 않는다.
정상 Expert t40은1.760mm/0.157°이며 segment가 교차한다. 이는 유한 pad surface clearance나 모든 유효 파지 Pose의 완전한 배제 검사가 아니다.

t27..34는 음수 gripper8개를 적용하며 추가로22.162mm 멀어졌다. Hidden update norm 평균은 접근 t19..26의0.362에서1.927로 커졌다. 실제 관측/행동 전환의 동반 변화이며 GRU 오류로 판정하지 않는다.
전체250step에서 파지·Peg lift가 관측되지 않았다는 기존 trace와도 일치한다.

![자율 접근·Action·실제 state 변화](../../experiments/de3e7e82202045238aa4c2526fd82d0a/autonomous_progress.png)

## C. 상태 분포와 History 차이

| 기술 통계 (각자의 접근 구간) | Expert t0..39 | Autonomous t0..26 |
| --- | --- | --- |
| Peg-frame TCP XYZ 평균 mm | -61.822/-5.538/56.394 | -81.146/-3.677/73.072 |
| World TCP velocity XYZ 평균 m/s | -0.003/-0.022/-0.064 | 0.030/-0.067/-0.071 |
| Opening 평균 mm | 80.000 | 79.335 |
| Peg/Box point count 평균 | 182.325/96.125 | 178.704/106.222 |
| Graph node/edge count 평균 | 281.450/1492.800 | 287.926/1574.000 |
| Hidden norm 평균 | 4.389 | 4.632 |
| Hidden update norm 평균 | 0.690 | 0.573 |

이 표는 **기간/진행도가 다른 접근 창의 기술 통계**다. 같은 t, 같은 phase 또는 같은 분포를 의미하지 않으며 통계적 모집단 검정이나 학습 OOD 판정이 아니다.
Expert 접근의 Peg-frame X 범위는−68.730..−32.204mm, 자율 t0..26은−138.645..−32.204mm다. 자율 궤적이 Peg의 다른 수평 상대 위치로 이동했다.
정상 Close t40과 자율 첫 Close t27의 Peg-frame 상대 위치 차이는 약−78.353/+3.876/+35.950mm다. 이 상태 차이를 Action prediction error라고 부르지 않았다.

접근 진행도도 별도 비교했다. 각 자율 t를 **목표 거리만 가장 가까운 Expert t0..40**에 대응시킨 자료를 `state_hidden_comparison.json`에 보존했다.
예를 들어 자율 t19와 Expert t12는 target 거리 차이2.160mm지만 상대 위치 벡터 차이는 **38.645mm**, 상대 orientation 차이는 **15.621°**, hidden 벡터 거리는1.244다.
같은 scalar 거리도 같은 geometry/history를 보장하지 않는다. 이 짝에서 Action 차이를 정답 오차로 계산하지 않았고 다른 Hidden을 이식하지 않았다.

양 궤적의 접근 구간 모두 Peg/Box Presence=1이다. Point count·F5 valid fraction·edge count는 연속적으로 달라진다. 기존 Mask montage에서도 각자의 관측 시야가 달라지지만 새로운 segmentation/geometry 처리를 실행하지 않았다.
07-F의 기존 t27 geometry는 Peg p95 표면 오차50.179mm의 비대상 outlier를 보였고, Oracle은 그 오염을 제거했어도 해당 seed의 접근/파지를 회복하지 못했다.
**Point count/Presence가 유지됐다는 사실은 정보 충분성의 증거가 아니며, Oracle의 실패도 Vision이 무관하다는 증거가 아니다.**

Hidden은 각각의 t0부터 순차 저장된 post-observation/pre-action 값이다. Update norm은 h_t−h_(t−1), t0은 zero hidden 대비다. 다른 History의 Hidden 차이는 정상일 수 있다.
제한적 offline forward에서 저장 Graph와 같은 Robot76/normalizer를 사용해 GNN readout도 재계산했다. Expert t0..40, 자율 t0..34만 범위로 삼았다. 같은 시점의 readout 차이만으로 표현 손실을 판정하지 않는다.

- expert t0..40: Action 최대 절대차 2.742e-06, Hidden 7.510e-06; 고정 허용치 통과=False, 같은 CPU Runtime 반복은 exact.
- autonomous t0..34: Action 최대 절대차 4.172e-06, Hidden 7.331e-06; 고정 허용치 통과=False, 같은 CPU Runtime 반복은 exact.

새 추론에 Expert Action/Stage/GT pose/goal/contact는 들어가지 않았다. 보존 readout은 독립 Graph probe나 decoder 학습 결과가 아니다. CPU 반복은 exact였지만 원래 CUDA 저장값과의 고정 tolerance는 모든 원소에서 통과하지 못했다. 허용치를 완화하지 않았다. Expert t38 rotation Z에서 약1.505e-6 차이와 여러 Hidden 원소의 초과가 있어 재계산 Action/Hidden으로 원본을 교체하지 않았다. GNN readout은 이 재계산의 진단용 배열로만 보존하고 가설 판정이나 검증된 상태 분포 근거에는 사용하지 않았다. 핵심 분석의 모든 Action/Hidden은 원본 저장값이다.

![각자의 실제 TCP 경로](../../experiments/de3e7e82202045238aa4c2526fd82d0a/tcp_paths_3d.png)

![각자의 상대 geometry·point·hidden timeline](../../experiments/de3e7e82202045238aa4c2526fd82d0a/state_history_comparison.png)

![보존 RGB와 Mask — 각자의 상태](../../experiments/de3e7e82202045238aa4c2526fd82d0a/stored_mask_montage.png)

## 가설과 Q1–Q8 직접 답변

| 질문 | 상태 | 실제 증거와 해석 |
| --- | --- | --- |
| **Q1 정상 Expert Reach–Descend Action은 얼마나 정확한가?** | `CONFIRMED_OBSERVATION` | Reach/Descend Action7 MSE0.0115518/0.00097384, translation command 벡터 오차 평균5.193/1.939mm. 초기 주 방향과 대부분 Descend는 모방하지만 국소 보정 오차가 크다. 충분성에 임의 PASS/FAIL을 붙이지 않는다. |
| **Q2 어디에서 큰가?** | `CONFIRMED_OBSERVATION` | 가장 큰 translation 오차는 Reach t13(17.945mm), 다음 t15/14/18/19. 전환 t26에도6.924mm. Descend 전체보다 Reach 보정과 Reach→Descend 시작이 두드러진다. |
| **Q3 방향·감속·정지·회전·gripper 중 무엇인가?** | `CONFIRMED_OBSERVATION` | t13 방향 반전 누락, t18..25 작은 보정 때 큰 이동 잔차, t26 하강 부족, t39 감속 부족. 회전 native 잔차는 norm clip 뒤 작아지는 부분이 크다. 조기 음수 gripper는 없지만 t15의 가상 finger target21.533mm로 부분 닫기 경향은 있다. 정확한 정지 target은 분석40frame에 없어 정지 실패 단정은 불가. |
| **Q4 왜 t19 뒤 멀어지는가?** | `CONFIRMED_OBSERVATION` + `SUPPORTED_HYPOTHESIS` | t19..26 목표 반대 radial 명령8개와 실제 거리 증가8개, 수평+38.025/−21.612mm, e+28.567mm. 회전은 개선돼 위치·회전의 불균형이 관측된다. 이를 최초 인과 원인으로 확정할 수는 없다. |
| **Q5 상대 geometry 차이를 모방 오차와 구분했는가?** | `CONFIRMED_OBSERVATION` | 정상 Close1.760mm와 자율 Close84.967mm, Peg-frame 수평/높이 차이와 손가락 배치를 비교했다. 같은 target 거리의 t19/t12도 상대 위치38.645mm 차이다. 서로 다른 궤적의 Action은 prediction error로 빼지 않았다. |
| **Q6 기본 Action Learning 부족은 얼마나 지지되는가?** | `SUPPORTED_HYPOTHESIS` | **`SUPPORTED_ON_EXPERT_STATE_IMITATION_ERROR`**. 실제 정상 관측에서 방향·크기·감속 오차가 직접 측정돼 H1의 근거가 강하다. 순수 BC Loss 원인, 추가 epoch 효과, 각 원인 기여율은 `INCONCLUSIVE`. |
| **Q7 Closed-loop/History 근거와 한계는?** | `SUPPORTED_HYPOTHESIS` | **`SUPPORTED_DISTRIBUTION_SHIFT_HYPOTHESIS`**. 같은 reset/t0에서 실제 state/history가 갈라지고, 자율은 상대 수평 위치를 이탈하며 닫는다. 오류 누적 설명과 일치하지만 정상 state에서도 잔차가 있어 H2 단독 설명은 부족하다. GRU 자체 오류·state와 history의 독립 기여는 `INCONCLUSIVE`. |
| **Q8 다음 우선순위는?** | `SUPPORTED_HYPOTHESIS` | **기존 BC의 Reach 보정/감속·전환 학습 검증을 먼저 수행할 근거가 가장 강하다.** 동일 Expert 관측의 큰 잔차가 직접 증거다. 기존 저장 train/validation에서 해당 구간의 target coverage·sampling·phase별 validation을 먼저 감사하고 그 결과로 학습 개선 비교를 정한다. 무조건 epoch/loss/architecture 변경은 지지하지 않는다. 이후 최소 Expert-prefix handoff로 오류 누적/회복 가능성을 검사한다. Graph probe는 아래 정보 원인 분리가 필요할 때 다음 순서다. |

H3 판정은 **`INPUT_REPRESENTATION_REQUIRES_FURTHER_TEST` / `INCONCLUSIVE`**다. Geometry/readout 변화와 Action 변화의 동반 관측 및 기존 Oracle 결과는 추가 검사 필요성을 지지한다. 입력 정보 손실이 원인임을 확정하지 않는다.
H1/H2/H3는 배타적이지 않으며 현재 자료에서 기여율은 **`INCONCLUSIVE`**로 남긴다.

최소 후속 근거는 두 층으로 나뉜다. 먼저 기존 train/validation Expert 기록에서 Reach t13과 같은 보정 반전·작은 명령 및 Descend 시작의 잔차가 반복되는지 stage별로 검사한다. 이것은 새 rollout 없이 BC 개선의 표적과 coverage 부족을 좁힌다.
그 다음 인과 분리를 위해서는 별도 승인한 **소수 prefix 길이의 Expert-prefix handoff**가 최소 물리 비교다. 매번 fresh reset, t0부터 실제 Expert history를 누적한 뒤 현재 policy로 넘겨 접근 회복 여부를 비교해야 한다. 시작점을 t19라는 최초 실패로 전제하지 않고, 보정 전/후 경계를 비교한다.
이후 자율 상태의 Expert label 검증/회복 demonstration은 해당 실제 상태에서 필요한 action을 알기 위한 추가 증거다. Graph information probe는 그런 조건에서도 입력에서 필요한 상대 geometry를 회수할 수 있는지 분리할 때 의미가 있다.
이번에는 handoff, 새 expert label, recovery data, decoder/probe 학습, BC 개선 학습을 **모두 `NOT_EXECUTED`**로 유지했다.

## 기존 Training/Test 지표와 범위

기존 epoch39 train Action7 MSE **0.0027367143**, validation **0.0050171197**, 단일 최종 offline test **0.0043559006**(1524 targets)이다.
Test translation MSE0.0005544628, rotation MSE0.0063099188, gripper MSE0.0098981597다. 원본 share manifest SHA와 대조한 복사본을 읽었으며 test를 재실행하지 않았다.
이번 Seed의 Reach MSE는 해당 전체 test 평균보다 크지만, phase/seed/frame 분포가 달라 동일 모집단의 성능 저하율로 환산할 수 없다.
원본 frozen test reset 재현 한계와 offline test Action metric은 다른 문제다. 07-B에서 제외한 closed-loop frozen-test 성공률을 이번 offline test 수치로 대신하지 않는다.

## 계약·보존·실행 검증

- Source run manifest에 등록된 실제 파일을 사용했고 존재하지 않는 자료를 보완하지 않았다. Expert Action은 수집 당시 controller entry와 exact이며, source prefix의 Expert Action hash40개/관측+Graph hash41개/Shadow Action+Hidden41개도 exact다. Positive Control과 Expert A의 prefix도 별도 대조했다.
- Expert88/Autonomous250의 Action/Robot76/물리 before-after clock, controller scaled command, packet timestep/episode/source index/EE pose를 검사했다. Hidden은 t0만 reset하고 연속 timestep을 유지하는 source metadata와 offline forward로 확인했다.
- 07-F A의 전체250 Action은 기존41005 저장 Action과 exact다. 같은 reset의 RGB-D/calibration/Robot76와 최초 Action/Hidden도 exact다. World raw pose/qpos/qvel 시간 연결은 exact이며, 사후 재정규화된 derived Peg-relative quaternion만 기존 물리 trace 허용치 atol1e-6/rtol0으로 검사했다.
- Quaternion 상대 회전은 부호 불변이다. Translation과 velocity는 m/m·s⁻¹, 요약 위치는mm, orientation은deg/rad를 구분했다. Controller 전처리는 원본 audit에 atol2e-8/rtol1e-6, gripper atol1e-8/rtol0으로 일치했다.
- 신규 수치 계약 테스트 **7개 PASS**. 실제 데이터 계약은 `contract_checks.log`, forward의 고정 tolerance 미통과와 local exact 반복은 `offline_forward.json`, decode/link/protection은 `verification.json`에 기록했다.
- 기존 Source/docs, 07-C/D/E/F 로컬 tree, checkpoint bundle, 원본 training/41005 review copy의 실행 전 SHA-256과 실행 후 값을 대조했다. 변경0개다. 신규 분석 entrypoint/report/test/이 문서와 새 run만 추가했다. 기존 dirty/untracked 파일은 유지했다.
- Checkpoint weight/normalizer 메모리 hash도 forward 전후 동일하다. 학습 원본 디렉터리 전체/원격 서버 전체를 새로 감사한 것은 아니며 **로컬 보존 범위**만 주장한다.
- CPU array 분석과 config-only controller 전처리, 제한적 CPU GNN–GRU forward만 실행했다. **Simulator reset/step0, optimizer step0, Vision/Graph 재생성0, dependency 설치0, git commit0, upload/publishing0**이다. macOS simulator, MPS, CUDA forward 및 CUDA PhysX는 **`NOT_EXECUTED`**다. 기존 다른 process를 중지하지 않았다.

## 보존 산출물과 재현

- [Input manifest](../../experiments/de3e7e82202045238aa4c2526fd82d0a/input_manifest.json) · [Source hashes](../../experiments/de3e7e82202045238aa4c2526fd82d0a/source_hashes.json) · [원본 불변 검사](../../experiments/de3e7e82202045238aa4c2526fd82d0a/protection.json)
- [Action comparison CSV](../../experiments/de3e7e82202045238aa4c2526fd82d0a/action_comparison.csv) · [NPZ](../../experiments/de3e7e82202045238aa4c2526fd82d0a/action_comparison.npz)
- [Reach progress CSV](../../experiments/de3e7e82202045238aa4c2526fd82d0a/reach_progress.csv) · [NPZ](../../experiments/de3e7e82202045238aa4c2526fd82d0a/reach_progress.npz)
- [State/History JSON](../../experiments/de3e7e82202045238aa4c2526fd82d0a/state_hidden_comparison.json) · [NPZ](../../experiments/de3e7e82202045238aa4c2526fd82d0a/state_hidden_comparison.npz)
- [핵심 시점 diagnostic](../../experiments/de3e7e82202045238aa4c2526fd82d0a/key_diagnostics.json) · [Offline forward 검증](../../experiments/de3e7e82202045238aa4c2526fd82d0a/offline_forward.json) · [Forward/Readout NPZ](../../experiments/de3e7e82202045238aa4c2526fd82d0a/offline_forward.npz)
- [Training/Test reference](../../experiments/de3e7e82202045238aa4c2526fd82d0a/training_metrics_reference.json) · [Test log](../../experiments/de3e7e82202045238aa4c2526fd82d0a/test_log.txt) · [데이터 계약 log](../../experiments/de3e7e82202045238aa4c2526fd82d0a/contract_checks.log)
- [설치 Controller Source](../../experiments/de3e7e82202045238aa4c2526fd82d0a/installed_controller_source.txt) · [Implementation diff](../../experiments/de3e7e82202045238aa4c2526fd82d0a/implementation.diff) · [실행 환경](../../experiments/de3e7e82202045238aa4c2526fd82d0a/execution.json)

기존 Python 환경에서 `WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m graph.integration_baseline.reach_action_distribution_analysis`로 동일 저장 자료를 분석한다.
사전 보호 snapshot(`/tmp/task07g-protected-before.json`)은 현재 세션의 실행 전 기준이다. 다른 세션에서는 기존 source/입력 파일의 SHA-256 snapshot을 먼저 생성해 사용해야 한다. 각 실행은 새 Logger UUID를 생성한다. 선행 forward 시도는 고정 tolerance 미통과 assertion으로, 다음 시도는 보고서 렌더링 문법 오류로 종료됐다. 두 failed Logger UUID와 로그는 `prior_attempts.json`에 보존했으며 문법 오류를 수정한 새 UUID에서 최종 분석을 완료했다. 최종 분석에서는 해당 차이를 제한사항으로 기록하고 핵심 저장 자료 분석을 완료했다.
추가 학습이나 물리 실험은 자동으로 이어지지 않는다.
