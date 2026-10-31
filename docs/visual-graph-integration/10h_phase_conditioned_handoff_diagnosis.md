# Task 10-H — Phase-conditioned expert-prefix handoff 진단


**해석 범위:** H2는 원본 Expert의 align→insert 경계이며 peg tip은 hole 앞105–142 mm에 있었다. 축/lateral 정렬은 제공됐지만 접촉 직전 몇 mm만을 시험한 조건은 아니다. 아래 독립 분석에 설치 패키지와 실제 Anchor 근거를 명시했다.

기존 B step9000을 freeze하여 평가했다. 새 학습·checkpoint 재선택·GT policy 입력은 없다. 독립 Parent108200–108211의 E/B0/H1/H2 전체48회를 보존했다.

| 조건 | Grasp | Lift | Alignment | Insertion (전체12) | 유효 Anchor | Anchor 조건부 Insertion |
|---|---:|---:|---:|---:|---:|---:|
| E | 12/12 | 12/12 | 12/12 | 12/12 | 12 | 12/12 |
| B0 | 3/12 | 3/12 | 2/12 | 0/12 | 12 | 0/12 |
| H1 | 12/12 | 12/12 | 1/12 | 0/12 | 12 | 0/12 |
| H2 | 12/12 | 12/12 | 12/12 | 6/12 | 12 | 6/12 |

H1/H2의 Grasp/Lift에는 Expert prefix 성과가 포함된다. 최종 insertion은 유효 handoff 이후 성공만 계산했다. ANCHOR_UNREACHABLE Parent를 교체하지 않았다. Wilson95%와 exact discordant test는 CSV에 기록했으며 독립 표본 단위는 Parent다. 한 checkpoint와12 Parent 결과로 일반화하지 않는다.

## 실행 계약

B SHA256: `8ab82f390dccf054b0bd6539dbd55f7a60af2ae7934f53234702ab66c1369ea3`. F1 초기 SHA와 구분한다. CPU PhysX, CUDA renderer/vision/policy,20Hz,250 Action을 사용했다. 로컬 macOS CUDA 및 GPU PhysX는 실행하지 않았다.

E/H1/H2는 reset부터 Expert를 실제 실행했고 모든 관측을 독립 causal tracker와 B GRU로 처리했다. Handoff 관측은 한 번 forward하며 직전 hidden을 그대로 이어간다. 초기 snapshot/RGB-D/Graph/Robot76 및 공통 prefix의 state/input/hidden/action identity를 검사했다. PhysX 내부 solver 전체의 bit-exact 복원을 주장하지 않는다.

ScriptedExpert.act는 현재 phase의 명령을 생성하면서 다음 phase를 설정한다. 따라서 H1/H2는 마지막 lift/align 명령을 실행한 다음 관측이다. H2는 goal*Pose([-0.02-half_len,0,0])에 대한 peg 위치<2mm/회전<.02rad, 파지·lift를 실제 검사한다. 단순 timestep anchor가 아니다. Expert의 phase는 B 입력에 전달되지 않는다. B 이후 phase 열은 GT Expert shadow의 진단 상태다.

## 연구 질문

Q1. B0 insertion=0/12, H1=0/12. 관찰된 차이를 paired CSV로 평가한다.

Q2. H2 insertion=6/12. 유효 Anchor 조건부 분모는 위 표와 같다.

Q3–Q4. H2의 성공 증가가 관찰됐다. Alignment 구간을 후속 진단의 우선 대상으로 볼 근거가 있으나 Geometry와 BC fit 중 원인을 분리하지 못한다.

Q5 (H1). 첫10 step 중 direction reversal=0/120, error 증가=0/120 frame. 즉시 반응과 누적 오차의 Parent별 원자료는 first10 CSV와 trajectory figure에 보존했다. Frame 비율은 통계적 독립 표본이 아니다.

Q5 (H2). 첫10 step 중 direction reversal=35/120, error 증가=28/120 frame. 즉시 반응과 누적 오차의 Parent별 원자료는 first10 CSV와 trajectory figure에 보존했다. Frame 비율은 통계적 독립 표본이 아니다.

| Handoff | Parent | 첫 action 방향 반대 | 첫 step tip error 증가 | 첫10 step tip error 증가 | 첫10 missing | 첫10 arm saturation | 첫10 open |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1 | 12 | 0 | 0 | 0 | 8 | 11 | 0 |
| H2 | 12 | 1 | 2 | 3 | 4 | 2 | 2 |

이 표는 Parent 단위의 현상 집계다. Tip norm은 축 깊이도 포함하므로 정상 삽입의 진행 여부는 공식 성공 및 lateral error와 함께 판정해야 한다.

Q6. Geometry missing과 GT 상대 오차는 동일 step trace와 phasewise_geometry_error.csv에 기록했다. 동시 발생은 직접 원인의 증거가 아니다.

Q7. Gripper 명령·실제 finger qpos·controller preprocessing·arm saturation·실제 TCP 이동을 기록했다. Failure category는 관찰 현상이며 인과 검증은 아니다. 전체 saturation은 지속적인 gripper=-1도 포함하므로 arm_saturation과 구분한다.

Q8. Pilot 및 Final의 prefix hidden/input/state 연속성 검증을 통과했다. Episode 시작에만 hidden=None이며 handoff reset/중복 forward는 없다.

Q9. 확인한 것은 이 checkpoint의 실제 성공/실패와 trace 현상이다. 인식, BC, 접촉 관측, GRU 상태 분포, 제어 정밀도 간 인과 분해는 이번 실험만으로 완료하지 못했다.

Q10. Alignment 상대 위치 표현과 alignment action correction을 우선 조사할 근거가 있다. Geometry/BC를 구분하는 비교를 먼저 설계하고 이번 Task에서는 실행하지 않았다.

## 지표의 한계와 산출물

Failure label 계산 규칙: grasp True→False=GRASP_LOST; 파지 상태에서 open>0=PREMATURE_GRIPPER_OPEN; 비파지 TCP–Peg>3cm에서 close<0=PREMATURE_GRIPPER_CLOSE; object slot 누락=GEOMETRY_MISSING; arm |a|>.999=ACTION_SATURATION; B/Expert translation dot<0=ACTION_DIRECTION_ERROR; handoff 대비 최종 tip error>5mm 증가=ALIGNMENT_DIVERGENCE; 최종 축 각도 기존 threshold 초과=INSERTION_MISALIGNMENT. TIMEOUT은 horizon/환경 time limit이며 HORIZON_LIMIT는 prefix 포함250 Action 소진이다. 더 긴 horizon이면 성공한다고 주장하지 않는다. Collision은 원본 contact audit를 보존하며 접촉 자체를 실패 원인으로 label하지 않는다. PHASE_TRANSITION_ERROR는 검증 실패 시 실행을 중단한다.

같은 상태의 Expert action은 GT 및 내부 phase를 사용하는 counterfactual이므로 B와의 차이를 순수 인식 오차로 해석하지 않는다. 최종 tip-hole 벡터의 norm은 깊이 성분을 포함하며 성공 predicate를 대체하지 않는다. 전체 quaternion 오차와 기존 축 정렬각을 구분한다.

[Functional CSV](../../outputs/task10h/final_results/functional_success_comparison.csv) · [Paired CSV](../../outputs/task10h/final_results/paired_handoff_success_analysis.csv) · [Anchor](../../outputs/task10h/final_results/handoff_anchor_state_metrics.csv) · [첫10 step](../../outputs/task10h/final_results/handoff_first10_action_metrics.csv) · [검증](../../outputs/task10h/final_results/verification.json)

![functional_success](../../outputs/task10h/final_results/figures/functional_success.png)

![H1_tip_hole_error](../../outputs/task10h/final_results/figures/H1_tip_hole_error.png)

![H2_tip_hole_error](../../outputs/task10h/final_results/figures/H2_tip_hole_error.png)

![H1_first10](../../outputs/task10h/final_results/figures/H1_first10.png)

![H2_first10](../../outputs/task10h/final_results/figures/H2_first10.png)

![failure_distribution](../../outputs/task10h/final_results/figures/failure_distribution.png)

![phase_failure_distribution](../../outputs/task10h/final_results/figures/phase_failure_distribution.png)

대표 영상은 결과 확인 전 첫 Parent108200으로 고정했다. 모든 Parent 영상도 보존했다.

[E](../../outputs/task10h/final_results/rollouts/final/seed108200/E/rollout.mp4) [B0](../../outputs/task10h/final_results/rollouts/final/seed108200/B0/rollout.mp4) [H1](../../outputs/task10h/final_results/rollouts/final/seed108200/H1/rollout.mp4) [H2](../../outputs/task10h/final_results/rollouts/final/seed108200/H2/rollout.mp4) 

공통 ExperimentLogger UUID: `eab73e236d954f348a564a02753e51fb`. 기존 source/checkpoint/data는 수정하지 않았다. Git commit, 외부 publishing, 새 인스턴스 생성은 하지 않았다.

운영 기록: 최초 Final108200 B0는 CUDA OOM으로 중단됐다. 실패 trace/log를 보존하고 allocator expandable_segments:True만 적용해 동일 reset부터 재시도했다. 정책/seed/anchor/budget 변경은 없다. 완료한 pilot/E는 SHA 검사 후 재사용했으며 재개 실행은 새 ExperimentLogger UUID를 생성했다. 이전 UUID도 experiments에 남아 있다.


## 독립 readback과 상세 연구 판단

이 절은 전체 Final 종료 후 저장된 trace만 재분석했다. 추가 rollout, 학습, checkpoint 선택은 하지 않았다. [Parent별 상세 수치](../../outputs/task10h/independent_parent_diagnostics.csv)와 [설치 패키지 Anchor 해석](../../outputs/task10h/anchor_semantics_readback.json)을 함께 보존한다.

### H2의 정확한 의미와 공식 성공

설치된 ManiSkill에서 `goal_pose = hole_pose * peg_head_offset.inverse()`다. 원본 ScriptedExpert의 align 목표는 다시 `goal_pose * Pose([-0.02-half_len,0,0])`를 적용한다. 따라서 이 실험에서 동결한 H2는 원본 `align→insert` 경계이며, **peg tip이 hole 앞 105.0–142.3 mm에 있는 축/lateral 정렬 상태**다. 실제 H2 lateral 오차는 평균0.948 mm, 축 정렬각은0.057–0.167°였다. “접촉 직전20 mm” 또는 “마지막 몇 mm만 수행”하는 조건으로 해석할 수 없다. 원본 경계와 threshold를 Final 결과에 맞춰 변경하지 않았다.

공식 성공은 hole frame에서 `head_x >= -0.015`, `abs(head_y) <= radius`, `abs(head_z) <= radius`다. 이는 lateral norm에 대한 원형 threshold와 다르다. 독립 분석의 clearance 초과도 이 개별 축 기준을 사용했다. Final9,809개 control step에서 저장된 success와 이 predicate가 모두 일치했다. 별도의 orientation 기준으로 공식 성공을 다시 정의하지 않았다.

### Q1–Q4: Prefix 효과와 남은 실패 구간

- **Q1:** B0와 H1 insertion은 모두0/12다. 정확한 Reach/Grasp/Lift 제공만으로 insertion 성공이 증가하지 않았다. H1 Grasp/Lift12/12는 Expert prefix 성과이며 B의 자율 파지 능력으로 해석하지 않는다.
- **Q2:** H2 insertion은6/12로 B0 대비+50%p였다. H1/H2 anchor가 모두12/12 유효해 전체 사전 등록 분모와 조건부 분모가 같다. H2의 Wilson95% 구간은25.4–74.6%, B0/H1의0/12 구간은0–24.3%다.
- **Q3:** H2−H1은 gained6/lost0, exact discordant nominal p=0.03125다. H2−B0도 같다. 여러 비교에 대한 multiplicity 보정은 하지 않은 탐색적 수치다. **Alignment를 Expert가 대신 수행했을 때의 물리 상태와 관측 이력 묶음이 이 checkpoint의 성공을 개선했다.** 이를 Geometry 단독의 인과 효과나 일반적 우월성으로 확대하지 않는다.
- **Q4:** B가 정렬 상태에서도 항상 insertion action을 만들지 못한다는 주장은 틀리다.6회는 공식 성공했다. 나머지6회 실패는 최종 구간의 안정성 부족을 보여주지만, H2에105–142 mm 축 방향 접근도 포함되므로 순수 접촉/최종 미세 삽입만의 실패로 분리하지 못한다.

H1에서는 handoff 후167–198 Action, H2에서는118–167 Action이 남았다. 같은 Parent의 Full Expert는 각각41–69 Action,14–32 Action으로 남은 구간을 완료했다. 남은 budget이 문자 그대로 전혀 부족했던 조건은 아니다. 다만 이 비교로 B에 시간을 추가하면 성공할지 판정할 수 없다. `HORIZON_LIMIT`는 budget 소진 현상이며 원인 확정 label이 아니다.

### Q5: 즉시 반대 방향인가, 오차 누적인가

H1의 첫 B translation은 Expert translation과 반대 방향인 Parent가0/12였고, tip–hole norm은 첫 step과 첫10 step에서 모두12/12 감소했다. 따라서 모든 H1 실패를 “첫 action부터 전역적으로 반대 방향”이라고 설명할 수 없다. 한편 첫10 step의 arm saturation은11/12, geometry missing은8/12였다. 이후 전체 구간에서는12/12에 Expert와의 direction disagreement 및 arm saturation이 적어도 한 번 나타났다. 초기 거리 감소가 정렬 완료를 보장하지 않았다.

H2는 첫 translation 반대 방향1/12, 첫10 step tip norm 증가3/12였다. 그러나 **lateral 오차는 첫10 step 동안12/12에서 증가**했다. 성공6 Parent의 lateral 증가 중앙값은1.95 mm, 실패6 Parent는26.74 mm였다. 실패군은9.64–95.70 mm 증가했다. 이 집계는 결과별 사후 기술 비교이며 새로운 threshold나 정책을 선택하는 데 사용하지 않았다. 축 방향 접근으로 tip norm이 감소하면서 lateral 정밀도는 악화될 수 있다. 즉 일부 실패는 첫10 step부터 드러나는 lateral 제어 오차와 관련된 관찰이며, gross translation reversal만으로 설명되지 않는다.

### Q6–Q7: Geometry, gripper, saturation의 관찰 범위

H1 실패12회 중8회에서 geometry missing이 관찰됐고4회는 해당 missing 없이 실패했다. H2 실패6회 중4회에서 missing이 있었지만2회는 missing 없이 실패했다. H2 성공6회 중1회에서도 missing이 관찰됐다. **Missing은 이번 표본에서 실패의 필요조건도 충분조건도 아니다.** Presence가 유지됐다고 공간 추정의 정밀성이 보장되는 것도 아니다. GT 물리 상대 오차와 graph presence를 기록했으며, B에 별도의 GT pose head나 probe를 새로 학습하지 않았다.

H1에서는 파지 중 open 명령7 Parent, grasp loss4 Parent가 관찰됐다. H2 실패군에서는 파지 중 open과 grasp loss가 각각3 Parent였고 arm saturation은5/6 실패에서 관찰됐다. H2 성공군의 arm saturation은0/6이었다. 예를 들어108203은 open128→grasp loss129,108209는 open109→grasp loss126으로 기록됐다. 이는 시간적 선후관계와 동시 발생의 증거다. Gripper 명령이나 saturation만을 독립적으로 조작한 인과 검증은 하지 않았다. 모든 saturation과 arm saturation을 구분했으며, gripper=-1만으로 arm saturation을 세지 않았다. Contact/penetration 및 impulse trace는 보존했고 접촉 자체를 collision 원인으로 단정하지 않았다.

### Q8–Q10: 검증과 후속 우선순위

**Q8:** Pilot8회와 Final48회 검증은 PASS다. 모든 Parent의 E/H1/H2 공통 prefix에서 physical state, RGB-D/Graph/Robot76 identity, hidden 전후 연결과 Expert action을 비교했다. Final24개 handoff branch가 해당 검사를 통과했다. Handoff reset·중복 forward·GT hidden injection은 없었다. Weight 비교 및 freeze/eval 검사도 통과했다. 이는 hidden 구현의 정합성 증거이며, 학습 당시 이력 분포에 대한 적합성을 입증하지 않는다.

**Q9:** 확인한 것은 B0/H1의0/12와 H2의6/12, 초기 lateral drift, missing/open/grasp loss/saturation 현상이다. 분리하지 못한 것은 Geometry 정밀도, BC의 alignment/insertion action fit, 접촉 관측 부족, GRU history 분포의 각각의 기여다. H2는 hidden 이력과 물리 상태를 함께 바꾸는 처치다. Graph–Recurrent의 일반적 가능/불가능을 판정하지 않는다.

**Q10:** 우선순위는 **Alignment의 상대공간 표현과 제어/action correction을 구분하는 진단**이다. Hole-centered Geometry 개선은 합리적인 후보이나 이번 결과만으로 Geometry 수정이 효과적이라고 확정할 수 없다. 동일 anchor/history/budget의 GT38D 비교는 원인 분리를 위한 후속 후보로 남긴다. H1이 전부 실패했으므로 초기 Reach/Grasp/Lift의 recovery/DAgger만을 최우선 해결책으로 단정할 근거도 부족하다. DAgger 효과를 부정하는 결과는 아니다. 이번 Task에서는 어떤 추가 학습·수집·모델 변경도 시작하지 않았다.

### 실제 지연, 실행 복구와 보존

| 조건 | End-to-end 진단 추론 s | Policy ms | CUDA GNN/readout ms | Simulator step wall ms |
|---|---:|---:|---:|---:|
| E |1.274|4.64|2.90|14.15|
| B0 |1.172|5.38|3.16|14.53|
| H1 |1.247|4.59|2.92|14.12|
| H2 |1.271|4.54|2.88|14.13|

각 열은 Parent별 평균을 다시 평균했다. 추론 시간에는 입력 hash/snapshot 및 pre-action 진단 overhead도 포함된다. 과거 Task10-G의0.81–0.87s와 완전히 동일한 계측 구간으로 간주하지 않는다. Simulator control dt는50 ms로 유지했고 wall-clock 실시간20Hz 달성을 주장하지 않는다.

Engineering 시도v1–v4, 최초 Final108200 B0의 OOM 시도, report kind 등록 실패를 각각 기록했다. OOM 후 allocator만 `expandable_segments:True`로 바꾸어 같은 Episode를 reset부터 재시도했다. 완성 Episode는 trace/video SHA 검사 후 재사용했다. 최종 보고 등록 kind는 공통 Logger의 `reports`를 사용했다. 일시적인 SSH 장애 때 원격 실행은 계속됐고 인스턴스 재시작은 하지 않았다. 실행/재개/후처리마다 공통 Logger가 발급한 UUID를 유지했으며 최종 후처리 UUID는 `eab73e236d954f348a564a02753e51fb`다. 이전 실행 metadata도 함께 보존했다.

로컬10개 계약/분석 테스트 PASS,476개 indexed artifact의 SHA/크기 readback PASS, 원본 source/checkpoint/config 등623개 파일 변경 없음, 공식 success9,809 step 재계산 PASS다. 원격 source report와 artifact index는 원문 그대로 보존했고 이 독립 보완 분석은 별도 artifact로 기록했다. CUDA policy/vision/renderer와 CPU PhysX는 지정 RTX3090에서 실행했다. GPU PhysX 및 로컬 macOS CUDA는 NOT_TESTED다. Git commit·외부 publishing·새 인스턴스 생성은 하지 않았다.
