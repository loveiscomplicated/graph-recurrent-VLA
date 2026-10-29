# Task10-B 최종 연구 판단 — 추가 인과 진단 반영

현재 근거는 **Expert-state fit 오차, 부정확한 Geometry-to-Action 반응, 초기 Action7 오차의 Model feedback 증폭이 함께 존재한다**는 해석을 지지한다. 실제 물체 이동과 동일 상태 짧은 분기로 이전 Phase1/2의 공백을 보완했다. GNN에서 정보가 소실되는지, GRU/head가 그 정보를 제대로 쓰지 못하는지는 아직 분리하지 못했다. 따라서 바로 Loss Weighting이나 GNN 구조 변경을 우선하자는 이전 권고는 보류한다.

## 핵심 추가 결과

| 항목 | E0 Last | E1 Last |
|---|---:|---:|
| 원래09-A 기준 유효 Peg 개입 | 22/24 | 22/24 |
| Action Δ와 Expert Δ의 평균 direction cosine | 0.254 | 0.308 |
| 평균 signed gain | 0.185 | 0.116 |
| 평균 Readout320 relative L2 변화 | 13.1% | 4.00% |
| Zero-response보다 오차가 작은 조건 | 11/22 | 14/22 |
| k0, 첫 action 후 TCP−Peg 상대 위치 impulse 효과 | 2.116mm | 2.214mm |
| 10 actions 후 고정 Expert suffix의 impulse 효과 | 2.583mm | 2.514mm |
| 10 actions 후 Model feedback suffix의 impulse 효과 | 16.862mm | 14.931mm |
| Model/Expert suffix 효과 비율 (10 actions) | 6.528 | 5.939 |

Geometry primary 유효 집계는 t=0/10/25의 같은 Train30003 Expert prefix 상태에서 나온다. 물리적으로 유효한40개 Peg 개입 중4개는 Graph centroid의 기준 반응이 부족했고, 후반 t40/48은 기존09-A reset-time target consistency 검사를 통과하지 않아 primary alignment 집계에서 제외했다. 실패 조건을 제거하지 않고 원자료와 판정을 보존했다. 유효 입력의 관측 source purity 최솟값은0.907이었다. 이는 완벽한 geometry/feature 정확성을 보장하지 않는다.

후반을 검토하기 위해 **별도 current-OBB baseline anchor와 같은 phase/counter의 reference**를 검증했다. 기존 기준의3.144µm/3.874µm target drift 실패를 그대로 두고, 원래2µm threshold도 유지했다. 두 reference의55개 Native Action 값은 모두 byte-identical이며, 수치를 모델에 맞게 조정한 것이 아니다. 차이는 reset-time target 일관성과 current-target 일관성이라는 검증 anchor다. 이 별도 reference는 원래 demonstration policy의 frozen target과 구분한다.

Current-OBB reference에서 t40의 유효7조건은 Expert translation response 평균2.206mm에 비해 Model response가 E0 0.0423mm, E1 0.0150mm였다. t48에서는 Expert10.791mm에 비해 E0 0.186mm, E1 0.0581mm였다. 후반에서도 실제 geometry 변화에 대한 수정 크기가 매우 작다는 조건부 근거다. 전체 secondary36개 유효 Peg 조건의 signed gain 평균은 E0 0.114, E1 0.0715였다. 원래 primary와 secondary 집계는 섞지 않았다.

## Q1–Q10 직접 답변

**Q1. 정상 Expert 상태부터 Action Error가 있는가? — CONFIRMED.** 이전 Train96/Val12 분석 그대로, seed30003 네 Best/Last 모두 t0에서 controller translation error>1mm였다. 새 실험으로 이 오차를 없던 것으로 해석하지 않는다. 다만 이번 live prefix는 원본 HDF5와 byte-identical한 관측이 아니며, 같은 current-runtime 분기끼리의 exact 비교다.

**Q2. MSE가 숨긴 제어 정밀도 문제가 있는가? — CONFIRMED.** Last Val translation vector error E0 1.927mm/E1 1.837mm, 반전 구간8.287mm/7.510mm라는 이전 결과에 더해, 물체를 실제 이동시켰을 때 Action 수정 방향·gain이 부족한 조건을 확인했다. 작은 전체 MSE만으로 적절한 공간 조건부 제어를 주장할 수 없다.

**Q3. 실제 자율 rollout은 언제 이탈하는가? — 관측 유지.** seed30003 Last의 최소거리는 E0 post33의92.8mm, E1 post14의177.4mm였고, 3연속 외향 명령 후보는 t35/t14였다. 이 시점을 인과적 최초 오류로 확정하지 않는다. 새 분기는 k0에서 한 번의 Action7 차이를 넣어 이후 차이를 따로 측정했다. 서로 다른 Expert/Autonomous의 동일 timestep을 정답 pair로 만들지 않았다.

**Q4. 그때 Geometry/Presence는 어땠는가? — 복합 증거.** 기존 seed30003 Last 이탈 후보에서 Peg presence와 wrist gate가 유지됐다. 새 물리 개입에서도 대체로 object source가 지원되는 Graph와 Readout 변화가 있었다. 반대로 기존 Dev에는 많은 Peg 누락이 있었고, 새40개 Peg 조건 중4개는 centroid response 기준을 통과하지 못했다. “모든 실패가 Geometry 누락”이나 “Geometry는 완전히 정확” 모두 근거가 없다.

**Q5. 실제 Peg 이동에 적절히 반응하는가? — 약점을 지지하는 인과 개입 증거.** Robot76·카메라·다른 물체·previous hidden·intervention 전 vision history를 고정하고 실제 Peg를 world XY±5/10mm 이동시켰다. 유효한 입력 변화와 Readout 변화가 있어도, primary Last 평균 direction cosine0.254/0.308, signed gain0.185/0.116으로 반응이 충분히 일치하지 않았다. E1의 zero 대비 개선 조건이 더 많아도 gain과 크기가 작아 적절한 correction 능력을 입증하지는 않는다. 이는 한 seed의 제한된 상태에서의 conditional sensitivity 결과다.

**Q6. 어느 단계가 문제인가? — Geometry→Action 전체의 활용 약점 SUPPORTED, 모듈 위치 INCONCLUSIVE.** Raw point 변화·source purity·Readout/GRU 변화와 Action 반응을 연결했다. 그래서 Graph를 완전히 무시한다는 설명은 맞지 않는다. 그러나 Readout이 변한다는 것만으로 필요한 위치 정보가 유용하게 표현됐다고 주장할 수 없다. GNN 표현 손실과 GRU/head 활용 문제는 frozen decoder probe나 별도 통제 없이는 구분되지 않는다. Probe는 이번 요청의 두 최소 진단 범위를 넘어 실행하지 않았다.

**Q7. E0/E1은 어떻게 다른가? — 기술적 차이만 확인.** E0가 더 큰 Readout/Action 반응을 내지만 방향 정합성이 낮은 조건도 있다. E1은 더 작은 반응과 낮은 gain을 보이면서 zero-response 대비 개선 조건은 더 많았다. Best와 Last의 수치는 거의 같고 어느 선택도 바꾸지 않았다. 두 구조의 parameter 수 차이(245,447/194,759), 별도 학습과 seed 하나 때문에 message passing의 순수 효과로 해석할 수 없다. 중간 성공 사례(E1 step1000 Dev42000, E0 step10000 Dev42004)는 이전 보고서의 사례로 유지하며 새 모델 선택에 쓰지 않았다.

**Q8. 초기 오류가 feedback에서 실제 증폭되는가? — k0 조건에서 CONFIRMED.** 같은 시작 input/Graph/hidden에서 E/E·EM·ME·MM을 교차했다. n1에서 ||ME−EE||=||MM−EM||가 정확히 같았다. n10에서 Model suffix의 영향이 고정 Expert suffix보다 E0 6.53배, E1 5.94배 컸다. 단순히 자율 trajectory가 Expert와 다르다는 상관이 아니라, 같은 첫 행동 차이의 suffix별 causal contrast다. 첫 차이는 translation만이 아니라 rotation/gripper를 포함한 **전체 Action7**이다. 고정 Expert suffix를 adaptive Expert recovery로 해석하지 않는다. 10 actions=0.5초이며, 이 한 초기 조건을 모든 상태/후반 failure로 일반화할 수 없다.

사전등록한 k25는 Robot76은 exact였으나 baseline RGB의1픽셀,3채널 값이 달라(max46) strict guard가 차단했다. 해당 branch action은0개다. 실패 당시 full physical-state check는 RGB 뒤에 있어 도달하지 못했고, 전체 current RGB도 저장되기 전이었다. traceback과 실패 logger를 보존했다. 원인을 reset 문제나 단순 render noise로 확정하거나 guard를 완화하지 않았다. Train30003 reset SHA 일치와 k0의 유효성은 별도 계약이며 k25의 실패를 숨기지 않는다.

**Q9. 유력한 원인 1–3개와 등급.**

| 후보 | 증거 등급 | 범위 |
|---|---|---|
| Expert-state action fit 부족 | CONFIRMED 오차 존재 / SUPPORTED 실패 기여 | 이전 offline command 오차. 독립 기여율은 미확정 |
| Geometry-to-Action 반응의 부정확성·약한 correction | SUPPORTED | 같은-state 실제 Peg 개입. 필요한 정보가 어디서 부족해지는지 미분리 |
| 초기 Action7 오차의 closed-loop feedback 증폭 | CONFIRMED 조건부 메커니즘 | Last E0/E1, Train30003 k0의10-action 대조. k25 미검증 |

Perception 품질 문제의 동시 기여, GNN 정보 소실, 독립적인 GRU history 효과, optimizer/자료 분포의 기여는 여전히 INCONCLUSIVE다. 각 원인의 전체 실패 기여율을 추정하지 않았다.

**Q10. Task10-C 우선순위는?**

1. **특정 loss/구조 변경을 고르기 전에 표현과 사용을 구분한다.** 기존 episode 단위 split에서 Raw Geometry와 frozen Object Slot/Readout의 TCP-relative Peg 위치·변화 복원력을 작은 decoder로 비교하는 진단을 우선한다. 이번에는 지원되는 input과 큰 Readout 변화에도 correction이 약했으므로, 유용한 정보가 encoder에서 부족한지 downstream에서 쓰이지 않는지 확인할 필요가 있다. Probe 실패를 정보 소멸의 증명으로 해석하지 않는다.
2. **Geometry-conditioned correction 개선을 검증할 평가축을 고정한다.** 이 Task의 valid intervention direction/gain/response error와 fine-command 오차를 train/dev 진단으로 보존한다. 이후 소규모 단일요인 실험에서 자료 커버리지·학습 최적화·conditional mapping 중 무엇을 개선할지 위 probe 결과에 따라 정한다. 전체 MSE 감소만으로 개선을 승인하지 않는다. 현재 결과는 Loss Weighting이나 GNN 변경 중 하나가 해결책이라는 증거가 아니다.
3. **Feedback robustness를 별도 확인한다.** 이번 k0 causal contrast를 재사용해 초기 fit 개선이 feedback 증폭도 줄이는지 확인한다. 향후 다른 상태로 확장하려면 k25 같은 input identity 실패부터 검사한다. Task10-A Test 복구를 변경하거나 새 Test 결과로 이 설정을 조정하지 않는다.

이번 Task에서 BC·GNN·GRU·head의 재학습, loss/구조 변경, recovery/DAgger 도입, checkpoint 변경, Git commit, 외부 publishing은 하지 않았다. 기존 Phase1/2 보고서의 생략 판단을 수정하되 원자료와 과거 보고서는 보존하고 이 보고서를 최신 판단으로 추가했다.


# Task10-B 추가 진단 — 실제 Geometry 개입과 짧은 feedback 분기

기존 Phase1/2의 Expert-state 오차 확인은 원인 분해의 완료가 아니다. 이 추가 진단은 그 공백을 메우기 위해 기존09-A/09-B의 물리 개입·causal memory·response metric과08-F의 짧은 분기 구조를 현재 E0/E1에 연결했다. 모델/Geometry/Codec을 변경하거나 재학습하지 않았다. 기존 보고서와 원자료는 보존했다. 최종 연구 판단은 `causal_findings.md`를 따른다.

## 실험 identity와 범위

- Parent comparison: `3788798a719f4ec4a66c224ca42f5d63`.
- 이전 Phase1/2 UUID: `35877452d6d94d9691964fc8a8490858`.
- Reset/Prefix guard UUID: `b659dc3c41494ea89abeee09ca421c59`.
- Geometry collection UUID: `bf2ab695001a487494a732727ece1be8`.
- Geometry analysis UUID: `8a7c11f84a8a43c7b6c67783db71c096`.
- Branch driver UUID (k25 guard로 failed; k0 원자료 보존): `4647218956e044dbb90852f1a1361f5c`.

새 결과는 `/workspace/task10b_extension_20261029`에만 저장했다. frozen source는 별도의 기존 Task10-B source copy를 재사용했고 원본 체크포인트·Graph cache·rollout·Task10-A source/process는 수정하지 않았다. 신규 진단은 Train30003만 사용하며 Test12를 seed/threshold/가설/모델 선택에 사용하지 않았다. Task10-A 관리 상태/복구 코드 확인 중 읽힌 Test reset audit는 진단 설정이나 결과 계산에 사용하지 않았다. 새 Test cache/prediction/rollout은 열지 않았다.

## Reset와 동일 상태 유효성

Train30003의 reset hash는 `27fd426300899a6a44b55aa04aab72e373cee7e61881762fd03d467d274d9f62`로 원본과 정확히 일치했다. Pose 복원이나 hash guard 우회는 없었다. 원본 Expert action prefix를 두 번 독립 reset에서49 actions까지 재생한 qpos/qvel/TCP/물체 pose/velocity 등이 모두 정확히 일치했다.

원래 HDF5와는 Robot76 최대 0.00010915613, TCP pose 최대 6.3627958e-06의 차이가 있다. 따라서 원본 Expert 입력과 byte-identical하다고 주장하지 않는다. 인과 개입은 **동일 current-runtime physical state 안**에서 로봇·카메라·previous hidden·causal vision history를 고정한 비교다. 원본 상태 근접성과 paired-state 동일성을 별개로 기록했다.

## Geometry 개입

시점은 사전 고정한 t=0,10,25,40,48이다. 각 시점 baseline, Peg world X/Y ±5/10mm, Box X+10mm negative control, baseline repeat를 수집했다. 총55조건이며 물체를 teleport한 뒤 physics step 없이 새 RGB-D를 렌더했다. SAT 정적 collision/near-contact 검사이며 teleport 후 cached PhysX contact를 새 contact force로 해석하지 않았다. 동적 안정성 실험은 아니다. 또한 다른 물체·Robot76·보정의 exact 검사를 통과한 물리 조건은 55/55다.

실제 causal RGB prefix를 sequential tracker로 처리하고, current frame 직전의 tracker memory를09-B `memory/restore`로 복원했다. 네 Best/Last policy의 hidden은 각자의 같은 history에서 얻은 값을 고정했다. Student 입력은 원래 whitelist의 RGB-D·예측 mask·Robot76뿐이다. GT pose는 개입 정의/Expert reference/평가, segmentation ID는 source purity 평가에서만 사용했다. 저장 NPZ 전체를 sensor input으로 넘기려던 최초 시도는 whitelist가 차단했으며, sensor-only key adapter로 수정한 뒤 계산했다. whitelist/geometry forward는 바꾸지 않았다.

Baseline repeat에서는 mask·Graph·Readout·GRU input/hidden·Native Action을 exact 비교했다. 각 baseline의 원래 forward와 adapter도 action/hidden exact, forward repeat exact였다. 모든 frame의 current source timestamp를 검사했고 미래 frame이나 Expert Action은 Student input에 넣지 않았다.

## Expert counterfactual 검증

09-A `expert_reference`를 그대로 사용해 같은 ScriptedExpert phase/counter와 새 OBB target으로 다시 계산했다. 목표 translation 이동량·방향/자세와 독립 servo 계산이 기준을 만족한 경우에만 유효 reference로 사용했다. 이전 원본 action을 임의의 perturbation 정답으로 넣지 않았다. 원래 HDF5 action과의 작은 차이는 별도 수치로 남기며, 동일 current-runtime reference의 유효성과 혼동하지 않는다.

| t | Phase | baseline target 재계산 차 m | baseline reference | 유효 물리 조건 |
|---|---|---:|---|---:|
| 0 | reach | 0 | VALID_STAGE_CONDITIONED | 11/11 |
| 10 | reach | 1.058e-06 | VALID_STAGE_CONDITIONED | 11/11 |
| 25 | reach | 1.729e-06 | VALID_STAGE_CONDITIONED | 11/11 |
| 40 | descend | 3.144e-06 | INVALID_EXPERT_REFERENCE | 11/11 |
| 48 | descend | 3.874e-06 | INVALID_EXPERT_REFERENCE | 11/11 |


후반 reference가 무효면 direction/gain 유효 집계에서 제외하되 원자료와 사유를 보존했다. Threshold를 완화하거나 target을 임의로 교정하지 않았다. Reference Δ≤0.1mm는09-A의 NON_DISCRIMINATIVE_REFERENCE 기준이다. 과거09-A 입력 purity≥0.5, 이동방향으로 centroid response≥1mm 기준을 C3의 양 camera source provenance에 적용했다. Point-centroid 변화는 전체 물체 pose 복원의 정확도와 같지 않으며 sampling/visibility 영향도 포함한다.

## Geometry → Readout → Action 결과

유효성 분류(모델 중복 전)는 {'VISIBLE_VALID': 46, 'GEOMETRY_UNSUPPORTED_RESPONSE': 4}이다. `GEOMETRY_UNSUPPORTED_RESPONSE`는 source purity는 통과했지만 기준 centroid 반응이 부족한 경우로 따로 남겼다. 아래는 physical·visible/purity/centroid·Expert discriminative reference를 모두 통과한 Peg 개입만 기술적으로 집계했다. 실패 조건과 box control/repeat도 CSV/JSON에는 남아 있다. 같은 시점의 ±offset, 같은 graph를 쓰는4모델을 독립 Episode로 취급하지 않는다.

| 모델 | t | 유효 Peg 개입 | Readout 상대L2 | Expert Δ mm | Model Δ mm | 방향 cosine | Signed gain | Zero보다 개선 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0_last | 0 | 7 | 0.1742 | 0.396 | 0.4041 | 0.2476 | 0.6803 | 2/7 |
| E0_last | 10 | 7 | 0.1327 | 0.5703 | 0.1931 | 0.1389 | -0.1308 | 4/7 |
| E0_last | 25 | 8 | 0.09204 | 1.249 | 0.1501 | 0.3612 | 0.02891 | 5/8 |
| E0_last | 40 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |
| E0_last | 48 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |
| E0_validation_best | 0 | 7 | 0.1747 | 0.396 | 0.4028 | 0.2477 | 0.6776 | 2/7 |
| E0_validation_best | 10 | 7 | 0.1331 | 0.5703 | 0.1916 | 0.1508 | -0.1284 | 4/7 |
| E0_validation_best | 25 | 8 | 0.09202 | 1.249 | 0.1505 | 0.364 | 0.02913 | 5/8 |
| E0_validation_best | 40 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |
| E0_validation_best | 48 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |
| E1_last | 0 | 7 | 0.04892 | 0.396 | 0.1443 | 0.3298 | 0.3302 | 3/7 |
| E1_last | 10 | 7 | 0.03584 | 0.5703 | 0.06382 | 0.2122 | -0.002062 | 4/7 |
| E1_last | 25 | 8 | 0.0357 | 1.249 | 0.06782 | 0.3741 | 0.03141 | 7/8 |
| E1_last | 40 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |
| E1_last | 48 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |
| E1_validation_best | 0 | 7 | 0.04895 | 0.396 | 0.145 | 0.3228 | 0.3332 | 3/7 |
| E1_validation_best | 10 | 7 | 0.03587 | 0.5703 | 0.06406 | 0.2147 | -0.001276 | 4/7 |
| E1_validation_best | 25 | 8 | 0.03572 | 1.249 | 0.06758 | 0.3733 | 0.03117 | 7/8 |
| E1_validation_best | 40 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |
| E1_validation_best | 48 | 0 | 미정의 | 미정의 | 미정의 | 미정의 | 미정의 | 0/0 |


Readout 변화만으로 유용한 정보 전달을 입증하지 않는다. Raw Geometry가 유효하게 변하고 Readout/GRU가 변해도 Action 방향·gain·zero-response 대비 오차를 함께 봐야 한다. 이 실험은 encoder/GRU/head 전체의 반응을 측정하며 개별 모듈의 독립 원인 기여를 분리하지 않는다. Frozen probe를 학습하지 않았으므로 정보 소멸 여부는 확정할 수 없다. E0/E1은245,447/194,759 parameters로 구조와 용량이 달라 message passing 단독 인과 효과를 주장할 수 없다.

## 별도 검증한 Current-OBB Reference

기존09-A 기준의 후반 무효 판정은 그대로 유지했다. 추가로 current live state에서 fresh OBB target을 만든 baseline과 perturbation을 같은 phase/counter로 비교하는 별도 reference를 사전 정의·검증했다. 원래 demonstration action과 같은 reference라고 주장하지 않는다. Threshold를 완화하지 않았으며 아래 결과는 기존 primary 표와 섞지 않았다. Model/Graph/Action 파일은 그대로 읽어 새 reference와만 비교했다. source physical state는 원래 수집 before/after와 정확히 일치했다.

| 모델 | t | 유효 개입 | Expert Δ mm | Model Δ mm | 방향 cosine | Signed gain | Zero보다 개선 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0_last | 0 | 7 | 0.396 | 0.4041 | 0.2476 | 0.6803 | 2/7 |
| E0_last | 10 | 7 | 0.5703 | 0.1931 | 0.1389 | -0.1308 | 4/7 |
| E0_last | 25 | 8 | 1.249 | 0.1501 | 0.3612 | 0.02891 | 5/8 |
| E0_last | 40 | 7 | 2.206 | 0.0423 | 0.1376 | -0.0009844 | 4/7 |
| E0_last | 48 | 7 | 10.79 | 0.186 | 0.2916 | 0.002254 | 4/7 |
| E0_validation_best | 0 | 7 | 0.396 | 0.4028 | 0.2477 | 0.6776 | 2/7 |
| E0_validation_best | 10 | 7 | 0.5703 | 0.1916 | 0.1508 | -0.1284 | 4/7 |
| E0_validation_best | 25 | 8 | 1.249 | 0.1505 | 0.364 | 0.02913 | 5/8 |
| E0_validation_best | 40 | 7 | 2.206 | 0.04244 | 0.1518 | -0.0009463 | 4/7 |
| E0_validation_best | 48 | 7 | 10.79 | 0.1846 | 0.2915 | 0.002192 | 4/7 |
| E1_last | 0 | 7 | 0.396 | 0.1443 | 0.3298 | 0.3302 | 3/7 |
| E1_last | 10 | 7 | 0.5703 | 0.06382 | 0.2122 | -0.002062 | 4/7 |
| E1_last | 25 | 8 | 1.249 | 0.06782 | 0.3741 | 0.03141 | 7/8 |
| E1_last | 40 | 7 | 2.206 | 0.01497 | 0.572 | 0.002799 | 6/7 |
| E1_last | 48 | 7 | 10.79 | 0.0581 | 0.3618 | 0.0009949 | 5/7 |
| E1_validation_best | 0 | 7 | 0.396 | 0.145 | 0.3228 | 0.3332 | 3/7 |
| E1_validation_best | 10 | 7 | 0.5703 | 0.06406 | 0.2147 | -0.001276 | 4/7 |
| E1_validation_best | 25 | 8 | 1.249 | 0.06758 | 0.3733 | 0.03117 | 7/8 |
| E1_validation_best | 40 | 7 | 2.206 | 0.01497 | 0.5767 | 0.00278 | 6/7 |
| E1_validation_best | 48 | 7 | 10.79 | 0.05785 | 0.3621 | 0.0009779 | 5/7 |

![Current OBB response](geometry_action_response_fresh_reference.png)

Baseline command 차이는 `reference_definition_comparison.json`, 정의·검증은 `fresh_reference_protocol.json`과 `fresh_expert_reference.json`에 보존했다. 이 reference도 학습 target이나 Test12 fitting에는 쓰지 않았다.


## 10-action feedback 분기

08-F의 첫 행동×suffix 구조를 재사용했다. Train30003, Last E0/E1, k=0에서 E/E·EM·ME·MM 각10 actions(모델별4개, 총8개)을 완료했다. 사전등록한 k=25는 RGB strict input guard가 차단해 policy branch action0개이며 BLOCKED_OBSERVATION_IDENTITY로 남겼다. 기준을 완화하지 않았다. 10 actions는20Hz에서0.5초다. E/E는 원래08-F reference replay처럼 첫 input 검사 뒤 policy 추론 없이 원본 Expert action을 적용한다. E suffix도 새 feedback Expert가 아니라 고정된 원본 Expert 명령이다. 이 제어 조건을 실제 scripted feedback recovery와 혼동하지 않는다.

분기 시작의 reset hash·물리상태·Robot76·RGB-D/보정·current Graph·previous hidden·prediction은 정확히 일치했다. 고정된 합법적 causal prefix의 tracker/hidden을 복원했고, suffix의 실제 새로운 관측에서 hidden을 갱신했다. 임의 hidden 교환·다른 시점 history 이식은 없다. E/E와 EM의 첫 post-state, ME와 MM의 첫 post-state도 정확히 일치했다.

초기 model-versus-expert 행동의 영향은 Model suffix에서 ||MM−EM||, Expert suffix에서 ||ME−EE||로 측정했다. 벡터는 TCP−Peg 상대 위치이며 시점 n은 분기 후 적용 action 수다. n=1 두 값이 같다는 계약을 확인한다. 비율의 분모가 작으면 mm 절대값을 우선 해석한다. EE가 피드백 Expert가 아니므로 “Expert가 보정해주면 회복한다”는 실험과는 다르다.

| 모델 | k | n | Expert suffix impulse mm | Model suffix impulse mm | 비율 | MM–EE mm |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0 | 1 | 2.116 | 2.116 | 1 | 2.116 |
| E0 | 0 | 3 | 2.587 | 5.791 | 2.239 | 4.194 |
| E0 | 0 | 5 | 2.59 | 9.877 | 3.814 | 5.42 |
| E0 | 0 | 10 | 2.583 | 16.86 | 6.528 | 16.73 |
| E1 | 0 | 1 | 2.214 | 2.214 | 1 | 2.214 |
| E1 | 0 | 3 | 2.528 | 3.738 | 1.478 | 4.293 |
| E1 | 0 | 5 | 2.521 | 8.112 | 3.218 | 6.941 |
| E1 | 0 | 10 | 2.514 | 14.93 | 5.939 | 20.87 |


## 실행 환경과 산출물

Linux Python3.12 / Torch2.14.1+cu130 / NumPy2.5.4. CPU PhysX와 SAPIEN host image readback을 실행했다. 이 호스트의 NVIDIA renderer 초기화에는 CUDA device가 보여야 했고 CUDA context를 초기화했다. Policy와 DINO/SAM2 inference는 CPU에서 실행해 Task10-A의 GPU 추론을 점유하지 않았다. CUDA PhysX/MPS는 실행하지 않았다. 설치된 SAM2의 optional `_C` 후처리 누락 warning은 보존했으며 패키지/모델/geometry filter는 변경하지 않았다.

원자료는 physical_collection의 RGB-D/보정/robot/평가 segmentation, geometry_inference의 실제 Graph/예측 mask/source records/Readout/GRU/Action, branches의 시작 input/step trace/physics states/MP4다. `geometry_sensitivity.csv`, `graph_to_action_response.csv`, `geometry_validity_audit.json`, `branch_metrics.csv`, `error_amplification_analysis.json`, `prefix_intervention_analysis.json`과 대조할 수 있다. 소규모 분기 영상은 새 실제 branch에만 생성했고 기존 rollout 영상을 재생성하지 않았다.

![Geometry response](geometry_action_response.png)
![Graph to action](graph_readout_action_response.png)

![Impulse divergence](divergence_curves.png)
