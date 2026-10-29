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
