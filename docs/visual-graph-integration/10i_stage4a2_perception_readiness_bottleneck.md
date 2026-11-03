# Task 10-I Stage4A-2 — Perception readiness bottleneck

작성일: 2026-11-02. 저장된 BF16 외형 프롬프트 평가 결과의 사후 진단이다. 새 추론·모델·Prompt·선택 정책·Threshold·Production 변경은 없다.

현재 Box 선택 결과를 신뢰할 만한 학습 입력이 확보된 상태로 판정하지 않는다. Peg/Box 동시 성공은30/58(51.7%), 후기 Main Peg는0/15다. 학습 자체가 불가능함을 증명한 실험은 아니지만, 이 관측 경로를 그대로 신뢰해 후속 학습에 연결할 근거가 부족하다. SAM2 초기 Mask 성능도 이번 구성에서는 아직 측정하지 않았다.

## 같은58개 장면의 단계별 손실

GT가16px 이상 보이는 두 물체를 모두 대상으로 하며 Box IoU≥0.5를 정답으로 정의했다. 두 Query 후보 풀을 고정해 아래 단계를 사후 계산했다.

| 단계 | 두 물체 조건을 모두 충족한 장면 | 비율 |
|---|---:|---:|
| DINO 후보 풀에 두 정답 Box가 존재 | 52/58 | 89.7% |
| Qwen이 두 물체의 정답 Box를 적어도 하나씩 승인 | 36/58 | 62.1% |
| 승인 후보 중 DINO Score1위를 골랐을 때 두 Box 모두 정답 | 30/58 | 51.7% |

실패28개를 서로 배타적으로 나누면 정답 후보 부재6개, 적어도 한 Object의 정답 후보 전부 거부16개, 두 정답을 승인했으나 선택 순위에서 손실6개다. 즉 이 코호트에서는 후보 생성 이후의 검증·선택 손실이 더 크다. 후보 존재89.7%는 GT로 확인한 사후 상한이며, 이를 달성하는 GT-free 선택기가 이미 있다는 뜻은 아니다. Qwen을 완전히 제거한 원시 Top1의 동시 성공은14/58(24.1%)이므로 단순히 검사기를 제거하는 것이 해법이라고 추론할 수도 없다.

후보 역할 분류 정확도86.0%는 최종 위치 정확도나 학습 가능한 Object Mask 정확도를 뜻하지 않는다. 정답 물체로 분류한 후보도 너무 느슨하거나 일부만 포함한 Box일 수 있다. 현재 결과는 역할 판별, Box 품질 선택, SAM2 초기 Mask 평가를 분리해야 함을 보여 준다.

## 80%를 목표로 삼을 때의 해석

두 물체 각각의 Recall80%와 두 물체 동시 성공80%는 다른 조건이다. 현재 목적에는 두 물체 동시 성공률과 Camera/시점별 실패를 함께 확인하는 것이 적절하다. 특정 평균을 넘었다는 이유로 후기 Main0/15 같은 집중 실패를 통과 처리하지 않는다. 80%는 다음 Prototype 진행을 위한 운영 목표로 설정할 수 있으나 학습 가능성·Closed-loop 성공을 보장하는 통계적 임계값으로 입증된 것은 아니다.

남은 검증의 초점은 올바른 Peg 후보의 거부를 줄이고, 역할 승인과 Box 품질 판정을 분리한 선택을 확인하는 것이다. 이후 사전 고정한 표본에서 Qwen/DINO Box로 얻은 실제 SAM2 Mask를 평가해야 학습 입력의 품질을 판단할 수 있다. 이번 진단으로 그 새 방법을 구현·검증했다고 주장하지 않는다. Stage4A-2 PAUSED_SCOPE_REVIEW를 유지한다.

- [전체 대응 평가 보고서](10i_stage4a2_qwen35_appearance_bf16_independent_evaluation.md)
- [고정 후보별 평가](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/candidate_evaluation.csv)
- [최종 선택](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/selected_boxes.csv)
- [GT 가시성과 평가 지표](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/evaluation_metrics.json)

## 후속 거부 원인 진단의 보완

위의 “정답 후보 거부”는 Box IoU 기준의 Query Gate 손실이다. 후속 감사에서 Box 거부7개 중6개는 Actor 구성상 Robot/Peg 우세였고 Qwen이 그 역할을 출력했다. 모두를 순수한 시각 인식 오류라고 해석해서는 안 된다. Peg 거부35개에는 실제 Peg 우세25개의 오분류가 포함돼 있다. [326회 통제 진단과 한계](10i_stage4a2_qwen35_rejection_mechanism.md)를 함께 읽는다. 위 손실 수치는 수정하지 않았다.
