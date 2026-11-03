# GT 지도 GNN 사전학습부터 Perception 진단까지의 커밋 정리

작성일: 2026-11-02. 범위: `b68f999`부터 `1fb0ecf`까지, 시작 커밋을 포함한 13개 커밋.

출발 질문은 기존 GT-state recurrent 정책의 40~60%대 성공을 바탕으로, GNN을 GT state 예측으로 먼저 학습한 뒤 recurrent 모듈을 붙여 BC를 수행하면 실제 시각 입력에서도 제어할 수 있는가였다. 해당 발언 자체가 Git 기록에 저장됐다는 근거는 찾지 못했으므로, 내용에 대응하는 최초 구현인 `b68f999`를 시작점으로 잡았다. 문서에 적힌 실험 날짜와 Git 커밋 날짜는 다를 수 있다.

실제 GNN 사전학습과 GNN–GRU BC 비교는 완료됐다. GNN의 평균 상태 오차는 기준 모델보다 낮아졌지만 작은 물체 이동 복원은 충분하지 않았고, 후속 세 정책의 독립 Final insertion은 모두 0/12였다. 이후 실패를 단계별로 조사하면서 초기 시야 조건을 갖춘 데이터를 다시 수집하고, perception과 그래프 변환에서 정보가 사라지는 지점을 확인하는 연구로 이어졌다. 새 데이터와 새 perception 후보를 적용한 정책의 성공률 개선은 아직 검증하지 않았다.

## 커밋별 진행

| 커밋 | 작업 | 얻은 결과와 역할 |
|---|---|---|
| `b68f999` | GT 지도 GNN 사전학습 입력 준비 | 원래 recurrent의 GT38D 계약, 예측할 물체 상태와 측정 로봇 상태, 데이터 및 비교 조건 정리 |
| `aa47c1d` | GT sidecar 인계 검증과 보완 | 기존 corrective 자료에 timestep별 물체 GT가 없는 문제를 해결하고 승인 데이터의 정합성 확보 |
| `2290732` | GNN 상태 예측 사전학습 및 평가 | 상태 예측의 평균 개선과 작은 이동 복원의 한계를 정량 확인 |
| `19745b4` | F1으로 초기화한 recurrent 정책 비교 | GNN 동결 A, BC 공동학습 B, BC와 geometry 보조학습 C 비교 |
| `a047bbb` | 학습과 평가 실행 관리 및 검증 | miniature, 데이터·체크포인트·추론 계약, 실행 이력과 결과 검증 보강 |
| `1c5b621` | 인식과 초기 가시성 진단 | 초기 Peg의 끝과 파지 영역이 Main 시야 밖인 데이터가 많음을 확인 |
| `a997275` | 단계별 Expert prefix 이후 정책 인계 | 파지 이후보다 정렬 이후 인계에서 insertion 성공이 증가하는 현상 확인 |
| `417ad0b` | Main에서 보이는 초기 상태 후보 풀 | 926회 reset으로 유효 고유 후보 300개 확보 |
| `9588868` | 후보 풀 회수와 사본 검증 | 후속 선정에 사용할 후보 자료의 보존과 동일성 확인 |
| `aa6843f` | 무작위 및 다양성 기반 데이터 선정 | R/D 각 96 Parent와 공통 Validation 16 Parent 구성 동결 |
| `7e9d8e4` | 선정 상태에서 Expert 데이터 수집 | Full-task와 corrective trajectory를 동기화된 RGB-D·로봇·GT로 수집 |
| `9e77aad` | 수집 결과 감사와 인계 | R/D 각 272 logical episode, Validation 48개 및 분할·동기화 검증 |
| `1fb0ecf` | Stage4A 및 Stage4A-2 보관 | 독립 perception 평가, 보정 후보, 실패 원인·점 예산 국소 진단 등 182개 파일 보존 |

마지막 커밋은 단일 실험의 완료 커밋이 아니라 여러 연구 단계와 중지 상태를 함께 보존한 snapshot이다. 이후 생긴 미커밋 파일은 이 범위에 포함하지 않는다.

## 출발점인 GT recurrent 성공률의 의미

Task10-E가 확인한 원래 GPU Clean GRU 결과는 학습 seed별 25/30, 15/30, 14/30이며 합계 54/90, 60%였다. 별도의 CPU 조건 평가에서는 insertion 35/54였지만 초기 조건·물리 backend·평가 목표가 달라 같은 성공률 재현으로 합산하지 않았다. 이 기록은 GT 입력 정책이 실제 조작을 수행할 수 있다는 근거이며, 시각 입력 정책이 같은 수준에 도달한다는 근거는 아니다.

관련 근거: [GT GRU 비교](10e_gt_gru_vs_graph_reach.md).

## GNN 사전학습에서 확인한 것

F1은 로봇 측정값을 그대로 사용하고, GNN이 물체 위치·자세·크기를 예측하도록 학습했다. 물체 출력 18D와 측정 로봇 15D, 파생 tip-to-hole 3D, freshness 2D를 합쳐 원래 GT38D 형식도 구성할 수 있게 했다. GRU나 action loss는 이 단계에 없었다.

Train 272 episode/96 Parent, Validation 48/16, 신규 Final 12/12를 사용했다. 기존 corrective 자료에는 필요한 GT가 없어 별도 212개를 동기 수집했고, 실행 환경 간 Graph 차이를 발견한 뒤 F1 환경에서 재캐시했다. 단순 seed replay나 물체 중심점으로 과거 GT를 대신하지 않았다.

Final 12 Parent/1,350 frame의 평균 위치 오차는 다음과 같다.

| 모델 | Peg | Hole | Peg tip과 Hole 상대 벡터 |
|---|---:|---:|---:|
| GNN | 15.291 mm | 51.071 mm | 57.170 mm |
| Robot-only | 21.033 mm | 63.312 mm | 70.585 mm |
| Raw Geometry Ridge | 81.424 mm | 109.430 mm | 130.149 mm |

GNN은 이 기준 모델들보다 평균 absolute 오차가 작았다. 그러나 실제 ±5/10 mm 물체 이동의 signed gain은 약 0.5로 움직임을 작게 추정했다. 5 mm 이동에서 Peg Δ 오차는 4.986 mm, Box/Hole은 5.246 mm여서, 이동에 전혀 반응하지 않는 5 mm 오차와 거의 같거나 더 나빴다.

따라서 공간 정보를 일부 학습했다는 근거는 있지만 정밀 제어에 충분한 GT state를 복원했다는 판정은 얻지 못했다. 부분 표면과 대칭성 때문에 상태의 일부가 현재 관측만으로 유일하게 식별되지 않는 문제도 남았다.

관련 근거: [F1 최종 보고서](10f1_gt_supervised_gnn_pretraining_final.md).

## Recurrent BC까지 실제로 수행한 결과

Task10-G는 F1 GNN 가중치로 초기화한 세 정책을 각 11,487 optimizer update 학습했다. 공통 BC sampling schedule과 action codec을 사용했고 학습 seed는 하나였다.

| 조건 | GNN 학습 방식 | Final Grasp | Final Lift | Final Insertion |
|---|---|---:|---:|---:|
| A | 사전학습 GNN 동결, 제어 모듈 BC | 0/12 | 0/12 | 0/12 |
| B | GNN과 제어 모듈을 BC로 공동학습 | 4/12 | 4/12 | 0/12 |
| C | 공동 BC와 GT geometry 보조 loss | 3/12 | 3/12 | 0/12 |

중요한 구현 차이가 있다. 이 비교에서 GRU에 들어간 것은 예측 GT38D가 아니라 **GNN Readout320 + Presence2 + 측정 Robot76**이다. C의 상태 예측 head는 학습 보조 loss에만 사용했고 추론 action 경로에는 넣지 않았다. 제어 모듈은 새로 초기화했으며, 기존 60% GT-GRU 가중치를 그대로 재사용한 실험이 아니다.

즉 원래 구상의 사전학습 후 recurrent BC 흐름은 실행했지만, 예측 GT38D를 원래 GT-GRU에 직접 넣는 방식의 성패까지 시험한 것은 아니다. B/C에서 일부 파지·들기가 관찰됐어도 한 학습 seed와 12 Parent로 일반적 우월성을 확정하지 않았다. Geometry 보조 loss도 정밀 이동 반응이나 insertion 성공을 확보하지 못했다.

관련 근거: [A/B/C 최종 보고서](10g_f1_initialized_graph_recurrent_abc.md), [실제 정책 입력 구현](../../graph/graph_recurrent10g/model.py).

## 실패 이후 데이터와 정렬 구간을 조사한 이유

초기 가시성 감사에서 학습 190/272 episode, Validation 25/48 episode가 Peg 양 끝과 파지 영역의 Main 가시성 기준을 통과하지 못했다. 고유 Parent로는 각각 63/96, 9/16이었다. 미통과 사례 전부에서 white 끝이 프레임 밖이었다. 이는 전체 성능 실패를 단독으로 설명하는 인과 증거는 아니지만, 완전한 물체 상태를 현재 시각 입력에서 추정하기 어려운 데이터 조건을 보여준다.

Task10-H는 동결 B 정책에 Expert prefix를 제공했다. 독립 12 Parent에서 자율 시작은 insertion 0/12, Expert가 lift까지 수행한 뒤 인계해도 0/12, align까지 수행한 뒤 인계하면 6/12였다. H2는 hole 앞 105–142 mm에서 시작하므로 접촉 직전 미세 삽입만의 시험은 아니다. 물리 상태와 GRU 관측 이력을 함께 바꾼 실험이라 geometry와 BC fit의 기여도는 분리하지 못했다.

관련 근거: [초기 가시성 감사](10g_initial_main_visibility_dataset_audit.md), [단계별 인계 진단](10h_phase_conditioned_handoff_diagnosis.md).

## 초기 가시성을 갖춘 데이터 재구축

Stage1은 926회 reset에서 가시성 통과 301개를 찾고 최종 유효 고유 후보 300개를 확보했다. Stage2는 기존 통과 자료를 보존하면서 Random R과 다양성 기반 D를 각각 96 Parent로 구성하고 공통 Validation 16 Parent를 동결했다.

D는 선정용 집합에서 R보다 평균 최근접 거리를 6.40% 줄였지만, 독립 Holdout60에서는 평균 거리가 1.57% 더 컸다. 다양성 선정이 독립 분포나 정책 성능을 개선했다는 결론은 얻지 못했다.

Stage3는 Expert trajectory를 수집하고 R/D 각각 Full-task96 + Corrective176 = 272 logical episode, Validation 48개를 확보했다. 원래 신규 선정 90 Parent 중 87개가 full-task 성공했고 실패 3개는 동결 reserve 규칙으로 대체했다. R/D는 78 Parent와 236 logical episode를 공유하므로 두 데이터셋이 독립적인 집합은 아니다.

이 단계의 성공은 데이터 수집 Expert의 성공이다. 새 데이터로 학습한 student의 성공률은 아니다.

관련 근거: [Stage1](10i_stage1_main_visible_candidate_pool.md), [Stage2](10i_stage2_diversity_aware_dataset_selection.md), [Stage3](10i_stage3_expert_dataset_collection.md).

## Perception과 Graph 변환에서 확인한 병목

Stage4A는 후보 동결 후 Internal Test와 Shared Validation까지 완료했고 최종 NO_GO였다. Validation Wrist Peg pixel recall은 약 14.30%p 높아졌지만 올바른 source 기반 Graph coverage는 약 0.75%p 낮아졌다. 잘 보이는 초기 상태도 실제 인식 성공을 보장하지 않았다.

Stage4A-2의 보정 후보는 Development36에서 올바른 Graph coverage를 48.19%에서 49.60%로 높였지만, Box 오염 두 기준과 Peg 끝부분 보존 기준을 통과하지 못했다. Internal 후보 추론은 38/60 episode 완료 후 중지됐고, 독립 성능 판정과 신규 Final 평가는 미완료다.

후속 국소 진단은 서로 다른 손실 경로를 확인했다.

- 물체 초기화: 유용한 DINO 후보가 consumer의 Null·Top-5·면적 규칙에서 탈락할 수 있었다. threshold 완화는 잘못된 선택도 늘렸다.
- 의미적 소유권: 0355 Main Peg는 정상 추적 후 Box로 이동한 것이 아니라 t30 최초 초기화부터 Box였다. 잘못된 Mask가 Memory에 들어가고 재검출 부재로 지속됐다.
- 겹침 필터: 정상 Box Mask와 잘못된 Peg Mask가 같은 표면을 주장해, 겹침 제거가 정상 Box를 지웠다. 안전한 Owner 교정은 입증하지 못했다.
- 점 예산: 0645의 지정 프레임에서 Wrist 점 추가가 cap128 안의 Main 끝점을 밀어냈다. t86은 cap256에서 Ends coverage 42.68%→100%로 복구됐지만 t6은 전체 입력 점을 유지해도 62.09%였다.
- 구조점 우선: 같은 지정 두 프레임에서 개선과 악화가 엇갈려 일괄 우선 보존을 정당화하지 못했다.
- Robot 필터: 손가락 근처 정상 Peg 표면 제거가 확인됐지만 실제 Robot 오염도 제거하므로 단순 완화의 안전성은 확보하지 못했다.

일반적인 깊이 단위·카메라 변환 오류나 정상 Wrist Mask의 Gate 거부가 주원인이라는 가설은 확인한 범위에서 지지되지 않았다. 현재 backprojection에는 픽셀 중심 +0.5 규약도 이미 적용돼 있었다.

관련 근거: [Stage4A 최종](10i_stage4a_final_report.md), [Stage4A-2 중지 상태](10i_stage4a2_paused_scope_review.md), [소유권 진단](10i_stage4a2_box_semantic_ownership_fix.md), [Mask에서 Graph까지](10i_stage4a2_mask_to_graph_surgical_fix.md), [점 예산](10i_stage4a2_point_budget_sensitivity.md), [구조점 비교](10i_stage4a2_structural_budget_2x2.md).

## 현재 답할 수 있는 연구 질문

| 질문 | 현재 답 |
|---|---|
| GNN을 GT state supervision으로 먼저 학습했는가 | 완료. 평균 상태 오차 개선을 확인 |
| 정밀한 GT-state 대체 입력을 확보했는가 | 충분성 미확보. 특히 Hole과 작은 이동 반응에 큰 한계 |
| 그 GNN에 recurrent 제어 모듈을 붙여 BC까지 했는가 | 완료. A/B/C 비교 Final insertion은 모두 0/12 |
| 예측 GT38D와 기존 GT-GRU 가중치를 직접 연결해 평가했는가 | 해당 비교 범위에서는 미실행 |
| Graph–Recurrent 접근이 불가능하다고 판정했는가 | 아님. 현재 구현·학습 조건의 실패와 부분적 복구를 확인 |
| 초기 가시성 조건을 만족하는 대체 학습 자료가 있는가 | R/D와 Validation 수집·검증 완료 |
| 새 자료와 새 perception으로 정책 성능이 개선됐는가 | 미검증. perception 후보 채택과 후속 정책 학습은 미완료 |

이 결과들이 시사하는 것은 GT 사전학습 loss, 실제 물체 이동 반응, 최종 Graph의 표면·끝부분 보존, closed-loop 성공을 각각 확인해야 한다는 점이다. 현재 실패를 GNN이나 recurrent 구조 하나의 문제로 확정하거나 새 최종 아키텍처를 결정할 근거는 없다.

## 실행과 검증 범위

이 정리는 저장된 커밋·보고서와 Task10-G 실제 입력 코드를 대조한 문서 분석이다. 새로운 학습, simulator, 모델 추론, 원격 상태 확인이나 게시를 수행하지 않았다. 역사적 실험은 지정 Linux RTX3090의 CUDA 모델·렌더러와 CPU PhysX 경로를 사용했으며 GPU PhysX나 이 macOS 호스트에서의 해당 실험 실행을 주장하지 않는다.

공통 logger와 artifact publisher의 인터페이스는 변경하지 않았다. 문서의 로컬 상대 링크 및 Git 공백 검사를 확인한다. 이 보고서는 자동 커밋하거나 push하지 않는다.
