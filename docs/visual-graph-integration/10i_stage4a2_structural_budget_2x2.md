# Task 10-I Stage 4A-2 — Structural Point Preservation × Point Budget 2×2

**판정: `LOCAL_ABLATION_COMPLETED`. Stage 4A-2는 `PAUSED_SCOPE_REVIEW` 유지.**

기존 구조점을 우선 보존하면 Point 수 증가 없이 항상 Peg 끝과 표면을 더 잘 보존한다는 근거는 얻지 못했다.
BC t6의 B128에서는 Ends가 .239535→.381395, Surface가 .198210→.207159로 개선됐다.
그러나 BC t86의 B128은 Ends .426829→.353659, Surface .755289→.561723으로 모두 악화됐다.
BC t6 B256은 Ends를 .434884→.534884로 높이면서 Surface를 .391946→.375391로 낮췄다.
이는 지정된 두 Frame의 국소 trade-off이며 Production 정책 채택이나 일반화 판정이 아니다.

## 범위와 실제 실행

- Episode `10i_stage3_10i_s1_0645_full_task`, t6/t86, B0/동결 `BC_WB_HANDOFF`만 검사했다. 추가 정상 대조는 0개다.
- A=Original128, B=Structural128, C=Original256, D=Structural256이다. Budget은 기존 코드처럼 **물체별** 적용하며 Peg 지표와 전체 Graph 수를 구분한다.
- 기존 A/C 8행을 이전 Point Budget 산출물에서 읽었다. B/D 8조건만 새로 선택하고 Graph를 조립했다.
- 신규 B/D 전에 원래 B128 Builder 4회를 실행해 저장 XYZ/Object/Source Group/Gate와 Node21/Edge4의 다섯 Tensor를 byte 단위로 재현했다.
- A/C Graph를 다시 생성하지 않았다. 이전 선택 ID로 같은 Pool의 실제 대표점을 읽고 Point Hash와 기존 지표를 재확인했다. B256은 기존 Trace/Tensor Hash/CSV Readback이다.
- 저장 RGB-D, Mask, Robot State, Calibration, Shape-First, Feature, Cross-view Identity, 3mm Dedup, 최종 Pool 순서와 평가 Reference를 유지했다.

읽은 선행 보고서는 [Point Budget](10i_stage4a2_point_budget_sensitivity.md),
[Mask-to-Graph 진단](10i_stage4a2_mask_to_graph_surgical_fix.md),
[Pause Review](10i_stage4a2_paused_scope_review.md)다.

원격 Instance **58576054**, 실행 Python `/workspace/venvs/task10i-stage4a-cu130/bin/python`에서 CPU Geometry/Graph와 사후 NumPy/SciPy 평가만 수행했다.
동결 Source Root는 `/workspace/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/source`,
이전 실험 Root는 `/workspace/task10i_stage4a2_point_budget_sensitivity_20261102_v1`,
신규 Root는 `/workspace/task10i_stage4a2_structural_budget_2x2_20261102_v1`다.
로컬 산출물 Root는 `outputs/task10i/stage4a2/structural_budget_2x2_20261102_v1/`다.

## 구조점 Identity 감사와 동결 선택 규칙

실제 호출 경로는 `make_packet → packet_from_frame → prepare_object/select_object/make_packet`이다.
`prepare_object`는 현재 예측 Mask/Depth의 PCA5 orientation >20° 후보와 관측 가능한 외곽 Silhouette를
각각 기존 nominal metric Voxel에서 실제 Source 대표점으로 압축한다. `select_object`는 기존 keep_ratio=.5,
phase=0으로 두 대표점 집합을 결정론적으로 thinning한 뒤 안정적인 합집합 `P`를 만든다.
그 뒤 Peg 최대97개/Box 최대95개의 추가 Surface FPS를 수행한다. 구조점은 이 **기존 P**이며 PCA 극점이나 GT End-band를 새로 정의하지 않았다.

Camera Packet의 `provenance.source_index`, `object_id`, `surface_selected`,
`selected_by_structural_source`(PCA/Silhouette 2열)에 실제 Source가 저장된다.
`surface_selected=False`와 두 Structural Source flag의 OR가 모든 Packet 행에서 일치했다.
`observations_from_packet`은 Source Pixel을 전달하고 `consistent_views`는 해당 관측의 부분집합만 남긴다.
`current_records`의 첫 Source `(episode,timestep,camera,pixel)`에 Packet 정보를 연결한다.

`PlanA.represent`는 그 Source Pixel로 `record.structural`을 부여한다.
3mm `deduplicate`는 `(object_id, 구조점 우선, Source key)` 순서로 처리하며 XYZ/F5를 평균하지 않는다.
병합 Source ID를 모두 남기고 구조점 flag를 OR한다. 모든 최종 Pool Group에서 원래 Packet Source별 flag를 복원했고
그 OR가 Pool flag와 일치했다. 구조점 Group의 대표 측정 Source 자체도 실제 구조점이었다.
삭제된 Cross-view Source를 되살리거나 Dedup Source Group을 변경하지 않았다.

따라서 기존 `budget(pool,K,hybrid=True) → select(anchors=True)`는 이번 정의와 일치하여 그대로 재사용했다.
별도 Sampling 구현이나 Anchor 정의는 추가하지 않았다. 실행 전 `selection_rules_frozen.json`에 다음을 동결했다.

1. 구조점≤K: 모든 구조점을 persistent_node_id 오름차순으로 seed하고, seed까지 최소 제곱 Euclidean XYZ 거리가 최대인 점으로 보충한다.
2. 구조점>K: 구조점에만 원래 `farthest_rows`를 적용한다. 첫 점은 사전식 persistent_node_id 최솟값이며 동일 거리 tie도 같은 ID 순서다. 일반 Surface점은 추가하지 않는다.
3. Pool≤K: 기존 코드처럼 원래 Pool 순서를 그대로 반환한다. 모든 선택은 물체별 min(K,Pool 수)다.

각 B/D 선택을 동일 입력으로 두 번 계산해 전체 Record Hash 일치를 확인했다. 새로운 정책의 반복은 결정론성 검사이며 추가 후보가 아니다.
GT·미래 Frame은 선택 함수 입력에 없고, GT Label/End-band 평가와 좌표 변환 그림은 모든 선택·Graph가 완료된 뒤 수행했다.
원래 `D.inputs`는 평가용 Reference를 함께 읽지만 Hook/FPS/Graph에는 센서·예측 Mask만 전달했다.

아래는 **최종 Dedup Peg 대표점** 기준이다. Packet 구조점 수와 Dedup 대표점 수를 혼용하지 않는다.

| Frame | 입력 | Pool | 구조/Surface | Dedup 구조 Main/Wrist | Packet 구조 Main/Wrist | 정상 구조 End L/R |
|---|---|---|---|---|---|---|
| 6 | B0 | 201 | 104/97 | 104/0 | 104/0 | 26/14 |
| 6 | BC | 354 | 170/184 | 104/66 | 104/92 | 32/28 |
| 86 | B0 | 114 | 20/94 | 20/0 | 20/0 | 12/0 |
| 86 | BC | 261 | 80/181 | 20/60 | 20/121 | 12/0 |

t86 BC의 Wrist Packet 구조점121개는 Dedup 구조 대표점60개로 줄지만 Source Identity는 Group에 남는다.
전체 구조점80개 중 정상 Left End 구조점은 Main의12개뿐이며 Wrist는0개다.
선행 보고서의 Packet/Shape-First 단계 수를 최종 Pool 구조점 수로 대체하지 않았다.

## 동일 입력과 원본 보존

각 Frame/Perception 조합의 Pool을 한 번 캡처하고 XYZ, Feature, 첫 Source ID, Camera,
Source Group, 구조점 및 전체 Packet Metadata, Record 순서의 SHA256을 계산했다.
4개 조합 모두 이전 실험의 모든 Budget 입력 Hash와 정확히 같았다.
A/C 선택 Point Hash도 이전 Trace와 일치했다. B/D는 같은 Pool의 깊은 복사본만 입력으로 사용했고 선택 후 원본 Pool Hash도 유지됐다.
전체 Hash와 Source별 연결은 Audit/Verification에 저장했다.

| 동결 파일 | SHA256 |
|---|---|
| development/source_config_frozen.json | 8844d2bc2f779acfec48bd62fd2b16d8f51dc15586e8d2b786dd1920412b5de5 |
| paired_graph/source_identity.json | 105a35018b486130acae9eb4310870a709ae1b6db8a1f6120d222b369ec1b5ee |
| source_config_frozen.json | beaf98d64b7fe4399ec2fe92e390641ea1ddcac1f4f2bb06cadcafdd39a96ba9 |

이전 실험의 Source of Truth Artifact Hash도 실제 원격 파일과 대조했다.

| 이전 Artifact | SHA256 |
|---|---|
| point_budget_comparison.csv | 93bc2864734ebc8be4c37dd6ab3a9563ed8ac48c0210ae626f53810a8cb6b90f |
| budget_input_identity.json | 8b72fea2f5239aa99e13fc3caa2077ab590f537e8ecaf37c1f425c69fee2133b |
| point_budget_trace.json | 7fedc8ab81e98c74aacfc6312066b9ae2615f5a6d2b7703feaea041a339a3197 |
| point_budget_comparison.png | 23ea5d6a8cda411d148d0cff374fab70ccb53c36cc75f9245b56fb4c3395b5b9 |

신규 실험 전후 보호 파일 **722개**(기존 Source/Config/Mask/Raw/GT/Freeze/Stop Stub/보존 Original 및 이전 실험 파일)를 Hash 검사했다.
Production Source, 기존 보고서/산출물, Dataset, Checkpoint, Graph Schema를 수정하지 않았다.

## 실제 16조건 결과

Coverage는 이전 평가 함수 그대로, 정상 Source **대표점**에서 거리 **<4mm**인 현재 관측 GT 표본 비율이다.
GT Mesh 최장 축의 양 끝10%를 사용한다. t6 분모는 Surface2235, Left238/Right192이며
 t86은 Surface9170, Left82/Right0이다. t86 Right는 모든 조건에서 **N/A**다.
정상 대표점 비율은 병합 Pixel 수로 가중하지 않는다. 고유 병합 Source 정상 비율도 CSV에 별도로 남겼다.

| Frame | 입력 | Arm | 선택/구조(탈락 구조) | Main/Wrist | 정상/전체 | 오점 R/BG/O | Ends | Surface | 오염률 |
|---|---|---|---|---|---|---|---|---|---|
| 6 | B0 | A | 128/58(46) | 128/0 | 124/128 | 0/4/0 | 0.313953 | 0.234004 | 0.031250 |
| 6 | B0 | B | 128/104(0) | 128/0 | 124/128 | 0/4/0 | 0.355814 | 0.217002 | 0.031250 |
| 6 | B0 | C | 201/104(0) | 201/0 | 197/201 | 0/4/0 | 0.453488 | 0.374497 | 0.019900 |
| 6 | B0 | D | 201/104(0) | 201/0 | 197/201 | 0/4/0 | 0.453488 | 0.374497 | 0.019900 |
| 6 | BC | A | 128/62(108) | 79/49 | 123/128 | 0/5/0 | 0.239535 | 0.198210 | 0.039062 |
| 6 | BC | B | 128/128(42) | 82/46 | 124/128 | 0/4/0 | 0.381395 | 0.207159 | 0.031250 |
| 6 | BC | C | 256/108(62) | 145/111 | 251/256 | 0/5/0 | 0.434884 | 0.391946 | 0.019531 |
| 6 | BC | D | 256/170(0) | 159/97 | 251/256 | 0/5/0 | 0.534884 | 0.375391 | 0.019531 |
| 86 | B0 | A | 114/20(0) | 114/0 | 106/114 | 1/6/1 | 1.000000 | 0.133479 | 0.070175 |
| 86 | B0 | B | 114/20(0) | 114/0 | 106/114 | 1/6/1 | 1.000000 | 0.133479 | 0.070175 |
| 86 | B0 | C | 114/20(0) | 114/0 | 106/114 | 1/6/1 | 1.000000 | 0.133479 | 0.070175 |
| 86 | B0 | D | 114/20(0) | 114/0 | 106/114 | 1/6/1 | 1.000000 | 0.133479 | 0.070175 |
| 86 | BC | A | 128/40(40) | 59/69 | 120/128 | 1/6/1 | 0.426829 | 0.755289 | 0.062500 |
| 86 | BC | B | 128/80(0) | 47/81 | 120/128 | 1/6/1 | 0.353659 | 0.561723 | 0.062500 |
| 86 | BC | C | 256/78(2) | 111/145 | 248/256 | 1/6/1 | 1.000000 | 0.986150 | 0.031250 |
| 86 | BC | D | 256/80(0) | 110/146 | 248/256 | 1/6/1 | 1.000000 | 0.982988 | 0.031250 |

BC의 좌우 끝과 선택된 정상 End 대표점은 다음과 같다. L/R은 화면의 좌우가 아니다.

| Frame | Arm | Left Coverage | Right Coverage | 최종 End L/R | 선택 구조 End L/R |
|---|---|---|---|---|---|
| 6 | A | 0.214286 | 0.270833 | 16/16 | 12/12 |
| 6 | B | 0.407563 | 0.348958 | 23/21 | 23/21 |
| 6 | C | 0.441176 | 0.427083 | 26/25 | 19/17 |
| 6 | D | 0.567227 | 0.494792 | 36/33 | 32/28 |
| 86 | A | 0.426829 | N/A | 35/0 | 6/0 |
| 86 | B | 0.353659 | N/A | 29/0 | 12/0 |
| 86 | C | 1.000000 | N/A | 82/0 | 12/0 |
| 86 | D | 1.000000 | N/A | 82/0 | 12/0 |

B0 t6의 A/B/C/D 정상 End L/R은20/16,29/15,34/19,34/19이며 B0 t86은 모두82/0이다.
각 Camera별 Input/Final 정상점과 End 수, Robot/BG/Other 비율, 정상 대표점 비율은 16행 CSV에 저장했다.

## 네 효과 비교

아래 Δ는 뒤 조건−앞 조건이며 **%p**다. 각 대비를 동일 Frame·Perception 안에서만 계산했다.

| Frame | 입력 | 대비 | Δ Ends %p | Δ Surface %p |
|---|---|---|---|---|
| 6 | B0 | A→B | +4.1860 | -1.7002 |
| 6 | B0 | A→C | +13.9535 | +14.0492 |
| 6 | B0 | C→D | +0.0000 | +0.0000 |
| 6 | B0 | B→D | +9.7674 | +15.7494 |
| 6 | BC | A→B | +14.1860 | +0.8949 |
| 6 | BC | A→C | +19.5349 | +19.3736 |
| 6 | BC | C→D | +10.0000 | -1.6555 |
| 6 | BC | B→D | +15.3488 | +16.8233 |
| 86 | B0 | A→B | +0.0000 | +0.0000 |
| 86 | B0 | A→C | +0.0000 | +0.0000 |
| 86 | B0 | C→D | +0.0000 | +0.0000 |
| 86 | B0 | B→D | +0.0000 | +0.0000 |
| 86 | BC | A→B | -7.3171 | -19.3566 |
| 86 | BC | A→C | +57.3171 | +23.0862 |
| 86 | BC | C→D | +0.0000 | -0.3162 |
| 86 | BC | B→D | +64.6341 | +42.1265 |

- **A→B:** t6 BC는 Ends+14.1860%p, Surface+.8949%p로 같은128개에서 개선됐다. t6 B0는 Ends+4.1860%p와 Surface−1.7002%p가 동반됐다. t86 BC는 Ends−7.3171%p, Surface−19.3566%p로 악화됐다.
- **A→C:** 이전 Budget 실험 결과를 그대로 재사용했다. BC 두 Frame의 Ends/Surface 동시 개선은 유지된다.
- **C→D:** 특히 t6 BC는 Ends+10%p이지만 Surface−1.6555%p다. 256개에서도 최종 선택의 효과가 남는다. t86은 Ends1.0 그대로이며 Surface−.3162%p다.
- **B→D:** BC t6/t86 모두 예산 증가로 Ends/Surface가 함께 개선됐다. 구조점 우선 보존이 Budget의 영향을 제거하지 않는다.

t6 BC A→B에서 오점은5→4개(모두 Background)지만 C/D에서는5개다.
t86 BC는 A/B 각8개, C/D 각8개(Robot1/BG6/Other1)로 동일하다.
구조점에도 오점이 포함되어 있으므로 구조점 우대가 정화 정책이라는 해석은 하지 않는다.

## 구조점의 끝 표현 부족과 예산 탈락의 구분

**t6 BC:** 최종 Pool 구조점170개에 정상 End60개(L32/R28)가 있다.
B128은 구조점128개만 선택해 **42개 구조점을 탈락**시키며 일반 Surface점은0개다.
정상 구조 End는44개(L23/R21)만 남아 End 구조점16개도 탈락했다.
모든 구조점을 보존했을 때의 구조점-only Ends Coverage는 .467442인데 B는 .381395다.
따라서 기존 구조점이 가진 End Support 일부를 Budget이 추가로 잃는다.
D256은170개 구조점 전부와 일반점86개를 선택한다. 정상 End69개(L36/R33), Ends .534884지만
전체 Dedup Pool의 Ends .620930보다 낮다. 구조점 자체도 Pool의 모든 정상 End78개를 포괄하지 않는다.

**t86 BC:** 구조점80개가128 예산 이하이므로 B/D 모두 **100% 보존**, 구조점 탈락0개다.
그러나 전체 정상 End82개 중 구조점은12개이고 나머지70개는 일반 Surface점이다.
구조점-only Ends Coverage는 .146341이다. B128에서12개 구조 End를 모두 보존해도
일반 End17개만 더 남아 총29개가 된다. A는 구조 End6개+일반 End29개=35개였다.
이 Frame의 악화는 구조점 Budget 초과 때문이 아니라 **기존 구조점이 끝부분을 충분히 대표하지 못하고,
그 구조점을 모두 우선 보존하면서 유용한 일반 Surface점 선택을 바꾼 결과**다.
Main/Wrist 최종 점 수도 A59/69→B47/81로 변하지만 Camera Quota를 사용하지 않았다.

B0 t6 구조점104개와 B0 t86 구조점20개는 모두128 이하이며 B/D에서 전량 보존된다.
B0 t86은 Pool114개로 모든 조건이 같다. 이 네 경우 밖의 정상 Frame 안전성은 미평가다.
다른 Anchor/Weight/Region/Camera 정책을 탐색하거나 이 두 원인을 해결하는 새 알고리즘을 개발하지 않았다.

## Graph·검증·실행 제한

| Frame | 입력 | A Node/Edge | B Node/Edge | C Node/Edge | D Node/Edge |
|---|---|---|---|---|---|
| 6 | B0 | 259/80 | 259/212 | 351/222 | 351/222 |
| 6 | BC | 259/80 | 259/160 | 406/172 | 406/414 |
| 86 | B0 | 236/780 | 236/780 | 236/780 | 236/780 |
| 86 | BC | 250/480 | 250/468 | 378/1900 | 378/1878 |

Edge 알고리즘은 동일하다. 선택된 실제 XYZ가 달라져 같은 알고리즘이 출력하는 Edge 수가 달라졌다.
`CurrentGeometry.validate_graph` 원본 AST의 마지막 `point cap` 검사 하나만 독립 Wrapper 메모리에서 분리했다.
Node21/Edge4, 유한값, Edge Endpoint, Robot Node, Current-only 등 나머지 검사는 그대로 통과했다.
256개 Graph는 원래 Validator의 `point cap` 거부도 기록했다. 기존 Model/Loader/GNN의256 지원은 실행하지 않았고 주장하지 않는다.

독립 Readback은 16행의 선택/탈락 분할, Budget, 정상/오염/Camera/End 수,
구조점 overflow·전량 보존, 모든 병합 Source Group의 정확한 전달, Coverage 재계산,
A⊆C 및 B⊆D, PNG Decode, 보호 Hash를 통과했다. A/C 기존 지표는 모든 열에서 정확히 일치했다.
단지 같은 구현을 흉내 내는 단위 테스트를 추가하지 않고 실제 저장 Frame/Graph 계약을 검증했다.

공통 `ExperimentLogger`를 W&B disabled로 재사용했고 기존 UUID/Project/Group 규칙을 유지했다.
HF/W&B 게시, 새 Instance, 신규 Benchmark는 없다. 원격 Python3.12.15/Torch2.14.1+cu130의
**CPU만** 실행했다. CUDA/MPS, GPU PhysX, SAPIEN Render, macOS Geometry/Simulator는 미실행이다.
Import 시 Vulkan ICD 경고가 있었지만 Simulator/Renderer는 생성하지 않았다.

DINO/DINOv2/SAM2 Forward, Simulator Reset/Render/Rollout, 새 GT 생성, GNN/GRU Forward/학습,
전체 Dataset 평가/Graph 재캐시, Stage4A-2 Internal/Final 재개는 모두 **0회**다.
CPU 원래 B128 Builder4회와 신규 B/D Graph8회만 실행했고 A/C Graph 재생성0회다.
Stage4B 재캐시·GNN 학습·전체 Perception 평가를 시작하지 않았으며 Stop Stub/보존 Original과 Freeze를 유지했다.
최종 프로세스 Readback에서도 Task 모델/Runner/Waiter는0개다.

## 산출물

- [구조점 정의·Source·병합 Identity·사후 End 분포](../../outputs/task10i/stage4a2/structural_budget_2x2_20261102_v1/structural_identity_audit.json)
- [실제 16조건 지표](../../outputs/task10i/stage4a2/structural_budget_2x2_20261102_v1/structural_budget_2x2.csv)
- [Source Group·선택/탈락·End·구조점 보존 Trace](../../outputs/task10i/stage4a2/structural_budget_2x2_20261102_v1/structural_selection_trace.json)
- [재현·동일 Pool·A/C 재사용·B/D·원본 보호 검증](../../outputs/task10i/stage4a2/structural_budget_2x2_20261102_v1/verification.json)
- [실행 전 동결 선택 규칙](../../outputs/task10i/stage4a2/structural_budget_2x2_20261102_v1/selection_rules_frozen.json)
- [독립 실행 Wrapper](../../outputs/task10i/stage4a2/structural_budget_2x2_20261102_v1/run_structural.py)
- [독립 사후 Readback·그림 QA](../../outputs/task10i/stage4a2/structural_budget_2x2_20261102_v1/finish.py)

![지정 Frame 네 조건의 실제 Point 배치·Ends/Surface 비교. t86 확대도와 전체 범위 Inset.](../../outputs/task10i/stage4a2/structural_budget_2x2_20261102_v1/structural_budget_2x2.png)
