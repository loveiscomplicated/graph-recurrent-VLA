# Task 10-I Stage 4A-2 — Box Semantic Ownership

2026-11-02. **판정: `ROOT_CAUSE_CONFIRMED_NO_SAFE_FIX`.**

Main Peg는 정상 Peg를 추적하다 Box로 이동한 것이 아니다. **t0–t29에는 Main Peg Mask가
없었고, t30의 늦은 최초 초기화부터 이미 Box였다.** 잘못된 DINO 후보의 시간적 일치를
확인해 채택한 뒤, 이를 SAM2 Conditioning Memory에 저장했다. 기존 found 상태와 BC의
Main Peg 보정 비활성화가 오류를 유지했다. 안전한 RGB-D 기반 Owner 교정은 입증하지
못했으므로 Patch와 Filter 변경은 하지 않았다. `PAUSED_SCOPE_REVIEW`를 유지한다.

## 범위·Source·재현

실패는 `10i_stage3_10i_s1_0355_full_task`, Parent `10i_s1_0355`의 **t66/t85**만 조사했다.
그 Episode의 t0–t85 기록을 읽었고, 실제 정상 대조는 이전 Surgical Audit에도 있던
`10i_stage3_10i_s1_0645_full_task`의 t6/t86 **두 프레임**으로 고정했다.
새 Episode/Seed/Benchmark는 없다. 이전 Surgical/Paused/Stage4A 최종 보고서를 읽고
기존 `mask_to_graph_trace.csv`와 `first_loss_attribution.json`의 두 Box 사례를 대조했다.

Instance **58576054**에서 다음 보존 자료를 사용했다.

| 자료 | 실제 경로 |
|---|---|
| Raw RGB-D·Robot State | `/workspace/task10i_stage3_20261031_v3/run/parents/10i_s1_0355/full_task/episode.h5` |
| B0 원래 Source·Detection·Adapter 기록 | `/workspace/task10i_stage4a2_newdev_A_20261101_v1/{source,run/development}/` |
| 동결 BC Source·Mask·Correction/SAM/DINOv2 기록 | `/workspace/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/{source,run/development}/` |
| 기존 Graph·Provenance | 위 BC Root의 `run/paired_graph/episodes/{episode}/` |
| 기존 EXACT_GT | `/workspace/task10i_stage4a2_newdev_reference_20261101_v1/run/references/{episode}/` |
| 새 독립 Output Root | `/workspace/task10i_stage4a2_box_semantic_ownership_20261102_v1/` |

B0 Source/Config Manifest `cd2cf5a6…cadf04d`의 590항목, BC Coordinator
`beaf98d6…96ba9`의 689항목, BC Development `8844d2bc…2b5de5`의 689항목,
Paired Graph `105a3501…1b5ee`의 156항목을 원본 파일과 대조했다. 중복 항목을 포함한
Manifest 수이며 합산한 독립 파일 수가 아니다. 두 Episode의 Raw·Mask Receipt·GT SHA와
B0 복사본의 원본 일치도 확인했다. 전체 SHA는 [입력 Identity](../../outputs/task10i/stage4a2/box_semantic_ownership_20261102_v1/input_identity.json)에 있다.

**모델 Replay 전 재현 장벽 PASS:** 실패2 + 대조2 프레임 × B0/BC의 Graph **8개**가
저장본의 5개 Tensor, World Point, Provenance, Gate와 byte-equal이다. Geometry/Graph
재현에 GT Actor/Pose/Shape를 입력하지 않았다.

## 최초 오류: t25 후보 대기 → t30 잘못된 최초 초기화

| 시점 | 실제 Main Peg 경로 |
|---|---|
| t0 | 유효 후보 중 실제 Peg를 포함하는 검출이 있지만 score **.291640937**로 Null .3에 미달. 최종 Peg 선택은 `None`; SAM에 Peg Box를 전달하지 않음 |
| t0–t24 | Main Peg Mask 0. 5-step 미발견 Slot Scan을 지속 |
| t25 | Peg Phrase의 rank0 후보가 Box 영역을 가리킴. score **.393589884**. 첫 확인 후보로 대기; SAM 초기화 없음 |
| t29 | Main Peg Mask 여전히 0 |
| t30 | 같은 Box 영역의 rank0 후보 score **.448092550**. t25 후보와 IoU **.995154399**여서 연속 확인 승인. `new_object_ids=[1]` |
| t30 최초 Mask | **Peg TP2 / Box FP896 / Robot0 / Background0**, 총898 Pixel. EXACT_GT |
| t31 이후 | 같은 잘못된 Identity를 전파. t30 Conditioning Memory를 유지하고 현재 Prediction의 Memory를 계속 기록 |
| t66 / t85 | Main Peg가 여전히 Box를 주장. B0와 BC의 Main Peg Mask는 동일 |

t0의 유용한 Peg 검출 Rectangle은 평가상 Peg530/Box9/Robot0 Pixel을 포함한다.
이것은 **유용한 위치 후보가 있었다는 증거**이며, 실행하지 않은 t0 Peg SAM의 성공을
뜻하지 않는다. t0 Peg Mask의 Box/Robot 혼입은 0이지만 Mask 자체가 비어 있기 때문이다.

t30에 실제 전달된 Peg Box는 `[40.506973,39.060646,74.560875,71.436760]`이다.
그 Rectangle에는 평가상 **Peg33 / Box899 / Robot24 Pixel**이 있었다. SAM은 올바른
Peg Prompt를 받고 다른 물체로 Drift한 것이 아니라, 처음부터 Box 중심의 Prompt를 받았다.

원래 [`detect_initial`](../../visual/runtime/initialization.py)의 미발견 Slot Scan은
`other=result[other]['box']`를 고정하고 score≥.3, exclusion IoU<.3, 이전 Scan IoU≥.5를
확인한다. 이 Episode의 `other`는 **t0에 채택한 큰 Box 검출 상자**다.
t30 Peg 후보와 이 상자의 IoU는 **.286994759 < .3**이므로 제외되지 않았다.
현재 SAM Box Mask나 BC의 정제된 Box Mask와 비교하는 경로가 아니다.
동일 Box 영역의 반복 검출이 Semantic Peg 확인으로 채택된 것이 구체적인 잘못된 결정이다.

저장 Raw Proposal의 Query·Token Score·Phrase·Rank·실제 소비 Top5·Rectangle 평가와
Scan 재구성은 [DINO 감사](../../outputs/task10i/stage4a2/box_semantic_ownership_20261102_v1/dino_proposal_audit.json),
[Prompt Provenance](../../outputs/task10i/stage4a2/box_semantic_ownership_20261102_v1/prompt_provenance.json)에 있다.
Decoded Phrase도 `red white block`이어서 Phrase 일치만으로 Box를 배제할 수 없었다.
Score는 보정된 Semantic 확률이 아니다.

## Memory와 재검출: 최초 발생과 지속 원인을 분리

저장 BC SAM Trace는 보정용 별도 State만 관찰하며, 원래 Main Peg의 Logit/Memory는
없었다. 이 공백만 확인하기 위해 **B0 한 조건의 t0–t85 Causal Replay 1회**를 실행했다.
원래 Source·Checkpoint·JPEG 경로·Seed42를 유지하고 DINO는 저장 결과로 대체했다.
관찰 Hook은 반환값을 바꾸지 않았다. **86/86 프레임의 양 Camera/양 Slot Mask가 모두
저장 B0와 byte-equal**이었다. BC Main Peg는 원래 B0 복사본이므로 별도 BC Replay는 없다.

Replay에서 t30 Peg Prompt 출력은 처음 `run_mem_encoder=False`였으나, 바로 이어지는
설치 SAM2의 `propagate_in_video_preflight`가 이를 Memory Encoding하고
`cond_frame_outputs[30]`에 저장했다. 실제 Memory Shape는 `[1,64,64,64]`다.
t31은 이 Conditioning Output을 입력받았고, t66/t85에도 Conditioning Key30과 현재
Non-conditioning Memory가 존재했다. 저장된 Key 전체가 매번 Attention에 모두 사용된다는
뜻은 아니다. 이 검사는 잘못된 초기 표면이 **실제로 Memory에 채택됐다**는 증거다.
Memory가 없는 조건과의 인과효과 크기 비교는 하지 않았다.

[`OnlineVisionAdapter.process`](../../graph/integration_baseline/online_vision_adapter.py)는
한 번 채택한 Slot을 `found`에 유지한다. Main은 t30부터 두 Slot이 모두 found라 Scan을
중단했고 누적 Detector Call35가 t85까지 그대로였다. t66/t85의 GT Peg는 각각323/190
Pixel로 보이지만, 이를 보고 복구하는 실제 Main Peg 재검출은 없었다.

BC의 [`development_correction`](../../graph/task10i_stage4a2/development_correction.py)은
Wrist Peg/Main Box만 활성화한다. Main Peg는 `scope_enabled=false`이고 t31,36,…,81의
불확실성 기반 재검출 예정도 실제 `detection_requested=false`로 차단됐다.
Trace의 `REDETECTION_REQUESTED` 상태 이름만 보고 모델 호출이 있었다고 세지 않았다.
Main Peg를 지우는 보정도 없어 B0의 잘못된 Mask를 그대로 반환했다.

| 원인 유형 | 이 사례의 판정 |
|---|---|
| `INITIAL_DINO_WRONG_ASSIGNMENT` | **확인**, t0가 아닌 t30의 늦은 최초 Peg 초기화 |
| `INITIAL_SAM2_WRONG_MASK` | 최초 Mask부터 잘못됨은 확인. 올바른 Peg Prompt 이후 독립 SAM 실패라는 증거는 없음 |
| `SAM2_TRACKING_DRIFT` | 최초 발생 원인 아님. 앞선 정상 Main Peg Track이 없음 |
| `SAM2_MEMORY_CONTAMINATION` | **확인**, 잘못된 최초 Prompt/Mask의 Memory 채택과 지속 경로 |
| `REDETECTION_NOT_TRIGGERED` | **확인**, B0 found Latch 및 BC Main Peg Scope 차단 |
| `REDETECTION_WRONG_ACCEPTANCE` | **미발견 Slot Scan의 t30 채택**에 해당. 이미 found인 Peg의 복구 보정은 아님 |
| `CROSS_OBJECT_MASK_CONFLICT` | **확인**, 잘못된 Peg Mask와 Box Mask의 후속 충돌 |
| `UNKNOWN_OR_INSUFFICIENT_EVIDENCE` | DINO 내부 순위의 학습적 이유, 현재 재검출의 복구 가능성, 안전한 Owner 규칙은 미확정 |

따라서 경로는 **늦은 잘못된 Prompt 채택 → 처음부터 Box인 Peg Mask/Memory → 재검출
차단으로 지속 → 양 Slot 충돌 → Overlap 제거로 Box 손실**이다.

## 현재 출력으로 Owner를 고를 수 있는가?

**검증된 안전한 판정 근거는 없다.** 오류를 진단하는 평가 근거와 실행 시 Owner를
결정할 독립적인 근거를 구분한다.

| 신호 | 실제 확인과 한계 |
|---|---|
| DINO | Raw Query/Token과 t25/t30 선택을 보존. 잘못된 Box 후보도 높은 점수·맞는 Phrase·시간 일치를 통과. t66/t85 Main Peg의 새로운 검출 출력은 없음 |
| SAM Pixel Logit | Replay에서 저장한 원래 Logit을128×128로 복원하면 실제 Binary Mask와 일치. Prompt별 Foreground 점수이며 Peg/Box 의미 경쟁에 대해 보정되지 않음 |
| SAM Presence / Quality | 잘못된 Main Peg도 높은 값. 설치 Source의 Presence는 Prompted Instance의 존재, IoU Head는 Mask 품질 추정. 별도 State의 의미적 Owner 확률이 아님 |
| RGB-D | 충돌 Pixel은 두 Slot이 같은 Depth 측정을 공유. 둘 다 연결된 Nonrobot 표면을 가짐. 연결성·Depth 일관성만으로 Peg/Box를 구분하지 못함 |
| Cross-camera | BC 충돌 Pixel 중 Wrist Depth와4mm 이내 일치하는 것은 t66 **50/621**, t85 **38/602**. 이 중 Wrist Peg가 주장하는 Pixel은 모두0. 다른 View의 Mask 부재를 부정 증거로 취급할 수 없음 |
| DINOv2 | B0 Forward0. BC는 t8 Main Box/t28 Wrist Peg Reference와 t57 후보 비교에 실제 호출. Main Peg Reference·Re-ID 비교·보정은0. 새 Matching 시스템을 추가하지 않음 |

특히 SAM Presence/Quality는 다음과 같다. Quality는 실제 Decoder 후보 IoU 추정 중
최댓값이며, Semantic 정확도나 GT IoU가 아니다.

| Frame | 잘못된 Main Peg Presence / Quality | BC Main Box Presence / Quality |
|---|---|---|
| t66 | **10.705775 / .980459** | 10.394090 / .979327 |
| t85 | **10.811929 / .979974** | 10.655851 / .979001 |

큰 Presence/Quality를 고르면 이 두 예에서 오히려 잘못된 Peg 쪽이 이긴다.
별도 State의 Pixel Logit Argmax도 검증하지 않았다. 원래 BC 현재 Dense Logit은
저장되지 않았고 Prompt Logit만 있다. 이를 얻으려고 두 번째 Replay를 실행하지 않았다.
[SAM 신뢰 감사](../../outputs/task10i/stage4a2/box_semantic_ownership_20261102_v1/sam_confidence_audit.json),
[Geometry/Owner 감사](../../outputs/task10i/stage4a2/box_semantic_ownership_20261102_v1/geometry_ownership_evidence.json),
[설치/동결 Source 위치와 SHA](../../outputs/task10i/stage4a2/box_semantic_ownership_20261102_v1/source_sites.json).

## 고정 B0/BC의 국소 결과와 Filter 판단

아래는 실제 Actor Source 감사다. Mask는 Main Camera, Graph는 현재 Main/Wrist를
원래 계약으로 조립한 최종 대표점이다. `TP/전체`의 분모를 구분했다.

| Frame·조건 | Main Peg TP / Box FP | Main Box TP / Peg·Robot·Background FP | Filter 후 Box TP | 최종 Box Correct/전체 | 최종 Peg Correct/전체 |
|---|---|---|---:|---:|---:|
| t66 B0 | 1 /620 | 605 /0·1772·0 | 0 | 0/4 | 1/16 |
| t66 BC | 1 /620 | 620 /1·0·2 | 0 | 0/2 | 80/80 |
| t85 B0 | 0 /597 | 586 /0·1467·1 | 0 | 0/1 | 0/11 |
| t85 BC | 0 /597 | 597 /1·7·0 | 0 | 0/1 | 83/83 |

t85 Main Peg에는 Robot FP5도 있다. BC Box의 Precision/Recall/IoU는 t66
**.995185/.998390/.993590**, t85 **.986777/.998328/.985149**다. 이 좋은 Raw Mask가
Filter 뒤에는 정상 Box0이 된다. BC 최종 Peg는 Wrist Source로 정상이지만 Main Peg
Identity 오류를 해결한 결과가 아니다. B0 최종 Peg의 다른 물체 오염은 t66 **15/16**,
t85 **11/11**이며, BC Box Source Purity는 두 시점 모두 **0**이다.
전체 Mask Precision/Recall/IoU와 고유 병합 Source 기준 Graph Purity는
[국소 회귀 JSON](../../outputs/task10i/stage4a2/box_semantic_ownership_20261102_v1/local_regression_results.json)에 있다.

기존 [`filtered_masks`](../../graph/persistent_geometry/fusion.py)는
`overlap | confirmed_robot | uncertain`의 합집합을 양쪽에서 제거한다.
현재 충돌 영역에 Owner를 안전하게 부여할 수 없으므로 이 계약을 유지했다.
충돌을 Unknown으로 보류하는 진단도 현재 출력과 같으며 새 후보가 아니다.

**GT Owner Oracle은 평가용 Mask 상한만** 계산했다. Robot 제거를 유지하고 충돌 Pixel의
Owner만 GT로 고르면 BC t66/t85의 Box TP620/597과 Peg TP1/0을 보존할 수 있다.
Box를 무조건 우선하면 t66의 실제 Peg1도 Box에 넘길 수 있으므로 성공 규칙으로
사용하지 않는다. Oracle을 Practical 개선, Patch 결과 또는 새 Graph 성능으로 계산하지 않았다.

정상 대조0645 t6은 Peg TP842와 Box TP491이 분리되어 있고 Overlap0이다.
t86은 Peg TP106/Box TP116, Overlap0이며 일부 기존 FP가 있다.
두 대조의 저장 B0/BC Graph 재현은 통과했지만 **실제 Peg가 Box 앞을 가리는 경우의
Owner 수정 안전성은 이 두 사례로 확정하지 못했다.** 원래0355 t0 Box 자체에도
Robot1848/Background7 FP가 있으므로 Box Slot을 항상 신뢰할 수도 없다.
음성 대조의 미지원 항목을 PASS로 만들지 않았고 Synthetic 성공 수치도 만들지 않았다.

## Patch를 만들지 않은 이유와 남은 정보

잘못된 State Transition은 특정했다. 그러나 stale Box 검출 대신 현재 Box Mask를
참조하거나, 충돌 시 Main Peg를 삭제하는 것만으로는 어느 Slot이 틀렸는지 확정되지 않는다.
실제 Peg 가림이나 Box 자체의 오류에서 정상 Peg를 제거할 가능성을 현재 자료로 배제할
수 없다. 이 규칙을 Patch로 만들고 Box TP 복구만 성공으로 보고하지 않았다.

안전한 수정에는 현재 RGB-D/모델 출력에서의 **독립적인 Semantic Owner 근거와 그
거부/보류 조건**, 그리고 실제 Peg 가림·잘못된 Box Slot의 국소 음성 대조가 필요하다.
이 보고서는 그 새 규칙이나 추가 실험을 승인·설계·실행하지 않는다.
BC의 Main Box 재보정도 충돌 뒤 거절되는 부수 경로는 기록만 했으며 해결하지 않았다.

## 계산·보존·산출물

- Causal Replay **1회**, `0355 full_task` B0 t0–t85만. SAM2 Frame Inference228회.
  새 DINO/DINOv2 Forward0, BC Replay0, Patch0.
- CPU Graph 재현8개. 지정 실패2 + 저장 대조2 이외의 Graph 재현/재캐시는 없다.
  Simulator Rollout/GT Render/학습/전체 평가/외부 게시0.
- Instance의 Python3.12·RTX3090 CUDA에서 SAM2 Observer Replay를 실행했고,
  Geometry/Graph와 NumPy/SciPy 평가는 CPU에서 실행했다. GPU PhysX, SAPIEN Render,
  macOS Simulator/MPS는 실행하지 않았다. Import 시 Vulkan ICD 경고가 있었으나
  Simulator/Renderer를 생성하지 않았다.
- 입력·동결 Source·Candidate5개 파일·Launcher Stop Stub5개 및 보존 Original의
  시작/종료 SHA가 같다. Task 모델/Runner/Waiter 실행 프로세스0을 확인했다.
  기존 Raw/GT/Mask/Graph/Checkpoint/Production Source를 덮어쓰지 않았다.
- 공통 `ExperimentLogger`를 W&B disabled로 재사용했다. Replay UUID는
  `92bcbddfea034ce7aebf957fad568be8`, 전체 분석 UUID는 `5de1a8e257734d7eb8511ad481bc3608`이다.
  Project/Group/UUID 규칙과 기존 Pause 상태 파일은 바꾸지 않았다.

초기 저장 자료 감사에서 설치 Token Phrase Helper의 인자 수를 잘못 사용해 후처리가
한 번 중단됐고, 설치 Signature에 맞춰 고쳤다. 이때 모델 Forward와 Graph 재현은
시작 전이었다. 로그를 보존했으며 Causal Replay 재시도는 없다.
원본 파일 Container는95프레임이다. Actor Timeline/추론은 t0–t85로 제한했고,
한 Metadata Readback에 함께 표시된 t86/t91 비활성화 시각은 판정에 사용하지 않았다.

필수 산출물:

- [시간순 CSV](../../outputs/task10i/stage4a2/box_semantic_ownership_20261102_v1/main_peg_box_identity_timeline.csv):
  688행 =86시점 ×2조건 ×2Camera ×2Slot. GT_UNSUPPORTED t6/12/17/46/67의 Actor 수치는
  비워 두었고 추정으로 채우지 않았다. Memory는 저장 BC Trace/일치한 B0 Replay의 출처를 구분한다.
- [원인 JSON](../../outputs/task10i/stage4a2/box_semantic_ownership_20261102_v1/semantic_ownership_root_cause.json).
- [국소 회귀·보존 결과](../../outputs/task10i/stage4a2/box_semantic_ownership_20261102_v1/local_regression_results.json).
- [단일 비교 PNG](../../outputs/task10i/stage4a2/box_semantic_ownership_20261102_v1/case_comparison.png):
  최초 전환의 RGB-D와 t66/t85의 B0/BC Mask·GT·Overlap·Filter·실제 최종 Graph Object Subgraph.
- [완료 Receipt](../../outputs/task10i/stage4a2/box_semantic_ownership_20261102_v1/completion_receipt.json).

`minimal_patch/`는 생성하지 않았다. 이 작업은 원인 확인과 안전한 수정 부재의 보고로
종료한다. Internal30/Final Normal32·Stress12/추가 후보/학습은 계속 중지되어 있다.

![Box Semantic Ownership 비교](../../outputs/task10i/stage4a2/box_semantic_ownership_20261102_v1/case_comparison.png)
