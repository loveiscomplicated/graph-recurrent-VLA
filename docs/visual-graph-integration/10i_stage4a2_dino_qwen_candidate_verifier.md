# Task 10-I Stage4A-2 — DINO → Qwen Candidate Verifier

**판정: `CANDIDATE_VERIFIER_MIXED`.** 512 원본의 저장된 DINO 후보 251개를 Qwen3.5-9B로 검사했다. 단일 Label 형식 통과는 251/251 (100.0%). 동일 원시 후보 풀에서 Score Top-1 → Qwen 검사 후 선택의 PEG Recall은 10/16 (62.5%) → 8/16 (50.0%), BOX Recall은 8/16 (50.0%) → 11/16 (68.8%)였다. 기준은 visible pixel≥16, Box IoU≥0.5다. 기존 Train Parent 4개/Camera-Frame 16개를 재사용한 파일럿이며 독립 일반화나 Production 승인을 뜻하지 않는다.

**초기/후기 차이:** t0 Primary PEG/BOX 모두 8/8, t80 PEG0/8·BOX3/8. 초기 발견의 국소 개선과 후기 의미 검증 실패가 함께 관측됐다.


## 1. 연구 질문과 고정 구성

Grounding DINO가 후보 좌표를 생성하고 Qwen이 후보의 기능적 역할을 검사하면, 자유 좌표·JSON 생성 부담을 줄이고 DINO의 잘못된 후보 선택을 보완할 수 있는지 평가했다. Qwen은 새 좌표를 만들거나 Box를 수정하지 않는다.

- Vast.ai **58576054 / RTX3090 24GB**. `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, 전체 BF16·SDPA·cuda:0·offload 없음.
- 512 native RGB 16개: Parent 0531/0275/0197/0305, t0/t80, Main/Wrist. 새 Simulator Reset/Render/Step 없음.
- 직전 동결 DINO 원시 후보 251개 전체를 사용했다. Box/text threshold0.1 이후 후보이며 면적25%/Top5/Null0.3이나 GT로 Qwen 검사 대상을 줄이지 않았다. 새 DINO forward 없음.
- 기존 성공 Functional-role Prompt SHA `691ea9e22ec710dad77084db6aa812455d525190f7d6f589821bfaf082889a06` 그대로. PEG/BOX/ROBOT/OTHER/UNCERTAIN 중 정확히 단어 하나만 출력한다. 양끝 공백 제거 외 복구·JSON/fence 추출은 하지 않는다.
- Batch1, greedy, enable_thinking=False, max_new_tokens=16, use_cache=True. 모델 Load1회, 요청251회, warmup/retry/sampling/Prompt 수정0회.
- 과거 3후보의 성공 실험은 NF4·128 원본/256 Processor였다. 이번은 BF16·512이므로 그 결과와의 차이를 Prompt 하나의 인과 효과로 주장하지 않는다.

## 2. 입력과 평가 장벽

각 요청에는 ①원본 전체 RGB ②중립 청록색(0,180,220) 4px Rectangle을 그린 복사본 ③Box 각 변10% margin Crop 순서로 이미지3장과 기존 Prompt를 제공했다. Marker 4px는 128의1px와 같은 상대 두께다. Crop은 화면과의 교집합을 floor/ceil로 잘랐지만 원래 후보 좌표는 바꾸지 않았다. 전체 장면과 clean Crop은 Marker로 덮지 않았다.

Qwen에 DINO query/label/score/rank, 후보 좌표 숫자, Consumer decision, GT Actor/Segmentation/Pose, SAM2 mask를 전달하지 않았다. Request ID/파일명은 orchestration용이며 모델 text에 넣지 않았다. Processor는 각 이미지 min=max262144 pixels, 원본/표시 grid는 [1,32,32], Crop은 종횡비 유지 Patch 반올림이다. 한 요청에 한 후보만 검사했다.

모든 원시 응답251개와 예측CSV를 저장·SHA 동결한 뒤 GT sidecar와 후보 메타데이터로 평가했다. 과거 이미지·GT/성적은 이미 알려져 있으므로 연구자 맹검으로 표현하지 않는다. 모델 inference 프로그램은 RGB-only Request Manifest만 읽는다. DINO/SAM2/Qwen의 과거 동결 산출물은 보존했다.

## 3. 선택 정책과 대조군

기존 선택 Box만 검사하는 legacy_gate에서는 기존 오답 Box 선택13개 중10개를 제거했고, 기존 정답 선택16개는 모두 유지했다. 이는16개 이미지의29개 Object 출력에 대한 관측이며 독립 정답16개 Parent가 아니다. PEG Precision은8/13→8/8, BOX는8/16→8/11이었고, Miss를 복구하지 않는 이 rejection-only 경로의 Recall은 그대로였다.

모든 정책은 예측을 보기 전에 protocol.json에 고정했다. 순위는 기존 DINO score 내림차순, 동점은 원래 rank로 결정하며 GT는 선택에 사용하지 않는다.

| 정책 | 정의 |
|---|---|
| legacy | 저장된 기존 area25%/Top5/Null0.3/joint0.3 결과 |
| raw_top1 | 각 원래 Query의 전체 원시 후보에서 최고 score |
| legacy_gate | 기존 선택 Box를 Qwen이 해당 역할로 판단할 때만 유지, 교체 없음 |
| verifier_query_gate (Primary) | 같은 Query의 원시 후보 중 Qwen이 요청 역할로 판단한 후보에서 최고 DINO score, 없으면 null |
| verifier_union_roles (Secondary) | 두 Query의 전체 후보를 합쳐 Qwen의 역할 Label별 최고 DINO score 선택 |

Primary와 raw_top1은 후보 풀이 같고 의미 검사 유무가 차이다. legacy 대비 변화에는 면적/Top5/Null/joint 제한을 사용하지 않는 새 선택 경로의 효과도 포함된다. 따라서 legacy 대비 개선 전부를 Qwen의 효과로 돌리지 않는다. 독립 Object별 선택이며 새 score/overlap threshold를 만들지 않았다. Secondary는 사전 정의한 대안으로, 결과에 따라 선택한 Production 정책이 아니다. 모든 정책은 실험 전용이며 기존 Production을 수정하지 않았다.

## 4. 실제 Box 선택 결과

| 정책 | PEG Recall | BOX Recall | PEG 평균 IoU(Null 포함) | BOX 평균 IoU(Null 포함) | 두 물체 동시 성공 |
|---|---:|---:|---:|---:|---:|
| legacy | 8/16 (50.0%) | 8/16 (50.0%) | 0.571 | 0.506 | 5/16 (31.2%) |
| raw_top1 | 10/16 (62.5%) | 8/16 (50.0%) | 0.694 | 0.506 | 5/16 (31.2%) |
| legacy_gate | 8/16 (50.0%) | 8/16 (50.0%) | 0.480 | 0.506 | 5/16 (31.2%) |
| verifier_query_gate | 8/16 (50.0%) | 11/16 (68.8%) | 0.480 | 0.721 | 8/16 (50.0%) |
| verifier_union_roles | 8/16 (50.0%) | 11/16 (68.8%) | 0.481 | 0.724 | 8/16 (50.0%) |

| 정책 | 물체 | 출력 Box Precision | Null | Wrong-object / 출력 |
|---|---|---:|---:|---:|
| legacy | peg | 8/13 (61.5%) | 3/16 (18.8%) | 1/13 (7.7%) |
| legacy | box | 8/16 (50.0%) | 0/16 (0.0%) | 8/16 (50.0%) |
| raw_top1 | peg | 10/16 (62.5%) | 0/16 (0.0%) | 2/16 (12.5%) |
| raw_top1 | box | 8/16 (50.0%) | 0/16 (0.0%) | 8/16 (50.0%) |
| legacy_gate | peg | 8/8 (100.0%) | 8/16 (50.0%) | 0/8 (0.0%) |
| legacy_gate | box | 8/11 (72.7%) | 5/16 (31.2%) | 3/11 (27.3%) |
| verifier_query_gate | peg | 8/8 (100.0%) | 8/16 (50.0%) | 0/8 (0.0%) |
| verifier_query_gate | box | 11/16 (68.8%) | 0/16 (0.0%) | 4/16 (25.0%) |
| verifier_union_roles | peg | 8/8 (100.0%) | 8/16 (50.0%) | 0/8 (0.0%) |
| verifier_union_roles | box | 11/16 (68.8%) | 0/16 (0.0%) | 5/16 (31.2%) |

Box Precision은 선택 Box의 IoU≥0.5 비율이며 의미 Label 정확도와 다르다. Null은 가시 목표의 Miss로 계산한다. 미가시 물체의 Hallucination 분모는 JSON에 별도로 보존했고 분모가 없으면 N/A다.

## 5. 검사기 형식과 의미 분류

전체 strict 형식 통과 251/251 (100.0%); 출력 분포 `{"BOX": 187, "ROBOT": 30, "PEG": 34}`. GT 역할이 명확한 후보의 분류 일치율 91/193 (47.2%); Parent-macro 분류 일치율 0.481. GT role 분포 `{"ROBOT": 79, "AMBIGUOUS": 58, "PEG": 81, "BOX": 33}`.

GT 역할 진단은 Box 내부 peg/box/robot foreground가16px 이상이고 한 Actor가 foreground≥70%를 차지하면 그 역할, foreground<16이면 OTHER, 나머지는 AMBIGUOUS로 정의했다. 이는 intended object의 완벽한 정답을 증명하는 기준이 아니라 Actor 구성 기반의 진단이다. Mixed 후보는 분류 정확도 분모에서 제외하되 전체 선택 Recall에는 계속 포함한다. 동일 물체의 중복 후보 때문에 251개를 독립 Scene251개로 세지 않는다.

| Query | 후보 수 | IoU≥0.5 정답 후보 | 정답 후보 유지율 | 오답 Box 승인율 | 승인 Box 중 정답 | UNCERTAIN |
|---|---:|---:|---:|---:|---:|---:|
| peg | 124 | 30 | 16/30 (53.3%) | 6/94 (6.4%) | 16/22 (72.7%) | 0/124 (0.0%) |
| box | 127 | 20 | 20/20 (100.0%) | 79/107 (73.8%) | 20/99 (20.2%) | 0/127 (0.0%) |

여기서 오답 Box 승인은 IoU<0.5인 Box에 해당 역할을 출력한 경우다. 그 Box 안에 실제 목표 일부가 있어 의미 분류는 맞고 Box 범위만 부정확할 수도 있다. Qwen 역할 분류가 정확하다고 Box geometry가 자동으로 수정되지 않는다.

| GT role | Qwen 출력 분포 |
|---|---|
| PEG | `{"BOX": 53, "PEG": 28}` |
| BOX | `{"BOX": 33}` |
| ROBOT | `{"BOX": 49, "ROBOT": 30}` |
| OTHER | `{}` |
| AMBIGUOUS | `{"BOX": 52, "PEG": 6}` |

## 6. Camera/시점과 Parent 통계

| Primary 구분 | PEG Recall | BOX Recall |
|---|---:|---:|
| base_camera | 4/8 (50.0%) | 7/8 (87.5%) |
| hand_camera | 4/8 (50.0%) | 4/8 (50.0%) |
| F0 | 8/8 (100.0%) | 8/8 (100.0%) |
| F1 | 0/8 (0.0%) | 3/8 (37.5%) |

Primary Parent-macro Recall: peg=0.500, box=0.688. 물리 Parent 4개 단위 Bootstrap10,000회(seed42), 탐색적95% CI다. 다중 비교 보정이 없고 작은 파일럿에서의 CI/0오류가 일반화·희귀 오류 안전성을 보증하지 않는다.

| 대조군 | 물체 | Recall 차이(Primary−대조군) | 탐색적 95% CI | 평균 IoU 차이 |
|---|---|---:|---|---:|
| legacy | peg | 0.000 | [0.0, 0.0] | -0.091 |
| legacy | box | 0.188 | [0.0625, 0.25] | 0.215 |
| raw_top1 | peg | -0.125 | [-0.25, 0.0] | -0.214 |
| raw_top1 | box | 0.188 | [0.0625, 0.25] | 0.215 |

개별 Parent·회복/회귀는 [대응 변화 CSV](../../outputs/task10i/stage4a2/dino_qwen_candidate_verifier_20261102_v1/hybrid_transitions.csv)와 [전체 지표](../../outputs/task10i/stage4a2/dino_qwen_candidate_verifier_20261102_v1/hybrid_evaluation_metrics.json)에 보존했다.

### 초기 Discovery와 후기 Re-grounding을 분리한 관측

| 시점 | 정책 | PEG Recall | BOX Recall |
|---|---|---:|---:|
| F0 | legacy | 8/8 (100.0%) | 5/8 (62.5%) |
| F0 | raw_top1 | 8/8 (100.0%) | 5/8 (62.5%) |
| F0 | verifier_query_gate | 8/8 (100.0%) | 8/8 (100.0%) |
| F1 | legacy | 0/8 (0.0%) | 3/8 (37.5%) |
| F1 | raw_top1 | 2/8 (25.0%) | 3/8 (37.5%) |
| F1 | verifier_query_gate | 0/8 (0.0%) | 3/8 (37.5%) |

t0에서는 Primary가 PEG/BOX 모두 8/8이지만 t80에서는 PEG 0/8·BOX 3/8이었다. 이는 4개 Parent의 두 Camera씩을 센 값이며 독립 Parent8개가 아니다. 초기 Object Discovery용 검사기로 후속 검증 가치가 있지만 Later Re-grounding을 같은 성능으로 가정할 수 없다. 이 시점 구분은 사전에 정한 F0/F1 분석이며 t0 성적만으로 전체 결과를 대체하지 않는다.

Actor 구성 기반 진단에서 PEG 우세 후보81개 중53개와 ROBOT 우세79개 중49개를 BOX로 출력했다. BOX 출력187/251, UNCERTAIN0개라는 편향이 관측됐다. 형식 통과100%와 안정적인 의미 검증은 별개다. 특히 가림·삽입·파지 상태에서의 역할 혼동, marked region과 crop 문맥 해석은 원인 후보이며 이번 결과만으로 특정 문장이나 구현을 원인으로 확정하지 않는다.

## 7. 실제 비용

Model load 3.056s, loaded allocated 17.527 GiB, peak inference allocated 17.966 GiB. CPU sampled peak 18.309 GiB(50ms RSS). 모델 Hash 검사로 OS file cache는 warm일 수 있다.

| 비용 | n | Mean ms | Median ms | p95 ms |
|---|---:|---:|---:|---:|
| Warm 후보1개 E2E | 250 | 650.109 | 646.088 | 686.228 |
| Preprocessing | 250 | 28.358 | 28.006 | 34.593 |
| Vision CUDA | 250 | 82.721 | 82.978 | 83.756 |
| Prefill−Vision 근사 | 250 | 394.441 | 395.466 | 398.140 |
| Decode forward CUDA | 250 | 128.482 | 123.336 | 165.526 |
| 이미지별 전체 후보 직렬 요청 합 | 16 | 10245.367 | 9165.966 | 15502.047 |

Cold 첫 요청 1398.69ms는 Warm에서 제외했다. 이미지별 합은 후보마다 scene/marked/crop을 새로 처리한 비용의 합이며 이미지 작업을 연속 실행한 벽시계 값이 아니다. 한 이미지의 Vision feature/KV cache를 후보들 사이에 재사용하는 최적화는 구현하지 않았다. 요청마다 입력·세 이미지 Vision·Prefill이 새로 필요하며 20Hz 경로의 지연으로 볼 수 없다. 초기 Discovery/필요 시 검사 용도에서도 전체 후보 검사 비용을 고려해야 한다. 이 값은 exhaustive 평가 비용이다. 점수순으로 검사해 첫 합격 후 중단하는 온라인 구현이나 최적화된 최소 배포 비용은 측정하지 않았다.

이 실험에서는 저장 DINO를 재사용했으므로 Detector 비용은 위 합에 포함하지 않았다. 이전 동일 이미지 DINO 2프롬프트 비용은 별도 보고서에 있다. reference PyTorch Gated DeltaNet/causal-conv fallback 환경을 유지했고 새 kernel을 설치하지 않았다. DINO/SAM2/Qwen 동시 GPU 상주나 실제 온라인 Scheduling은 검증하지 않았다.

## 8. 해석과 한계

단일 Label은 자유 좌표·JSON 생성과 다른 출력 계약이며 실제 형식 통과율로 판단한다. 높은 분류 일치율이나 일부 후보 회복만으로 전체 시스템 안정성을 주장하지 않는다. Primary의 raw Top1 대비 이득과 잘못된 승인·정답 거부·geometry 오류를 함께 평가해야 한다. 고득점 오답이 같은 역할 Label을 받으면 최고 DINO score tie-break는 계속 실패할 수 있다. 후보 좌표가 부정확하거나 올바른 후보를 모두 거부하면 null/오류가 남는다.

이 결과는 이미 관측했던4개 Train Parent다. 추가 Parent/Prompt/threshold를 결과 뒤에 최적화하지 않았다. SAM2 초기 Mask·장기 Tracking·Object Graph·Recurrent controller·Closed-loop 성공은 미실행이다. 기존 Stage4A-2 PAUSED_SCOPE_REVIEW 및 Stop Stub을 유지하며 Production 통합·Git Commit·외부 게시를 하지 않았다.

## 9. 보존과 산출물

공통 ExperimentLogger를 W&B disabled 로컬 기록으로 재사용했다. 보호 파일 1843개 SHA, Qwen 모델/Processor15개 파일, package 목록이 전후 동일했다. 응답251개 strict parser 재해석, Primary 선택32개 GT-free replay, 선택 Box IoU 160개 독립 재계산을 통과했다. 요청 이미지3장 모두 SHA를 검증했다. 기존 역할 Prompt·모델·DINO/Qwen 과거 결과를 변경하지 않았다. GPU compute process는 종료 후 없었다. macOS GPU·Simulator·renderer는 실행하지 않았다.

- [Protocol](../../outputs/task10i/stage4a2/dino_qwen_candidate_verifier_20261102_v1/protocol.json) · [Prompt identity](../../outputs/task10i/stage4a2/dino_qwen_candidate_verifier_20261102_v1/prompt_identity.json) · [Model identity](../../outputs/task10i/stage4a2/dino_qwen_candidate_verifier_20261102_v1/model_identity.json) · [RGB-only Request manifest](../../outputs/task10i/stage4a2/dino_qwen_candidate_verifier_20261102_v1/request_manifest.json)
- [원시 응답](../../outputs/task10i/stage4a2/dino_qwen_candidate_verifier_20261102_v1/verifier_raw_responses.json) · [단일 Label 예측](../../outputs/task10i/stage4a2/dino_qwen_candidate_verifier_20261102_v1/verifier_predictions.csv) · [후보 전체 평가](../../outputs/task10i/stage4a2/dino_qwen_candidate_verifier_20261102_v1/candidate_evaluation.csv)
- [선택 Box](../../outputs/task10i/stage4a2/dino_qwen_candidate_verifier_20261102_v1/hybrid_selected_boxes.csv) · [전체 지표](../../outputs/task10i/stage4a2/dino_qwen_candidate_verifier_20261102_v1/hybrid_evaluation_metrics.json) · [실패 사례](../../outputs/task10i/stage4a2/dino_qwen_candidate_verifier_20261102_v1/hybrid_failure_cases.csv) · [Latency/VRAM](../../outputs/task10i/stage4a2/dino_qwen_candidate_verifier_20261102_v1/verifier_latency_vram.json) · [보존 검증](../../outputs/task10i/stage4a2/dino_qwen_candidate_verifier_20261102_v1/verification.json)

![사전 선정 Parent의 기존 선택과 Primary 검사 후 선택](../../outputs/task10i/stage4a2/dino_qwen_candidate_verifier_20261102_v1/hybrid_verifier_comparison.png)

주황=선택 PEG, 하늘색=선택 BOX, 초록=GT PEG, 자홍=GT BOX. 후보별 Marker/Crop은 별도 원자료이며 이 그림은 첫 고정 Parent의 네 Camera/Frame을 결과와 무관하게 표시한다. SAM2 Mask는 없다.

## Appendix — Frozen verifier prompt

```text
You are a visual object verifier for robotic manipulation.

The scene shows a robot performing a peg-insertion task. Your job is to identify the physical object inside a marked candidate region.

Important: Classify objects by their functional role in the manipulation task, not merely by their everyday appearance or shape.

Object categories:

PEG: The separate physical component that the robot is intended to grasp, move, and insert into a matching opening. A peg does not have to be cylindrical. Its shape, color, and material can vary. An object that resembles a block or another everyday item can still be a PEG if it serves this role.

BOX: The receiving object or fixture containing the opening into which the PEG is inserted.

ROBOT: Any part of the robotic manipulator, including its arm, gripper, and fingers.

OTHER: A clearly identifiable object that does not serve any of these roles.

UNCERTAIN: The visual evidence is insufficient to determine the object's role reliably.

You are given three images:
1. The complete scene.
2. The same scene with a colored rectangle marking the candidate region.
3. A crop of that candidate region.

Examine the marked region carefully. Use the complete scene to understand the object's role.

The candidate region is only a proposal and may contain an incorrectly identified object. Do not assume that the proposal is correct.

Distinguish the insertable component from the receiving fixture and the robot. Do not classify an insertable component as OTHER simply because its appearance resembles a familiar object.

If the marked region contains multiple objects and its intended object cannot be identified reliably, choose UNCERTAIN.

Output exactly one of these labels:

PEG
BOX
ROBOT
OTHER
UNCERTAIN

Do not output JSON, Markdown, explanations, punctuation, or additional text.
```

로컬 수신 검증: Artifact 537개 SHA, 역할 진단 251개 및 후보/선택 IoU 411개 독립 재계산 통과. Crop251개가 원본 RGB의 기록된 영역과 pixel-equal이며 로컬/원격 실행 Source가 일치한다.
