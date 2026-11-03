# Task 10-I Stage4A-2 — Initial RGB Reference Paired Comparison

**결과: 초기 RGB 한 장 추가로 후기 Grounding은 개선되지 않았다.** 현재 입력만 제공한 조건과 같은 Parent·같은 Camera의 초기 RGB를 추가한 조건 모두 PEG Recall0/8, BOX Recall3/8이었다. 정답 Peg 후보14개도 두 조건에서 모두 BOX로 분류돼0/14 유지였다. 문제가 됐던0275·0305 후기 Wrist Top-1 후보 역시 BOX→BOX로 남았다. 새로운 모델 추론298회와 사후 독립 검증을 완료했다.

최종 선택32개(2조건×8장×2물체)는 조건 사이에서 모두 동일했다. 후보 Label5개만 바뀌었고 모두 Robot 우세 영역에서의 ROBOT↔BOX 변화였다. 후보당 Warm 평균은655.90→798.58ms로 약21.8% 증가했다. **현재 구성에서는 표시 없는 초기 RGB만 추가하는 것으로 역할 혼동을 해결할 수 없었다.**

## 1. 고정 실험과 비교 조건

Qwen3.5-9B가 후기 후보의 역할을 혼동하는 원인에 초기 시각 맥락 부재가 관련되는지 검사했다. Vast.ai58576054 RTX3090에서 revision c202236235762e1c871ad0ccb60c8ee5ba337b9a, 전체 BF16·SDPA·cuda:0·offload 없음·greedy·thinking=False·max_new_tokens16·Batch1을 유지했다. 모델1회 Load, 후보149개×2조건=298요청. 기존4개 Train Parent(0531/0275/0197/0305)의 t80 Main/Wrist8장에 있는 모든 보존 DINO 후보를 사용했다. GT·이전 성적에 따른 후보 필터링이나 신규 DINO/SAM2 실행은 없다.

- current_only: 현재 전체 RGB + 현재 표시 RGB + 현재 Crop.
- with_initial_RGB: 같은 세 이미지에 같은 Parent·같은 Camera의 t0 전체 RGB를 네 번째 이미지로 추가.
- 초기 RGB는 native512 원본이며 Mark/Crop/GT/Label/이전 답변/물체 대응 정답을 제공하지 않았다. 전체 장면이 추가되므로 두 시점만으로 물체 연속성을 모델이 스스로 추론해야 한다.

두 조건의 모든 텍스트는 동일하다. 기존 기능적 역할 Prompt 원문은 그대로이고, 선택적 네 번째 이미지 설명을 두 조건에 똑같이 추가했다. 따라서 인과 비교는 새 current_only와 with_initial_RGB 사이에서 수행한다. 과거 검사기와는 안내 문구가 달라졌으므로 그 결과는 별도 재현 대조다. Prompt sweep/결과 후 수정은 없다.

역할 Prompt SHA `691ea9e22ec710dad77084db6aa812455d525190f7d6f589821bfaf082889a06`. 공통 Preamble SHA `9827a81266ff5c3aaf1494c48a7b4d64484f8f4ab5a03b6be96bce9b47a2d415`. 공통 추가 안내:

```text
If a fourth image is present, it is the initial RGB frame from the same scene and camera, provided only as earlier visual context. Always classify the candidate marked in image 2 of the current scene, not an object in the initial frame.
```

각 요청은 새 user 메시지 한 개이며 이전 응답이나 cache를 전달하지 않았다. 실행 Hook으로 최초 Forward의 cache length가0이고 현재 입력 전체를 Prefill했음을298/298 확인했다. 요청은 고정 Hash 순서로 두 조건을 교차했다. 모든 원시 응답과 예측CSV를 SHA 동결한 후 GT Actor Segmentation을 열었다. 이미 알려진 개발 장면의 기전 진단이며 연구자 맹검·독립 일반화 평가가 아니다.

## 2. 동일 후보 풀에서 실제 선택 Recall

Strict Label이 Query 역할과 같은 후보 중 최고 기존 DINO score를 선택하고 없으면 null이다. 기존 순위·좌표를 유지하며 면적/Top5/Null-score/joint 제한은 추가하지 않았다. 성공은 visible≥16px, Box IoU≥0.5다.

| 방법 | PEG Recall | BOX Recall | PEG 평균 IoU, null 포함 | BOX 평균 IoU, null 포함 | 두 물체 동시 성공 |
|---|---:|---:|---:|---:|---:|
| raw_top1 | 2/8 (25.0%) | 3/8 (37.5%) | 0.4283 | 0.4032 | 0/8 (0.0%) |
| historical_verifier | 0/8 (0.0%) | 3/8 (37.5%) | 0.0000 | 0.4639 | 0/8 (0.0%) |
| current_only | 0/8 (0.0%) | 3/8 (37.5%) | 0.0000 | 0.4687 | 0/8 (0.0%) |
| with_initial_RGB | 0/8 (0.0%) | 3/8 (37.5%) | 0.0000 | 0.4687 | 0/8 (0.0%) |

raw_top1과 historical_verifier는 이전 동결 출력을 동일8장으로 제한한 값이다. 초기 t0의8/8 결과를 이번 후기 Recall 분모에 섞지 않았다.

## 3. 정답 후보 거부·오답 승인 및 Camera별 결과

| 조건 | Query | 정답 후보 유지 | 오답 후보 승인 |
|---|---|---:|---:|
| current_only | peg | 0/14 (0.0%) | 0/58 (0.0%) |
| current_only | box | 12/12 (100.0%) | 57/65 (87.7%) |
| with_initial_RGB | peg | 0/14 (0.0%) | 0/58 (0.0%) |
| with_initial_RGB | box | 12/12 (100.0%) | 58/65 (89.2%) |

current_only: 형식 149/149 (100.0%), 명확 Actor-role 일치 28/111 (25.2%), Label 분포 `{"BOX": 132, "ROBOT": 17}`.


with_initial_RGB: 형식 149/149 (100.0%), 명확 Actor-role 일치 25/111 (22.5%), Label 분포 `{"BOX": 135, "ROBOT": 14}`.

후보 수는 중복 영역을 포함하며 독립 Parent 수가 아니다. 역할 Label이 맞아도 느슨한 Box는 IoU 기준에서 오답이다. 명확 Actor-role은 foreground≥16px에서 한 Actor가70% 이상인 경우이며 intended-object의 완벽한 정답을 뜻하지 않는다.

| 조건 | Camera | PEG Recall | BOX Recall |
|---|---|---:|---:|
| current_only | base_camera | 0/4 (0.0%) | 3/4 (75.0%) |
| current_only | hand_camera | 0/4 (0.0%) | 0/4 (0.0%) |
| with_initial_RGB | base_camera | 0/4 (0.0%) | 3/4 (75.0%) |
| with_initial_RGB | hand_camera | 0/4 (0.0%) | 0/4 (0.0%) |

## 4. 사전 지정 회귀 사례와 재현성

| 후보 | 과거 Label | current_only | with_initial_RGB |
|---|---|---|---|
| 10i_s1_0275_F1_hand_camera_native512_peg_rank000 | BOX | BOX | BOX |
| 10i_s1_0305_F1_hand_camera_native512_peg_rank000 | BOX | BOX | BOX |

current_only의 과거149개 Label 재현 일치 143/149 (96.0%). 공통 안내 문구가 있으므로 과거 대비 변화를 초기 이미지 효과로 계산하지 않는다. 두 새 조건 사이 Label 변화 5/149. 전체 전이는 지표JSON과 CSV에 보존했다.

![초기 참고 영상과 두 조건의 후기 선택](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/temporal_reference_comparison.png)

왼쪽은 초기 참고 RGB, 가운데·오른쪽은 현재 RGB에 사후 평가 Box를 표시한 그림이다. 주황=선택PEG, 하늘색=선택BOX, 초록=GT PEG, 자홍=GT BOX. SAM2 Mask는 없다. 두 회귀 사례는 사전에 지정한 진단 대상이며 대표 표본으로 해석하지 않는다.

## 5. Parent 단위 통계

독립 집계 단위는 물리 Parent4개다. 각 Parent의 두 Camera Recall을 평균한 paired 차이에 cluster bootstrap10,000회, seed42를 적용했다. 탐색적95% CI이고 다중 비교 보정이 없으며 작은 표본이다.

| Object | Parent-macro Recall 차이, 추가−현재 | 탐색적95% CI |
|---|---:|---|
| peg | 0.0000 | [0.0, 0.0] |
| box | 0.0000 | [0.0, 0.0] |

## 6. 실제 비용

| 조건 | Warm n | Mean ms | Median ms | p95 ms | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| current_only | 149 | 655.90 | 652.20 | 689.68 | 17.968 |
| with_initial_RGB | 148 | 798.58 | 796.86 | 835.96 | 18.047 |

Model Load 3224.54ms, global cold 첫 요청 with_initial_RGB 1632.48ms는 위 Warm에서 제외했다. 각 이미지에512² pixel budget을 적용해 초기 RGB는256개 visual token을 추가한다. 세 이미지의 첫 두 grid는[1,32,32]이고 Crop은 종횡비를 유지한다. 입력 Token·Vision·Prefill·Decode 상세는 Latency JSON에 있다. 기존 reference kernel fallback 환경을 유지했다. 모든 요청은 새 Vision/Prefill이며 과거 cache 재사용 최적화는 없다. DINO는 저장 결과를 썼으므로 비용에 포함하지 않았다.

## 7. 해석 범위와 보존

관측된 Label 전이는 BOX→BOX131개, ROBOT→ROBOT13개, ROBOT→BOX4개, BOX→ROBOT1개였다. 변한5개는 모두 Actor 구성상 ROBOT 우세이며 PEG로 회복된 후보는 없었다. 초기 RGB가 응답에 전혀 영향을 주지 않았다고 말할 수는 없지만, 필요한 정답 Peg 복구에는 도움이 되지 않았다. BOX의 IoU<0.5 후보 승인도57/65→58/65로 감소하지 않았다. 현재 영상에서의 Peg 우세 후보49개는 두 조건 모두 BOX였다. 명확 Actor-role 일치율은28/111→25/111이었다.

이 실험은 초기 RGB가 없다는 사실만으로 이전 실패를 설명하기 어렵다는 근거를 제공한다. 초기 RGB를 추가하면 정체 연결이 자연스럽게 회복될 것이라는 가설은 이 조건에서 지지되지 않았다. 다만 명시적 Object correspondence·초기 모델 판정의 전달·중간 프레임·연속 Tracking은 시험하지 않았으므로 시간적 추론 전체의 한계를 확정하지 않는다. 출력 형식은 양쪽149/149 통과했고 빈 시작 cache도298/298 확인했다. JSON/좌표 생성이나 대화 누적 문제로 이번 실패를 설명할 증거는 없다.

8장 모두 PEG/BOX GT가16px 이상 보여 평가 가능했다. 미가시 물체나 TINY_VISIBLE 사례가 없으므로 해당 조건의 Hallucination 성능을 추정하지 않는다. PEG는 두 조건 모두8장 전부 null이므로 Precision을 성공률100%로 표현할 수 없다. Parent Bootstrap CI[0,0]은 네 Parent에서 관측된 차이가 모두0인 데 따른 재표집 결과이며 모집단 동등성을 입증하지 않는다.

초기 RGB 추가로 성능이 변하면 이 고정 입력 구성에서 참고 영상 제공의 효과를 지지한다. 다만 추가 이미지 유무는 시각 Token 수와 문맥량도 함께 바꾸므로, 개선만으로 시간적 정체 추적이 내부적으로 성공했다고 증명할 수 없다. 반대로 개선이 없더라도 모든 시간 정보 방식이 무효라는 뜻은 아니다. 이번에는 초기 RGB 한 장만 제공했고, 중간 궤적·초기 확정 Label·Object correspondence·명시적 Tracking은 제공하지 않았다. Parent 추가나 새로운 Prompt 탐색은 수행하지 않았다.

기존 보호 파일2381개 SHA와 Qwen 모델/Processor 파일·package 목록이 전후 동일했다. Strict Label298개 재해석, 빈 최초 cache298회, GT-free 선택 재현32개, 선택 IoU 독립 재계산64개를 통과했다. 공통 ExperimentLogger를 W&B disabled로 재사용했다. Production·PAUSED_SCOPE_REVIEW·Stop Stub을 유지했다. 새 Simulator/Training/SAM2/Graph/Controller/Closed-loop 실행이나 Git Commit·외부 게시는 없다. 종료 후 GPU compute process가 없었다. macOS에서는 GPU/Simulator를 실행하지 않았다.

## 8. 산출물

- [Protocol](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/protocol.json), [Prompt identity](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/prompt_identity.json), [Model identity](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/model_identity.json), [Input manifest](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/input_manifest.json), [Request manifest](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/request_manifest.json)
- [원시 응답](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/verifier_raw_responses.json), [Label 예측](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/verifier_predictions.csv), [후보 평가](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/candidate_evaluation.csv), [Label 변화](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/paired_label_changes.csv)
- [선택 Box](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/selected_boxes.csv), [전체 지표](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/temporal_reference_metrics.json), [Latency/VRAM](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/verifier_latency_vram.json), [Verification](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/verification.json)

## 9. 로컬 독립 검증

원격 Artifact44개 SHA와 실행 Source7개가 로컬 수신본과 일치했다. 요청 이미지 SHA1,043건, Actor-role298건, 후보/선택 IoU660건을 별도로 재계산했다. GT 없이 두 조건 선택32개를 재현하고 Recall과 Parent-macro를 독립 검증했다. 실행 시작 cache0과 strict Label298건도 다시 확인했다. 비교 그림을 직접 열어 두 조건의 동일한 선택과 GT/예측 구분을 검사했다.

- [로컬 검증 기록](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/local_verification.json) · [검증 스크립트](../../outputs/task10i/stage4a2/qwen35_temporal_reference_20261102_v1/local_verification.py)
- [이전 원인 진단](10i_stage4a2_dino_qwen_verifier_failure_audit.md) · [기존 검사기 파일럿](10i_stage4a2_dino_qwen_candidate_verifier.md)
