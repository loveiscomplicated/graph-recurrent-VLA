# Task 10-I Stage4A-2 — Frozen Appearance Prompt, BF16 Parent-held-out Evaluation

**판정: `APPEARANCE_PROMPT_HELDOUT_GAIN`.** 외형 프롬프트 개발에 쓰지 않은 Parent15개, Native512 Camera-Frame60개, 후보986개에서 두 프롬프트를 대응 비교했다. PEG Recall은 기존 역할 Prompt 31/60 (51.7%) → 외형 Prompt 37/60 (61.7%), BOX는 33/58 (56.9%) → 45/58 (77.6%)였다. 기준은 EXACT Actor visible pixel≥16, 선택 Box IoU≥0.5다.


**최종 해석: 기존 역할 Prompt 대비 개선은 재현됐지만, 일반적으로 안정적인 Peg/Box 선택기로 확인된 것은 아니다.** 외형 Prompt의 Peg Recall61.7%는 DINO 원시 Top1의65.0%보다 낮고, 후기 Main Peg는 **0/15**다. 기존 Prompt 대비 Peg Recall 차이의 Parent-cluster95% CI는0을 포함한다. BOX는 같은 비교에서 양의 CI를 보였다. 위 상태명 `APPEARANCE_PROMPT_HELDOUT_GAIN`은 사전 규칙에 따른 두 Object의 점추정 개선을 뜻하며, 통계적 보장·모든 Camera/시점 개선·Production 적합성의 판정이 아니다.

## 1. 사전 고정 범위와 자료의 독립성

직전 Native512/외형 Prompt 진단에 쓰인 Parent0531/0275/0197/0305를 제외한 기존20개 Parent를 전부 계획 Manifest에 고정했다. 그중5개는 기록된 physics_qpos/qvel 및 Actor pose가 없어 Native512 복원이 불가능했다. 추론 전에 필드 존재만 확인해 제외하고 대체 Parent/Frame을 뽑지 않았다. 나머지15개는 모두 평가했다. 과거 다른 Prompt의 Qwen 실험에는 사용됐으므로 전체 연구에서 미관측 자료나 외부 분포라는 의미의 독립성은 아니다.

| 항목 | 실제 범위 |
|---|---|
| Parent | 계획20, 평가15, 복원 자료 부재5 |
| Frame / Camera | 기존 F0=t0, F1=t80 또는 기존 fallback; Main/Wrist 각각 별도 입력 |
| 이미지 / 후보 / Qwen 요청 | 60 / 986 / 1972 |
| Qwen | Qwen/Qwen3.5-9B, revision c202236235762e1c871ad0ccb60c8ee5ba337b9a |
| 실행 | Vast.ai58576054, RTX3090, 전체BF16, SDPA, cuda:0, offload 없음 |
| Generation | Batch1, Greedy, thinking=False, max_new_tokens16, fresh user/cache |
| Processor | 동일 Source, min=max262144px, Native512; Crop 종횡비 유지/Patch 배수 반올림 |
| DINO | 기존 Tiny Weight/Revision/두 Prompt/threshold0.1, FP32, 120 forwards, Load1회 |
| 예산 | Qwen Load1회, SAM2/Training/Rollout/Physics Step/자동Retry/추가Warmup/Prompt 수정0회 |

새 Rollout 대신 기존 상태를 정적으로 복원했다. 환경 Constructor15회와 명시적 Seeded Reset15회가 있었고, 이후 저장 qpos/qvel·Peg/Box pose를 설정해 Camera만 Render했다. 원래128과 Native512 Camera의 Pose/FOV/Mount/Intrinsic scaling 및 TCP/Hole pose를 확인하고 get_obs 전후 물리 상태가 바뀌지 않음을 검증했다. Solver 상태의 동적 Replay를 주장하지 않는다. 두 해상도를 관측했지만 Qwen/DINO 정확도 평가는512만 사용했다.

평가 Parent: 10i_s1_0686, 10i_s1_0044, 10i_s1_0921, 10i_s1_0134, 10i_s1_0767, 10i_s1_0362, 10i_s1_0157, 10i_s1_0023, 10i_s1_0924, 10i_s1_0193, 10i_s1_0327, 10i_s1_0720, 10i_s1_0449, 10i_s1_0621, 10i_s1_0844.
자료 부재 Parent: existing_30040, existing_30124, existing_30082, existing_30017, existing_30032.

제외된5개는 모두 기존 `existing_full_task` 자료이고, 평가된15개는 복원 필드가 있는 Stage3 Parent다. 이 자료 가용성 차이로 평가 범위가 제한됐다. 복원 불가능한 Source군까지 같은 성능으로 일반화하지 않는다. 실제 F1은0134만t78이고 나머지는t80이다.

## 2. 입력·Prompt·선택 정책

전체 RGB, 청록색4px 후보 Rectangle 복사본, 각 변10% Margin Crop을 같은 순서로 입력했다. 원본과 Crop은 Marker가 없다. DINO Query/Score/Rank/좌표 숫자/기존 Consumer 결과/GT/Mask를 Qwen 텍스트에 넣지 않았다. 기존 역할 Prompt와 외형 Prompt는 직전 진단의 원문·Preamble을 그대로 복사했다. A/B에서 각 후보의 이미지 경로와 SHA가 동일하다. 결과에 따른 Prompt 수정은 없다.

두 Prompt가 제공하는 사전정보와 지칭·길이 등이 함께 다르므로 외형 Prompt 패키지의 효과를 평가한다. 색상 한 문장만의 인과 효과를 분리한 것은 아니다. 외형 Prompt는 알려진 Asset의 색·형태를 사용하므로 다른 색/모양에 대한 불변 일반화로 표현하지 않는다.

DINO의 두 고정 Query(red and white block/cardboard box)가 threshold0.1 이후 출력한 후보 전체를 검사했다. GT/면적/Top5/Null0.3으로 후보를 먼저 추리지 않았다. Qwen은 좌표를 수정하지 않는다. Primary는 같은 원래 Query 안에서 해당 역할 Label을 받은 후보 중 DINO Score1위를 선택하며 없으면 Null이다. 동점은 원래 Rank로 결정한다. 원시 Top1, 기존 Production 선택 로직, 기존 역할 Prompt의 동일 Query Gate를 함께 비교했다. 별도 Threshold 최적화나 Cross-query 재분류는 없다.

모든 Qwen 응답과 예측 CSV의 SHA를 동결한 뒤 Actor Segmentation을 열어 평가했다. Capture의 Privileged 복원 정보는 RGB-only 추론 Manifest와 분리했다. 과거 연구 결과가 알려진 자료이므로 완전 맹검으로 표현하지 않는다.

## 3. 최종 선택 성능

| 정책 | PEG Recall | BOX Recall | PEG 평균 IoU(Null 포함) | BOX 평균 IoU | 두 물체 동시 성공 |
|---|---:|---:|---:|---:|---:|
| legacy | 31/60 (51.7%) | 24/58 (41.4%) | 0.569 | 0.414 | 17/58 (29.3%) |
| raw_top1 | 39/60 (65.0%) | 20/58 (34.5%) | 0.674 | 0.346 | 14/58 (24.1%) |
| original_query_gate | 31/60 (51.7%) | 33/58 (56.9%) | 0.487 | 0.595 | 26/58 (44.8%) |
| appearance_query_gate | 37/60 (61.7%) | 45/58 (77.6%) | 0.627 | 0.781 | 30/58 (51.7%) |

| 정책 | 물체 | 선택 Box Precision | Null | Wrong-object / 출력 |
|---|---|---:|---:|---:|
| legacy | peg | 31/49 (63.3%) | 11/60 (18.3%) | 4/49 (8.2%) |
| legacy | box | 24/59 (40.7%) | 1/60 (1.7%) | 30/59 (50.8%) |
| raw_top1 | peg | 39/60 (65.0%) | 0/60 (0.0%) | 4/60 (6.7%) |
| raw_top1 | box | 20/60 (33.3%) | 0/60 (0.0%) | 35/60 (58.3%) |
| original_query_gate | peg | 31/32 (96.9%) | 28/60 (46.7%) | 0/32 (0.0%) |
| original_query_gate | box | 33/58 (56.9%) | 2/60 (3.3%) | 19/58 (32.8%) |
| appearance_query_gate | peg | 37/44 (84.1%) | 16/60 (26.7%) | 0/44 (0.0%) |
| appearance_query_gate | box | 45/57 (78.9%) | 3/60 (5.0%) | 1/57 (1.8%) |

Box Precision은 IoU≥0.5 비율이다. Wrong-object는 IoU<0.5이며 다른 명확한 Actor가 Box 내부 foreground의70% 이상일 때다. 느슨한 Box, 혼합 영역과 Wrong-object를 구분한다. VISIBLE은16px 이상, TINY_VISIBLE은1–15px, ABSENT는0px이며 모든 표본의 실제 분모는 JSON에 있다. 미가시/Tiny 사례가 없으면 해당 오류율의 검증 근거가 없는 것이다.

## 4. 후보 분류·정답 거부·오승인

| Prompt | 형식 준수 | 명확한 Actor 역할 일치 | Parent-macro 역할 일치 |
|---|---:|---:|---:|
| original_three | 986/986 (100.0%) | 377/736 (51.2%) | 0.518 |
| appearance_three | 986/986 (100.0%) | 633/736 (86.0%) | 0.858 |

AMBIGUOUS는 역할 정확도에서 양 조건 모두 제외하며 최종 선택 평가에는 그대로 포함했다. 역할 Proxy는 Box 내부 Actor 구성에 대한 진단이며 혼합 후보의 intended object를 완벽히 정의하지 않는다. 동일 물체 중복 Proposal을 독립 Parent로 세지 않는다.

| Prompt | Query | IoU≥0.5 정답 후보 유지 | IoU<0.5 오답 Box 승인 | 승인 Box 중 IoU≥0.5 |
|---|---|---:|---:|---:|
| original_three | peg | 66/131 (50.4%) | 28/359 (7.8%) | 66/94 (70.2%) |
| original_three | box | 76/76 (100.0%) | 276/420 (65.7%) | 76/352 (21.6%) |
| appearance_three | peg | 96/131 (73.3%) | 99/359 (27.6%) | 96/195 (49.2%) |
| appearance_three | box | 69/76 (90.8%) | 94/420 (22.4%) | 69/163 (42.3%) |

| Prompt | GT Actor 역할 | 출력 Label 분포 |
|---|---|---|
| original_three | PEG | `{"PEG": 132, "BOX": 196, "ROBOT": 2}` |
| original_three | BOX | `{"BOX": 115}` |
| original_three | ROBOT | `{"BOX": 160, "PEG": 1, "ROBOT": 130}` |
| original_three | OTHER | `{}` |
| original_three | AMBIGUOUS | `{"BOX": 212, "PEG": 18, "ROBOT": 20}` |
| appearance_three | PEG | `{"BOX": 41, "ROBOT": 22, "PEG": 267}` |
| appearance_three | BOX | `{"BOX": 111, "PEG": 4}` |
| appearance_three | ROBOT | `{"ROBOT": 255, "BOX": 31, "PEG": 5}` |
| appearance_three | OTHER | `{}` |
| appearance_three | AMBIGUOUS | `{"BOX": 139, "ROBOT": 36, "PEG": 75}` |

오답 Box 승인에는 목표가 보이지만 Box 범위가 부정확한 경우도 포함된다. 의미 Label 개선만으로 Localization이 개선됐다고 간주하지 않는다. Oracle 후보 풀 Recall은 peg=58/60 (96.7%), box=54/58 (93.1%)이며 실제 선택 성능이 아니다.

## 5. Camera·시점·Parent 통계

| 정책 | 구분 | PEG Recall | BOX Recall |
|---|---|---:|---:|
| legacy | base_camera | 18/30 (60.0%) | 23/30 (76.7%) |
| legacy | hand_camera | 13/30 (43.3%) | 1/28 (3.6%) |
| legacy | F0 | 28/30 (93.3%) | 15/28 (53.6%) |
| legacy | F1 | 3/30 (10.0%) | 9/30 (30.0%) |
| raw_top1 | base_camera | 19/30 (63.3%) | 19/30 (63.3%) |
| raw_top1 | hand_camera | 20/30 (66.7%) | 1/28 (3.6%) |
| raw_top1 | F0 | 29/30 (96.7%) | 11/28 (39.3%) |
| raw_top1 | F1 | 10/30 (33.3%) | 9/30 (30.0%) |
| original_query_gate | base_camera | 19/30 (63.3%) | 21/30 (70.0%) |
| original_query_gate | hand_camera | 12/30 (40.0%) | 12/28 (42.9%) |
| original_query_gate | F0 | 26/30 (86.7%) | 24/28 (85.7%) |
| original_query_gate | F1 | 5/30 (16.7%) | 9/30 (30.0%) |
| appearance_query_gate | base_camera | 15/30 (50.0%) | 25/30 (83.3%) |
| appearance_query_gate | hand_camera | 22/30 (73.3%) | 20/28 (71.4%) |
| appearance_query_gate | F0 | 29/30 (96.7%) | 26/28 (92.9%) |
| appearance_query_gate | F1 | 8/30 (26.7%) | 19/30 (63.3%) |

Parent별 자료는 evaluation_metrics.json의 per_parent에 있다. 같은 Parent의 두 Frame·두 Camera를 네 독립 표본으로 세지 않았다. Parent-macro Recall:

| 정책 | PEG | BOX |
|---|---:|---:|
| legacy | 0.517 | 0.422 |
| raw_top1 | 0.650 | 0.350 |
| original_query_gate | 0.517 | 0.567 |
| appearance_query_gate | 0.617 | 0.772 |

| 대조군 | 물체 | 외형 Prompt Recall 차이(Parent-macro) | Parent-cluster Bootstrap 95% CI | 회복 / 회귀 |
|---|---|---:|---|---:|
| legacy | peg | 0.100 | [0.016666666666666666, 0.2] | 9 / 3 |
| legacy | box | 0.350 | [0.2, 0.48333333333333334] | 23 / 2 |
| raw_top1 | peg | -0.033 | [-0.1, 0.03333333333333333] | 2 / 4 |
| raw_top1 | box | 0.422 | [0.2944444444444444, 0.55] | 25 / 0 |
| original_query_gate | peg | 0.100 | [0.0, 0.2] | 11 / 5 |
| original_query_gate | box | 0.206 | [0.12222222222222222, 0.2888888888888889] | 12 / 0 |

물리 Parent15개 Cluster Bootstrap10,000회(seed42)이며 탐색적 CI다. 다중 비교 보정이 없고, 작은 수의 Train Parent만으로 희귀 오승인이나 실환경 안정성을 입증하지 않는다.

## 5.1 사후 Camera × 시점 교차 분해

아래는 완료된 같은 예측의 사후 기술 통계다. 이 분해 결과로 Prompt·대상·선택 정책을 수정하지 않았다.

| Camera / 시점 | 원시 Top1 PEG | 기존 Prompt PEG | 외형 Prompt PEG | 원시 Top1 BOX | 기존 Prompt BOX | 외형 Prompt BOX |
|---|---:|---:|---:|---:|---:|---:|
| base_camera / F0 | 15/15 | 14/15 | 15/15 | 10/15 | 12/15 | 13/15 |
| base_camera / F1 | 4/15 | 5/15 | 0/15 | 9/15 | 9/15 | 12/15 |
| hand_camera / F0 | 14/15 | 12/15 | 14/15 | 1/13 | 12/13 | 13/13 |
| hand_camera / F1 | 6/15 | 0/15 | 8/15 | 0/15 | 0/15 | 7/15 |

외형 Prompt는 후기 Wrist PEG를0/15→8/15, BOX를0/15→7/15로 회복했지만, 후기 Main PEG는5/15→0/15로 퇴행했다. 전체 평균의 상승과 특정 조건의 집중 실패가 함께 존재한다. 초기에는 PEG29/30·BOX26/28로 상대적으로 높았지만 후기 재발견에 같은 성능을 가정할 수 없다.

## 5.2 남은 실패의 사후 회계와 관측·가설

각 가시 Object 실패를 같은 Query 후보 풀에서 서로 배타적으로 분해했다. 정답은Box IoU≥0.5다.

| Prompt / 물체 | 성공 | 정답 DINO 후보 없음 | 정답 후보 전부 거부 | 정답도 승인했지만 IoU 실패 후보가 더 높은 점수로 선택됨 |
|---|---:|---:|---:|---:|
| original_three/peg | 31 | 2 | 27 | 0 |
| appearance_three/peg | 37 | 2 | 15 | 6 |
| original_three/box | 33 | 4 | 0 | 21 |
| appearance_three/box | 45 | 4 | 1 | 8 |

외형 Prompt의 Peg 실패23개 중15개는 정답 후보를 모두 거부했고,6개는 승인된 정답 후보보다 IoU<0.5 후보를 위에 놓은 DINO Score 순위를 그대로 따랐다. 정답 후보 자체가 없는 경우는2개였다. BOX 실패13개는 후보 없음4개, 정답 거부1개, 승인 후보 간 순위 실패8개였다. 이 숫자는 저장된 Label·좌표·Score로 확인한 선택 단계의 원인 분해이며 모델 내부 기전에 대한 인과 증명은 아니다.

Peg Query의 정답 후보 유지율은50.4%→73.3%로 올랐지만, IoU<0.5 후보 승인도7.8%→27.6%로 늘었다. 실제 선택된 Peg의 IoU 실패7개는 Actor 구성상PEG6개·AMBIGUOUS1개였다. 단순 Wrong-object 판정과 Box 범위 실패가 다르다. Box Query의 오답 후보 승인은65.7%→22.4%로 줄었고, 최종 BOX Wrong-object도19/58→1/57로 줄었다.

GT 역할이 명확한 후보의 분류 일치율은377/736→633/736(51.2%→86.0%), ROBOT→BOX는160/291→31/291(55.0%→10.7%)이었다. 개선됐지만 이 오류가 사라지지는 않았다. 이 후보 풀에는 OTHER 정답 사례가 없고, 두 Prompt 모두 OTHER/UNCERTAIN 출력이0이었다. PEG 미가시 이미지도0개다. BOX 미가시는2개뿐이며 외형 Prompt의 Hallucination은0/2였다. 이 작은/없는 분모로 일반적인 부재 인식·불확실성 보정을 검증했다고 주장하지 않는다.

### DINO 원시1위 대비 Peg 회귀4개

| Parent / 후기 Main | 원시1위 Peg IoU | 외형 Prompt가 그 후보에 붙인 Label | 최종 출력 |
|---|---:|---|---|
| 10i_s1_0767 / F1 Main | 0.862 | BOX | Null |
| 10i_s1_0924 / F1 Main | 0.933 | BOX | Null |
| 10i_s1_0193 / F1 Main | 0.811 | BOX | Null |
| 10i_s1_0327 / F1 Main | 0.927 | BOX | Null |

네 사례 모두 정답 위치의 후보를 BOX로 판단해 최종 PEG가 Null이 됐다. 0193/0327은 Gripper가 섞인 AMBIGUOUS 역할 Proxy이지만, GT Peg Box와의 IoU는0.81/0.93으로 위치 평가상 정답이다. 혼합 영역이라는 이유로 이 Detection Miss를 분모에서 제외하지 않았다.

직접 확인한0767/0924의 Crop에서는 흰색·회색 Peg 부분이 두드러지고 빨간 면은 보이지 않았다. 이는 관측이다. 고정된 red-and-white 외형 설명의 일부가 가림·회전으로 보이지 않을 때 분류가 흔들렸을 가능성은 가설이며, 해당 문구만 바꾼 대응 실험이나 내부 특징 분석은 하지 않았다. 색상 설명이 모든 시점의 정체성 연결을 해결했다고 볼 수 없다.

기존 Prompt 대비 Parent별 Recall 방향은 PEG 개선7/동일6/퇴행2, BOX 개선10/동일5/퇴행0이다. 대응 이미지 단위로는 PEG11개 회복·5개 회귀, BOX12개 회복·0개 회귀였다. PEG의 Parent-macro +10.0%p에 대한 탐색적95% CI는[0.0,+20.0]%p, BOX의 +20.6%p는[+12.2,+28.9]%p다. BOX의 이득은 이 코호트에서 더 일관됐고, PEG 개선의 강도는 불확실하다.

[실패 분해 원자료](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/remaining_failure_analysis.json), [Camera/시점 교차 지표](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/camera_time_cross_analysis.json).

## 6. 비용과 검증

Qwen Load 5.695s, loaded GPU 17.527GiB, CPU sampled peak 18.261GiB. 첫 요청 하나만 Cold로 구분했으며 별도 Warmup은 없다. 조건별 Warm 비용:

| Prompt | n | Mean / Median / p95 ms | Peak allocated GiB | 이미지별 전체 후보 요청 합 Mean ms |
|---|---:|---|---:|---:|
| original_three | 985 | 653.27 / 647.65 / 689.69 | 17.966 | 10748.72 |
| appearance_three | 986 | 615.14 / 604.89 / 647.70 | 17.913 | 10108.78 |

전체 후보를 각자 새로 인코딩하는 exhaustive 진단 비용이다. 후보 간 Vision/KV Cache 공유나 Score순 조기 종료는 구현하지 않았다. 20Hz Control 경로의 비용으로 해석하지 않는다. DINO 비용은 dino_latency_vram.json에 별도 기록됐고 Qwen 수치에 포함하지 않았다. DINO와 Qwen은 순차 실행했으며 동시 상주·SAM2 연계·Tracking 비용을 측정하지 않았다.

공통 ExperimentLogger(W&B disabled)를 재사용했다. 기존 보호파일2430개 SHA/모델·Processor/환경이 전후 동일했다. 모든1972개 요청 strict parser·초기 Cache0을 확인했고, 동일 이미지986쌍, GT-free 선택 replay240개, 선택 IoU472개를 별도 재계산했다. 원시 응답의 GT 채점 전후 SHA는 동일하다. CUDA/Renderer는 원격58576054에서 실행했고 macOS에서 실행하지 않았다. Production/Checkpoint/Dataset/Threshold/Stop Stub을 변경하지 않았으며 PAUSED_SCOPE_REVIEW를 유지했다. Commit·외부 게시·후속 Graph/Controller 실행은 없다.

## 7. 해석의 범위

이번 비교는 외형 Prompt의 효과가 개발 사례 밖의 같은 Task Parent로 이어지는지, 그리고 해당 후보 풀에서 DINO 점수1위보다 유용한 선택을 하는지 평가한다. 색상·형태 불변성, 완전히 새로운 배경/Asset, DINO-free Box 생성, SAM2 Mask, 장기 시간적 정체성, Closed-loop Success는 평가하지 않았다. 후보 역할 정확도와 실제 선택 Recall을 함께 보고, 나쁜 Camera/시점의 결과를 제거하지 않는다. 어떤 결과이든 Prompt 추가 수정이나 Parent 추가 실행 없이 이 고정 평가로 종료한다.

## 8. 산출물

- [Protocol](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/protocol.json), [선정·누락 Manifest](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/selection_manifest.json), [Capture 검증](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/capture_verification.json), [입력 Manifest](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/input_manifest.json), [Prompt Identity](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/prompt_identity.json), [Model Identity](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/model_identity.json).
- [Qwen 원시 응답](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/verifier_raw_responses.json), [DINO 원시 후보](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/dino_raw_responses.json), [후보 평가](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/candidate_evaluation.csv), [최종 선택](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/selected_boxes.csv), [실패 사례](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/failure_cases.csv), [대응 회복·회귀](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/paired_transitions.csv).
- [전체 지표](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/evaluation_metrics.json), [Qwen 비용](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/verifier_latency_vram.json), [DINO 비용](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/dino_latency_vram.json), [보존 검증](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/verification.json), [Artifact Index](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/artifact_index.json).

![첫 고정 Parent 네 Camera/시점의 세 정책 비교](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/comparison.png)

주황=선택Peg, 청록=선택Box, 초록=GT Peg, 자홍=GT Box. 사전 고정 첫 Parent를 표시했으며 성공 사례를 골라 만든 그림이 아니다. SAM2 Mask는 없다.

## 부록 — Frozen appearance prompt

```text
Identify the object shown in the last image, which is a crop of the candidate.
If other images are provided, the first shows the full scene and the second shows the candidate outlined.
Classify the candidate itself using its visible appearance:
ROBOT: Gray or white robot arm links, or dark or black gripper fingers. A separate object held by the gripper is not a robot part.
PEG: A red-and-white elongated rectangular piece. It can be held by the gripper or partly outside the image.
BOX: A tan or brown block-shaped fixture. An opening may be visible, depending on the view.
Use color, shape, and visible parts together. Do not classify a different object elsewhere in the scene.
Choose OTHER for a clearly different object, or UNCERTAIN if the candidate cannot be identified.
Return exactly one label: PEG, BOX, ROBOT, OTHER, UNCERTAIN.
Do not add explanations or punctuation.
```

로컬 수신 독립 검증: Artifact 2279개 SHA, 원격/로컬 실행 Source 8개, 후보 Crop 986개 pixel-equal, Actor 역할 1972행, 후보 IoU 3944개, 선택 IoU 472개 재계산을 통과했다. [상세 검증](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/local_verification.json).

실행 전 준비 스크립트 requests.py가 SAPIEN의 requests 라이브러리 import를 가리는 오류가 있어 build_requests.py로 이름을 바꿨다. 첫 Capture와 모델 Load 전에 발견해 교정했고 원래 Source와 Log를 보존했다. 입력·Prompt·선정 규칙은 바뀌지 않았다.

실행 전 메타데이터 점검에서 복원 가능 Parent 수를14개로 가정한 Assertion이 중단됐다. 전체 필드 검사로15개 가능/5개 부재임을 확인해 모델 Load 전에 수만 교정했고, 이전 Source/Log를 preflight_revisions에 보존했다. 선정 기준이나 Prompt를 결과에 따라 바꾸지 않았으며 GPU 추론 재실행은 없었다.
