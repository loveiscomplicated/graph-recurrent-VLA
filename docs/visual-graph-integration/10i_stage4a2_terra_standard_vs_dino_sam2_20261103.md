# Task 10-I Stage 4A-2: Terra Standard와 기존 DINO–SAM2의 저장 결과 비교

**같은 native512 60장에 대한 정적 초기화에서, 기존 DINO 선택 규칙 + SAM2의 두 물체 동시 Mask 성공은 16/58 (27.6%), Terra Standard + SAM2는 47/58 (81.0%)였다.** 성공률 **+53.4%p**, 성공 이미지 **31장 증가**, 실패 **42→11장 (73.8% 감소)**다.

앞서 비교에 사용한 27/58 (46.6%)는 **DINO 단독이 아니라 Qwen 외형 검증을 포함한 A 경로**였다. 그 개선된 경로와 비교해도 Terra는 **+34.5%p, +20장**이다. 두 기준선을 구분한다.

## 자료와 비교 방법

- 기존 Train Parent 15개, F0=t0/F1=t80 (Parent0134만 t78), Main/Wrist 총 60장. 기존 두 실험의 Image ID·Parent·Camera·시점·PNG SHA·RGB SHA가 60/60 모두 일치했다.
- 성공은 GT Visible Pixel≥16인 대상의 **실제 SAM2 Mask IoU≥0.8**이다. Peg 60, Fixture 58, 두 물체 동시 58장이 분모다. Null·미초기화는 실패로 포함한다.
- 모든 경로의 SAM2.1 Hiera Small checkpoint SHA가 일치한다. 기존 실행은 FP32 Weight/BF16 autocast, multimask_output=False, logit>0 및 이미지 Embedding 재사용을 사용했다.
- 기존 DINO 선택 규칙은 저장된 `legacy` 선택 그대로다. 면적25% 제외·Top5·Null/Joint 점수 규칙으로 고른 후보를 사용한다. 원시 Top1은 각 Query의 원래 DINO Score 1위다. Qwen A는 같은 Query에서 외형 Label이 맞는 후보 중 DINO Score 1위다.
- **새 DINO/Qwen/Terra/SAM2 호출은 0회다.** 기존 986개 DINO 후보에 실제 실행해 둔 SAM2 Mask의 점수를 저장된 legacy/Top1 선택과 연결했다. 결과를 보고 후보를 다시 고르거나 GT로 최적 후보를 선택하지 않았다.
- legacy/Top1의 Mask 수치는 이번 사후 재집계다. 기존 보고서의 Box IoU≥0.5 수치를 Mask 성공률로 전용하지 않았다. 후보별 A/B/C Mask 지표 동일성과 A 선택 120개의 복원 결과를 기존 발표 값에 대조했다.

## 실제 Mask 초기화 결과

| 경로 | Peg 성공 | Fixture 성공 | 두 물체 동시 성공 | Peg Wrong-object | Fixture Wrong-object |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 DINO 선택 규칙 + SAM2 | 29/60 (48.3%) | 26/58 (44.8%) | 16/58 (27.6%) | 5/49 (10.2%) | 33/59 (55.9%) |
| DINO 원시 Top1 + SAM2 | 34/60 (56.7%) | 19/58 (32.8%) | 12/58 (20.7%) | 8/60 (13.3%) | 40/60 (66.7%) |
| DINO + Qwen 외형 검증 + SAM2 (기존 A) | 33/60 (55.0%) | 48/58 (82.8%) | 27/58 (46.6%) | 1/44 (2.3%) | 4/57 (7.0%) |
| Terra Standard + SAM2 | 49/60 (81.7%) | 56/58 (96.6%) | 47/58 (81.0%) | 0/57 (0.0%) | 0/57 (0.0%) |

Wrong-object는 다른 Peg/Fixture/Robot Actor가 예측 Mask Pixel의 50% 이상일 때다. 분모는 Tiny를 제외한 비어 있지 않은 예측 Mask이며 가시 대상 성공 분모와 다르다. Terra의 0건은 이번 표본에서 관측되지 않았다는 뜻이다.

| 평균 Mask IoU (가시 대상, 누락=0) | 기존 DINO 선택 + SAM2 | DINO+Qwen A+SAM2 | Terra Standard+SAM2 |
| --- | ---: | ---: | ---: |
| Peg | 0.582 | 0.633 | 0.871 |
| Fixture | 0.447 | 0.851 | 0.945 |

## Camera·시점별 동시 성공

| 범위 | 기존 DINO 선택 + SAM2 | DINO+Qwen A+SAM2 | Terra Standard+SAM2 |
| --- | ---: | ---: | ---: |
| Main | 16/30 (53.3%) | 12/30 (40.0%) | 25/30 (83.3%) |
| Wrist | 0/28 (0.0%) | 15/28 (53.6%) | 22/28 (78.6%) |
| 초기 t0 | 13/28 (46.4%) | 22/28 (78.6%) | 26/28 (92.9%) |
| 후기 t80/t78 | 3/30 (10.0%) | 5/30 (16.7%) | 21/30 (70.0%) |

## Parent 대응 비교

같은 물리 Parent 15개를 독립 단위로 삼아 Parent별 동시 성공률 차이를 계산했다. Parent-cluster Bootstrap 10,000회, Seed42다. 아래 차이는 전체 이미지 성공률 차이와 달리 Parent-macro 차이다.

| 대응 비교 | Parent-macro 차이 | 95% CI | 기존 실패→Terra 성공 | 기존 성공→Terra 실패 |
| --- | ---: | ---: | ---: | ---: |
| Terra − 기존 DINO 선택 규칙 + SAM2 | +53.33%p | +45.00–+61.67%p | 31 | 0 |
| Terra − DINO 원시 Top1 + SAM2 | +60.00%p | +49.44–+70.56%p | 35 | 0 |
| Terra − DINO + Qwen 외형 검증 + SAM2 (기존 A) | +34.44%p | +23.33–+44.44%p | 20 | 0 |

Camera/시점을 60개의 독립 Parent로 세지 않았다. 이는 고정된 알려진 장면에서의 탐색적 경로 비교이며 모델이나 Prompt 한 요소만의 인과 효과를 분리하지 않는다.

## 속도·비용에서 비교할 수 있는 범위

Terra Standard의 기존 HTTP 요청 평균 6.53초, 중앙값 4.59초, 60장 Usage 기반 API 비용 약 $0.147435를 재사용한다. 확정 청구액이 아니며 GPU 비용은 별도다. Qwen 후보 검증에는 조건당 약 986회×0.61초/60장, 약 10초/이미지 규모가 기록됐지만 DINO Forward·Load·SAM2·I/O를 포함한 동일 정의의 파이프라인 E2E는 확보되지 않았다. 따라서 DINO 경로 대비 정확한 속도 배수나 GPU/API 총비용 절감률은 주장하지 않는다.

## 판단과 보존

현재 자료에서는 Terra Standard로 Grounding을 바꾸면 **두 물체가 모두 올바르게 분할되는 이미지가 크게 늘고, 기존 선택 경로의 다른 Actor 오초기화가 줄었다**고 판단할 수 있다. Qwen 검증을 이미 더한 기준선보다도 개선됐다. 후기 프레임의 초기화 수치는 매 프레임 독립 재발견 조건이며 t0 초기화 뒤 SAM2로 유지한 Tracking 성공률과 섞지 않는다.

이번 결론은 **초기화/독립 재탐지 이미지의 Mask 품질**에 해당한다. 기존 DINO 런타임의 5프레임 간격 탐색·2회 일치 후 초기화·Video Predictor 전파를 새로 실행한 전체 시스템 비교가 아니다. Terra 초기화 후 Tracking과 학습/Controller 성공률은 이번 비교에서 측정하지 않았다. 이 한계 때문에 추가 장면 실험을 자동으로 시작하지 않는다.

공통 ExperimentLogger(W&B disabled)를 재사용했다. 기존 입력·평가 파일 17개 SHA 보존, 60개 이미지의 PNG/RGB·시점 일치, GT-free 선택 재현 240개, 기존 A 결과 120개 대조를 통과했다. 기존 지표를 재집계했으며 원시 GT Pixel과 Mask IoU를 새로 계산한 검사는 아니다. 로컬 macOS CPU 파일 분석만 실행했다. 원격 접속, GPU/CUDA/MPS, Simulator Render/Rollout, Graph/학습/Controller, Production 변경, Commit, 게시를 실행하지 않았다. PAUSED_SCOPE_REVIEW·Stop Stub과 기존 산출물을 유지했다.

별도 검증 구현으로 물체 480개 행의 임계값·성공 수·평균 IoU, 이미지 174쌍의 성공 전환, Parent Bootstrap 차이·CI를 다시 계산해 PASS했다. 원시 Pixel IoU 재계산과 구분한다.

산출물: [비교 JSON](../../outputs/task10i/stage4a2/terra_vs_dino_saved_comparison_20261103_v1/comparison.json) · [물체별 재집계](../../outputs/task10i/stage4a2/terra_vs_dino_saved_comparison_20261103_v1/object_cases.csv) · [이미지별 성공 전환](../../outputs/task10i/stage4a2/terra_vs_dino_saved_comparison_20261103_v1/paired_scene_transitions.csv) · [입력 SHA](../../outputs/task10i/stage4a2/terra_vs_dino_saved_comparison_20261103_v1/input_provenance.json) · [검증](../../outputs/task10i/stage4a2/terra_vs_dino_saved_comparison_20261103_v1/verification.json) · [독립 재계산](../../outputs/task10i/stage4a2/terra_vs_dino_saved_comparison_20261103_v1/independent_verification.json).

근거 보고서: [기존 SAM2 초기화·추적](10i_stage4a2_sam2_mask_verifier_tracking.md) · [DINO 선택 규칙과 Box 평가](10i_stage4a2_qwen35_appearance_bf16_independent_evaluation.md) · [Terra Standard 및 계열 모델 비교](10i_stage4a2_api_openai_family_grounding_comparison.md). 선택 규칙의 런타임 설명은 [Visual README](../../visual/runtime/README.md)를 따른다.
