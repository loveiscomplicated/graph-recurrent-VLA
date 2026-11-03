# Task 10-I Stage 4A-2: GPT-6.1 Sol · Kimi K3 Full-image Grounding 및 SAM2 비교

## 목적과 해석 범위

동일한 전체 RGB에서 Peg와 Fixture를 독립적으로 찾고, 예측 Box만으로 SAM2 초기 Mask를 생성했다. 기존 Astra와 같은 장면에서 정확도·비용·지연을 비교한다. 이미 분석한 Train Parent 15개이므로 미사용 Parent 일반화 실험은 아니다.

## 실행 설정과 입력 보존

**범위 변경:** Kimi K3는 응답 시간이 너무 오래 걸려 사용자 요청으로 중도 탈락했다. 정확도에 따른 탈락이 아니다. 완료된 응답·시간·비용을 보존하고, 진행 중이던 Kimi 요청1개는 클라이언트에서 중단했다. 서버 처리·청구 여부는 확인되지 않아 예약 비용을 유지했다. 이후 Kimi 재호출은 없으며 Sol만 기존60장 평가를 계속했다. Kimi의 미시도 이미지를 인식 실패로 해석하거나 전체 성능 순위를 매기지 않는다.

- 신규 모델: GPT-6.1 Sol `gpt-6.1-sol` / Ollama Cloud Kimi K3 `kimi-k3`. 실제 신규 요청 72/120회, 자동 Retry 0회.
- 동일 native512 RGB 60장: 15 Parent × F0/F1 × Main/Wrist. F0=t0, F1=t80; 0134만 t78. 누락·실패 이미지 교체 없음.
- 프롬프트 SHA256: `df06d5fc6f2cfe2005afa2605b4f35db6f7a602855a2f1abd363df0921aec709`. 이전 외형 중심 프롬프트 원문과 동일하며 위치 힌트·Few-shot·Functional-role 설명 없음.
- 한 요청은 전체 RGB 하나와 프롬프트만 사용한다. DINO 후보·점수·GT·Mask·Parent ID·이전 대화·도구를 입력하지 않았다.
- Sol: Responses API, detail=original, reasoning=high, 생성 한도5120. Kimi: think=true, temperature=1, top_p=.95, top_k=64, num_predict=8192. 두 API 모두 Schema 강제 없음.
- 유한한 숫자4개, bool 제외, 0–1000 범위와 순서를 검사한 뒤 .512를 곱했다. Clipping·Box 확대·GT 보정·응답 재생성 없음.
- 모든 신규 API 원시 응답과 두 파싱 결과를 동결한 뒤 SAM2를 실행하고, 모든 Mask 동결 후 EXACT_GT Actor Segmentation으로 채점했다.
- 기존 Astra 원시 결과·Mask를 재사용했다. Gemma는 기존 동결 응답에 새로 사전 고정한 코드펜스 제거 규칙을 적용하여 SAM2를 실행했다. Gemma API 재호출은 0회다.
- 서비스 모델은 alias이며 서버 내부 Weight revision·양자화·GPU는 확인할 수 없다. 반환 모델 ID·Usage·요청 ID·UTC 시각은 원시 기록에 보존했다.

공식 설정·단가 출처: [GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol), [Kimi K3](https://ollama.com/library/kimi-k3). 입력/출력 100만 토큰당 Sol $2/$10, Kimi $3/$15 기준으로 동결했다.

## 두 출력 계약의 결과

**Strict**는 전체 응답이 정확한 JSON일 때만 인정한다. **Fence**는 정확한 JSON 또는 응답 전체를 감싼 단일 코드펜스만 제거한다. 설명에서 좌표 검색·추론 내용 추출·좌표 수정은 하지 않는다. 두 조건은 같은 원시 응답과 같은 Box를 사용하며, 같은 유효 Box의 Mask를 두 번 생성하지 않았다.

### Strict JSON

| Model | Peg Mask >=.8 | Fixture Mask >=.8 | Both Masks >=.8 | Valid coordinates incl. null |
| --- | ---: | ---: | ---: | ---: |
| GPT-6 Astra | 50/60 (83.3%) | 55/58 (94.8%) | 49/58 (84.5%) | 57/60 (95.0%) |
| GPT-6.1 Sol | 50/60 (83.3%) | 57/58 (98.3%) | 49/58 (84.5%) | 60/60 (100.0%) |
| Kimi K3 | 미완료 | 미완료 | 사용자 요청으로 시간상 중단 | 미완료 |
| Gemma4 31B | 0/60 (0.0%) | 0/58 (0.0%) | 0/58 (0.0%) | 0/60 (0.0%) |

### 사전 고정한 코드펜스 제거 허용

| Model | Peg Mask >=.8 | Fixture Mask >=.8 | Both Masks >=.8 | Valid coordinates incl. null |
| --- | ---: | ---: | ---: | ---: |
| GPT-6 Astra | 50/60 (83.3%) | 55/58 (94.8%) | 49/58 (84.5%) | 57/60 (95.0%) |
| GPT-6.1 Sol | 50/60 (83.3%) | 57/58 (98.3%) | 49/58 (84.5%) | 60/60 (100.0%) |
| Kimi K3 | 미완료 | 미완료 | 사용자 요청으로 시간상 중단 | 미완료 |
| Gemma4 31B | 47/60 (78.3%) | 40/58 (69.0%) | 35/58 (60.3%) | 60/60 (100.0%) |

주 분모는 GT Visible Pixel >=16인 물체다. Peg60개, Fixture58개, 두 물체 모두 보이는 이미지58장이다. GT 1–15pixel은 Tiny, 0pixel은 Absent로 별도 저장했다. 서비스 실패·미시도·null·무효 응답은 해당 가시 물체 초기화 실패에 포함했다. Format 실패와 물체 인식 실패를 동일 원인으로 해석하지 않는다.

## Camera·시점별 동시 Mask 성공

| Model (Fence) | Main | Wrist | F0 | F1 |
| --- | ---: | ---: | ---: | ---: |
| GPT-6 Astra | 26/30 (86.7%) | 23/28 (82.1%) | 27/28 (96.4%) | 22/30 (73.3%) |
| GPT-6.1 Sol | 27/30 (90.0%) | 22/28 (78.6%) | 26/28 (92.9%) | 23/30 (76.7%) |
| Gemma4 31B | 24/30 (80.0%) | 11/28 (39.3%) | 23/28 (82.1%) | 12/30 (40.0%) |

## Box Grounding과 잘못된 물체 초기화

| Model (Fence) | Peg Box IoU>=.5 | Fixture Box IoU>=.5 | Peg Wrong-object | Fixture Wrong-object |
| --- | ---: | ---: | ---: | ---: |
| GPT-6 Astra | 57/60 (95.0%) | 51/58 (87.9%) | 0/57 (0.0%) | 0/55 (0.0%) |
| GPT-6.1 Sol | 59/60 (98.3%) | 54/58 (93.1%) | 0/60 (0.0%) | 0/58 (0.0%) |
| Gemma4 31B | 56/60 (93.3%) | 40/58 (69.0%) | 1/60 (1.7%) | 0/58 (0.0%) |

Wrong-object는 다른 Peg/Fixture/Robot Actor가 예측 Mask의 50% 이상인 경우다. 낮은 Mask IoU 또는 일부 Robot 오염과 구분한다. 분모는 Tiny를 제외한 실제 비어 있지 않은 예측 Mask다. Box IoU>=.3/.5/.7, 중심 오차, Pixel Precision/Recall, GT Actor 구성·배경 오염, null·누락·Absent Hallucination은 [상세 지표](../../outputs/task10i/stage4a2/api_sol_kimi_grounding_20261103_v1/model_comparison.json)에 저장했다.

## Parent 단위 불확실성과 대응 비교

Physical Parent15개를 독립 단위로 사용했다. Parent별 동시 성공률을 평균하고, Seed42로 Parent를 10,000회 재표집했다. 같은 Parent의 Camera·시점을 독립 Parent로 세지 않는다.

| Model (Fence) | Parent-macro | 95% Cluster Bootstrap CI |
| --- | ---: | ---: |
| GPT-6 Astra | 83.9% | 75.6–91.7% |
| GPT-6.1 Sol | 83.9% | 75.6–91.7% |
| Gemma4 31B | 60.0% | 51.1–67.8% |

| Difference (Fence) | Parent-macro pp | 95% CI pp |
| --- | ---: | ---: |
| astra_minus_sol | +0.0 | -5.0–+5.0 |

완료 비교 모델의 API가 정상 응답한 공통 이미지 57장에 대한 별도 대응 결과도 저장했다. 그 집합에서 동시 Mask 성공은 GPT-6 Astra 49/55 (89.1%), GPT-6.1 Sol 48/55 (87.3%), Gemma4 31B 35/55 (63.6%)다.
Astra 기존 실행은 Mac 절전과 겹친 전송 실패3건을 포함한다. 이번 새 모델과 가용성을 그대로 비교하지 않는다. 전체 할당 분모 결과와 공통 정상응답 집합을 함께 본다. 작은 알려진 표본의 CI가 넓다는 점과 alias·시간·서비스 설정 차이를 고려해야 한다.

## API 지연·비용 및 총 소요 시간

| New Model | API status | OK latency mean / median / p95 | First request | Accounted USD | USD / successful joint init |
| --- | --- | ---: | ---: | ---: | ---: |
| GPT-6.1 Sol | {'OK': 60} | 10.64 / 8.27 / 21.59 s (n=60) | 9.69 s | $0.27609 | $0.00563 |
| Kimi K3 | {'OK': 7, 'TOKEN_LIMIT': 4, 'USER_INTERRUPTED_LATENCY': 1} | 25.33 / 19.61 / 55.89 s (n=7) | 32.89 s | $0.99339 | N/A |

Kimi는 완료된 요청 11개(정상 응답과 토큰 한도 종료 모두 포함) 기준 평균 41.87초, 중앙값 32.89초, p95 75.78초였다. 그중 토큰 한도 종료 4개는 평균 70.83초였다. 표의 OK 지연만으로 중도 탈락 당시의 대기 비용을 설명하지 않는다. 사용자 중단1건의 시간은 UTC 기준 클라이언트 취소 시점까지의 경과이며 완료 응답 시간이 아니다.

신규 API 경과 시간 **18.67분**. 순차 HTTP 측정 합계 18.65분. 시작 2026-11-02T20:14:50.368728+00:00, 종료 2026-11-02T20:33:30.851877+00:00 (UTC).
신규 비용 보수 집계 **$1.26948**, 이전 실험 및 승인된 첫 이미지 반복 **$2.33405**를 합쳐 **$3.60353/$20**. 알려진 Usage×공식 단가의 추정과 Usage 누락 예약액을 구분했다. 청구서 확정값이 아니며 캐시 할인은 적용하지 않았다. 기존 3090 요금은 별도다.
HTTP 시간은 로컬에서 측정한 서비스 호출 E2E다. 서버 Vision Encoding·Prefill·Decode는 분리 관측할 수 없다. 첫 요청을 Cold라고 확정하지 않는다. 이번에는 최초 이미지 추가 반복을 실행하지 않았다.
SAM2는 58576054 RTX3090에서 **1회 로드**, Mask **250개**를 생성했다. Load 963.7ms, Peak allocated 0.515GiB. FP32 Weight·BF16 autocast, multimask_output=False, logit>0, 장기 Tracking/기존 Memory Commit 없음.
VLM은 초기 발견·필요 시 재Grounding 용도다. 이번 API 지연으로 Control20Hz 실행을 주장하지 않는다.

## 실패 분해

| Model (Fence) | Visible object failures | Correct Box IoU>=.7 but Mask<.8 | Wrong-object failures |
| --- | ---: | ---: | ---: |
| GPT-6 Astra | 13 | 7 | 0 |
| GPT-6.1 Sol | 11 | 10 | 0 |
| Gemma4 31B | 31 | 7 | 1 |

관측: Box가 정확해도 SAM2가 Peg 일부만 분할하거나 Robot 일부를 포함할 수 있다. 반대로 느슨한 Box에서도 정확한 Mask가 나올 수 있다. 이러한 사례를 VLM의 의미 오인식으로 일괄 분류하지 않았다. 색상 경계·가림·그리퍼 접촉이 원인일 수 있다는 설명은 가설이며 이번 실험에서 인과적으로 분리 검증하지 않았다. 모델별 모든 실패 ID와 Actor 구성은 Strict/Fence 실패 CSV에 남겼다.

## 소규모 비교 그림

![Full-image grounding comparison](../../outputs/task10i/stage4a2/api_sol_kimi_grounding_20261103_v1/full_image_grounding_comparison.png)

빨강=예측 Peg, 파랑=예측 Fixture, 초록=사후 GT Box. 반투명 색은 실제 실행한 SAM2 Mask다. Hash 순서 첫4장 10i_s1_0157_F1_base_camera_native512, 10i_s1_0686_F0_hand_camera_native512, 10i_s1_0924_F0_hand_camera_native512, 10i_s1_0157_F1_hand_camera_native512을 표시했다. 성공 사례 선택으로 그림을 구성하지 않았다.

## 검증·보존과 결론의 한계

- 공통 ExperimentLogger를 W&B disabled로 재사용했다. Provider별 별도 W&B/HF 게시기는 추가하지 않았다.
- 형식·좌표·최종응답/추론 분리·단일 요청 입력·예산·Retry 차단·재개 중복 방지·Strict/Fence Mask gating 테스트12개 통과. Synthetic GT 테스트는 실제 GT를 열지 않았다.
- 독립 PIL ImageChops/getbbox 구현으로 Box·Mask IoU와 Wrong-object를 재계산하고 주요 원시 응답·Mask SHA를 검증했다. Checkpoint·SAM2 소스·설치 패키지는 원격 전후 보존 검사를 수행했다.
- API 키와 인증 Header는 원격으로 보내거나 로그에 저장하지 않았다. 기존 모델·데이터·이전 실험 산출물을 변경하지 않았다.
- 로컬 macOS에서 CUDA를 실행하지 않았다. SAM2 CUDA는 지정된 기존 RTX3090에서만 실행했다. Simulator·Training·Video Tracking·Hole 추정·Graph/Controller·Closed-loop는 실행하지 않았다.
- Production 변경, DINO 삭제, Git Commit, 외부 게시 없음. Stage4A-2 PAUSED_SCOPE_REVIEW와 기존 Stop Stub 유지.
- 모델의 크기·다른 일반 벤치마크 점수로 이 태스크 성능을 대신 판정하지 않는다. 이번 관측은 같은60장·동일Prompt·고정예산과 설정에서의 서비스 경로 비교다.
- DINO 제거 또는 Production 준비 판단은 **INSUFFICIENT_SUPPORT**다. 새 Parent, 희귀 오인식, 초기화 후 Tracking, Hole 위치·정밀도와 실제 제어 성공률은 별도 검증이 필요하다.

상세 산출물: [비교 지표](../../outputs/task10i/stage4a2/api_sol_kimi_grounding_20261103_v1/model_comparison.json), [비용·지연](../../outputs/task10i/stage4a2/api_sol_kimi_grounding_20261103_v1/grounding_latency_cost.json), [Strict 실패](../../outputs/task10i/stage4a2/api_sol_kimi_grounding_20261103_v1/grounding_failure_cases.csv), [Fence 실패](../../outputs/task10i/stage4a2/api_sol_kimi_grounding_20261103_v1/fence_grounding_failure_cases.csv), [독립 검증](../../outputs/task10i/stage4a2/api_sol_kimi_grounding_20261103_v1/independent_verification.json), [원격 보존](../../outputs/task10i/stage4a2/api_sol_kimi_grounding_20261103_v1/remote_preservation_verification.json).
