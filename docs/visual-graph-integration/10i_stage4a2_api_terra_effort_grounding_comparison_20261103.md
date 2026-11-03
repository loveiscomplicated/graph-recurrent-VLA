# Task 10-I Stage 4A-2: Terra Standard reasoning effort 비교

같은 native512 60장에서 `none`, `low`, `medium`, `high`의 전체 RGB Grounding과 실제 SAM2 초기 Mask를 비교했다. 기존 `high` 60개 응답·Mask·점수는 재사용했고 새로운 `high` API 호출과 SAM2 재실행은 없었다. 표의 결과는 이번 알려진 장면과 고정 설정에서의 관측이다.

## 고정한 실험 조건

- Train Parent 15개 × 초기/후기 × Main/Wrist, 총60 native512 PNG. F0=t0, F1=t80, Parent0134만 t78. 미사용 Parent 일반화 실험이 아니다.
- `gpt-6.1-terra` Responses API, Standard (`service_tier=default`), detail=original, max_output_tokens=5120, store=False. 새 요청은 `none`, `low`, `medium` 60장씩 최대180회다. 기존 high는 요청에서 tier를 생략했지만 실제 응답이 default였다.
- 같은 외형 설명 Prompt SHA256 `df06d5fc6f2cfe2005afa2605b4f35db6f7a602855a2f1abd363df0921aec709`. 매 요청은 원본 전체 RGB 하나와 Prompt만 사용하며 GT·DINO·SAM2·Crop·Point·Parent ID·이전 대화·도구를 전달하지 않는다.
- 0–1000 정규화 좌표, x/y에 .512를 곱한 Pixel 좌표. 범위·유한값·순서·bool을 검사하며 GT 보정, Clipping, Box 확대·좌표 수리, Schema 강제, 자동 Retry와 실패 이미지 대체는 없다.
- 원시 응답·예측을 먼저 동결하고 동일 SAM2.1 Hiera Small을 FP32 Weight + BF16 autocast로 실행, multimask_output=False·Logit>0·이미지 Embedding 재사용·독립 Box Prompt. 전체 Mask 동결 후 GT로 평가했다. 추적은 실행하지 않았다.
- 주 지표: Peg·Fixture의 SAM2 Mask IoU가 모두 ≥0.8인 이미지 비율. Visible≥16 pixel이 주 분모이며 Peg 60·Fixture 58·동시 58장이다. Tiny 1–15와 Absent 0은 별도다.

## 실제 Mask 성공: Strict JSON

| Effort | Peg | Fixture | 두 물체 동시 | Peg Wrong-object | Fixture Wrong-object | 좌표 계약 (null 허용) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 47/60 (78.3%) | 54/58 (93.1%) | 44/58 (75.9%) | 0/54 (0.0%) | 0/55 (0.0%) | 60/60 (100.0%) |
| low | 50/60 (83.3%) | 55/58 (94.8%) | 48/58 (82.8%) | 0/59 (0.0%) | 0/56 (0.0%) | 60/60 (100.0%) |
| medium | 49/60 (81.7%) | 56/58 (96.6%) | 47/58 (81.0%) | 2/58 (3.4%) | 0/57 (0.0%) | 60/60 (100.0%) |
| high | 49/60 (81.7%) | 56/58 (96.6%) | 47/58 (81.0%) | 0/57 (0.0%) | 0/57 (0.0%) | 60/60 (100.0%) |

Wrong-object는 다른 Peg/Fixture/Robot Actor가 예측 Mask Pixel의 50% 이상인 경우다. 분모는 Tiny를 제외한 비어 있지 않은 예측 Mask이며 Visible Recall 분모와 다르다. 서비스 오류·토큰 한도·무효 출력·null·미시도는 해당 가시 대상 초기화 실패에 포함하며 낮은 IoU를 모두 오인식으로 분류하지 않는다.

## Fence 허용 결과

| Effort | Peg | Fixture | 두 물체 동시 | 좌표 계약 (null 허용) |
| --- | ---: | ---: | ---: | ---: |
| none | 47/60 (78.3%) | 54/58 (93.1%) | 44/58 (75.9%) | 60/60 (100.0%) |
| low | 50/60 (83.3%) | 55/58 (94.8%) | 48/58 (82.8%) | 60/60 (100.0%) |
| medium | 49/60 (81.7%) | 56/58 (96.6%) | 47/58 (81.0%) | 60/60 (100.0%) |
| high | 49/60 (81.7%) | 56/58 (96.6%) | 47/58 (81.0%) | 60/60 (100.0%) |

Fence는 응답 전체를 감싼 단일 코드펜스만 제거한다. Strict의 형식 실패를 소급 성공으로 변경하지 않고, 설명이나 추론에서 좌표를 검색·수정하지 않는다. 같은 유효 Box의 SAM2 Mask는 한 번만 생성했다.

## 응답 시간과 비용

| Effort | 요청 상태 | OK 평균 / 중앙값 / p95 | 완료(OK+TOKEN_LIMIT) 중앙값 / n | Usage 추정 | 미확인 예약 | 총 계상 | 추론 토큰 합계 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | {'OK': 60} | 1.86 / 1.77 / 2.42초 (n=60) | 1.77초 / 60 | $0.042480 | $0.000000 | $0.042480 | 0 |
| low | {'OK': 60} | 2.65 / 2.50 / 4.00초 (n=60) | 2.50초 / 60 | $0.061305 | $0.000000 | $0.061305 | 3614 |
| medium | {'OK': 60} | 4.11 / 3.51 / 7.86초 (n=60) | 3.51초 / 60 | $0.092895 | $0.000000 | $0.092895 | 9907 |
| high | {'OK': 60} | 6.53 / 4.59 / 13.67초 (n=60) | 4.59초 / 60 | $0.147435 | $0.000000 | $0.147435 | 20820 |

새 요청 보수 계상 **$0.196680**, 기존 계상 **$4.04610386**, 합계 **$4.24278386/$20**. 기존 high 비용은 이전 계상에 이미 포함돼 있으므로 중복 가산하지 않았다. Usage 추정과 미확인 요청 예약은 확정 청구액이 아니며 RTX3090 요금은 별도다. 입력 $1/출력 $5 per million, 캐시 할인 미적용. [공식 Terra 가격·설정](https://developers.openai.com/api/docs/models/gpt-6.1-terra).
HTTP 시간에는 네트워크와 응답 파싱이 포함되고 SAM2는 포함되지 않는다. 요청을 순차 실행했고 high는 이전 시점·Terra/Luna 교대 실행 결과이므로 시간·서비스 부하·추론 토큰 차이를 포함한다. 성공 응답 시간뿐 아니라 TOKEN_LIMIT 완료 시간을 함께 보며 순수 추론 속도나 실시간 제어 성능을 주장하지 않는다.

## Parent 단위 대응 비교

| Effort − high | Parent-macro 차이 | Parent-cluster 95% CI |
| --- | ---: | ---: |
| none − high | -5.56%p | -12.78~+1.67%p |
| low − high | +1.67%p | -3.33~+6.67%p |
| medium − high | +0.00%p | -5.00~+5.00%p |

물리 Parent 15개를 독립 단위로 사용하고 Seed42·10,000회 Parent-cluster Bootstrap을 적용했다. 차이의 CI에 0이 포함되면 우열이 확정되지 않은 것이며, 이것만으로 동등성을 입증한 것은 아니다. 여러 수준 비교는 탐색적이며 다중 비교 보정은 하지 않았다.

## 카메라·시점

| Effort | Main | Wrist | t0 | 후기 t80/t78 |
| --- | ---: | ---: | ---: | ---: |
| none | 25/30 (83.3%) | 19/28 (67.9%) | 26/28 (92.9%) | 18/30 (60.0%) |
| low | 26/30 (86.7%) | 22/28 (78.6%) | 26/28 (92.9%) | 22/30 (73.3%) |
| medium | 25/30 (83.3%) | 22/28 (78.6%) | 26/28 (92.9%) | 21/30 (70.0%) |
| high | 25/30 (83.3%) | 22/28 (78.6%) | 26/28 (92.9%) | 21/30 (70.0%) |

## 남은 실패와 해석

| Effort | Visible 물체 실패 | Box IoU≥0.7이지만 Mask IoU<0.8 | Wrong-object 실패 |
| --- | ---: | ---: | ---: |
| none | 17 | 6 | 0 |
| low | 13 | 7 | 0 |
| medium | 13 | 6 | 2 |
| high | 13 | 7 | 0 |

정확한 Box 뒤 부분 분할·오염이 나올 수 있으므로 실패를 모두 VLM 의미 인식 오류로 해석하지 않는다. 기존에 여러 차례 본 15개 Parent의 정적 초기 Mask 비교이며, 새 Parent·초기화 이후 Tracking·Graph·Controller·Closed-loop 결과는 아니다.

## 고정 이미지 예시

![동일 이미지 effort별 실제 Mask](../../outputs/task10i/stage4a2/api_terra_effort_grounding_20261103_v1/effort_grounding_comparison.png)

입력 Manifest의 고정 순서 첫 4장이다. 빨강=예측 Peg Box, 파랑=예측 Fixture Box, 초록=사후 GT Box, 반투명 영역=실제로 생성된 SAM2 Mask. 성공 여부를 보고 이미지를 고르지 않았다.

## 검증과 근거

- 공통 ExperimentLogger(W&B disabled), 기존 HTTP Adapter·좌표 Parser·SAM2 Segment·GT 평가·별도 PIL ImageChops 검증을 재사용했다. 기존 high와 새 예측·Mask를 중복 실행하지 않았다.
- 기존 RTX3090 인스턴스에서만 새 SAM2 Mask를 실행했다. API 키는 로컬에서만 사용했고 원격 GPU로 전송하지 않았다. 기존 원격 checkpoint·설치 SAM2 Source·패키지를 전후 대조했다.
- 신규 Simulator Render/Rollout·Tracking·Graph·학습·Controller·Production 변경·Git Commit·외부 게시 없음. PAUSED_SCOPE_REVIEW와 Stop Stub 유지. 로컬 macOS CUDA/MPS 및 Simulator는 실행하지 않았다.

[비교 JSON](../../outputs/task10i/stage4a2/api_terra_effort_grounding_20261103_v1/effort_comparison.json) · [시간·비용](../../outputs/task10i/stage4a2/api_terra_effort_grounding_20261103_v1/effort_latency_cost.json) · [요청 Ledger](../../outputs/task10i/stage4a2/api_terra_effort_grounding_20261103_v1/request_ledger.json) · [Strict 실패](../../outputs/task10i/stage4a2/api_terra_effort_grounding_20261103_v1/grounding_failure_cases.csv) · [Fence 실패](../../outputs/task10i/stage4a2/api_terra_effort_grounding_20261103_v1/fence_grounding_failure_cases.csv) · [독립 검증](../../outputs/task10i/stage4a2/api_terra_effort_grounding_20261103_v1/independent_verification.json) · [원격 보존](../../outputs/task10i/stage4a2/api_terra_effort_grounding_20261103_v1/remote_preservation_verification.json) · [기존 high 보고서](10i_stage4a2_api_openai_family_grounding_comparison.md).
