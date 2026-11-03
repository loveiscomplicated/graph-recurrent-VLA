# Task 10-I Stage 4A-2: Terra Standard 채택 및 DINO 설계 제외 결정

작성일: **2026-11-03 KST**. 기존 API Grounding 비교, Terra Fast 실측, DINO–SAM2 저장 결과 분석을 통합한 모델 선택 보고서다.

**앞으로 Peg·Fixture의 초기 발견과 필요한 시점의 재탐지에는 GPT-6.1 Terra Standard를 사용한다. DINO는 향후 모델 설계에서 제외한다.** 전체 RGB에서 Terra가 두 물체의 Bounding Box를 직접 출력하고, 그 Box로 SAM2를 초기화한다. 이후 관측 갱신은 SAM2가 담당하는 방향을 유지한다.

이 결정의 근거는 세 가지다. Terra Standard는 다른 완료 API 모델과 비교해 정확도·시간·비용의 균형이 좋았고, Fast의 추가 비용에 비해 실측 지연 단축은 작았다. 또한 같은 이미지에서 기존 DINO 기반 경로보다 실제 SAM2 초기 Mask 품질이 크게 개선됐다.

## 1. 채택 사항과 용도

| 항목 | 결정 |
| --- | --- |
| VLM 모델 | **GPT-6.1 Terra (`gpt-6.1-terra`)** |
| 서비스 모드 | **Standard**. 후속 구현에서 `service_tier="default"`를 명시해 사용 |
| VLM 역할 | 전체 RGB에서 Peg와 Fixture 초기 발견, 필요한 경우 재탐지 |
| 추론 설정 | Responses API, `reasoning.effort="high"`, 이미지 `detail="original"`, `max_output_tokens=5120`, `store=False` |
| SAM2 역할 | Terra의 Box로 초기 Mask 생성, 이후 관측 갱신 |
| DINO | **향후 모델 설계에서 제외**. DINO 후보 생성과 이를 전제로 하는 Qwen 후보 검증 경로를 채택 경로에서 제외 |
| Fast | 지원은 확인했으나 기본 운용 모드로 채택하지 않음 |

VLM은 매 제어 프레임이나 20Hz로 실행하는 모듈이 아니다. 초기 발견과 재탐지에서 오인식·미인식을 줄여 Object Graph와 학습 자료의 입력 품질을 높이는 데 목적이 있다. 따라서 모델 선택에서는 최소 지연만을 추구하기보다, 실제 Mask 성공률과 잘못된 물체 초기화, 응답 형식, 비용을 함께 고려했다.

위 표는 이번에 확정한 설계 방향과 후속 요청 설정이다. 기존 Standard 실험은 응답의 실제 `service_tier="default"`를 확인한 결과이며, 이 문서를 작성하면서 실행 코드나 프로젝트 서비스 설정을 변경하지 않았다. 재탐지의 구체적인 조건, Graph 구성, 학습 인터페이스는 이 보고서에서 새로 정하지 않는다.

## 2. 비교 자료와 평가 기준

모델 비교는 이미 여러 차례 분석한 **Train Parent 15개·원본 native 512×512 PNG 60장**을 사용했다. 각 Parent의 초기·후기와 Main·Wrist를 포함한다. 초기 시점은 t0, 후기 시점은 t80이며 Parent0134만 t78이다.

API 입력은 전체 RGB 한 장과 공통 영어 외형 설명 Prompt다. Peg는 빨강·흰색의 길쭉한 직사각형 부품, Fixture는 황갈색·갈색 블록 형태의 받침, Robot은 회색·흰색 링크와 어두운 그리퍼 손가락으로 설명했다. DINO 후보·점수, GT, SAM2 Mask, Crop, Point, Parent 이름, 이전 프레임·응답은 제공하지 않았다. 각 요청은 새 대화이며 Few-shot·특정 위치 힌트·Functional-role 설명을 사용하지 않았다.

Prompt SHA256은 `df06d5fc6f2cfe2005afa2605b4f35db6f7a602855a2f1abd363df0921aec709`다. 출력은 Peg·Fixture 각각의 Box 또는 null을 갖는 JSON이며, 원본 전체 이미지 기준 0–1000 좌표를 사용했다. 유한 숫자·범위·순서를 검사하고 Pixel 좌표로 변환했다. Boolean, 좌표 수리, Clipping, GT 보정, 자동 Box 확대, 자동 Retry·실패 이미지 대체는 허용하지 않았다. API Schema 강제도 사용하지 않았다.

**주 지표는 Peg와 Fixture의 실제 SAM2 Mask가 모두 IoU≥0.8인 이미지 비율**이다. GT Visible Pixel≥16을 주 평가에 사용하고 Tiny 1–15와 Absent 0은 분리했다. Peg는 60개, Fixture는 58개, 동시 성공은 두 물체가 모두 보이는 58장이 분모다. Null·서비스 실패·무효 출력·미초기화는 가시 대상 성공률에서 실패로 포함했다.

SAM2는 기존 **SAM2.1 Hiera Small**, FP32 Weight·BF16 autocast, `multimask_output=False`, Logit>0을 사용했다. 이미지 Embedding은 재사용하고 각 Object Box Prompt는 독립 실행했다. API 예측 전체를 저장·SHA 동결한 뒤 SAM2를 실행했으며, Mask 전체를 동결한 다음 GT로 채점했다.

따라서 아래 정확도는 VLM의 단독 물체 인식 점수가 아니라 **VLM Box로 SAM2를 초기화한 전체 경로의 성공률**이다. 모델·서비스 설정과 출력 계약의 차이도 포함한다. [API 통합 평가](10i_stage4a2_api_openai_family_grounding_comparison.md)

## 3. 다른 API 모델과 비교해 Terra를 선택한 이유

### 3.1 완료된 60장 비교

Latency는 SAM2를 제외한 API HTTP 요청 시간의 중앙값이다. 비용은 각 모델의 주 평가 60건에 대한 Usage 기반 추정과 미확인 요청 예약을 합친 보수 계상액이며 확정 청구액이 아니다. GPU 사용료는 별도다.

| 모델 | Peg Mask 성공 | Fixture Mask 성공 | 두 Mask 모두 성공 | API 중앙값 | 60장 API 계상액 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-6 Astra | 50/60 · 83.3% | 55/58 · 94.8% | 49/58 · 84.5% | 10.03초 | $2.216830¹ |
| GPT-6.1 Sol | 50/60 · 83.3% | 57/58 · 98.3% | 49/58 · 84.5% | 8.27초 | $0.276090 |
| **GPT-6.1 Terra Standard** | **49/60 · 81.7%** | **56/58 · 96.6%** | **47/58 · 81.0%** | **4.59초** | **$0.147435** |
| GPT-6 Luna | 42/60 · 70.0% | 54/58 · 93.1% | 39/58 · 67.2% | 6.41초 | $0.020708 |
| Gemma 4 31B (Fence 허용) | 47/60 · 78.3% | 40/58 · 69.0% | 35/58 · 60.3% | 약 4.97초 | $0.033365 |

¹ Astra는 Usage 확인 추정 $1.351840과 미확인 요청 예약 $0.864990의 합계다. 예약액을 실제 청구로 해석하지 않는다. Astra/Gemma 지연 통계는 기존 절전 중첩 요청을 제외한 정상 응답 기준이고, 병행 호출 등 실행 조건도 달라 순수 모델 추론 속도 비교는 아니다.

**Gemma의 60.3%는 Fence 허용 조건이다.** 모든 응답이 Markdown 코드펜스로 감싸져 Strict JSON에서는 0/60이었다. 저장 응답에 사전 고정한 전체 코드펜스 제거 규칙을 적용하고 실제 SAM2 Mask를 실행한 결과가 35/58이다. 좌표 수정이나 추가 Gemma API 호출은 없었다. 다른 네 모델은 Strict와 Fence 결과가 같았다.

OpenAI 네 모델은 공통으로 Responses API·original 이미지·high 추론·생성 한도 5120을 사용했다. Gemma·Kimi·GLM의 Thinking 설정을 OpenAI의 reasoning high와 동일한 추론 수준으로 간주하지 않는다. [모델별 설정과 결과](10i_stage4a2_api_openai_family_grounding_comparison.md)

### 3.2 Sol·Astra와의 차이보다 비용과 시간이 유리했다

Terra의 동시 성공은 Sol·Astra보다 2장 적었다. Sol−Terra의 Parent-macro 차이는 +3.33%p이고 95% CI는 약 **−3.33~+10.00%p**로 0을 포함한다. 이는 정확도 우열이 확정되지 않았다는 뜻이며, 통계적 동등성이 입증됐다는 뜻도 아니다. 독립 단위는 물리 Parent 15개이고 Parent-cluster Bootstrap 10,000회·Seed42를 사용했다.

그 상태에서 Terra는 Sol 대비 60장 API 비용이 **46.6% 낮았고**, 관측된 요청 중앙값도 8.27초에서 4.59초로 짧았다. Astra도 점추정 정확도가 높았지만 비용과 지연에서 Terra를 선택할 이유가 더 컸다. 따라서 이번 선택은 최고 점추정 정확도 하나보다 **가까운 정확도에서의 비용·응답 시간**을 중시한 결정이다.

초기 Astra 실행에는 macOS 절전과 겹친 전송 실패 3건이 있다. 최초 실험 전체의 전송 오류 6건도 절전 구간과 겹쳤으므로 이를 Provider 고유 안정성 문제로 단정하지 않는다. 정상 공통 응답 집합과 전체 할당 분모는 원본 보고서에 별도 보존돼 있다.

### 3.3 Luna·Gemma의 저렴한 비용보다 입력 품질을 우선했다

Terra는 가장 싼 모델은 아니다. Luna는 비용이 훨씬 낮았지만 동시 성공이 39/58로 Terra보다 8장 적었고, Peg Wrong-object도 3/56이었다. Terra는 Peg·Fixture Wrong-object가 각각 **0/57, 0/57**이었다. Wrong-object는 다른 Peg/Fixture/Robot Actor가 예측 Mask Pixel의 50% 이상을 차지하는 경우다. 단순한 낮은 IoU나 일부 Robot 오염과 구분한다.

Gemma도 비용은 낮았지만 Fence 허용 후 동시 성공이 35/58에 머물렀고 출력 형식 처리도 필요했다. Object Graph와 학습 자료에 잘못된 물체를 넣는 오류를 줄이려는 목적에서 Terra의 추가 비용을 수용하기로 했다. 성공 1건당 비용만 최소화하는 기준과는 구분한다.

Terra Standard의 이번 Usage 기준 비용은 **장당 약 $0.002457**, 동시 초기화 성공 1건당 약 **$0.003137**이다. 이는 이번 입력·출력 Usage에 대한 평균이며 미래 모든 요청의 고정 가격은 아니다.

### 3.4 중단된 Kimi·GLM은 운영 사유를 분리했다

| 모델 | 관측된 실행 상태 | 선택 판단에 반영한 내용 |
| --- | --- | --- |
| Kimi K3 | 완료 11건: OK 7, TOKEN_LIMIT 4. 진행 중 요청 1건 로컬 취소 | 응답 시간이 너무 오래 걸려 사용자 요청으로 중단. 정확도 탈락이 아니며 전체 60장 정확도 순위에서 제외 |
| GLM-5.3-Flash | 29회 시도: OK 21, TOKEN_LIMIT 5, Timeout 1, Connection Error 2. 미실행 31장 | 연속 서비스 오류로 중단. 미시도를 포함한 운영 실패와 정상 응답의 시각 정확도를 구분. 절전 교란 때문에 Provider 단독 문제로 귀속하지 않음 |

Kimi의 완료 11건 지연 중앙값은 32.89초였으며 취소 요청의 서버 청구 여부는 확인되지 않아 예약 비용을 유지했다. 이 결과를 Kimi의 전체 정확도 점수로 사용하지 않았다. [Sol·Kimi 보고서](10i_stage4a2_api_sol_kimi_grounding_comparison.md), [최초 Astra·GLM·Gemma 보고서](10i_stage4a2_api_full_image_grounding_comparison.md)

## 4. Fast가 지원돼도 Standard를 선택한 이유

### 4.1 지원 조건과 토큰 단가

Terra는 Fast 요청을 지원하며 실제 60회 호출에서도 `service_tier="fast"`가 확인됐다. 공식 문서는 요청별 Fast 설정과 응답의 실제 tier 확인을 안내하며 이미지 입력도 지원한다. 프로젝트의 기본 서비스 설정에 영향을 받지 않도록, 후속 Standard 구현에서는 `service_tier="default"`를 명시하기로 한다. [공식 Fast 문서](https://developers.openai.com/api/docs/guides/fast-mode)

2026-11-03 KST 확인 기준, 비캐시 단문맥 입력·출력의 USD/100만 토큰 단가는 다음과 같다.

| Terra 모드 | 입력 | 출력 | 동일 Usage일 때의 비용 관계 |
| --- | ---: | ---: | --- |
| **Standard** | **$1** | **$5** | 기준 |
| Fast | $2 | $10 | Standard의 2배 |

이는 토큰 단가이며 장당 고정 과금이 아니다. 출력에는 추론 과정에 소비된 Usage가 포함돼 요청마다 비용이 달라진다. 아래 실측 계상에는 캐시 할인을 적용하지 않았고, Fast 실행의 기록된 입력 캐시 토큰 합계는 0이었다. [Terra 공식 모델·가격 문서](https://developers.openai.com/api/docs/models/gpt-6.1-terra)

### 4.2 같은 60장에 대한 Fast 실측

원본 이미지·Prompt·high 추론·출력 한도·HTTP Adapter는 유지하고 요청에 `service_tier="fast"`만 추가했다. 순차 60회, Warm-up·자동 Retry·Standard 재실행은 0회였다. Fast 60/60이 정상 응답했고 실제 tier도 60/60이 `fast`였다. Strict JSON·null 허용 좌표 계약은 모두 60/60 통과했다.

| 지표 | 기존 Standard | 이번 Fast | 관측 차이 |
| --- | ---: | ---: | ---: |
| API 요청 중앙값 | 4.592초 | 3.781초 | **0.811초 감소 · 17.7% 감소** |
| 평균 | 6.530초 | 4.706초 | 27.9% 감소 |
| p95 | 13.668초 | 10.137초 | 25.8% 감소 |
| 60장 API Usage 추정 비용 | $0.147435 | $0.274430 | **86.1% 증가** |
| 장당 평균 Usage 추정 비용 | $0.002457 | $0.004574 | 출력 Usage 차이 포함 |
| Mask 정확도 | 동시 성공 47/58 · 81.0% | 미평가 | 이번 Fast 측정은 시간·비용·응답 형식에 한정 |

전체 중앙값의 비는 약 1.21배였고, 대응 이미지 60장 중 50장은 Fast가 더 빨랐다. Fast 60회 루프 경과는 283.11초, 약 4분 43초였다. 프로세스에 한정한 caffeinate를 적용했으며 측정 구간의 Sleep/Wake 전환과 2초 초과 시계 불일치는 각각 0건이었다. [Fast 실측 보고서](10i_stage4a2_api_terra_fast_latency_20261103.md)

동일 Usage라면 Fast의 토큰 비용은 2배지만 이번 실제 계상비는 약 1.86배였다. 요청별 출력·추론 토큰 수가 달랐기 때문이다. 또한 Standard와 Fast는 서로 다른 시점에 측정했고 기존 Standard는 Terra/Luna 교대 호출, 이번 Fast는 Terra만 순차 호출했다. 따라서 17.7%를 tier 하나만의 인과적 속도 향상으로 확정하지 않는다. Fast의 Mask 정확도가 Standard와 같다는 주장도 하지 않는다.

**이번 용도에서는 중앙값 약 0.81초의 절약보다 Standard의 낮은 비용을 선택한다.** VLM을 초기 발견·필요 시 재탐지에 사용하므로, 이 시간 차이가 Fast를 기본 모드로 사용할 만큼 중요하지 않다고 판단했다. Fast가 지원되지 않아서 Standard를 고른 것이 아니라, 지원과 실측을 확인한 뒤 비용 대비 이득을 비교해 결정했다.

## 5. 기존 DINO 기반 파이프라인 대비 개선

### 5.1 같은 이미지에서의 실제 SAM2 초기 Mask 비교

사용한 PNG·RGB SHA, Parent·Camera·시점은 기존 DINO 실험과 Terra 실험의 60장 모두에서 일치했다. SAM2 checkpoint도 같았다. 따라서 아래는 서로 다른 장면의 평균을 나란히 놓은 비교가 아니다.

| 경로 | Peg 성공 | Fixture 성공 | 두 물체 모두 성공 |
| --- | ---: | ---: | ---: |
| 기존 DINO 선택 규칙 + SAM2 | 48.3% | 44.8% | **16/58 · 27.6%** |
| DINO + Qwen 검증 + SAM2 | 55.0% | 82.8% | **27/58 · 46.6%** |
| **Terra Standard + SAM2** | **81.7%** | **96.6%** | **47/58 · 81.0%** |

물체별 실제 성공 수는 DINO가 Peg 29/60·Fixture 26/58, DINO+Qwen이 33/60·48/58, Terra가 49/60·56/58이다. DINO 선택 규칙은 저장된 legacy 선택을 그대로 사용했다. 앞서 46.6%로 제시한 기준선은 Qwen 외형 검증까지 포함한 경로이며, DINO 단독 선택 경로와 구분한다.

DINO 단독·Top1 Mask 수치는 기존 후보 986개에 실제 실행해 둔 SAM2 Mask 지표와 저장 선택을 연결한 사후 재집계다. 새 DINO·Qwen·Terra·SAM2 실행은 없었고 GT로 더 좋은 후보를 골라 바꾸지도 않았다. DINO 원시 Top1의 동시 성공은 별도로 12/58, 20.7%였다. Box IoU≥0.5 결과를 Mask 성공률로 대체한 분석이 아니다. [DINO–Terra 대응 비교 보고서](10i_stage4a2_terra_standard_vs_dino_sam2_20261103.md)

### 5.2 개선 규모와 실패 회복

기존 DINO 선택 규칙+SAM2 대비 동시 성공률은 **+53.4%p**, 성공 이미지는 **31장 증가**했다. 실패는 **42→11장으로 73.8% 감소**했다. 기존 성공 16장은 모두 Terra에서도 성공했고, 기존 실패 중 31장이 성공으로 바뀌었다. 기존 성공에서 Terra 실패로 바뀐 이미지는 0장이었다.

Qwen 검증을 더한 경로와 비교해도 동시 성공은 27→47장, **+34.5%p**였다. 이 경로의 기존 성공 27장도 모두 유지하고 실패 20장을 추가로 회복했다.

| 대응 비교 | Parent-macro 동시 성공 차이 | Parent-cluster 95% CI | 실패→성공 | 성공→실패 |
| --- | ---: | ---: | ---: | ---: |
| Terra − 기존 DINO 선택 + SAM2 | +53.33%p | +45.00~+61.67%p | 31 | 0 |
| Terra − DINO+Qwen+SAM2 | +34.44%p | +23.33~+44.44%p | 20 | 0 |

독립 단위는 같은 물리 Parent 15개이고 Bootstrap은 10,000회·Seed42다. Parent-macro는 Parent별 비율의 평균이므로 전체 이미지 비율 차이와 수치가 조금 다르다. 두 대응 차이 모두 CI가 0보다 높았다.

### 5.3 초기 발견에서도 이득이 확인됐다

| 범위 | 기존 DINO 선택 + SAM2 | DINO+Qwen+SAM2 | Terra Standard+SAM2 |
| --- | ---: | ---: | ---: |
| Main | 16/30 · 53.3% | 12/30 · 40.0% | 25/30 · 83.3% |
| Wrist | 0/28 · 0.0% | 15/28 · 53.6% | 22/28 · 78.6% |
| 초기 t0 | 13/28 · 46.4% | 22/28 · 78.6% | **26/28 · 92.9%** |
| 후기 t80/t78 | 3/30 · 10.0% | 5/30 · 16.7% | **21/30 · 70.0%** |

초기 발견이라는 실제 사용 목적에서도 t0 동시 성공이 46.4%에서 92.9%로 개선됐다. 후기 값은 해당 RGB를 독립적으로 재탐지해 초기화한 성능이며, t0부터 SAM2로 추적한 성공률과 섞지 않았다.

### 5.4 다른 Actor를 초기화하는 오류가 줄었다

| 경로 | Peg Wrong-object | Fixture Wrong-object |
| --- | ---: | ---: |
| 기존 DINO 선택 + SAM2 | 5/49 · 10.2% | 33/59 · 55.9% |
| DINO+Qwen+SAM2 | 1/44 · 2.3% | 4/57 · 7.0% |
| **Terra Standard+SAM2** | **0/57 · 0.0%** | **0/57 · 0.0%** |

분모는 Tiny를 제외한 비어 있지 않은 예측 Mask다. Terra의 0건은 이번 표본에서 다른 Actor가 우세한 오초기화가 관측되지 않았다는 뜻이다. 부분 Mask나 일부 Robot 오염까지 없었다는 뜻은 아니다.

누락을 0으로 포함한 가시 대상 평균 Mask IoU도 Peg **0.582→0.871**, Fixture **0.447→0.945**로 개선됐다. 따라서 이득은 동시 성공의 임계값 하나에서만 나타난 변화가 아니라 물체별 Mask 품질과 오초기화에서도 확인됐다.

## 6. 최종 결정과 적용 범위

**향후 연구 모델의 Grounding 모듈은 Terra Standard로 확정한다. DINO는 설계에서 제외한다.** 전체 RGB를 이해해 Peg·Fixture Box를 직접 만드는 Terra 경로를 사용하고, DINO 후보 생성·후보 점수·Qwen의 DINO 후보 승인에 의존하는 경로를 설계의 전제로 두지 않는다. SAM2는 Box 기반 초기화와 이후 관측 갱신 역할을 유지한다.

이 결정에는 다음 근거가 함께 작용했다.

1. Sol·Astra와의 관측 성공 수 차이는 2장이었고 Sol 대비 정확도 우열은 확정되지 않았지만, Terra의 비용은 낮고 관측 지연도 짧았다.
2. Luna·Gemma보다 비용은 높아도 실제 동시 Mask 성공과 잘못된 물체 초기화에서 현재 목적에 더 적합했다.
3. Fast는 실제 지원됐지만 중앙값 약 0.81초 단축에 Usage 비용이 86.1% 증가해 Standard를 선택했다.
4. 기존 DINO+SAM2 대비 동시 성공이 16→47장으로 늘었고, DINO+Qwen+SAM2보다도 20장을 추가로 성공시켰다.

추가 장면 검증을 이번 모델 채택의 선행 조건으로 두지 않는다. 이 보고서는 완료된 근거에 따라 앞으로 사용할 모델을 정한 결정 문서다. 과거 보고서의 `DINO_REPLACEMENT: INSUFFICIENT_SUPPORT`는 당시 선택·통합을 보류한 기록으로 보존하고, **향후 모델 선택과 DINO의 설계상 사용 여부는 이번 채택 결정을 따른다.**

실측 범위는 알려진 15 Parent의 초기화·독립 재탐지 이미지다. Terra의 가시 물체 실패 13개 중 7개는 Box IoU≥0.7인데 Mask IoU<0.8인 사례여서 남은 실패를 전부 VLM 인식 오류로 분류하지 않는다. Terra 초기화 후 장기 Tracking, Hole 정밀도, Graph·Controller·Closed-loop 성공은 이번 선택 근거의 실측 항목에 포함하지 않는다. 이 범위 표기는 채택 결정을 보류하는 조건이 아니라 실제 측정한 성과의 정의다.

DINO의 설계 제외 결정을 문서화하면서 기존 실험 기록과 재현용 코드는 보존했다. 이번 작업 범위는 보고서 작성이며 실행 파이프라인 변경·기존 코드 삭제·학습 재개는 하지 않았다. 기존 `PAUSED_SCOPE_REVIEW`와 Stop Stub은 유지한다.

## 7. 근거 자료와 검증 기록

| 근거 | 확인한 내용 |
| --- | --- |
| [API 계열 통합 비교](10i_stage4a2_api_openai_family_grounding_comparison.md) | Astra·Sol·Terra·Luna·Gemma의 Mask 성능, Strict/Fence, 비용·지연, Parent 대응 CI |
| [Sol·Kimi 및 Gemma Fence 평가](10i_stage4a2_api_sol_kimi_grounding_comparison.md) | Sol 완료 결과, Kimi의 시간상 중단, Gemma의 저장 응답 재사용 |
| [최초 Astra·GLM·Gemma 평가](10i_stage4a2_api_full_image_grounding_comparison.md) | 서비스 실패·형식 실패와 시각 실패, 절전 교란, Astra의 비용 예약 |
| [Terra Fast 실측](10i_stage4a2_api_terra_fast_latency_20261103.md) | 실제 Fast tier 60회, 요청 시간·비용, Standard와의 비교 한계 |
| [Terra–DINO 저장 결과 대응 비교](10i_stage4a2_terra_standard_vs_dino_sam2_20261103.md) | 동일 60장·SHA·선택 재현, legacy/Top1/A와 Terra의 Mask 및 성공 전환 |
| [기존 SAM2 초기화·추적 평가](10i_stage4a2_sam2_mask_verifier_tracking.md) | DINO+Qwen A 경로의 원래 초기 Mask 결과, 추적 성능과 정적 성능의 구분 |

주요 원자료는 [API 가족 비교 JSON](../../outputs/task10i/stage4a2/api_terra_luna_grounding_20261103_v1/openai_family_comparison.json), [Terra Standard 비용·지연](../../outputs/task10i/stage4a2/api_terra_luna_grounding_20261103_v1/grounding_latency_cost.json), [Fast 요약](../../outputs/task10i/stage4a2/api_terra_fast_latency_20261103_v1/latency_summary.json), [DINO 대응 비교 JSON](../../outputs/task10i/stage4a2/terra_vs_dino_saved_comparison_20261103_v1/comparison.json)에 있다. 이 문서에 핵심 결과와 판단을 모두 수록했으며 링크는 원자료 추적을 위한 것이다.

기존 실험은 공통 `ExperimentLogger`를 사용했고 원시 응답·예측·Mask 동결과 독립 검증 결과를 보존했다. Fast 실측은 오프라인 보호 검사 5개, 시간·비용 독립 재계산, 기존 파일 1,528개 SHA·원시 응답 60개 SHA·비밀 검사 PASS 기록이 있다. DINO 대응 비교는 선택 재현 240개·A 결과 120개 복원과 별도 구현의 480개 물체 행·174개 이미지 쌍·Parent Bootstrap 재계산 PASS 기록이 있다.

이 통합 보고서 작성에서는 기존 보고서·JSON 수치와 로컬 링크를 대조하고 근거 파일 SHA 보존을 확인했다. 보고서 생성 스크립트나 과거 실험을 재실행하지 않았다. 신규 유료 API 호출, GPU/CUDA/MPS, 원격 실행, Simulator Render/Rollout, Graph 생성, 학습, Controller 실험, Git Commit, 외부 게시도 수행하지 않았다. 현재 누적 API 보수 계상액은 Fast 실측을 포함한 **$4.04610386/$20**이며, 확정 청구서와 기존 GPU 사용료는 별도다.
