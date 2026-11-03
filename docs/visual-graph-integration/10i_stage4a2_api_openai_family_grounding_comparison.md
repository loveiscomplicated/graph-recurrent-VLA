# Task 10-I Stage 4A-2: Astra · Sol · Terra · Luna Grounding/SAM2 비교

동시 Mask 성공률의 전체 할당 분모 점추정은 **GPT-6 Astra, GPT-6.1 Sol (84.5%)**가 가장 높았다. 이는 동일한 알려진60장·고정 설정에서의 관측이며, 대응 CI와 기존 Astra 전송 실패 영향을 함께 해석해야 한다.
탐색적 동시 성공80%를 관측한 OpenAI 모델 중 보수 집계 API 비용이 가장 낮은 모델은 **GPT-6.1 Terra**였다.

## 목적과 고정 조건

전체 RGB 한 장에서 Peg와 Fixture Box를 직접 출력하고, 그 Box만으로 SAM2 초기 Mask를 생성하는 경로를 비교했다. 같은 알려진 Train Parent15개·60장 비교이며 새 Parent 일반화 실험이 아니다.

- 새 모델: GPT-6.1 Terra `gpt-6.1-terra`, GPT-6 Luna `gpt-6-luna`, 각 최대60회. 기존 Astra와 이번에 완료한 Sol 결과를 재사용했다.
- Kimi K3는 **응답 시간이 너무 오래 걸려 사용자 요청으로 중도 탈락**했다. 정확도 때문에 탈락한 것이 아니며, 전체 모델 정확도 순위에서 제외했다. 완료된 응답·시간·비용과 중단1건의 비용 예약은 별도 보존했다.
- 동일 native512 RGB60장: 15 Parent × F0/F1 × Main/Wrist. F0=t0, F1=t80; Parent0134만 t78. 실패 이미지 대체·추가 Sampling 없음.
- 동결 Prompt SHA256 `df06d5fc6f2cfe2005afa2605b4f35db6f7a602855a2f1abd363df0921aec709`. 모든 모델에 동일한 외형 중심 영어 Prompt 원문을 사용했다.
- 한 요청에는 원본 전체 RGB 하나와 Prompt만 전달했다. DINO·GT·Mask·Crop·Point·Parent ID·이전 대화·도구 없음. 요청마다 새 대화다.
- 네 OpenAI 모델 모두 Responses API, detail=original, reasoning=high, 생성 한도5120, Schema 강제 없음. 모델 alias와 실제 응답 ID·Usage·UTC·원시 응답을 기록했다.
- 전체 이미지 기준0–1000 좌표를 .512배로 변환했다. 유한 숫자4개·bool제외·범위·순서 검증, GT 보정·Clipping·Box 확대·응답 재생성 없음.
- Strict JSON과, 응답 전체를 감싼 단일 코드펜스만 제거하는 Fence 모드를 사전에 고정했다. 설명/추론에서 좌표를 검색하거나 수정하지 않았다.
- 각 실험의 예측을 모두 저장·SHA 동결한 뒤 SAM2를 실행하고, 해당 Mask 전체 동결 후에만 GT 배열로 채점했다. 병행 API 실행 때문에 지연은 서비스 부하·네트워크 영향을 포함한다.
- 키는 로컬 .env에만 두었다. 신규 Instance·모델 다운로드·렌더·학습·Tracking·Graph/Controller·Production 변경·Commit·게시 없음. PAUSED_SCOPE_REVIEW 유지.

공식 문서: [Terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra), [Luna](https://developers.openai.com/api/docs/models/gpt-6-luna). 입력/출력100만 토큰당 Terra $1/$5, Luna $0.10/$0.50으로 동결했다.

## 실제 Mask 초기화 성공

GT Visible Pixel>=16을 주 평가 대상으로 삼았다. Peg60개, Fixture58개, 둘 다 보이는 이미지58장이다. Tiny1–15pixel과 Absent0pixel은 별도 집계했다. 서비스 실패·무효 출력·null은 가시 대상 초기화 실패로 포함한다.

### 엄격한 JSON

| Model | Peg Mask IoU>=.8 | Fixture Mask IoU>=.8 | Both Masks IoU>=.8 | Coordinate contract incl. null |
| --- | ---: | ---: | ---: | ---: |
| GPT-6 Astra | 50/60 (83.3%) | 55/58 (94.8%) | 49/58 (84.5%) | 57/60 (95.0%) |
| GPT-6.1 Sol | 50/60 (83.3%) | 57/58 (98.3%) | 49/58 (84.5%) | 60/60 (100.0%) |
| GPT-6.1 Terra | 49/60 (81.7%) | 56/58 (96.6%) | 47/58 (81.0%) | 60/60 (100.0%) |
| GPT-6 Luna | 42/60 (70.0%) | 54/58 (93.1%) | 39/58 (67.2%) | 60/60 (100.0%) |
| Gemma4 31B | 0/60 (0.0%) | 0/58 (0.0%) | 0/58 (0.0%) | 0/60 (0.0%) |

### 사전 고정한 코드펜스 제거 허용

| Model | Peg Mask IoU>=.8 | Fixture Mask IoU>=.8 | Both Masks IoU>=.8 | Coordinate contract incl. null |
| --- | ---: | ---: | ---: | ---: |
| GPT-6 Astra | 50/60 (83.3%) | 55/58 (94.8%) | 49/58 (84.5%) | 57/60 (95.0%) |
| GPT-6.1 Sol | 50/60 (83.3%) | 57/58 (98.3%) | 49/58 (84.5%) | 60/60 (100.0%) |
| GPT-6.1 Terra | 49/60 (81.7%) | 56/58 (96.6%) | 47/58 (81.0%) | 60/60 (100.0%) |
| GPT-6 Luna | 42/60 (70.0%) | 54/58 (93.1%) | 39/58 (67.2%) | 60/60 (100.0%) |
| Gemma4 31B | 47/60 (78.3%) | 40/58 (69.0%) | 35/58 (60.3%) | 60/60 (100.0%) |

Gemma의 Fence 결과는 기존 응답을 재사용하여 실제 SAM2 Mask를 추가 측정한 것이다. 추가 Gemma API 호출은0회다. 형식 실패와 시각 실패를 구분하며 Strict 결과를 소급 수정하지 않았다.

## Camera·시점과 Grounding 오차

| Model (Fence) | Main joint | Wrist joint | F0 joint | F1 joint | Peg Box>=.5 | Fixture Box>=.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-6 Astra | 26/30 (86.7%) | 23/28 (82.1%) | 27/28 (96.4%) | 22/30 (73.3%) | 57/60 (95.0%) | 51/58 (87.9%) |
| GPT-6.1 Sol | 27/30 (90.0%) | 22/28 (78.6%) | 26/28 (92.9%) | 23/30 (76.7%) | 59/60 (98.3%) | 54/58 (93.1%) |
| GPT-6.1 Terra | 25/30 (83.3%) | 22/28 (78.6%) | 26/28 (92.9%) | 21/30 (70.0%) | 55/60 (91.7%) | 54/58 (93.1%) |
| GPT-6 Luna | 22/30 (73.3%) | 17/28 (60.7%) | 25/28 (89.3%) | 14/30 (46.7%) | 50/60 (83.3%) | 50/58 (86.2%) |
| Gemma4 31B | 24/30 (80.0%) | 11/28 (39.3%) | 23/28 (82.1%) | 12/30 (40.0%) | 56/60 (93.3%) | 40/58 (69.0%) |

| Model (Fence) | Peg Wrong-object | Fixture Wrong-object | Peg mean Mask IoU | Fixture mean Mask IoU |
| --- | ---: | ---: | ---: | ---: |
| GPT-6 Astra | 0/57 (0.0%) | 0/55 (0.0%) | 0.887 | 0.916 |
| GPT-6.1 Sol | 0/60 (0.0%) | 0/58 (0.0%) | 0.925 | 0.963 |
| GPT-6.1 Terra | 0/57 (0.0%) | 0/57 (0.0%) | 0.871 | 0.945 |
| GPT-6 Luna | 3/56 (5.4%) | 0/57 (0.0%) | 0.771 | 0.913 |
| Gemma4 31B | 1/60 (1.7%) | 0/58 (0.0%) | 0.860 | 0.685 |

Wrong-object는 다른 Actor가 예측 Mask Pixel의50% 이상인 경우이며, 분모는 Tiny를 제외한 실제 비어 있지 않은 Mask다. Robot 일부 오염·낮은 IoU와 구분한다. 상세 Box IoU>=.3/.5/.7, 중심 Pixel 오차, Mask Precision/Recall, Actor 구성·배경 오염, null·미검출·Absent Hallucination과 실제 분모는 JSON/CSV에 저장했다.

## Parent 단위 대응 비교

| Model (Fence) | Parent-macro joint | Parent-cluster 95% CI |
| --- | ---: | ---: |
| GPT-6 Astra | 83.9% | 75.6–91.7% |
| GPT-6.1 Sol | 83.9% | 75.6–91.7% |
| GPT-6.1 Terra | 80.6% | 72.8–88.3% |
| GPT-6 Luna | 67.2% | 60.6–74.4% |
| Gemma4 31B | 60.0% | 51.1–67.8% |

| Paired difference | Parent-macro pp | 95% CI pp |
| --- | ---: | ---: |
| astra_minus_sol | +0.0 | -5.0–+5.0 |
| astra_minus_terra | +3.3 | -5.0–+11.7 |
| astra_minus_luna | +16.7 | +6.7–+26.7 |
| sol_minus_terra | +3.3 | -3.3–+10.0 |
| sol_minus_luna | +16.7 | +8.3–+25.0 |
| terra_minus_luna | +13.3 | +6.7–+21.7 |

Physical Parent15개를 독립 단위로 사용하고 Seed42·10,000회 Cluster Bootstrap을 계산했다. Camera/시점을 별도 Parent로 세지 않았다. 표본이 작고 이미 알려진 장면이라는 한계가 있다.
모든 완료 비교 모델의 API가 정상 응답한 공통 이미지 57장의 Fence 동시 성공: GPT-6 Astra 49/55 (89.1%), GPT-6.1 Sol 48/55 (87.3%), GPT-6.1 Terra 45/55 (81.8%), GPT-6 Luna 38/55 (69.1%), Gemma4 31B 35/55 (63.6%).
기존 Astra 실행에는 Mac 절전과 겹친 전송 실패3건이 있다. 전체 분모와 공통 정상응답 집합을 함께 보며 제공자 가용성 차이로 단정하지 않는다.

## 지연·비용·소요 시간

| New model | API status | Mean / median / p95 OK latency | First request | Accounted cost | Cost / joint success |
| --- | --- | ---: | ---: | ---: | ---: |
| GPT-6.1 Terra | {'OK': 60} | 6.53 / 4.59 / 13.67 s (n=60) | 4.18 s | $0.14743 | $0.00314 |
| GPT-6 Luna | {'OK': 60} | 8.06 / 6.41 / 19.15 s (n=60) | 10.36 s | $0.02071 | $0.00053 |

Terra/Luna 신규 API 경과 14.63분, 신규 보수 비용 $0.16814. 이전 실험·첫 이미지 반복·Sol·부분 Kimi까지 중복 없이 합산한 보수 비용 **$3.77167/$20**다. 이 값은 Usage×공식 단가 추정 및 Usage 누락 예약이며 청구 확정값이 아니다. 캐시 할인은 적용하지 않았고 3090 요금은 별도다.
사전 예약 검증에는 동시 실행 중이던 외부 실험의 최악 예약 상한 $16.32923를 사용했다. 최종 비용에 그 상한을 실제 지출로 더하지 않았다.
- HTTP E2E를 측정했으며 서버 Vision/Prefill/Decode는 개별 관측하지 못했다. 첫 요청을 Cold라고 단정하지 않는다. 모델별 첫 요청과 나머지 요청 통계는 별도 보존했다.
- Terra/Luna SAM2는 기존58576054 RTX3090에서 1회 로드, 227개 Mask를 생성했다. Load 967.8ms, Peak allocated 0.515GiB.
- SAM2.1 Hiera Small 기존 checkpoint SHA를 검증했다. FP32 Weight, BF16 autocast, multimask_output=False, logit>0. 임시 단일 프레임 실행이며 Memory Commit·장기 Tracking 없음.
- VLM은 초기 발견/필요 시 재Grounding 용도다. 이 API 지연으로 Control20Hz 또는 실제 Closed-loop 성공을 주장하지 않는다.

### 전체 모델의 비용과 정상 응답 지연

| Model | Usage-based estimate | Unknown-usage reservation | Conservative total | Median OK latency | Total / joint success |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-6 Astra | $1.35184 | $0.86499 | $2.21683 | 10.03s (n=56) | $0.04524 |
| GPT-6.1 Sol | $0.27609 | $0.00000 | $0.27609 | 8.27s (n=60) | $0.00563 |
| GPT-6.1 Terra | $0.14743 | $0.00000 | $0.14743 | 4.59s (n=60) | $0.00314 |
| GPT-6 Luna | $0.02071 | $0.00000 | $0.02071 | 6.41s (n=60) | $0.00053 |
| Gemma4 31B | $0.03336 | $0.00000 | $0.03336 | 4.97s (n=59) | $0.00095 |

위 표는 모델별 주 평가60건의 API 비용이다. 추가 첫 이미지 반복과 부분 Kimi 비용은 모델별 표에서 제외하고 전체 예산에만 포함했다. Astra/Gemma 지연은 기존 절전 중첩 요청을 제외한 정상 응답 통계이며, 병행 실행 조건 차이가 있어 순수 모델 속도 비교로 단정하지 않는다.

## 실패 분해와 판단

| Model (Fence) | Visible object failures | Box IoU>=.7 but Mask<.8 | Wrong-object failures |
| --- | ---: | ---: | ---: |
| GPT-6 Astra | 13 | 7 | 0 |
| GPT-6.1 Sol | 11 | 10 | 0 |
| GPT-6.1 Terra | 13 | 7 | 0 |
| GPT-6 Luna | 22 | 6 | 3 |
| Gemma4 31B | 31 | 7 | 1 |

관측된 Box 성공 후 Mask 실패는 SAM2의 부분 분할 또는 Robot 오염과 구분해서 해석한다. 색 경계·가림·그리퍼 접촉은 가능한 원인 가설이며 이번에 인과적으로 검증하지 않았다. 최종 모델 선택에는 동시 Mask 성공·Wrong-object·응답 안정성·지연·성공당 비용을 함께 사용해야 한다.
새 Parent·희귀 오류·Hole 위치와 정밀도·초기화 후 Tracking·Graph/Controller·실제 제어는 검증하지 않았다. 작은 동일 장면 비교만으로 Production 준비나 DINO 제거를 선언하지 않는다. **DINO_REPLACEMENT: INSUFFICIENT_SUPPORT.**

## 소규모 비교 그림

![모델별 실제 Grounding 및 Mask](../../outputs/task10i/stage4a2/api_terra_luna_grounding_20261103_v1/full_image_grounding_comparison.png)

Hash 순서 첫4장이다. 빨강=예측 Peg Box, 파랑=예측 Fixture Box, 초록=사후 GT Box. 반투명 색은 실제 실행한 SAM2 초기 Mask이며 제목 아래 숫자는 Mask IoU다. 결과를 보고 성공 사례를 선택하지 않았다.

## 검증·산출물

- 공통 ExperimentLogger(W&B disabled)를 재사용했다. Adapter·형식·좌표·입력 격리·예산·Retry/재개 계약 테스트10개 통과. 유료 API를 테스트에서 호출하지 않았다.
- 독립 PIL ImageChops/getbbox 구현으로 Box·Mask IoU와 Wrong-object를 재계산하고 SHA를 검증했다. 원격 checkpoint·SAM2 소스·패키지를 전후 비교했다.
- API 키/인증 Header는 로그·Manifest·원격에 저장하지 않았다. 로컬 CUDA는 미실행이며 실제 CUDA SAM2는 지정된 RTX3090에서만 실행했다.
- 기존 데이터·Checkpoint·이전 결과·PAUSED_SCOPE_REVIEW를 보존했다. Git Commit·외부 게시 없음.

[가족 모델 비교 지표](../../outputs/task10i/stage4a2/api_terra_luna_grounding_20261103_v1/openai_family_comparison.json) · [비용·지연](../../outputs/task10i/stage4a2/api_terra_luna_grounding_20261103_v1/grounding_latency_cost.json) · [Strict 실패](../../outputs/task10i/stage4a2/api_terra_luna_grounding_20261103_v1/grounding_failure_cases.csv) · [Fence 실패](../../outputs/task10i/stage4a2/api_terra_luna_grounding_20261103_v1/fence_grounding_failure_cases.csv) · [독립 검증](../../outputs/task10i/stage4a2/api_terra_luna_grounding_20261103_v1/independent_verification.json) · [원격 보존](../../outputs/task10i/stage4a2/api_terra_luna_grounding_20261103_v1/remote_preservation_verification.json)

Sol·Gemma 및 Kimi 중단의 전체 기록: [선행 보고서](10i_stage4a2_api_sol_kimi_grounding_comparison.md).
