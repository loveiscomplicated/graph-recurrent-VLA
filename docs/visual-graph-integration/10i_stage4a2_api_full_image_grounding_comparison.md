# Task 10-I Stage4A-2 — API Full-image Grounding·SAM2 초기화 비교

실험일: **2026-11-03 KST** · Identity: `20261103_v1`

**Astra가 독립적으로 생성한 Box로 SAM2를 초기화했을 때, 두 물체 동시 성공은 49/58장(84.5%)이었다.** 동일 이미지의 기존 DINO+Qwen+SAM2는 27/58장(46.6%)이었다. 이번 코호트에서는 강한 VLM의 직접 Grounding에 유망한 근거가 생겼다.

다만 **Gemma의 형식 실패, GLM의 중도 중단, 로컬 호스트 절전**을 분리해야 한다. 특히 연결 오류·타임아웃 6건 모두 절전 구간과 겹쳐, 이번 실행으로 API 제공자의 가용성을 비교할 수 없다. 새 Parent 일반화나 Production 준비 판정도 하지 않는다.

## 1. 목적·자료·입력 계약

목표는 전체 RGB에서 Peg·받침 Fixture의 위치를 직접 찾고, 그 Box만으로 올바른 SAM2 초기 Mask를 생성하는지 확인하는 것이다. Hole 입구 위치를 추정하는 실험은 아니다.

- 기존 Train **15 Parent, 초기 F0/후기 F1, Main/Wrist 총 60장**을 그대로 사용했다. native512 PNG이며 F1은 t80, Parent0134만 t78이다. 이미 연구에 사용한 자료이므로 미사용 Parent 독립 일반화 평가가 아니다.
- 세 모델 모두 동일한 전체 RGB 한 장과 동일한 영어 외형 Prompt를 받았다. 빨강·흰색 직사각형 PEG, 황갈색·갈색 Fixture, 회색·흰색 Robot 및 어두운 Finger의 외형 설명을 사용했다. Functional-role·Few-shot·좌표 힌트는 없다.
- DINO Proposal/Score/Label, GT Actor Mask/Pose, SAM2 Mask, Parent 이름과 대화 이력을 API에 제공하지 않았다. 60개 원본은 RGB512 PNG이고 **임베디드 PNG Metadata도 모두 비어 있음**을 확인했다.
- 출력은 `{"peg": [x1,y1,x2,y2], "box": [x1,y1,x2,y2]}` 또는 Object별 `null`이다. 0–1000 정규화 좌표를 원본512 Pixel로 변환했다. 비유한 수·Boolean·범위 초과·역전/영 면적은 Invalid로 처리하며 Clipping·GT 보정·자동 확대를 하지 않았다.
- Strict JSON을 주 계약으로 삼았다. 최종 응답의 사후 좌표 진단은 별도로 평가했으며, 복구 좌표를 주 성공률이나 SAM2 입력으로 승격하지 않았다.

Prompt SHA256: `df06d5fc6f2cfe2005afa2605b4f35db6f7a602855a2f1abd363df0921aec709`. [원문](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/grounding_prompt.txt)과 입력·모델·프로토콜 Identity를 보존했다.

## 2. 모델 설정과 실제 실행

| 모델 | 고정 설정 |
|---|---|
| OpenAI `gpt-6-astra` | Responses API, original detail, reasoning high, 전체 생성 최대5,120토큰 |
| Z.ai `glm-5.3-flash` | 일반 Model API, Thinking 활성/max, temperature1·top_p.95, 최대8,192토큰 |
| Ollama Cloud `gemma4:31b` | Thinking 활성, temperature1·top_p.95·top_k64, 최대8,192토큰 |

도구·검색·Crop·Schema 강제·자동 Retry·모델 대체를 사용하지 않았다. GLM의 Thinking을 끄지 않았으며, 최종 JSON 전에 토큰 한도가 소진되면 `TOKEN_LIMIT`으로 기록했다. 모델별 설정과 제공 환경이 다르므로 모델 크기만의 효과가 아니라 **해당 예산·설정에서의 서비스 경로 비교**다. API의 내부 Weight·정밀도·GPU 구성은 관측하지 못했고, Alias의 고정 Revision도 보장하지 않는다.

| 모델 | 실제 요청 | 정상 API 최종 응답 | Strict JSON |
|---|---:|---:|---:|
| gpt-6-astra | 60/60 | 57 | 57/60 |
| glm-5.3-flash | 29/60 | 21 | 17/29 |
| gemma4:31b | 60/60 | 60 | 0/60 |

Astra는 연결 오류1건·타임아웃2건이었다. GLM은 정상21건·토큰 한도5건·전송 오류3건 뒤, 사전 정의한 연속 서비스 오류3회 중단 규칙에 따라 종료했다. 남은31장은 미시도로 기록했고 이미지 교체·재호출을 하지 않았다. Gemma는60건 모두 정상 API 응답이었으나 **모두 Markdown 코드블록으로 감싸 Strict JSON에는 실패**했다.

주 평가149요청에 사용자가 추가 지시한 첫 이미지 반복3건을 더해 **총152요청**이다. 반복은 정확도와 SAM2 평가에 포함하지 않았다. 모든 API 원시 응답·좌표를 먼저 동결하고, 유효한 주 Box에 대해서만 기존 Vast58576054/RTX3090에서 SAM2.1 Hiera Small을 실행했다. SAM2는1회 로드, 실제 Mask는144개다. Mask도 동결한 다음 GT Actor Pixel을 열어 채점했다.

## 3. 주 결과 — 실제 SAM2 초기 Mask

GT Visible Pixel≥16을 주 평가 대상으로 삼았다. Peg60개, Fixture58개, 두 물체 동시58장이 분모다. Tiny1–15와 Absent0을 구분하며, Null·Invalid·API 오류·미시도는 전체 초기화 성공에서 실패다. 주 성공 기준은 **Mask IoU≥0.8**이다.

| 모델 | Peg | Fixture | 두 물체 동시 |
|---|---:|---:|---:|
| gpt-6-astra | 50/60 (83.3%) | 55/58 (94.8%) | 49/58 (84.5%) |
| glm-5.3-flash | 11/60 (18.3%) | 10/58 (17.2%) | 8/58 (13.8%) |
| gemma4:31b | 0/60 (0.0%) | 0/58 (0.0%) | 0/58：형식 실패로 SAM2 미실행 |

GLM13.8%는 미시도31장까지 포함한 운영 결과이며 모델의 시각 정확도13.8%를 뜻하지 않는다. Gemma의0%도 물체 인식0%를 뜻하지 않는다. Strict 계약 위반으로 SAM2를 실행하지 않은 결과다.

| 모델 | Main 동시 | Wrist 동시 | 초기 F0 동시 | 후기 F1 동시 |
|---|---:|---:|---:|---:|
| gpt-6-astra | 26/30 (86.7%) | 23/28 (82.1%) | 27/28 (96.4%) | 22/30 (73.3%) |
| glm-5.3-flash | 3/30 (10.0%) | 5/28 (17.9%) | 8/28 (28.6%) | 0/30 (0.0%) |
| gemma4:31b | 0/30 (0.0%) | 0/28 (0.0%) | 0/28 (0.0%) | 0/30 (0.0%) |

다른 Actor가 예측 Mask Pixel의50% 이상일 때 Wrong-object로 정의했다. 분모는 Tiny를 제외한 비어 있지 않은 예측 Mask이며, Recall 분모와 다르다.

| 모델 | Peg Wrong-object | Fixture Wrong-object |
|---|---:|---:|
| gpt-6-astra | 0/57 (0.0%) | 0/55 (0.0%) |
| glm-5.3-flash | 2/15 (13.3%) | 3/17 (17.6%) |
| gemma4:31b | N/A (분모 0) | N/A (분모 0) |

Astra의 Parent-macro 동시 성공은 **83.9%, 95% CI[75.6, 91.7]%**다. 동일60장의 기존 DINO+Qwen 조건A 대비 차이는 **+37.78pp, 95% CI[+25.56,+49.44]pp**다. 독립 집계 단위는 물리 Parent15개이며 Bootstrap10,000회/Seed42를 사용했다. 60장을60개의 독립 Parent로 세지 않았다.

세 모델 공통 시도29장의 동시 성공은 Astra23/29, GLM8/29, Gemma0/29였다. 정상 Strict JSON 응답에 조건화하면 Astra49/55, GLM8/17이며 Gemma는 분모0이다. 조건부 결과는 서로 다른 부분집합이므로 직접 순위나 전체 운영 성공률을 대신하지 않는다. 실제 분모·Parent별 결과·대응 차이 CI는 [평가 지표](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/grounding_evaluation_metrics.json) 및 [최종 해석](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/final_interpretation.json)에 보존했다.

## 4. Box 정확도·형식 실패·남은 오류

아래는 Strict 계약 기준이며, 유효 Box가 없는 Visible 대상의 IoU를0으로 포함했다. 중심 오차는 유효 Box가 존재하는 경우만 계산하고 분모를 명시했다.

| 모델 / 물체 | IoU≥.3 | IoU≥.5 | IoU≥.7 | IoU 평균 | 중심 오차 평균 |
|---|---:|---:|---:|---:|---:|
| gpt-6-astra / peg | 57/60 (95.0%) | 57/60 (95.0%) | 57/60 (95.0%) | 0.920 | 1.09px / n=57 |
| gpt-6-astra / box | 55/58 (94.8%) | 51/58 (87.9%) | 51/58 (87.9%) | 0.882 | 5.16px / n=55 |
| glm-5.3-flash / peg | 13/60 (21.7%) | 13/60 (21.7%) | 13/60 (21.7%) | 0.199 | 24.20px / n=15 |
| glm-5.3-flash / box | 14/58 (24.1%) | 13/58 (22.4%) | 10/58 (17.2%) | 0.197 | 25.05px / n=17 |
| gemma4:31b / peg | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) | 0.000 | N/A |
| gemma4:31b / box | 0/58 (0.0%) | 0/58 (0.0%) | 0/58 (0.0%) | 0.000 | N/A |

사후 진단에서는 최종 응답에서 JSON 좌표만 추출했다. 추론 텍스트에서 좌표를 가져오지 않았고, 아래 Box로 SAM2를 추가 실행하지 않았다.

| 모델 | 전체 Peg 진단 Box IoU≥.5 | 전체 Fixture 진단 Box IoU≥.5 | 정상 API 완료에 조건화한 Peg / Fixture |
|---|---:|---:|---|
| gpt-6-astra | 57/60 (95.0%) | 51/58 (87.9%) | 57/57 (100.0%) / 51/55 (92.7%) |
| glm-5.3-flash | 17/60 (28.3%) | 17/58 (29.3%) | 17/21 (81.0%) / 17/21 (81.0%) |
| gemma4:31b | 56/60 (93.3%) | 40/58 (69.0%) | 56/60 (93.3%) / 40/58 (69.0%) |

Gemma의 Peg93.3%, Fixture69.0%는 의미 있는 위치 추정이 있었음을 보여준다. 다만 출력 후처리 계약을 바꾸고 실제 Mask 성능을 측정하는 후속 실험은 이번에 실행하지 않았다. GLM의 정상21개 응답에서 각17/21이라는 조건부 값도 전체 성공률을 대신하지 않는다.

**Astra의 정상 응답에서 남은 실패는 Peg Mask7개였다.** 모두 Box IoU≥.7이고6개는≥.968이다. 정상 응답·GT Visible Fixture55개는 모두 Mask IoU≥.8이었다. Box가 정확해도 SAM2 Mask가 불완전할 수 있다는 관측이다.

| Astra Peg 실패 Camera-Frame | Box IoU | Mask IoU | Mask 내 Robot Pixel |
|---|---:|---:|---:|
| 10i_s1_0686_F1_hand_camera_native512 | 0.997 | 0.776 | 0.0% |
| 10i_s1_0044_F1_hand_camera_native512 | 0.995 | 0.727 | 0.0% |
| 10i_s1_0157_F1_hand_camera_native512 | 0.993 | 0.356 | 0.0% |
| 10i_s1_0193_F1_base_camera_native512 | 0.758 | 0.784 | 19.4% |
| 10i_s1_0449_F1_base_camera_native512 | 0.968 | 0.798 | 17.4% |
| 10i_s1_0844_F0_hand_camera_native512 | 0.990 | 0.665 | 25.2% |
| 10i_s1_0844_F1_hand_camera_native512 | 0.990 | 0.560 | 0.0% |

네 사례는 Peg Pixel만 포함했지만 일부를 놓쳤고, 세 사례는 Robot Pixel을약17–25% 포함했다. **Wrong-object0건이 Robot 오염0건을 뜻하지 않는다.** 색/재질별 부분 분할이나 그리퍼 경계 혼동은 가능한 원인가설이며, 이번에 인과효과를 분리 검증하지 않았다. GT Box 대체·SAM2 알고리즘 변경도 하지 않았다.

Box/Mask IoU 분포, Precision/Recall, Null·누락·비가시 대상 Hallucination, Actor/Background 오염과 실패 사례 전체를 JSON/CSV에 보존했다.

## 5. 소요 시간·비용과 호스트 절전 제약

사용자 요청에 따라 첫 이미지를 각 모델에 한 번씩 추가 요청했다. PNG·Prompt·설정의 요청 SHA가 최초와 같고, 세 쌍 모두 절전 겹침0초다.

| 모델 | 최초 왕복 시간 | 동일 입력1회 반복 |
|---|---:|---:|
| gpt-6-astra | 16.55s | 14.93s |
| glm-5.3-flash | 26.52s | 32.06s |
| gemma4:31b | 6.14s | 8.63s |

각 조건n=1이며 반복 동안 주 평가도 진행 중이었다. 순수 Warm-up 효과로 단정하지 않는다. 최초 호출은 클라이언트의 첫 요청이며 서버 모델 Cold Load가 관측됐다는 뜻도 아니다.

절전과 겹치지 않은 정상 완료 응답의 왕복 시간은 다음과 같다.

| 모델 | n | Mean | Median | p95 |
|---|---:|---:|---:|---:|
| gpt-6-astra | 56 | 12.99s | 10.03s | 30.35s |
| glm-5.3-flash | 19 | 31.76s | 26.52s | 77.59s |
| gemma4:31b | 59 | 6.20s | 4.97s | 13.12s |

**HOST_SLEEP_CONTAMINATED:** macOS 전원 전환 로그에서 실험 중 Sleep 약139.77분을 확인했다. 11요청이 Sleep과 겹쳤고, 연결 오류·타임아웃6건 모두 그 구간과 겹쳤다. 오류의 직접 원인이 전부 절전이라고 확정할 수는 없지만, 이 실행으로 API 제공자의 가용성을 비교할 수 없다. GLM의 중단 원인도 제공자에게 단독 귀속하지 않는다.

주 평가 UTC 경과는 **01:30:40–04:48:21 KST, 197.69분**이었다. monotonic 기반 HTTP 측정 합계는 **58.78분**이다. 전체 경과를 모델의 순수 추론 시간으로 사용하지 않는다. 요청별 시계·절전 겹침은 [호스트 절전 진단](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/host_sleep_overlap.json)에 보존했다. 정상 응답의 시각 정확도는 유지하되 전송 실패와 분리한다.

실행 관리에서 로컬 idle sleep 방지를 누락했다. 종료 후 Launcher에 프로세스 수명 동안의 `caffeinate -i -w PID` 보호를 보강하고 테스트했다. 영구 전원 설정을 바꾸지 않으며 Lid 닫기·수동 Sleep까지 방지한다고 보장하지 않는다. 이 보강 뒤 유료 평가를 재실행하거나 실패를 대체하지 않았다.

가격은 실험 시작 시 동결한 USD/100만 토큰 단가다. Usage 기반 추정은 캐시 할인을 적용하지 않았고 Invoice 확인값이 아니다.

| 모델 | 입력 / 출력 단가 | 주 평가 Usage 기반 추정 | Usage 미확인 예약액 | 주 초기화 성공1건당 보수적 계상 |
|---|---:|---:|---:|---:|
| gpt-6-astra | $10 / $50 | $1.351840 | $0.864990 | $0.04524 |
| glm-5.3-flash | $0.15 / $0.5 | $0.038315 | $0.020194 | $0.00731 |
| gemma4:31b | $0.14 / $0.4 | $0.033365 | $0.000000 | N/A |

추가 반복3건의 추정은$0.025347이다. 이를 포함한 Usage 확인 요청의 합계 추정은 **$1.448867**, 미확인6건 예약은 **$0.885184**, 전체 보수적 예산 계상은 **$2.334051/$20**이다. 예약은 실제 청구로 주장하지 않는다. 3090 사용료는 별도다.

SAM2는1회 로드(1.128초), 이미지Encoding60개 평균31.09ms, Object Prompt·Mask 저장·Hash 포함144개 평균9.74ms, Peak allocated **0.515GiB**였다. Prompt 시간은 순수 GPU Decoder 시간이 아니며 VLM 동시 상주·두 카메라20Hz·Controller 포함 성능도 측정하지 않았다. API 내부 Vision/Prefill/Decode 시간과 VRAM은 노출되지 않아 추정하지 않았다.

## 6. 검증·보존·산출물

- 공통 ExperimentLogger를 W&B disabled로 재사용했다. 원시 응답·예측 동결→SAM2 Mask 동결→GT 채점의 순서를 검사했다.
- 별도 PIL 계산으로 Object 지표360개, Scene180개를 확인했다. 보호 파일137개, 초기 실행 Source11개, 주 예측 Artifact Hash299개 검증이 PASS했다.
- 원격 SAM2 Checkpoint·설치 Source26개와 Package138개의 전후 보존 검사가 PASS했다. 기존 Checkpoint SHA는 `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`다. FP32 Weight·BF16 autocast, multimask=False, Logit>0 및 설치 기본 후처리를 유지했다.
- 계약6개·실제 추론 Loop 오프라인7개·합성 평가 통합1개·공통 Logger/Artifact42개 테스트가 통과했다. 비유한 좌표 저장 방어를 보강한 뒤에도 저장된149개 응답의 파싱 결과가 모두 같음을 확인했다.
- 최초 실행 Source11개는 `frozen_inference_source/`에 보존했다. 배포 경로 수정과 실행 후 보강은 전후 SHA·이유를 기록했다. API Adapter·Prompt·모델 설정·원본 입력은 변경하지 않았다.
- Key는 로컬 `.env`에서만 읽고 API 인증 Header 외에 기록·원격 전송하지 않았다. 원격에는 GT 없는 RGB·예측·SAM2 실행 코드만 전달했다. macOS는 API/CPU 평가, Linux3090은 SAM2 CUDA를 실행했다.
- Production, 기존 데이터·환경·Checkpoint, PAUSED_SCOPE_REVIEW/Stop Stub을 보존했다. 새로운 DINO Forward·Simulator Reset/Render/Rollout·Tracking·Graph 생성·Training·Commit·외부 게시를 실행하지 않았다.

주요 자료: [프로토콜](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/protocol.json), [입력 Manifest](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/input_manifest.json), [원시 응답 Index](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/full_image_grounding_raw_responses.json), [평가 지표](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/grounding_evaluation_metrics.json), [실패 사례](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/grounding_failure_cases.csv), [시간·추론 토큰](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/observed_timing_summary.json), [독립 검증](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/independent_verification.json), [Artifact Index](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/artifact_index.json).

![전체 RGB, API Box, 사후 GT Box, 실제 SAM2 초기 Mask](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/full_image_grounding_comparison.png)

Hash 순서 첫6장의 고정 비교다. 빨강은 예측Peg, 파랑은 예측Fixture, 초록 얇은 선은 사후 GT Visible Box다. 실제 생성한 Mask만 표시하며 `JSON_INVALID`와 API 상태를 구분했다.

## 7. 최종 판단

**Astra의 현재 경로는 VLM_GROUNDING_PROMISING**이다. 강한 VLM이 DINO 후보 없이 전체 RGB에서 의미와 위치를 함께 찾아 SAM2 초기화를 수행할 수 있다는 근거를 얻었다. 남은 정상 응답 실패는 Peg Mask의 부분 분할·Robot 오염에 집중됐다.

**DINO_REPLACEMENT_CANDIDATE의 완전한 운영 근거는 INSUFFICIENT_SUPPORT**다. 동시 Mask≥80%는 관측됐지만 좌표 계약 성공은 전송 오류 포함95%이고 Fixture Box Recall은87.9%다. 15 Parent의 CI도 넓다. 희귀 오인식·미사용 Parent·새 외형/배경·장기 추적·Hole 위치·Graph 기하 품질·Closed-loop 학습 성능은 검증하지 않았다.

Gemma의 후처리 계약 변경과 GLM의 절전 영향 없는 재평가는 후속 검증 대상이다. 이번에는 추가 실행하지 않았고, Production 변경 없이 보고서를 작성하고 종료했다.
