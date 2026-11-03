# Task 10-I Stage 4A-2: Terra Fast API 소요 시간

Terra Fast 60건의 HTTP E2E 중앙값은 **3.781초**였다. 기존 Standard 4.592초와 비교하면 중앙값 기준 **1.21배**, **17.7% 감소**다.
이는 같은 알려진 60장에 대한 **서로 다른 시점의 관측 비교**다. Tier만의 인과적 속도 향상이나 실시간 제어 성능을 뜻하지 않는다.

## 승인 범위와 고정 계약

- 사용자 요청: “terra fast의 소요 시간을 측정하고 싶어.” 기존 Terra Standard의 원본 60장과 동일한 순서·Prompt·HTTP Adapter를 재사용했다. Standard 유료 재실행은 0회다.
- Train Parent 15개 × 초기/후기 × Main/Wrist, native512 PNG. 미사용 Parent 일반화 실험이 아니다.
- 모델 `gpt-6.1-terra`, Responses API, `reasoning.effort=high`, `detail=original`, `max_output_tokens=5120`, `store=False`. 유일한 요청 변경은 `service_tier=fast`다.
- Prompt SHA256 `df06d5fc6f2cfe2005afa2605b4f35db6f7a602855a2f1abd363df0921aec709`. 한 요청은 전체 RGB 한 장과 영어 외형 Prompt뿐이다. 이전 대화·GT·DINO·SAM2·Parent ID·도구·Schema 강제 없음.
- 순차 요청, 동시성 1, Warm-up 0회, 자동 Retry/실패 대체 0회. 응답 tier 미확인, 종료 오류, 연속 서비스 오류 3회 또는 예산 위반 시 중단한다. Durable RESERVED 요청은 재전송하지 않는다.
- 기존 `adapters.payload/call`과 엄격한 좌표 파서, 공통 `ExperimentLogger`(W&B disabled)를 재사용했다. API 응답·Ledger·요청과 이미지 SHA를 저장하고 동결했다.
- `caffeinate -i -w PID`는 측정 프로세스 수명에만 적용했다. 영구 전원 설정은 변경하지 않았다.

## 실제 요청과 지연

실행: **2026-11-03 06:01:25–06:06:08 KST**. 측정 루프 전체 경과 **283.11초 (4.72분)**. 상태 `COMPLETE`, 중단 사유 `None`.

| 경로 | 정상 완료 n | 평균 s | 중앙값 s | p95 s | 최소–최대 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 Standard (`default`) | 60 | 6.530 | 4.592 | 13.668 | 2.342–21.186 |
| 이번 Fast (`fast`/`priority` 확인) | 60 | 4.706 | 3.781 | 10.137 | 2.093–14.575 |

- 요청 상태: `{'OK': 60}`. 실제 응답 tier: `{'fast': 60}`.
- Fast 첫 요청 4.878초. 첫 요청 제외: n=59, 평균 4.703초, 중앙값 3.771초, p95 10.191초. 첫 요청을 Cold Start라고 단정하지 않는다.
- 대응 이미지 60장 중 Fast가 더 짧은 이미지 50장. 이미지별 Standard/Fast 비율의 중앙값 1.319배. 이는 전체 중앙값의 비와 다른 통계다.
- 엄격한 JSON 성공 60/60, null 허용 좌표 계약 통과 60/60. 좌표 수정·Clipping·Box 확대 없음. **이번 Fast Box의 Grounding/Mask 정확도는 채점하지 않았다.**
- 기록된 입력 캐시 토큰 합계 0. 캐시 상태와 출력·추론 토큰 수가 달라질 수 있으며 이미지별 출력 토큰 수는 대응 CSV에 보존했다.
- 호스트 전원 로그의 측정 구간 Sleep/Wake 전환 0건, wall clock와 perf_counter 차이 2초 초과 요청 0건.

## 비용과 공식 조건

[공식 Fast 문서](https://developers.openai.com/api/docs/guides/fast-mode)는 요청 `service_tier=fast` 설정 및 응답 tier 확인을 안내한다. [Terra 공식 모델 문서](https://developers.openai.com/api/docs/models/gpt-6.1-terra)는 high 추론·이미지 입력을 지원하고 Fast 단가를 Standard의 2배로 안내한다. 2026-11-03 KST 확인 기준 Fast 입력 $2 / 출력 $10, Standard 입력 $1 / 출력 $5 (100만 토큰당)로 동결했다.

- 이번 Usage 기반 추정 **$0.274430**, Usage 미확인 예약 **$0.000000**, 보수 합계 **$0.274430**.
- 기존 누적 **$3.77167386**에 이번 비용을 더한 누적 **$4.04610386/$20**. 이전 3개 주 Ledger와 최초 이미지 반복 Ledger를 한 번씩 합산했다.
- 사전 신규 60회 최대 예약액 $3.45996. 실제 추정 지출과 구분한다. 캐시 할인을 적용하지 않은 Usage 추정이며 확정 청구서가 아니다. Usage 미확인 요청은 최대 예약을 유지한다. 실제 default 강등 응답은 Standard 단가로 추정한다.
- 신규 GPU 사용은 0회이며 기존 3090 사용료는 이 API 계상에 포함하지 않는다.

## 해석·검증과 범위 보존

HTTP Adapter는 비스트리밍 단일 `requests.post` 시작부터 응답 JSON 파싱까지를 `perf_counter`로 측정한다. Payload 이미지 인코딩·로컬 저장·SAM2는 개별 요청 지연에서 제외한다. 기존과 같이 지속 HTTP Session은 사용하지 않아 연결 설정·네트워크 지연을 포함한다. Vision/Prefill/Decode·TTFT는 분리 관측하지 못했다.

기존 Standard는 Terra/Luna를 교대로 호출한 실행이고 이번에는 Terra Fast만 순차 호출했다. 호출 시각·부하·캐싱·출력 길이·확률적 추론량이 다를 수 있어 관측 비율을 Fast tier만의 효과로 확정하지 않는다. 이 비교를 위해 추가 Standard 호출을 실행하지 않았다. SAM2 초기화·Tracking·Hole 정밀도·Object Graph·Closed-loop 품질로 확장해 주장하지 않는다.

오프라인 보호 검사 5개 PASS. 별도 Python statistics/선형 보간 p95 및 raw Usage 산식으로 시간·비용을 독립 재계산했다. 기존 산출물·Source·PAUSED_SCOPE_REVIEW 자료 1528개 SHA, 이번 원시 응답 60개 SHA, 입력 및 요청 Hash 60개를 대조했다. 비밀 검사 82개 파일 PASS. 이는 과거 SAM2 정확도를 재실행한 검사가 아니다.

변경 범위는 신규 격리 실험 코드·출력과 이 보고서다. Production·기존 결과·공통 Logger/Publisher·PAUSED_SCOPE_REVIEW·Stop Stub은 유지했다. 로컬 macOS에서 API 측정과 CPU 분석만 실행했다. CUDA/MPS, 원격 Vast.ai, SAM2, 신규 Simulator Render/Rollout, Graph 생성, 학습, Controller, Commit, 외부 게시를 실행하지 않았다. DINO 제거와 Production 준비 판정은 여전히 INSUFFICIENT_SUPPORT다.

산출물: [요약 JSON](../../outputs/task10i/stage4a2/api_terra_fast_latency_20261103_v1/latency_summary.json) · [대응 이미지 CSV](../../outputs/task10i/stage4a2/api_terra_fast_latency_20261103_v1/paired_latency.csv) · [요청 Ledger](../../outputs/task10i/stage4a2/api_terra_fast_latency_20261103_v1/request_ledger.json) · [실행 계약](../../outputs/task10i/stage4a2/api_terra_fast_latency_20261103_v1/protocol.json) · [독립 검증](../../outputs/task10i/stage4a2/api_terra_fast_latency_20261103_v1/independent_verification.json) · [호스트 전원 검사](../../outputs/task10i/stage4a2/api_terra_fast_latency_20261103_v1/host_sleep_check.json) · [Artifact Index](../../outputs/task10i/stage4a2/api_terra_fast_latency_20261103_v1/artifact_index.json).
기존 비교: [OpenAI 계열 최종 보고서](10i_stage4a2_api_openai_family_grounding_comparison.md).
