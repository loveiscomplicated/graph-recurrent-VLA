# Task 10-I Stage 4A-2: DINO와 Terra Standard의 소요 시간 비교

**같은 원본 60장의 기록에서 DINO의 두 물체 Grounding 중앙값은 0.238초, Terra Standard의 API 요청 중앙값은 4.592초였다. 관측된 Grounding 지연은 Terra가 약 19.3배 길고, 이미지당 중앙값 차이는 약 4.35초였다.**

Terra 채택은 DINO 자체보다 빠르다는 판단에 근거하지 않는다. 기존 DINO 선택+SAM2의 동시 Mask 성공 27.6%를 Terra+SAM2가 81.0%로 높였다는 **입력 품질 개선을 위해 Grounding의 대기 시간을 수용한 선택**이다.

## 1. 같은 60장의 Grounding 기록

| 경로 | 표본 | 평균 | 중앙값 | p95 | 기록 구간 합계 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 Grounding DINO Tiny + 기존 선택 규칙 | 60장 | 0.250초 | **0.238초** | 0.253초 | 15.01초 |
| Terra Standard (`gpt-6.1-terra`, high) | 60장 | 6.530초 | **4.592초** | 13.668초 | 391.79초 (약 6.53분) |

DINO의 입력 Manifest와 Terra의 입력 Manifest에서 Image ID·PNG SHA·RGB SHA·Parent·Camera·시점이 60/60 모두 일치했다. Train Parent 15개, 초기/후기 × Main/Wrist, 원본 native512 자료다. 이 자료는 기존 성능 비교와 같은 코호트다.

- DINO: 기존 Vast.ai 58576054 RTX3090, Grounding DINO Tiny FP32. 이미지당 Peg Prompt 1회와 Fixture Prompt 1회, **2회 Forward**와 기존 후보 선택을 포함한다. 원본은 512이며 Processor 입력은 800×800이다.
- Terra: 로컬 호스트에서 Responses API로 전체 RGB 한 장을 보내 **한 번의 요청으로 두 Box를 출력**했다. `reasoning.effort=high`, `detail=original`, 생성 한도 5120, 실제 응답 tier `default`였다.
- 모든 시간은 과거 저장 로그를 재집계했다. 새 DINO/Terra/SAM2 호출, Simulator, GPU 실행, 원격 접속은 0회다.

## 2. 시간에 포함된 처리

| 항목 | DINO 이미지 시간 | Terra API 시간 |
| --- | --- | --- |
| 이미지 읽기 | PNG 읽기 포함 | Payload 생성·Base64 준비는 제외 |
| 물체 검출 | Peg·Fixture Prompt로 DINO 2회 호출, 전처리/H2D·GPU Forward·후처리 포함 | 이미지 업로드·네트워크·서버 처리·응답 다운로드 포함 |
| 선택/응답 처리 | 기존 candidates/assign 선택 포함 | 응답 JSON 파싱·기존 Adapter 정규화 포함 |
| 기록 오버헤드 | 호출별 시간 수집·원시 결과 파일 저장 일부 포함 | 요청 결과의 로컬 저장은 제외 |
| 모델 Load | 별도 | 서버 내부 Load는 분리 관측 불가 |
| SAM2 / Qwen / Tracking | 모두 제외 | 모두 제외 |

DINO의 측정 코드는 [기존 dino.py](../../graph/experiments/qwen35_appearance_bf16_independent_20261102_v1/dino.py)다. 60개 이미지 시간은 PNG 읽기 직전부터 두 Detector 호출과 선택이 끝난 시점까지다. 이미지별 최종 Ledger 기록·Hash 검사 등 측정 구간 바깥 작업은 위 합계에 모두 포함되지 않는다. 따라서 **기록 구간 합계는 실험 프로세스 전체 경과 시간이 아니다.** Terra 합계도 각 HTTP 요청 시간의 합이며 과거 Terra/Luna 교대 실행의 전체 Wall Time과 구분한다.

DINO는 로컬 GPU 경로이고 Terra는 네트워크를 포함한 API 경로다. 약 19.3배는 이번 운영 경로의 관측 지연 비율이며, 동일 장비에서 모델의 순수 Forward 속도를 비교한 수치가 아니다. DINO+SAM2 전체와 Terra+SAM2 전체를 같은 정의로 측정한 E2E 배수도 아니다.

## 3. 첫 DINO 호출과 Load를 구분한 보조 확인

DINO의 첫 이미지에는 첫 Forward가 포함돼 0.874초가 기록됐다. 모델 Load 3.965초는 모든 이미지 시간과 별도다. 장기 실행에서 이 Load를 이미지마다 반복하는 것으로 계산하지 않는다. 파일 Hash 사전 검사로 OS 파일 캐시가 warm일 수 있어 물리적인 완전 Cold Start로 단정하지 않는다.

| 범위 | n | DINO 중앙값 | Terra 중앙값 | Terra/DINO 중앙값 비 |
| --- | ---: | ---: | ---: | ---: |
| 전체 동일 이미지 | 60 | 0.238초 | 4.592초 | 19.3배 |
| DINO 첫 이미지 제외, 같은 59장 | 59 | 0.238초 | 4.580초 | 19.3배 |

첫 DINO 호출을 제외해도 Grounding 지연에 대한 판단은 동일하다. Terra의 첫 요청을 Cold라고 가정해 제거하거나 임의로 느린 응답을 제외하지 않았다.

## 4. DINO+Qwen 경로와의 구분

Qwen을 포함하면 비교의 대상이 달라진다. 같은 60장에 대한 저장된 외형 Prompt Qwen 후보 검증은 **검증 요청 시간의 이미지별 합계**가 중앙값 9.48초, 평균 10.11초, p95 16.00초였다. 986개 후보를 순차 검사한 `appearance_three` 조건이다.

이 값은 DINO 검출·후보 이미지 제작·SAM2·모델 Load를 제외한 Qwen 검증 구간의 합이다. DINO와 Qwen의 중앙값을 단순히 더해 전체 파이프라인 중앙값을 만들지 않는다. **DINO 자체는 Terra보다 훨씬 빠르고, DINO+Qwen의 후보 검증 단계는 Terra API보다 오래 걸렸다는 서로 다른 관측**을 구분한다.

## 5. 모델 채택에 대한 해석

Terra Standard는 DINO 대비 속도를 희생하면서 실제 초기 Mask 품질을 높이는 선택이다. 매 제어 프레임마다 VLM을 실행하지 않고 초기 발견·필요 시 재탐지에 한정하므로 이 대기 시간을 수용하기로 한 기존 채택 결정과 연결된다.

성능 비교의 동시 성공은 기존 DINO+SAM2 **16/58·27.6%**, DINO+Qwen+SAM2 **27/58·46.6%**, Terra+SAM2 **47/58·81.0%**였다. 이 품질 수치와 이번 Grounding 시간은 같은 60장의 근거지만, 시간 표에 SAM2 실행을 포함했다고 해석하지 않는다.

기존 [채택 보고서](10i_stage4a2_terra_standard_adoption_20261103.md)의 속도·비용 판단에서 Terra가 Sol보다 짧고 Standard가 Fast보다 저렴하다는 비교는 유지된다. 이 추가 비교는 DINO 자체와의 속도 차이를 명확히 한다. 같은 정의의 전체 파이프라인 E2E와 총비용은 여기서 새로 추정하지 않는다.

## 6. 원자료와 검증

- [DINO 60장 원시 시간](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/dino_latency_vram.json)
- [Terra Standard 요청 Ledger](../../outputs/task10i/stage4a2/api_terra_luna_grounding_20261103_v1/request_ledger.json)
- [Qwen 후보 검증 시간](../../outputs/task10i/stage4a2/qwen35_appearance_bf16_independent_20261102_v1/verifier_latency_vram.json)
- [이 비교의 시간 요약](../../outputs/task10i/stage4a2/dino_terra_latency_comparison_20261103_v1/comparison.json)
- [기존 Mask 품질 대응 비교](10i_stage4a2_terra_standard_vs_dino_sam2_20261103.md)

Python statistics와 정렬 기반 선형 p95 보간으로 저장된 시간들을 집계하고, 동일 60장 SHA·시점과 이미지당 두 DINO Prompt 호출을 확인했다. 첫 이미지 제외 결과도 같은 59장에 대응시켰다. 근거 파일 SHA와 로컬 링크를 검증했다. 완료 실험 재실행, 코드·모델·Production 변경, 외부 게시·Commit은 없었으며 PAUSED_SCOPE_REVIEW와 Stop Stub을 유지했다. 현재 호스트에서는 CPU 파일 분석만 수행했고 CUDA/MPS는 새로 실행하지 않았다.
