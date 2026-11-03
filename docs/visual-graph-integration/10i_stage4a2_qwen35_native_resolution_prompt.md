# Task 10-I Stage4A-2 — Native Resolution × Prompt 2×2 Pilot

**Native 512와 P1에서 위치를 읽는 능력은 개선 신호가 있었지만, 출력 계약 실패가 실제 사용을 막았다.** 사후 진단에서 512/P1 좌표는 PEG 13/16, BOX 11/16이 IoU≥0.5였으나 정식 JSON 계약 통과는 0/16이다. 이는 Accepted 성능이나 SAM2 성공으로 계산하지 않는다.

BF16 Qwen3.5-9B에서 시뮬레이터 원본 128/512 해상도와 두 프롬프트를 비교했다. 동일 물리 상태의 16개 Camera-Frame에 4조건을 적용했고, 64개 Qwen 요청과 64개 Temporary SAM2 대상을 완료했다. 이 결과는 이미 연구에 사용한 Train Parent 4개의 파일럿이며 독립 일반화 결과가 아니다.

## 1. 실험 계약과 Identity

- Vast.ai instance `58576054`, RTX 3090 24GB. 기존 모델·Processor·환경·SAM2 가중치 재사용. 새 모델/패키지 설치 없음.
- `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, 전체 BF16, SDPA, `cuda:0`, offload 없음. 모델 로드 1회.
- 모든 조건에서 Processor `min_pixels=max_pixels=262144`, 실제 grid `[1,32,32]`, vision token 256개. Native 128은 이 동일 Processor 안에서 리사이즈되고 Native 512는 처음부터 512로 렌더링된다.
- Batch 1, greedy, `enable_thinking=False`, `max_new_tokens=128`, KV cache ON. 별도 warmup, 자동 retry, sampling 없음.
- 좌표 0–1000을 각 원본 너비/높이로 변환한다. 128/512를 하드코딩한 동일 배율로 처리하지 않는다. 범위 밖 좌표를 clipping하거나 GT로 보정하지 않는다.
- P0/P1 모두 같은 native array schema와 동일 파서. bare JSON을 별도로 집계하고, 응답 전체를 감싼 단일 JSON fence도 기존 계약대로 허용한다. 중복 키, 부분 응답 복구, 설명문, nonfinite/bool/역전·범위 밖 좌표는 불허한다.
- 원래 128 RGB/256 Processor 점수를 이번 A조건 점수로 재사용하지 않았다. 네 조건 모두 새로 추론했다.

| Prompt | 단어 수 | SHA256 |
|---|---:|---|
| P0 | 889 | `40b8c68e18ab9d588f86760ea5023cf0366e85e216fe7d94e18f749348f6f3cb` |
| P1 | 245 | `1a58c877d6020d1d041f54a364fa0fa3ae09ad5864c14a5ae6be41b17f7d9f61` |

P0는 기존 상세 프롬프트를 그대로 유지했다. P1은 작업의 두 역할과 물체/로봇 구분, 가시 부분, 좌표·출력 규칙을 간결하게 정리한 단일 후보다. 색상·위치·Episode 정답·후보 Box를 넣지 않았고 결과 후 수정하지 않았다. 두 Prompt의 입력 토큰 차이는 Prompt 요인의 일부다.

## 2. 대상 선정과 Native Capture 검증

24개 기존 Train Parent에서 `SHA256("Task10I_native_resolution_prompt_pilot_v1|" + parent_id)` 오름차순 앞 4개를 고정했다. GT 가시성·DINO/Qwen 성적 필터는 사용하지 않았다. 선택 Parent: `10i_s1_0531`, `10i_s1_0275`, `10i_s1_0197`, `10i_s1_0305`. 각 t0/t80, base_camera/hand_camera를 모두 사용했다. 누락·대체 Frame은 없다. 나머지 20개 Parent는 이 파일럿에서 제외했다.

저장된 seed로 동일 geometry를 만들고 기록된 qpos/qvel, Peg/Box pose를 복원했다. TCP/hole pose·geometry·카메라 calibration을 검증한 뒤, 기존 카메라 설정에서 uid와 width/height만 바꾼 512 카메라를 같은 Scene에 추가했다. mount/pose/FOV는 동일하며, 두 해상도 RGB·Depth·Actor Segmentation은 한 번의 동일 상태 관측에서 얻었다. 물리 Step은 0회다. 이는 정적 렌더 복원이며 PhysX solver cache까지 동적 재현했다는 뜻은 아니다.

실행 backend는 `physx_cpu / sapien_cpu`이며 CUDA 물리 시뮬레이션은 실행하지 않았다. Native 512 렌더는 RTX 3090 호스트에서 실제 실행했다. 원래 보존 RGB와 새 128 RGB는 8/16에서 픽셀 일치했고, 나머지도 최대 평균 절대 RGB 차이 0.002 (0–255 척도)였다. 과거 이미지와 완전히 같다고 주장하지 않으며, 모든 비교는 이번 동시 캡처끼리 수행한다.

## 3. DINO-free 입력과 평가 장벽

Capture 프로세스는 장면 복원에 필요한 privileged state를 읽고 GT sidecar를 저장한다. Qwen inference 프로세스는 원본 전체 RGB 1장과 고정 Prompt만 사용한다. GT/로봇 상태/캡처 seed/Depth/Segmentation/후보 Box는 Qwen message에 들어가지 않는다. 64개 원시 응답과 좌표 CSV를 SHA 동결한 뒤 Qwen Box만으로 SAM2를 실행했고, 모든 Mask를 동결한 후 별도 CPU 평가 프로세스가 GT sidecar를 열었다. 과거 실험 결과가 이미 알려진 파일럿으로, 연구자 전체가 GT를 처음 보는 맹검 실험이라고 표현하지 않는다. DINO 원시 출력은 열지 않았고 새 DINO forward는 없다. 이전 보고서의 baseline 부족 요약을 확인했지만 이번 지표 계산이나 VLM 입력에 DINO 결과를 사용하지 않았다.

## 4. Grounding 결과

주 Recall은 가시 픽셀 ≥16인 목표에 대한 Bounding Box IoU≥0.5 비율이다. Null/invalid는 분모에서 제거하지 않고 Miss로 처리한다. 작은 가시 영역(1–15)은 TINY_VISIBLE, 0은 ABSENT로 구분한다.

| 조건 | PEG Recall | BOX Recall | PEG 평균 IoU | BOX 평균 IoU | JSON 허용 | Bare JSON |
|---|---:|---:|---:|---:|---:|---:|
| native128_P0 | 6/16 (37.5%) | 2/16 (12.5%) | 0.324 | 0.213 | 14/16 (87.5%) | 1/16 (6.2%) |
| native512_P0 | 2/16 (12.5%) | 2/16 (12.5%) | 0.120 | 0.149 | 4/16 (25.0%) | 1/16 (6.2%) |
| native128_P1 | 0/16 (0.0%) | 0/16 (0.0%) | 0.000 | 0.000 | 0/16 (0.0%) | 0/16 (0.0%) |
| native512_P1 | 0/16 (0.0%) | 0/16 (0.0%) | 0.000 | 0.000 | 0/16 (0.0%) | 0/16 (0.0%) |

| 조건 | PEG Parent-macro Recall | BOX Parent-macro Recall | 두 물체 동시 성공 | 하나 이상 Wrong-object |
|---|---:|---:|---:|---:|
| native128_P0 | 0.375 | 0.125 | 2/16 (12.5%) | 5/16 (31.2%) |
| native512_P0 | 0.125 | 0.125 | 2/16 (12.5%) | 2/16 (12.5%) |
| native128_P1 | 0.000 | 0.000 | 0/16 (0.0%) | 0/16 (0.0%) |
| native512_P1 | 0.000 | 0.000 | 0/16 (0.0%) | 0/16 (0.0%) |

| 조건 | 물체 | IoU≥0.3 | IoU≥0.7 | Hallucination / ABSENT | False discovery / emitted | Null |
|---|---|---:|---:|---:|---:|---:|
| native128_P0 | peg | 8/16 (50.0%) | 2/16 (12.5%) | 0/0 (N/A) | 8/14 (57.1%) | 0/32 (0.0%) |
| native128_P0 | box | 4/16 (25.0%) | 2/16 (12.5%) | 0/0 (N/A) | 12/14 (85.7%) | 0/32 (0.0%) |
| native512_P0 | peg | 2/16 (12.5%) | 2/16 (12.5%) | 0/0 (N/A) | 2/4 (50.0%) | 0/32 (0.0%) |
| native512_P0 | box | 3/16 (18.8%) | 2/16 (12.5%) | 0/0 (N/A) | 2/4 (50.0%) | 0/32 (0.0%) |
| native128_P1 | peg | 0/16 (0.0%) | 0/16 (0.0%) | 0/0 (N/A) | 0/0 (N/A) | 0/32 (0.0%) |
| native128_P1 | box | 0/16 (0.0%) | 0/16 (0.0%) | 0/0 (N/A) | 0/0 (N/A) | 0/32 (0.0%) |
| native512_P1 | peg | 0/16 (0.0%) | 0/16 (0.0%) | 0/0 (N/A) | 0/0 (N/A) | 0/32 (0.0%) |
| native512_P1 | box | 0/16 (0.0%) | 0/16 (0.0%) | 0/0 (N/A) | 0/0 (N/A) | 0/32 (0.0%) |

Null 열은 조건 전체의 Object-level 비율이다. ABSENT 분모가 없으면 Hallucination은 측정 불가이며 0%로 해석하지 않는다. False discovery는 평가 가능 상태에서 Box를 출력했으나 IoU≥0.5를 만족하지 못한 비율이다. 잘못된 물체는 IoU<0.5이면서 다른 Actor가 Box 내부 foreground의 ≥70%이고 그 Actor pixel≥16인 보수적 진단이다. Mislocalization과 별도로 저장했다.

## 5. Camera / 시점 / 공간 오차

| 조건 | 구분 | PEG Recall | BOX Recall |
|---|---|---:|---:|
| native128_P0 | base_camera | 0/8 (0.0%) | 0/8 (0.0%) |
| native128_P0 | hand_camera | 6/8 (75.0%) | 2/8 (25.0%) |
| native128_P0 | F0 | 4/8 (50.0%) | 2/8 (25.0%) |
| native128_P0 | F1 | 2/8 (25.0%) | 0/8 (0.0%) |
| native512_P0 | base_camera | 0/8 (0.0%) | 0/8 (0.0%) |
| native512_P0 | hand_camera | 2/8 (25.0%) | 2/8 (25.0%) |
| native512_P0 | F0 | 2/8 (25.0%) | 2/8 (25.0%) |
| native512_P0 | F1 | 0/8 (0.0%) | 0/8 (0.0%) |
| native128_P1 | base_camera | 0/8 (0.0%) | 0/8 (0.0%) |
| native128_P1 | hand_camera | 0/8 (0.0%) | 0/8 (0.0%) |
| native128_P1 | F0 | 0/8 (0.0%) | 0/8 (0.0%) |
| native128_P1 | F1 | 0/8 (0.0%) | 0/8 (0.0%) |
| native512_P1 | base_camera | 0/8 (0.0%) | 0/8 (0.0%) |
| native512_P1 | hand_camera | 0/8 (0.0%) | 0/8 (0.0%) |
| native512_P1 | F0 | 0/8 (0.0%) | 0/8 (0.0%) |
| native512_P1 | F1 | 0/8 (0.0%) | 0/8 (0.0%) |

| 조건 | PEG center error / 원본 px | BOX center error / 원본 px | PEG / 128px 환산 | BOX / 128px 환산 |
|---|---:|---:|---:|---:|
| native128_P0 | 14.880 | 33.345 | 14.880 | 33.345 |
| native512_P0 | 67.031 | 58.532 | 16.758 | 14.633 |
| native128_P1 | - | - | - | - |
| native512_P1 | - | - | - | - |

원본 pixel 오차는 128/512의 척도가 다르므로 환산값도 함께 제시했다. Center error는 Box를 출력한 경우만의 조건부 수치다. Null/invalid를 반영한 평균 IoU·Recall과 함께 해석한다.

## 6. 2×2 Paired 효과

두 해상도에서 모두 가시 pixel≥16인 공통 목표만 비교한다. 공통 분모 PEG 16, BOX 16. Parent별 차이를 평균했고 Parent 4개 단위 Bootstrap 10,000회, seed 42를 사용했다. CI는 작은 파일럿의 탐색적 수치다. 희귀 오류·독립 일반화·통계적 우월성의 증거로 사용하지 않는다. 다중 비교 보정은 없다.

| 대비 | 물체 | Parent-macro Recall 차이 | 탐색적 95% CI | 평균 IoU 차이 |
|---|---|---|---:|
| resolution_P0 | peg | -0.250 | [-0.4375, -0.0625] | -0.203 |
| resolution_P0 | box | 0.000 | [-0.1875, 0.1875] | -0.065 |
| resolution_P1 | peg | 0.000 | [0.0, 0.0] | 0.000 |
| resolution_P1 | box | 0.000 | [0.0, 0.0] | 0.000 |
| prompt_native128 | peg | -0.375 | [-0.5, -0.25] | -0.324 |
| prompt_native128 | box | -0.125 | [-0.25, 0.0] | -0.213 |
| prompt_native512 | peg | -0.125 | [-0.25, 0.0] | -0.120 |
| prompt_native512 | box | -0.125 | [-0.25, 0.0] | -0.149 |
| interaction | peg | 0.250 | [0.0625, 0.4375] | 0.203 |
| interaction | box | 0.000 | [-0.1875, 0.1875] | 0.065 |

resolution_P0/P1은 512−128, prompt_native128/512는 P1−P0다. interaction은 `(D−C)−(B−A)`이며 해상도 효과가 Prompt에 따라 달라지는지를 나타낸다.

## 7. Temporary SAM2 실제 초기화

동일한 사전 선정 16개 Camera-Frame 전부를 조건마다 평가했다. 총 64개 평가 대상 중 유효 계약을 통과한 18개에서 실제 Temporary State를 생성해 36개 Object Box를 초기화했고, 나머지는 invalid로 미초기화 실패 처리했다. SAM2.1 Hiera Small 기존 가중치, 임시 State, Qwen 유효 Box만 제공했다. RGB를 기존 경로대로 JPEG에 기록하고 SAM2 내부 1024 처리 후 원본 128/512 Mask를 얻었다. GT Box 대체·장기 추적·기존 Memory Commit은 없다.

| 조건 | PEG Mask IoU≥0.8 | BOX Mask IoU≥0.8 | PEG 평균 Mask IoU(실패 포함) | BOX 평균 Mask IoU(실패 포함) |
|---|---:|---:|---:|---:|
| native128_P0 | 2/16 (12.5%) | 2/16 (12.5%) | 0.230 | 0.210 |
| native512_P0 | 1/16 (6.2%) | 2/16 (12.5%) | 0.116 | 0.120 |
| native128_P1 | 0/16 (0.0%) | 0/16 (0.0%) | 0.000 | 0.000 |
| native512_P1 | 0/16 (0.0%) | 0/16 (0.0%) | 0.000 | 0.000 |

Native 512에서는 SAM2에 전달하는 RGB도 더 선명하므로 Mask 성능 차이는 Qwen Box와 SAM2 입력 정보가 함께 달라진 End-to-end 효과다. Qwen 자체의 위치 성능은 Box IoU로 구분하며, SAM2 개선 전부를 Qwen 개선으로 귀속하지 않는다. Pixel precision/recall, Actor composition, robot/background contamination, Wrong-object와 초기화 여부는 [SAM2 사례 CSV](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/sam2_initialization_cases.csv) 및 [전체 SAM2 지표](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/sam2_initialization_metrics.json)에 보존했다. Null/invalid 미초기화는 가시 목표의 End-to-end 성공률에서 실패로 유지했다.

## 8. Latency / 자원

Model load 3.80s; load 후 allocated 17.527 GiB. CPU process sampled peak 18.304 GiB (50ms RSS sampling), ru_maxrss 19236088 KiB. 모델 파일은 사전 hash 검사로 file cache가 warm일 수 있다.

| 조건 | Warm n | 평균 E2E ms | Median ms | p95 ms | 평균 output tokens | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| native128_P0 | 15 | 3971.567 | 3715.779 | 5793.817 | 83.250 | 18.005 |
| native512_P0 | 16 | 4716.703 | 5685.930 | 5867.248 | 102.250 | 18.005 |
| native128_P1 | 16 | 5403.313 | 5430.819 | 5545.935 | 127.000 | 17.770 |
| native512_P1 | 16 | 5134.883 | 5406.855 | 5493.937 | 120.625 | 17.770 |

Cold 첫 요청은 native128_P0, 4090.6ms이며 Warm 집계에서 제외했다. 조건 순서는 각 16개 pair 안에서 Latin rotation으로 교차시켜 각 순서 위치에 조건당 4회씩 배치했다. 한 번 로드한 모델을 재사용했다.

Qwen+SAM2 동시 GPU 상주: allocated 17.845 GiB, offload 없음. SAM2 warm forward를 포함한 peak와 preprocessing/vision/prefill/decode/tokens는 [Latency 원자료](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/grounding_latency_vram.json)에 저장했다. E2E는 PNG read + Processor + H2D + generation + decode이며 CUDA event first-forward에는 vision이 포함된다. Prefill−vision 차이는 근사치다. 설치 환경의 flash-linear-attention/causal_conv1d 부재로 일부 Gated DeltaNet·causal-conv 연산은 reference PyTorch 경로를 사용했다. 최적화 kernel 환경의 최고 속도 측정이 아니다. Camera rendering 시간은 별도 capture 총 5.61s로, 환경 생성·seed reset·복원·RGB/GT 저장을 포함하여 순수 sensor latency로 쓰지 않는다.

## 9. 실패 관측과 원인 해석

관측은 저장된 응답·Actor segmentation·Box/Mask 지표에 근거한다. 포맷 실패, 의미 혼동, 공간 오차를 분리했다. Input/output token 제한에 도달했다는 사실만으로 truncation이라 판정하지 않는다.

- native128_P0: JSON 허용 14/16 (87.5%); semantic confusion {"peg": {"peg->box": 4}, "box": {"box->peg": 5}}; 128 output token 도달 2/16.
- native512_P0: JSON 허용 4/16 (25.0%); semantic confusion {"peg": {"peg->robot": 1}, "box": {"box->robot": 1, "box->peg": 1}}; 128 output token 도달 8/16.
- native128_P1: JSON 허용 0/16 (0.0%); semantic confusion {"peg": {}, "box": {}}; 128 output token 도달 10/16.
- native512_P1: JSON 허용 0/16 (0.0%); semantic confusion {"peg": {}, "box": {}}; 128 output token 도달 10/16.

가설: Native 고해상도는 실제 경계·opening·가림 정보의 손실을 줄일 수 있지만, 의미 역할의 혼동이나 좌표 생성 능력 부족을 자동으로 해결하지는 않는다. P1 효과는 길이 축소·표현·중복 키 지시를 함께 바꾼 Prompt 전체 효과이며, 특정 문장 하나의 인과 효과로 해석할 수 없다. Processor 512를 공통으로 고정했기 때문에 이번 resolution 대비는 시각 토큰 수 증가의 효과와 분리된다.

### 출력 계약과 분리한 사후 좌표 진단

Primary accepted 결과의 0점은 좌표가 전혀 없었다는 뜻이 아니다. 모든 예측·SAM2 평가를 끝낸 뒤 사후 진단을 추가했다. 완전한 JSON에서 반복 키의 값이 정확히 일치하는 경우에만 좌표를 읽고, extra field는 진단에서만 무시했다. 서로 다른 반복 좌표나 불완전 JSON은 복구하지 않았다. GT로 좌표를 고르거나 보정하지 않았다. 이 파서는 Accepted 파서로 승격하지 않았고 이 좌표로 SAM2를 추가 실행하지 않았다.

| 조건 | 진단 가능 응답 | PEG 진단 IoU≥0.5 | BOX 진단 IoU≥0.5 | PEG 평균 IoU | BOX 평균 IoU |
|---|---:|---:|---:|---:|---:|
| native128_P0 | 16/16 | 6/16 | 3/16 | 0.369 | 0.272 |
| native512_P0 | 13/16 | 8/16 | 9/16 | 0.399 | 0.524 |
| native128_P1 | 15/16 | 10/16 | 5/16 | 0.475 | 0.353 |
| native512_P1 | 15/16 | 13/16 | 11/16 | 0.623 | 0.547 |

모든 조건에서 가시 분모는 동일하며 Parent당 물체별 4개이므로 이 표의 Image recall과 Parent-macro recall은 같다. 상세 Parent 결과와 진단 가능 공통 subset 비교는 [사후 진단 JSON](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/posthoc_coordinate_diagnostic.json), 각 좌표는 [진단 CSV](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/posthoc_coordinate_diagnostic.csv)에 보존했다. 이 진단은 사전 Primary 지표가 아니고 네 Parent의 사후 탐색 결과다.

P0에서 Native 128→512의 진단 좌표 Recall은 PEG 6→8/16, BOX 3→9/16이었다. P1에서는 PEG 10→13/16, BOX 5→11/16이었다. 이 파일럿을 근거로 고해상도나 P1이 공간 Grounding을 악화시켰다고 해석할 수 없다. 형식 실패를 분리하면 개선 가능성이 보이지만, 512/P1조차 Peg/Box의 운영 Recall 90%에 미달하고 이 좌표는 Mask로 검증되지 않았다.

정식 Accepted와 진단 값을 섞으면 결론이 왜곡된다. 현재 출력 계약 준수는 큰 병목이다. 동일 값의 반복 키를 포함한 응답 수는 Native128/P0 2개, Native512/P0 9개, Native128/P1 15개, Native512/P1 12개였다. 한 응답에서 동일 반복과 충돌 반복이 함께 나타나면 issue count는 중복될 수 있다. P1/512에는 추가 키 응답도 3개 있었다. 원인 후보는 Prompt/schema/runtime의 상호작용이며, 이번 데이터만으로 학습된 패턴이나 특정 구현 원인을 확정하지 않는다.

운영 경로의 판단은 `VLM_GROUNDING_INSUFFICIENT`다. 공간 능력에는 개선 신호가 있어 Native 512를 배제할 근거가 되지 않는다. 후속 작업을 한다면 고해상도를 유지하면서 출력 계약을 사전에 고정해 검증하는 것이 우선이며, 이번 결과 뒤에 추가 추론·프롬프트 변경·SAM2 보완 실행은 하지 않았다.

## 10. 판단과 범위

이 파일럿으로 DINO 제거 여부를 확정하지 않는다. 이번 Native512 및 전체 paired matrix에 대응하는 충분한 보존 DINO baseline을 구성하지 않았고 새 DINO를 실행하지 않았으므로 `MATCHED_DINO_BASELINE_INSUFFICIENT`다. DINO 대비 우월성 주장은 보류한다. 4개 Parent·16개 Camera-Frame은 운영 기준의 충분한 근거가 아니며 `DINO_REPLACEMENT_CANDIDATE`를 확정할 수 없다. Native 512와 Prompt의 효과는 위 paired 결과로 판단한다.

나머지 20개 Parent는 이번 Pilot에서 실행하지 않았다. 후속 확인을 한다면 Prompt·해상도·선택 기준과 새 실행 예산을 먼저 고정해야 한다. 이번 결과를 보고 같은 파일럿에서 Prompt를 추가 수정하거나 Parent를 늘리지 않았다. 장기 SAM2 tracking, Graph 생성, Controller, closed-loop success, 외부 분포는 검증하지 않았다. Production과 Stage4A-2 `PAUSED_SCOPE_REVIEW` 및 Stop Stub은 유지했다.

## 11. 보존·검증과 산출물

공통 ExperimentLogger를 재사용하고 W&B disabled 로컬 기록만 남겼다. 보호 파일 1620개 SHA, 모델/Processor 15개 파일, Python package 목록이 변하지 않았다. 원시 응답→파서 재해석 및 이미지 SHA 재검증 통과; Box IoU 128개와 Mask IoU 128개 독립 재계산 통과. 추론 종료 후 GPU compute process는 없다. 공통 simulator/logger API 변경 없음. 로컬 공통 logger/artifact 계약 테스트 42개 통과. macOS renderer·CUDA physics는 이 작업에서 실행하지 않았다.

- [Protocol](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/protocol.json) · [Input manifest](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/input_manifest.json) · [Capture audit](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/capture_verification.json) · [Preflight](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/preflight_verification.json)
- [P0](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/prompt_P0.txt) · [P1](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/prompt_P1.txt) · [Prompt identity](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/grounding_prompt_identity.json) · [Model identity](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/model_identity.json)
- [Raw responses](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/full_image_grounding_raw_responses.json) · [Predictions CSV](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/full_image_grounding_predictions.csv) · [Grounding metrics](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/grounding_evaluation_metrics.json) · [Failure cases](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/grounding_failure_cases.csv)
- [SAM2 metrics](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/sam2_initialization_metrics.json) · [SAM2 cases](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/sam2_initialization_cases.csv) · [Latency/VRAM](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/grounding_latency_vram.json) · [Verification](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/verification.json)

![동일 Parent t0 Main/Wrist의 4조건 비교](../../outputs/task10i/stage4a2/qwen35_native_resolution_prompt_20261102_v1/full_image_grounding_comparison.png)

그림: 주황 Qwen PEG, 하늘색 Qwen BOX, 초록 GT PEG, 자홍 GT BOX. 반투명 영역은 실제 SAM2 초기 Mask이며 invalid/null에는 Mask가 없다. Qwen Box는 정식 계약을 통과한 경우에만 표시한다. 사후 진단 좌표는 위 별도 CSV에 보존한다. 두 해상도를 같은 크기로 표시한 것은 보고서 비교용이며, 512 입력은 시뮬레이터의 native RGB다.

## Appendix — Frozen P1

```text
Locate the PEG and BOX in this complete RGB image of a robotic peg-insertion task.

PEG is the separate workpiece that the robot grasps, moves, and inserts into an opening. It may be a block or bar; it need not be cylindrical. Identify the workpiece itself, including when held or partly inserted. The robot arm, gripper, and fingers are not the PEG.

BOX is the receiving fixture containing the matching opening. Locate the fixture body, not just its hole. It need not look like a cardboard box. The PEG and BOX are different physical objects; use their insertion roles and visible boundaries to distinguish them. Do not assume fixed colors, shapes, orientations, or image locations.

For each object, return a tight axis-aligned box enclosing all of its visible parts. Exclude robot parts and background as far as a rectangle permits. Do not estimate hidden parts. Return null if the object is not visible or cannot be distinguished and located reliably.

Coordinates are normalized to the COMPLETE image: left/top is 0, right/bottom is 1000. A box is [x1, y1, x2, y2], where 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000. These are normalized coordinates, not resized-image pixel coordinates.

Return only a JSON array of exactly two objects in this order: PEG, then BOX. Each object has exactly two keys: "label" and "bbox_2d". Use four numbers or null for "bbox_2d". Include each key only once. No Markdown or other text.
```

추가 관측: Native512/P1 사후 좌표 진단은 Main PEG 6/8·BOX 7/8, Wrist PEG 7/8·BOX 4/8이었다. Wrist BOX의 실패가 남아 있어 전체 평균만으로 Camera별 안정성을 주장하지 않는다.

로컬 수신 검증: artifact index의 184개 파일 SHA가 일치했고, 사후 좌표 진단 IoU 128개도 독립 재계산했다. 로컬 Source와 원격 실행 Source가 일치한다.
