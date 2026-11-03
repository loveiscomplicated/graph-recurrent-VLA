# Task 10-I Stage4A-2 — Grounding DINO Native Resolution 128/512

**판정: `NATIVE_RESOLUTION_BENEFIT_OBSERVED`.** 동일 상태의 시뮬레이터 원본 128→512에서 기존 DINO 선택 경로의 PEG Recall은 6/16 (37.5%) → 8/16 (50.0%), BOX Recall은 6/16 (37.5%) → 8/16 (50.0%)였다. 기준은 visible pixel≥16, Box IoU≥0.5다. 이 결과는 기존 Train Parent 4개의 대응 비교 파일럿이며 일반화·희귀 오류 보증이 아니다.

## 1. 질문과 통제 조건

Qwen 실험에서 확보한 동일 상태의 원본 128/512 RGB·EXACT actor segmentation을 그대로 사용했다. 새 이미지 확대본을 고해상도 입력으로 대체하거나 새 장면을 선택하지 않았다. 각 해상도 16개 Camera-Frame, 총 32개 이미지다. Parent는 `0531`, `0275`, `0197`, `0305`; 각 t0/t80, Main `base_camera` / Wrist `hand_camera`다. 전체 32개를 모두 사용했다.

Vast.ai **58576054 / RTX 3090 24GB**, 기존 `IDEA-Research/grounding-dino-tiny` snapshot revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. 기존 `GDino`와 `initialization.candidates/assign`을 직접 재사용했다. FP32, `cuda:0`, quantization/offload 없음. 모델 로드 1회, 64 forwards(32장 × 2프롬프트), 별도 warmup/retry 0회.

- PEG prompt: `red and white block`
- BOX prompt: `cardboard box`
- Box/Text threshold 각각 0.1. 면적 < 이미지의 25%, top-5, Null score 0.3, 두 선택 Box exclusion IoU 0.3을 유지했다.
- Task10-J의 별도 최적화 프롬프트로 바꾸거나 Prompt/threshold를 탐색하지 않았다. 이 결과는 현재 운영 프롬프트 고정 조건이며 DINO가 낼 수 있는 최고 성능이라는 뜻이 아니다.
- Processor default: shortest edge800 / longest edge1333. 실제 두 해상도 모두 `[1,3,800,800]`을 확인했다. 내부 처리 크기를 늘린 실험이 아니므로 native 세부 정보 효과를 비교한다.
- Postprocessing target_sizes는 원본 H/W다. Box를 Processor의 800 좌표로 GT와 비교하지 않는다. 모델 원시 Box를 clipping/GT 보정하지 않았다. 이미지 밖으로 일부 나가는 Box도 기존 DINO consumer 계약대로 유지하고 별도 기록했다.

## 2. 입력 경로와 동결

DINO inference용 Manifest에는 RGB 경로·SHA·카메라/Frame ID·원본 크기만 두고 GT path/Actor ID/Privileged state/Qwen 결과를 제거했다. RGB와 고정 문자열만 Detector에 제공했다. 원시 후보·Score·기존 규칙의 선택 Box를 모두 저장하고 SHA 동결한 뒤 별도 평가 프로세스에서 GT를 열었다. 과거 GT/성적은 이미 알려져 있으므로 연구자 맹검이라고 표현하지 않는다.

이번에는 Simulator 생성/Reset/Render/Step 0, Qwen load/forward 0, SAM2 load/forward 0, training 0이다. 기존 캡처의 같은 상태·카메라 위치/FOV/마운트·Native GT 검증을 재사용하며 이번 작업에서 그 검증을 새 동적 replay로 주장하지 않는다.

## 3. Detector 후보와 실제 선택 정확도

| 원본 | 물체 | 기존 Joint 선택 Recall | Score Top-1 Recall | 후보 중 정답 존재(Oracle) | 선택 평균 IoU(Null 포함) |
|---|---|---:|---:|---:|---:|
| 128 | peg | 6/16 (37.5%) | 6/16 (37.5%) | 16/16 (100.0%) | 0.393 |
| 128 | box | 6/16 (37.5%) | 6/16 (37.5%) | 14/16 (87.5%) | 0.372 |
| 512 | peg | 8/16 (50.0%) | 10/16 (62.5%) | 16/16 (100.0%) | 0.571 |
| 512 | box | 8/16 (50.0%) | 8/16 (50.0%) | 16/16 (100.0%) | 0.506 |

Oracle 열은 GT로 후보 중 가장 좋은 Box를 찾아 진단한 coverage다. 실제 Consumer는 GT를 보지 않았고 이 Oracle Box를 출력하거나 SAM2에 제공하지 않았다. Detector가 후보를 발견하는 능력과 기존 선택 규칙의 동작을 구분해야 한다. 이 수치는 COCO AP/mAP가 아니다.

| 원본 | 물체 | IoU≥0.3 | IoU≥0.7 | Null | 잘못된 선택 / 출력 | Hallucination / ABSENT |
|---|---|---:|---:|---:|---:|---:|
| 128 | peg | 8/16 (50.0%) | 6/16 (37.5%) | 7/16 (43.8%) | 3/9 (33.3%) | 0/0 (N/A) |
| 128 | box | 6/16 (37.5%) | 6/16 (37.5%) | 2/16 (12.5%) | 8/14 (57.1%) | 0/0 (N/A) |
| 512 | peg | 11/16 (68.8%) | 8/16 (50.0%) | 3/16 (18.8%) | 5/13 (38.5%) | 0/0 (N/A) |
| 512 | box | 9/16 (56.2%) | 8/16 (50.0%) | 0/16 (0.0%) | 8/16 (50.0%) | 0/0 (N/A) |

Null은 가시 목표에 대한 Miss로 계산한다. ABSENT 분모가 0이면 해당 False Positive는 N/A이며 0%라고 해석하지 않는다. False discovery는 선택 Box 중 IoU<0.5의 비율이다. 원시 후보·선택 Box의 누락을 분리해서 보존했다.

## 4. Parent / Camera / 시점

| 원본 | PEG Parent-macro Recall | BOX Parent-macro Recall | 두 물체 동시 선택 성공 | 하나 이상 Wrong-object |
|---|---:|---:|---:|---:|
| 128 | 0.375 | 0.375 | 4/16 (25.0%) | 6/16 (37.5%) |
| 512 | 0.500 | 0.500 | 5/16 (31.2%) | 9/16 (56.2%) |

| 원본 | 구분 | PEG Recall | BOX Recall |
|---|---|---:|---:|
| 128 | base_camera | 4/8 (50.0%) | 6/8 (75.0%) |
| 128 | hand_camera | 2/8 (25.0%) | 0/8 (0.0%) |
| 128 | F0 | 6/8 (75.0%) | 4/8 (50.0%) |
| 128 | F1 | 0/8 (0.0%) | 2/8 (25.0%) |
| 512 | base_camera | 4/8 (50.0%) | 7/8 (87.5%) |
| 512 | hand_camera | 4/8 (50.0%) | 1/8 (12.5%) |
| 512 | F0 | 8/8 (100.0%) | 5/8 (62.5%) |
| 512 | F1 | 0/8 (0.0%) | 3/8 (37.5%) |

t80에도 같은 per-frame candidates/assign을 독립 적용한 결과다. Production이 실제 t80마다 다시 초기화한다거나 temporal scan을 replay했다는 뜻은 아니다. Saved t0/t80을 t5 단위 연속 시퀀스로 취급하지 않았다.

## 5. 대응 변화와 통계

같은 목표가 양쪽 해상도에서 visible≥16인 경우만 대응시켰다. 독립 집계 단위는 물리 Parent 4개이며 각 Parent 안의 Camera/Frame을 별개 Parent로 세지 않았다. Parent별 변화 평균, 10,000회 Parent-cluster Bootstrap(seed42)을 보고한다. 작은 파일럿의 탐색적 CI이며 다중 비교 보정은 없다. 퇴화 CI는 희귀 오류 안전성을 뜻하지 않는다.

| 물체 | 지표 (512−128) | Parent-macro 차이 | 탐색적 95% CI |
|---|---|---:|---|
| peg | selected_recall_delta | 0.125 | [0.0, 0.25] |
| peg | raw_top1_recall_delta | 0.250 | [0.0625, 0.4375] |
| peg | proposal_coverage_delta | 0.000 | [0.0, 0.0] |
| peg | selected_mean_IoU_delta | 0.177 | [0.042749168096206595, 0.3116959358404485] |
| box | selected_recall_delta | 0.125 | [0.0, 0.25] |
| box | raw_top1_recall_delta | 0.125 | [0.0, 0.25] |
| box | proposal_coverage_delta | 0.125 | [0.0, 0.25] |
| box | selected_mean_IoU_delta | 0.134 | [-0.00019190233758158582, 0.2676600947074386] |

선택 Recall의 복구/회귀 수: peg 복구 2, 회귀 0, box 복구 2, 회귀 0. [전체 대응 사례](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/dino_resolution_transitions.csv)에 실패 원인 변화를 보존했다.

## 6. 실패 원인과 공간 오차

| 원본 | 물체 | Drop reason | Semantic confusion | 평균 Center error / native px | 128px 환산 |
|---|---|---|---|---:|---:|
| 128 | peg | `{"NULL_SCORE": 5, "AREA": 3, "TOP5": 2, "SUCCESS": 6}` | `{}` | 14.159 | 14.159 |
| 128 | box | `{"SUCCESS": 6, "NO_GOOD_PROPOSAL_AT_THRESHOLD": 2, "NULL_SCORE": 7, "TOP5": 1}` | `{"box->robot": 6}` | 43.348 | 43.348 |
| 512 | peg | `{"NULL_SCORE": 5, "AREA": 3, "SUCCESS": 8}` | `{"peg->robot": 1}` | 55.366 | 13.841 |
| 512 | box | `{"SUCCESS": 8, "AREA": 1, "NULL_SCORE": 2, "JOINT_OR_RANKING": 4, "TOP5": 1}` | `{"box->robot": 8}` | 187.582 | 46.895 |

NO_GOOD_PROPOSAL은 0.1 threshold를 넘은 후보에 IoU≥0.5 정답이 없는 경우다. AREA/TOP5/NULL_SCORE/JOINT_OR_RANKING은 기존 후보 선택 경로에서의 손실을 나타낸다. Threshold 미만의 잠재 Query를 GT로 되살리지 않았다. NULL_SCORE는 정답과 일치하는 후보가 Null score 이하였다는 사후 손실 원인이다. 최종 출력 자체가 Null이라는 뜻은 아니며, 그 경우 다른 잘못된 Box가 선택될 수도 있다. Wrong-object는 다른 Actor가 선택 Box foreground의 ≥70%, pixel≥16이며 목표 IoU<0.5인 경우다. 오검출과 위치 오차를 동일시하지 않는다. Center error는 출력이 있는 경우만의 조건부 지표이며 128/512 원본 픽셀 척도를 직접 비교하지 않고 환산값을 병기했다.

고해상도가 경계/텍스처 정보를 바꾸어 score·rank·IoU·null 선택을 바꿀 수 있다는 것은 원인 가설이다. 이번 실험으로 어느 texture나 단일 feature가 원인이었는지 확정하지 않는다. 관측된 변화는 위 Frame별 원시 score·Box·GT와 consumer replay로 확인한 범위다.

## 7. Latency / GPU

기존 DINO 모델 로드 4.283s, 로드 후 allocated 0.645 GiB. 파일 SHA 사전 검사로 OS file cache는 warm일 수 있다. CPU process sampled peak 1.738 GiB(50ms RSS sampling).

| 원본 | Warm call n | 1프롬프트 평균 / median / p95 ms | 두 프롬프트 이미지 평균 / median / p95 ms | Peak allocated GiB |
|---|---:|---|---|---:|
| 128 | 31 | 104.735 / 105.829 / 106.943 | 217.457 / 217.101 / 224.784 (n=15) | 1.447 |
| 512 | 32 | 106.574 / 106.731 / 111.180 | 225.569 / 225.789 / 234.595 (n=16) | 1.447 |

Cold 첫 call 128/peg 742.50ms는 Warm call에서 제외했다. 첫 이미지도 Warm 이미지 집계에서 제외했다. 16개 대응 pair 안에서 128→512와 512→128 순서를 8개씩 교차했다. 입력 해상도별 내부 Tensor 크기는 동일했다. 두 프롬프트 이미지 E2E는 RGB PNG read+2회 Detector+실제 선택+호출별 측정/원시 결과 저장 overhead를 포함한다. 1프롬프트 E2E는 GDino 전체 호출이며 pre(H2D 포함)/forward/post 세부 원자료를 제공한다. 전처리·후처리·forward 시간은 [Latency JSON](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/dino_latency_vram.json)에 있다. 모델 외 다른 GPU resident 모델은 없었다.

## 8. 동일 이미지 Qwen과의 제한된 비교

이제 동일 Native RGB에 대응하는 DINO 결과가 생겼다. 기존 Qwen을 재실행하지 않고 동결 결과만 join했다. GT/카메라/시점/IoU 기준은 동일하지만 DINO와 Qwen의 Prompt·선택 경로·정밀도는 다르므로 파이프라인 조건의 제한된 비교다. DINO 1이미지=2회 text-conditioned forward, Qwen 1이미지=두 물체를 한 번에 생성한다. 모델의 본질적 우열·외부 분포 성능을 주장하지 않는다.

| 원본 | 물체 | DINO 실제 선택 | Qwen P0 Accepted | Qwen P1 Accepted | Qwen P1 사후 좌표 진단(미승인) |
|---|---|---:|---:|---:|---:|
| 128 | peg | 6/16 (37.5%) | 6/16 (37.5%) | 0/16 (0.0%) | 10/16 |
| 128 | box | 6/16 (37.5%) | 2/16 (12.5%) | 0/16 (0.0%) | 5/16 |
| 512 | peg | 8/16 (50.0%) | 2/16 (12.5%) | 0/16 (0.0%) | 13/16 |
| 512 | box | 8/16 (50.0%) | 2/16 (12.5%) | 0/16 (0.0%) | 11/16 |

Qwen 진단 값은 중복 키가 동일하거나 추가 필드가 있는 완전한 JSON에서 읽은 좌표이며 운영 Accepted/SAM2 성공으로 계산하지 않았다. DINO Oracle 후보 coverage와도 혼동하지 않는다. [동일 이미지 비교 JSON](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/matched_qwen_comparison.json)에 각각 분리했다.

## 9. 결론과 제한

관측상 Native512에서 PEG/BOX 선택 Recall이 각각 12.5 percentage points 증가했다. PEG 평균 선택 IoU는 0.393→0.571, BOX는 0.372→0.506이었다. 다만 Native512의 Main BOX는 7/8, Wrist BOX는 1/8로 집중 실패가 남았다. BOX→Robot 오인식은 128의 6건에서 512의 8건으로 증가했다. 높은 해상도가 모든 오류 유형을 줄였다는 주장은 성립하지 않는다. 탐색적 Recall 변화 CI는 두 물체 모두 [0,0.25]여서 이 표본으로 통계적 우월성을 단정하지 않는다.

이번 결과는 현행 DINO와 선택 규칙을 고정했을 때 native capture resolution 변화가 만든 성능 차이다. 원본 고해상도 효과를 Detector 후보 단계와 실제 선택 단계로 구분해 해석한다. 4개 Parent 표본만으로 현장 안정성이나 높은 confidence를 의미 정확성으로 보증하지 않는다. GT가 없는 경우/미가시 객체/아주 작은 객체의 충분한 검증은 별도다. 추가 Prompt나 Parent를 결과 뒤에 늘리지 않았다.

SAM2 Mask, Object Graph, 장기 Tracking, 정책 Closed-loop success는 이번에 평가하지 않았다. DINO Box가 좋아졌다고 Visual-to-Graph 전체 성능 향상을 확정할 수 없다. Production source·threshold와 기존 Stage4A-2 `PAUSED_SCOPE_REVIEW` 및 Stop Stub은 그대로다.

## 10. 검증·보존·산출물

공통 ExperimentLogger를 W&B disabled 로컬 기록으로 재사용했다. 보호 파일 1804개 SHA 및 Python package 목록을 전후 대조해 보존을 확인했다. 기존 Qwen 실험 전체 artifact와 DINO snapshot도 포함했다. 저장된 원시 후보만으로 실제 consumer 32개 이미지의 선택을 재생했고 추가 forward는 0회다. Native GT에서 Bounding Box를 별도 재구성하고 선택/전체 후보 IoU 658개를 독립 재계산했다. Raw/선택 CSV SHA는 GT 평가 전후 불변이다. 기존 DINO 평가 테스트 5개 및 새 Source compile 통과. 실제 GPU 실행은 RTX3090에서 했고 로컬 macOS GPU·Simulator·renderer는 실행하지 않았다.

- [Protocol / 모델·Source SHA](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/protocol.json) · [RGB-only input manifest](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/input_manifest.json) · [실제 모델 Identity](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/model_identity.json)
- [원시 DINO 응답](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/dino_raw_responses.json) · [선택 Box](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/dino_predictions.csv) · [객체별 전체 평가](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/dino_object_evaluation.csv) · [후보별 전체 평가](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/dino_proposal_evaluation.csv)
- [전체 지표](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/dino_evaluation_metrics.json) · [실패 사례](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/dino_failure_cases.csv) · [대응 변화](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/dino_resolution_transitions.csv) · [Latency / VRAM](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/dino_latency_vram.json) · [보존 검증](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/verification.json)

![동일 Parent의 t0/t80 Main/Wrist — Native128/512](../../outputs/task10i/stage4a2/dino_native_resolution_20261102_v1/dino_native_resolution_comparison.png)

주황=DINO 선택 PEG, 하늘색=DINO 선택 BOX, 초록=GT PEG, 자홍=GT BOX. 원본 해상도와 표시 크기는 별개이며, 표시용 Resize가 실험 입력이 아니다. SAM2 Mask는 그리지 않았다.

로컬 수신 후 검증: artifact 36개 SHA, Native RGB 32개 파일·배열 SHA와 GT sidecar SHA가 일치했다. GT에서 직접 복원한 Bounding Box와 IoU 658개가 일치했고 로컬 Source가 원격 실행 Source와 일치한다.
