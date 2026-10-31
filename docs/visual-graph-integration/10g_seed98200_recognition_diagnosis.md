# Task10-G seed98200: peg / hole·box 인식 진단

대상: Final / C / step011487 / seed98200. 분석일: 2026-10-31.

## 결론

**이 에피소드는 첫 프레임부터 base_camera가 실제 peg를 인식했다.** DINO 초기화 점수는 0.479687이고, raw peg mask의 GT IoU는 90.44%, precision은 99.64%이다. 저장된 초기 정책 그래프에도 peg 128노드, box 128노드가 있다. seed98202의 첫 peg 초기화 실패와는 다르다.

첫 프레임 base_camera의 구멍이 있는 **box도 대체로 올바르게 분할됐다**. DINO 점수 0.583972, GT IoU 93.76%이다. 단, 현재 visual/runtime에는 **hole 전용 검출 클래스나 마스크가 없다**. box 표면 분할의 정확성과 hole 중심/축/pose 인식의 정확성은 구분해야 한다. 이 rollout에서 후자는 독립 출력으로 검증할 수 없다.

반면 **hand_camera의 box 마스크는 그리퍼를 box로 잘못 잡았다**. 첫 프레임 box GT IoU는 0이며, 예측 픽셀의 98.48%가 로봇이다. 현재 C3 geometry는 wrist box를 항상 제외하므로 이 raw 오배정이 그대로 box 노드에 들어가지는 않는다.

## 첫 프레임의 raw mask와 정책 입력

[첫 프레임 RGB/peg/box 분리 그림](../../outputs/task10g/diagnostics/seed98200_recognition_v2/masks_t000.png). 원래 peg가 빨간 재질인 데 따른 혼동을 줄이기 위해 그림의 peg 마스크를 청록색으로 표시했다. 이는 검출 재실행이나 마스크 변경이 아니라 저장 bool mask를 그리는 색상만 바꾼 것이다.

| Camera | 대상 | 예측 / GT 픽셀 | IoU | Precision | 예측 중 robot 비율 |
| --- | --- | ---: | ---: | ---: | ---: |
| base_camera | peg | 845 / 928 | 90.44% | 99.64% | 0.00% |
| base_camera | box | 465 / 436 | 93.76% | 93.76% | 5.59% |
| hand_camera | peg | 0 / 29 | 0.00% | N/A | N/A |
| hand_camera | box | 263 / 438 | 0.00% | 0.00% | 98.48% |

Hand peg는 t=0에 0픽셀이고, 실제로 보이는 peg도 29픽셀뿐이다. t=15에 초기화되며, 그때 GT IoU는 88.70%, precision은 99.70%이다. Base peg가 처음부터 존재하므로 초기 wrist mask의 부재를 전체 peg 미검출로 해석하면 안 된다.

초기 최종 그래프의 source 측정 중 GT 물체 ID와 일치하는 비율은 peg 97.69%, box 96.88%이다. 따라서 올바른 물체를 주로 잡았지만 모든 경계/깊이 픽셀이 완벽한 것은 아니다. 이 비율은 측정 source별 통계이며 raw mask precision과 분모가 다르다.

## 전체 250스텝

| Camera | 대상 | GT가 보이는 프레임 | 예측 mask가 있는 프레임 | GT 가시 프레임 평균 IoU | 전체 예측 픽셀 가중 precision |
| --- | --- | ---: | ---: | ---: | ---: |
| base_camera | peg | 221 | 218 | 87.60% | 99.64% |
| base_camera | box | 203 | 205 | 92.88% | 94.45% |
| hand_camera | peg | 158 | 135 | 61.78% | 99.71% |
| hand_camera | box | 115 | 250 | 0.00% | 0.00% |

- 최종 graph peg 존재: **250/250**. 누락 구간: [].
- 최종 graph box 존재: **205/250**. 누락 구간(양끝 포함): **[[24, 44], [52, 75]]**.
- Box가 누락된 45스텝 중 base GT box가 0픽셀인 스텝: **45**. 소수라도 GT box가 보이는 누락 스텝: **[]**. 누락은 주로 로봇에 의한 base 시야 가림과 함께 발생한다.
- Wrist peg gate 통과 구간(양끝 포함): [[15, 146], [150, 151]]. 첫 프레임의 gate 거절은 wrist peg에 대한 기록이며 base peg 누락을 뜻하지 않는다.
- 전체 최종 graph source 측정의 GT ID 일치율: peg **98.33%**, box **97.11%**. Source 개수로 가중한 값이며 추정 hole pose의 정확도는 아니다.

[t=25 가림 예시](../../outputs/task10g/diagnostics/seed98200_recognition_v2/masks_t025.png) · [t=100](../../outputs/task10g/diagnostics/seed98200_recognition_v2/masks_t100.png) · [t=249](../../outputs/task10g/diagnostics/seed98200_recognition_v2/masks_t249.png)

## Hole에 대해 확인할 수 있는 범위

초기화 프롬프트는 peg=`red and white block`, box=`cardboard box` 두 개다. Hole용 별도 클래스, SAM2 ID, graph node type은 없다. 현재 graph slot은 TCP/Left/Right/Peg/Box다. 또한 C의 auxiliary state head는 이 rollout의 정책 추론에서 호출되지 않는다(`state_head_inference=false`).

Trace에 있는 `hole_pose`는 `evaluation_measure()`가 simulator GT에서 읽어 평가용으로 기록한 값이다. 이를 visual이 hole pose를 인식한 출력으로 해석하면 안 된다. **Box mask는 첫 프레임부터 대체로 정확했지만, 이 자료로 hole 자체의 pose 인식 성공을 주장할 수는 없다.**

## Hand camera가 첫 프레임에서 peg를 놓친 직접 원인

후속 질의에 따라 같은 저장 재현 자료를 세분 분석했다. 추가 inference나 threshold 변경은 하지 않았다. [4개 시점의 손목 영상/GT 위치/후보 박스](../../outputs/task10g/diagnostics/seed98200_recognition_v2/hand_initialization_timeline.png)와 [수치 원문](../../outputs/task10g/diagnostics/seed98200_recognition_v2/hand_initialization_diagnosis.json)을 함께 제공한다. 그림의 청록색은 **진단용 GT peg**이며 모델의 예측 마스크가 아니다. 노란 박스는 t=0/5에는 GT bbox와 가장 잘 맞는 후보, t=10/15에는 실제 선택 규칙이 사용하는 최고 점수의 유효 후보이다. GT는 후보 설명용 사후 대응에만 사용했다.

| 시점 | GT peg 가시 픽셀 | 설명한 peg 후보 점수 | SAM2 상태 |
| --- | ---: | ---: | --- |
| t=0 | 29 (영상의 0.177%) | 0.156561 | 기준 미달, 미초기화 |
| t=5 | 252 | 0.209258 | 기준 미달, 미초기화 |
| t=10 | 899 | 0.427347 | 첫 유효 검출, 다음 스캔 확인 대기 |
| t=15 | 1,139 | 0.382470 | 두 번째 일치 검출로 초기화, mask 1,016픽셀 |

첫 프레임의 GT peg bbox는 **x=[67,78), y=[123,128)**이다. 128×128 입력의 맨 아래 5개 행에 작은 일부만 걸쳐 보인다. Base camera처럼 peg의 넓은 면을 보는 상황이 아니다. 이 관측 차이는 낮은 검출 점수와 함께 확인되지만, 가시 크기·잘림·프롬프트 각각의 영향을 분리한 원인 실험은 아니다.

확정된 처리 경로는 다음과 같다. t=0의 화면 전체를 감싸는 score 0.381219 후보는 면적 100%로 25% 제한에 걸린다. 면적을 통과한 모든 후보도 최대 0.267961로 미검출 점수 0.3보다 낮다. 특히 실제 peg 조각과 대응하는 후보는 0.156561에 불과하다. 따라서 **DINO 후보 선택 단계에서 SAM2에 넘길 peg box가 결정되지 않았다**. Depth/self filter나 wrist gate가 이미 있던 peg 마스크를 지운 것이 아니다.

후속 스캔은 5스텝마다 실행되며 score≥0.3 후보가 두 연속 스캔에서 bbox IoU≥0.5를 만족해야 한다. t=10과 t=15의 선택 박스 IoU는 **0.510996**으로 조건을 통과한다. 두 박스는 이미 등록된 box 후보와의 IoU도 각각 0.101671, 0.145344로 배제 기준 0.3보다 낮아, 잘못된 wrist box 등록이 이 peg 초기화를 직접 막은 것은 아니다.

따라서 지연은 **t=0/5의 작은 부분 관측과 낮은 후보 점수**, 이어서 **t=10의 첫 유효 검출 이후 두 번째 확인을 기다리는 규칙**으로 설명된다. t=15의 SAM2 peg mask는 GT IoU 0.887040, precision 0.997047이고 wrist gate도 통과했다. 이때까지 model/threshold/prompt를 바꾼 것은 없다.

## 재현 및 독립 검증

원래 Linux 서버의 frozen source와 DINO/SAM2 checkpoint를 사용했다. Seed98200 reset 후 원본에 저장된 action을 그대로 재생했으며 정책 inference·학습을 실행하지 않았다. GT segmentation은 이미 촬영된 sensor texture에서 평가용으로만 읽었고, detector/tracker/geometry 입력에는 전달하지 않았다.

| 검증 | 결과 |
| --- | --- |
| 초기 simulator/controller/RNG snapshot SHA | 원본 exact 일치 |
| DINO 모델/전처리/토크나이저 파일 및 SAM2 checkpoint SHA | 원본 실행 identity와 일치 |
| 관련 frozen source SHA | 원본과 일치 |
| 초기 Graph21/Edge4/Robot76 모든 저장 배열 | 원본 exact 일치 |
| 250개 wrist gate 전체 필드 | 원본 exact 일치 |
| 250개 최종 graph 물체 존재 여부 | 원본 exact 일치 |
| 250개 wrist peg raw 픽셀 수 | 원본 exact 일치 |
| 249개 post-action peg/TCP pose 최대 차이 | 모두 0 |
| 전체 1,000개 camera/object/frame mask와 GT metric 독립 재계산 | 저장 값 exact 일치 |

첫 시도(v1)는 공유 GPU의 가용 메모리가 부족해 t=90 처리 중 CUDA OOM으로 중단됐고, 완료된 90스텝 기록과 로그를 보존했다. V2는 설치된 SAM2의 `offload_state_to_cpu=True`와 이미 CUDA에서 전처리된 과거 image tensor의 CPU 보관만 적용했다. 계산 전에 원래 CUDA device로 다시 이동하며 모델 가중치, 전처리 계산, threshold, prompt, sampling, action은 바꾸지 않았다. V2 전체 250스텝 결과는 위 원본 비교를 통과했고, V1과 겹치는 90스텝의 mask GT metric도 모두 exact 일치했다.

실행 backend: **BackendInfo(device=device(type='cpu'), sim_device=Device("cpu"), sim_backend='physx_cpu', render_device=Device("cuda:0"), render_backend='sapien_cuda')**. Vision은 **NVIDIA GeForce RTX 3090 / 2.14.1+cu130**에서 실행했다. macOS CPU/MPS의 simulator 또는 vision은 실행하지 않았다. 공통 `ExperimentLogger`를 W&B disabled로 재사용했다. 새 파일은 task 진단 스크립트, 출력, 보고서이며 공유/정책/vision 소스는 변경하지 않았다. 외부 artifact 게시나 다른 GPU 작업의 중단은 하지 않았다.

## 근거 파일

- [재현 요약](../../outputs/task10g/diagnostics/seed98200_recognition_v2/summary.json), [전체 재현 기록](../../outputs/task10g/diagnostics/seed98200_recognition_v2/replay.json), [검출 후보/초기화 점수](../../outputs/task10g/diagnostics/seed98200_recognition_v2/detector_calls.json)
- [전체 raw mask와 GT segmentation](../../outputs/task10g/diagnostics/seed98200_recognition_v2/all_masks_segmentation.npz), [독립 mask metric CSV](../../outputs/task10g/diagnostics/seed98200_recognition_v2/mask_gt_metrics.csv), [검증/집계 JSON](../../outputs/task10g/diagnostics/seed98200_recognition_v2/analysis_verification.json)
- [V2 재현 코드](../../outputs/task10g/diagnostics/seed98200_recognition/reproduce_recognition_v2.py), [로컬 검증 코드](../../outputs/task10g/diagnostics/seed98200_recognition/analyze.py)
- [원본 영상](../../outputs/task10g/final_results/rollouts/final/C/step011487/seed98200/rollout.mp4), [원본 trace](../../outputs/task10g/final_results/rollouts/final/C/step011487/seed98200/trace.json)
- 코드: [초기화](../../visual/runtime/initialization.py), [C3 wrist box 제외](../../graph/plan_a_geometry/core.py), [GT 평가용 hole_pose 기록](../../graph/plan_a_bc/rollout.py), [rollout inference](../../graph/graph_recurrent10g/eval_runner.py).
