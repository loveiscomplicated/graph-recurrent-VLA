# Task 10-I — SAM2 입력 원본 해상도 128/512/768 소요 시간 비교

2026-11-03 KST. **동일 조건의 native 해상도 비교 완료.**
128→768에서 정적 분할 평균은 **1.168배 (+6.282ms)**,
연속 추적 평균은 **1.001배 (+0.066ms)**였다.
세 조건 모두 SAM2 내부 크기는1024²로 같았다. Native 픽셀 수 증가와 추론 시간 증가를 비례 관계로 해석하지 않는다.

## 실행 조건과 표본

- 기존 Vast.ai58576054, RTX3090 24GB. 실험 시작 시 다른 CUDA Compute Process0개.
- SAM2.1 Hiera Small, 동일 checkpoint SHA `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`.
- FP32 Weight, CUDA BF16 autocast, TF32 허용, Batch1, CPU4 thread, compile 없음.
- Torch `2.14.1+cu130`, torchvision `0.29.1+cu130`, SAM-2 `1.0`. 전체 Package/설치 SAM2 Source Hash는 Completion에 있다.
- 기존 네 Train Parent `10i_s1_0197, 10i_s1_0275, 10i_s1_0305, 10i_s1_0531`의 Main/Wrist를 모두 사용했다. 성적·가시성으로 대상을 거르지 않았다.
- 같은 seed의 geometry를 만들고 저장 qpos/qvel·Peg/Box pose를 복원했다. 세 해상도를 같은 scene/상태에서 직접 렌더링했다. 단순 이미지 확대가 아니다.
- 4 Parent×2 Camera×22시점(t0..20,t80)×3해상도=528 PNG. 카메라 pose/FOV/mount 동일, intrinsic 배율과 extrinsic 일치, capture 전후 물리 상태·TCP·geometry를 검사했다. Physics Step0, 새 Action/Policy Rollout0.
- Renderer/Simulator는 원격 CPU backend(`physx_cpu/sapien_cpu`)에서 실행했다. CUDA는 SAM2 추론에 사용했다. macOS에서 Simulator/MPS/CUDA는 실행하지 않았다.

## 프롬프트와 비교 통제

Detector/API 변동을 제거하기 위해 native768 Actor Segmentation의 bbox를 한 번 정규화하고 세 해상도에 같은 비율로 적용했다.
정적 분할은 각 프레임, 추적은 t0에만 두 Box를 사용했다. 미가시 Actor에는 full-image Box를 주고 **항상 두 객체를 실제 실행**했다.
정적16개 Camera-Frame 중 미가시 Actor 포함 장면은 0개다. 이를 제외하거나 추론을 건너뛰지 않았다.
이는 **Oracle Box를 쓰는 시간 측정**이며 Detection/Mask 품질·일반화 평가가 아니다. GT bbox 생성 비용은 SAM2 시간에 포함하지 않았다.

정적 분할은16장×6회×3해상도=288회다. 각 장면에서 여섯 순열을 모두 실행해 순서 위치를 균형화했다.
해상도마다3회 warmup을 제외했다. 추적은8개 독립 Camera Sequence×3회×3해상도=72 Sequence,
1512 Frame이다. 세 순환 순서로 위치를 균형화했고 해상도마다21-frame warmup Sequence1개를 제외했다.
Steady 통계는 각 Sequence의 t1..20만 포함해 **해상도당480 Frame**이다. t0와 JPEG 전체 로딩·초기화는 별도다.
두 모드 모두 반복 간 같은 입력·같은 해상도의 최종 Mask Hash가 정확히 일치했다.

## 정적 분할: 한 이미지 인코딩 + 두 Box

단위 ms. RGB는 미리 메모리에 올렸다. Encode는 `set_image`의 전처리·H2D·인코딩,
Prompt/Output은 두 번의 순차 `predict`, 원본 크기 Mask 복원 및 NumPy 전송을 포함한다.
각 단계 전후 CUDA 동기화로 wall time을 측정했다. 모델 Load·PNG 읽기·Hash·저장은 제외했다.

| Native | n | Encode 평균 | 두 Prompt/Output 평균 | 전체 평균 | 전체 중앙값 | 전체 p95 |
|---|---:|---:|---:|---:|---:|---:|
| 128² | 96 | 26.124 | 11.208 | 37.332 | 36.859 | 40.103 |
| 512² | 96 | 28.904 | 11.384 | 40.288 | 39.396 | 45.112 |
| 768² | 96 | 31.447 | 12.166 | 43.614 | 42.968 | 47.772 |

## 연속 추적: 프레임당 두 물체

단위 ms, VRAM은GiB. 각 조건에서 JPEG quality100/subsampling0을 사용했다. 원본 해상도에 따른 JPEG Decode 차이는 초기화에 포함한다.
`init_state(offload_video_to_cpu=True, offload_state_to_cpu=False, async_loading_frames=False)`로21개 Frame을 먼저 로드한다.
따라서 여기서 초기화는 **21-frame 파일 로딩+첫 인코딩+두 Box 입력**이며 온라인 최초1Frame latency가 아니다.
Steady는 `next(propagate_in_video)`부터 두 원본 크기 Bool Mask의 CPU 전송까지 포함한다. I/O·Hash·저장·Renderer는 제외했다.

| Native | Steady n | 평균 | 중앙값 | p95 | 1000/평균 FPS | 초기화 평균 | Peak allocated VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128² | 480 | 46.790 | 48.895 | 52.321 | 21.37 | 513.977 | 0.611 |
| 512² | 480 | 46.778 | 48.959 | 52.478 | 21.38 | 777.623 | 0.612 |
| 768² | 480 | 46.857 | 48.994 | 52.523 | 21.34 | 780.316 | 0.615 |

GPU 입력 Tensor는 정적 `[1,3,1024,1024]`, Video CPU Frame Buffer는 `[21,3,1024,1024]`로 모두 같다.
원본 크기 출력 복원·Bool 변환·CPU 전송과 입력 전처리는 해상도에 따라 달라질 수 있다.
추적은 동기화 wall time이며 GPU kernel만의 시간이나 Simulator/Detector/Graph/Controller 포함20Hz 검증이 아니다.
JPEG 경로와 무압축 RGB 정적 경로는 서로 다른 측정이므로 두 모드 사이의 시간 차이를 해상도 효과로 해석하지 않는다.

## 대응 차이와 한계

Parent별 평균의 대응 차이를 계산하고 Parent4개 단위 bootstrap10,000회(seed42)를 보조 기록했다.
반복·프레임을 독립 장면으로 세지 않았다. 작은4 Parent에서 나온 탐색적 구간이며 다른GPU/모델/긴Sequence 성능 보장은 아니다.

- 정적 128→512: 평균 차이 +2.956ms, 비율 1.079배, 탐색적95% 구간 [+2.684, +3.228]ms.
- 정적 512→768: 평균 차이 +3.325ms, 비율 1.083배, 탐색적95% 구간 [+3.110, +3.477]ms.
- 정적 128→768: 평균 차이 +6.282ms, 비율 1.168배, 탐색적95% 구간 [+6.048, +6.521]ms.
- 추적 128→512: 평균 차이 -0.012ms, 비율 1.000배, 탐색적95% 구간 [-0.164, +0.108]ms.
- 추적 512→768: 평균 차이 +0.078ms, 비율 1.002배, 탐색적95% 구간 [+0.027, +0.148]ms.
- 추적 128→768: 평균 차이 +0.066ms, 비율 1.001배, 탐색적95% 구간 [-0.080, +0.213]ms.

모든 결과는 한 서버의 순차 실행 조건이다. 균형화로 순서 영향을 줄였지만 GPU clock/온도 변동을 제거했다고 주장하지 않는다.
Mask 정확도·3D 역투영 정밀도·삽입 성공률은 이번에 측정하지 않았다. 해상도 변경이 이 지표들을 개선한다는 결론은 별도 평가가 필요하다.

처음 정적 측정은 Package metadata에서 distribution 이름을 `sam2`로 조회해 모델 Load 전 실패했다.
실제 이름 `SAM-2`로 수정했고 원래 Source/Protocol·실패 Log를 보존했다. 시간 측정·샘플·모델 설정은 바꾸지 않았다.
중단된 Logger 상태를 실제 종료에 맞게 failed로 정리했고, Capture/정적/추적의 정상 Run3개는 모두 finished다.

## 검증과 산출물

공통 `utils.experiment_logger.ExperimentLogger`를 재사용했고 W&B는 disabled다.
정적 Run `b911682cc6474847adbe3a54be635bf5`, 추적 Run `9d1b59b190054421b616bce5493d82d1`.
Production/공통 Simulator API·Model Config·Checkpoint·기존 Stage4A-2 Runner/Stop Stub은 변경하지 않았다.
새 코드는 독립 `visual/experiments/sam2_resolution_latency_20261103_v1/`에만 추가했다. 외부 게시·학습·DINO/Qwen은 실행하지 않았다.

독립 CPU readback은 JSON/CSV 모든 값 일치, 표본 수, 균형화, 동일 정규화 Box, 반복 Mask Hash,
모든 요약 통계 재계산의 정확 일치, capture/calibration 검증을 확인했다. Python 구문 검사도 통과했다.
측정 종료 후 PNG/JPEG 전체1,032개·기존 Raw/GT·Checkpoint Hash를 재검사했고 CUDA Compute Process0개를 확인했다.
실행 원자료/환경/프로토콜46개 파일은 SHA256 일치를 확인해 로컬 Output에 복사했다.
PNG/JPEG1,032개는 원격 `/workspace/task10i_sam2_resolution_latency_20261103_v1/run/inputs/`에 보존돼 있다.
로컬에는 파일별 Hash와 전체 입력 Manifest가 있으며, 원본 이미지의 로컬 복사를 완료했다고 주장하지 않는다.

- [실행 프로토콜](../../outputs/task10i/stage4a2/sam2_resolution_latency_20261103_v1/run/protocol.json) · [입력 manifest](../../outputs/task10i/stage4a2/sam2_resolution_latency_20261103_v1/run/input_manifest.json) · [Capture 검증](../../outputs/task10i/stage4a2/sam2_resolution_latency_20261103_v1/run/capture_verification.json)
- [정적 원자료 CSV](../../outputs/task10i/stage4a2/sam2_resolution_latency_20261103_v1/run/static_raw.csv) · [추적 Frame CSV](../../outputs/task10i/stage4a2/sam2_resolution_latency_20261103_v1/run/tracking_frames.csv) · [추적 Sequence CSV](../../outputs/task10i/stage4a2/sam2_resolution_latency_20261103_v1/run/tracking_sequences.csv)
- [정적 요약](../../outputs/task10i/stage4a2/sam2_resolution_latency_20261103_v1/run/static_summary.json) · [추적 요약](../../outputs/task10i/stage4a2/sam2_resolution_latency_20261103_v1/run/tracking_summary.json) · [대응 차이](../../outputs/task10i/stage4a2/sam2_resolution_latency_20261103_v1/paired_differences.json)
- [독립 readback 검증](../../outputs/task10i/stage4a2/sam2_resolution_latency_20261103_v1/verification.json) · [실행 코드](../../visual/experiments/sam2_resolution_latency_20261103_v1/benchmark.py)
- [실행 완료·입력 보존·환경](../../outputs/task10i/stage4a2/sam2_resolution_latency_20261103_v1/completion.json) · [아티팩트 SHA256](../../outputs/task10i/stage4a2/sam2_resolution_latency_20261103_v1/artifact_sha256.json)

![Native 해상도별 Parent 평균 소요 시간](../../outputs/task10i/stage4a2/sam2_resolution_latency_20261103_v1/resolution_latency.png)
