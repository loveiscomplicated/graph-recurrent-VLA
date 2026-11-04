# Task10-I: 로봇 필터 live 연결 누락 반영

2026-11-04 KST. 최종 CUDA 실행 ID: `51dbf8b49da54fe2953c3c06577689d4`.

요청한 네 항목을 현재 코드에 반영했다. **원격 RTX3090에서 관련 테스트 79개가
전부 통과했고, 실제 저장 입력 512·두 카메라 2사례의 일반/보정 경로를 검증했다.**
손목 gate의 규칙이나 구현은 이번 변경에서 수정하지 않았다.

## 반영 내용

| 누락 | 변경 후 |
| --- | --- |
| 기본 live가 공용 Torch depth lookup 사용 | `TensorRobotPreparation`이 기존 필터의 fused Triton `depth_tensor`를 호출하고, 결과 float64 depth를 필터·후속 단계와 공유 |
| Frozen inference의 CPU mask 출력·직렬 준비 | 실제 SAM2 adapter에 CUDA mask 출력을 요청하고 producer 함수 자체를 `CurrentGeometry.build`에 전달해 CPU 준비를 먼저 제출 |
| 보정 중 CPU 로봇 분류와 최종 GPU 필터 중복 | 현재 프레임 전체 화면을 GPU에서 한 번 분류하고 `RobotFilterFrame`으로 보정 후보와 최종 clean mask가 공유 |
| 과거 live caller의 CUDA RGB/CPU adapter 불일치 | 기존 tensor vision adapter와 지연 CUDA-mask producer로 연결하고, 영상·NPZ export만 명시적으로 CPU 변환 |

### Fused depth와 입력 경계

[tensor_ingress.py](../../graph/plan_a_bc/tensor_ingress.py)는 signed-int16 depth를
`CudaRobotFilter.depth_tensor`에 전달한다. 원래 256KiB 상수 lookup과 Triton 변환
kernel을 그대로 사용하며, invalid/overflow depth의 NaN 의미와 float64 단위를
유지한다. 변환 중 예외가 나면 이미 제출된 worker job을 discard한 뒤 원래 예외를
전파한다. 두 카메라 각각 한 번 변환한 depth는 기존 후속 GPU 경로에 재사용된다.

[rollout.sensors](../../graph/plan_a_bc/rollout.py)는 native CUDA RGB-D는 다운로드
없이 snapshot으로 유지한다. 과거 renderer가 CPU 이미지를 반환하면 같은 원본
uint8 RGB/signed-int16 depth를 관측 경계에서 올린다. CUDA가 없는 명시적 CPU
진단은 NumPy 관측을 유지한다. Renderer, Physics, 해상도 및 controller 설정은
이번에 변경하지 않았다.

### Frozen live와 보정 결과 공유

[frozen_inference.infer_frame](../../graph/task10i_stage4a2/frozen_inference.py)은
기존 JPEG/PNG 전처리·모델·prompt를 유지하면서 `mask_device='cuda'`를 요청한다.
현재 mask producer가 build 안에서 실행되므로 CPU world-shape/Qhull/bounds 준비가
vision보다 먼저 제출된다. Lossless adapter와 기존 trace wrapper도 keyword 출력을
전달하도록 확장했다. Frame timing/receipt에 filter backend, schedule 및
shared classification 여부를 기록한다.

보정은 새 마스크 픽셀을 제안할 수 있으므로 원래 raw mask union만 분류해서는
충분하지 않다. 보정이 있는 프레임에만 전체 화면 coverage로 GPU robot classification을
한 번 계산한다. 보정 없는 경로는 기존 compact mask-union filtering을 그대로 쓴다.

- [CurrentGeometry.build](../../graph/plan_a_bc/geometry.py)의 선택적
  `mask_transform(raw_cuda, robot_frame)`이 이 공유를 연결한다.
- [RobotFilterFrame](../../graph/plan_a_bc/robot_frame.py)은 현재 CPU 보정 consumer에
  필요한 전체 boolean 판정 및 진단을 한 번 명시적으로 export한다.
- [CausalCorrection.process](../../graph/task10i_stage4a2/tri_model_integration.py)는
  공급된 frame의 `prepare`를 사용한다. 확인/불확실 비율, supported pixels 등
  기존 evidence 계산과 상태머신은 유지된다.
- [PlanA.prepare](../../graph/plan_a_geometry/core.py)의 `robot_diagnostics=` 경로는
  CPU `classify_robot`를 호출하지 않고 공급된 flags로 기존 mask 제거를 수행한다.
- 최종 clean mask는 같은 GPU confirmed/uncertain flags와 현재 proposal의 overlap으로
  계산한다. Robot hull/ray clipping을 두 번째 실행하지 않는다.
- Frame은 callback 종료/예외 후 무효화되고 다음 시점에 재사용할 수 없다.
  Memo wrapper는 새 keyword와 diagnostic 배열·metadata를 cache identity에 포함한다.

보정 상태머신 전체를 GPU로 옮긴 변경은 아니다. 현재 보정은 NumPy raw mask 및
robot evidence를 소비하므로 GPU 분류 **이후** 명시적 host 경계를 유지한다.
변하지 않은 SAM2 mask는 기존 CUDA tensor를 그대로 보관하고, 실제 바뀐 proposal만
다시 올린다. Standalone CPU 진단/과거 offline correction이 frame을 공급하지 않는
경로는 기존 CPU reference 동작을 유지한다.

### 과거 live 호출자

다음 호출자들은 CUDA 입력을 받는 `TensorOnlineVisionAdapter`와 지연 producer를
사용하도록 연결했다. Handoff/Predstate는 eval_runner의 adapter alias를 공유한다.

- [graph_recurrent10g/eval_runner.py](../../graph/graph_recurrent10g/eval_runner.py)
- [geometry_reach10d/evaluate.py](../../graph/geometry_reach10d/evaluate.py)
- [handoff10h/runner.py](../../graph/handoff10h/runner.py)
- [predstate10i/rollout.py](../../graph/predstate10i/rollout.py)

영상용 raw mask export는 Graph 이후 `numpy_packet`, NPZ/기록용 CUDA Graph/image는
기존 `numpy` adapter를 명시적으로 사용한다. 필터 입력을 CPU로 내렸다 올리는
우회는 추가하지 않았다. Episode finally에서 geometry worker도 close한다.
모델·상태머신·action codec·물리 제어 로직은 변경하지 않았다.

## 최종 검증

### 단위·호출 계약

원격 suite **79/79 PASS, skip/실패/오류 0**이다. 다음 범위를 포함한다.

- 기존 GPU 필터의 empty/full/overlap/invalid depth, threshold, owner/tie,
  row compaction/tile, pose/intrinsic 갱신, batch 공유, signed-int16 전수 검사
- 기존 worker/bridge, online prefix 및 cache/live entrypoint 계약
- 새 frame의 CPU 분류 생략·보정용 flags 유지·새 후보 픽셀 처리·프레임 만료
- depth 변환 실패 시 pending 정리 및 다음 submit 가능 여부
- 기존 보정 상태머신·후보 검증·late handoff·raw 입력 불변 및 새 frame routing
- Frozen/Reach 실제 함수의 producer 지연·CUDA mask 요청
- 네 과거 live adapter alias에서 native CUDA RGB의 초기화/추적 mock 실행
- Native CUDA 입력의 dense readback 금지 및 CPU 센서 업로드/원래 signed depth 유지
- 기존 Plan A 및 per-frame memo 계약

로컬에서는 기존 관련 35개와 Plan A 16개가 통과했다. 최종 새 suite는
9개 중 CPU 6개 통과, CUDA 전용 3개 skip이다. Mac에는 CUDA가 없으므로 GPU 검증은
기존 RTX3090 환경에서 실행했다. Fixture simulator/model 의존성을 교체한 호출
테스트를 실제 폐루프 rollout 실행으로 해석하지 않는다.

### 실제 저장 입력과 GPU 계산

일반 경로는 두 사례 모두 CUDA raw/clean 반환, 직접 기존 filter와 bitwise 일치,
shared prep/batch 1회, ray 재사용, 현재 prepared 갱신, vision 예외 후 재사용을
확인했다. NoReadback 및 global synchronize 금지 guard도 통과했다.

보정 경로는 실제 `infer_frame`부터 robot handoff까지 실행하되 모델 출력은 저장된
actual SAM2 mask로 공급하고 최종 Graph는 가로챘다. 원래 마스크 밖 4×4 proposal도
추가해 full-frame coverage를 검사했다.

| 입력 / 512 두 카메라 | CPU 로봇 분류 | 준비 | GPU classification batch | fused depth 변환 | flags·raw/clean·보정 gate 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| `10i_s1_0921_t080` | 0 | 1 | 1 | 2 | CPU reference와 정확히 일치 |
| `10i_s1_0531_track_t011` | 0 | 1 | 1 | 2 | CPU reference와 정확히 일치 |

여기서 flags는 confirmed/uncertain/candidate/interior의 전체 화면 boolean 배열이다.
현재 프레임의 원래 mask와 proposal을 반복 prepare해도 분류를 다시 수행하지 않았으며,
memo의 CPU classifier miss도 0이었다.

## 부가 진단의 수치 차이와 검증 한계

최초 검증은 독립 NumPy classifier의 `link_owner`까지 bitwise 같다고 요구해
실패했다. 실패 원자료를 보존했고, 조사 후 실제 보정 소비 필드와 부가 진단의
검증 계약을 구분했다. 기존 CUDA filter의 kernel/threshold/tie 구현은 이번에
변경하지 않았다. 원래 CUDA owner/tie suite는 그대로 통과했다.

| 입력 | NumPy 대비 owner 차이: base / hand | predicted depth 최대 차이 |
| --- | ---: | ---: |
| `0921_t080` | 9 / 67 픽셀 | 1.721e−15 m |
| `0531_track_t011` | 11 / 6 픽셀 | 1.499e−15 m |

기계 정밀도 수준의 depth 차이와 기존 CUDA tie 검사 결과를 근거로, 독립 NumPy
계산과 CUDA 계산의 링크별 산술·동률 차이로 해석한다.
보정은 owner를 읽지 않으며, 소비하는 confirmed/uncertain flags와 최종 선택 mask는
두 실제 입력 모두 정확히 같았다. CPU와 모든 diagnostic byte가 같다고 주장하지 않는다.
공급 진단에는 `backend='cuda'`를 표시했다. 일반 standalone NumPy diagnostics는
원래 구현을 유지한다. 이 두 사례가 모든 장면의 NumPy/GPU bitwise 동등성을
증명하는 것은 아니다.

최종 실행의 경고는 기존 SAPIEN Vulkan ICD import 경고다. 최초 read-only mask를
Torch gate가 감싸며 발생한 경고는 writable owned mask를 전달하도록 수정했고,
최종 로그에서는 발생하지 않았다.

실제 SAM2/DINO/feature 모델의 새 forward, 외부 API, 전체 simulator/policy rollout,
새 latency 측정, 학습 또는 기존 frozen deployment 재개는 수행하지 않았다.
79개 테스트의 실행 시간을 inference latency로 발표하지 않는다. 이번 변경의
새 ms 개선율이나 전체 모델 속도는 아직 측정하지 않았다.

## 기록과 완료 상태

공통 `ExperimentLogger`를 재사용했고 W&B disabled, HF publishing 없음이다.
최종 Logger 상태는 `finished`다. 실패한 초기 검증과 최종 결과를 별도 출력 경로에
보존했으며, 원래 paused PID 7개는 종료 점검에서도 모두 `T` 상태였다.
종료 GPU utilization은 0%, 기존 resident allocation은 636MiB였다.

실제 실행 코드는 격리 snapshot으로 고정했다. 다른 작업의 변경을 되돌리거나
commit/stage하지 않았다. 원격은 Git 없는 source copy여서 Logger의 null commit/
dirty=false를 clean checkout 증명으로 사용하지 않으며, source archive 및 overlay
해시로 확인한다. 기존 출력을 덮어쓰거나 과거 protocol/seed/model ID를 바꾸지 않았다.

- [검증 runner/방법](../../graph/experiments/robot_filter_live_completion_10i_20261104_v1/README.md)
- [최종79개·실제 입력 결과](../../outputs/task10i/robot_filter_live_completion_20261104_v1/remote_final/results.json)
- [최종 테스트 로그](../../outputs/task10i/robot_filter_live_completion_20261104_v1/remote_final/tests.log)
- [최종 실행 로그](../../outputs/task10i/robot_filter_live_completion_20261104_v1/measured_v2.log)
- [최초 실패 로그](../../outputs/task10i/robot_filter_live_completion_20261104_v1/measured.log)
- [로컬 최종 새 suite](../../outputs/task10i/robot_filter_live_completion_20261104_v1/tests_local_final.log)
- [고정 source identity](../../outputs/task10i/robot_filter_live_completion_20261104_v1/source_identity.json)
- [최종 실행 source archive](../../outputs/task10i/robot_filter_live_completion_20261104_v1/executed_source.tar.gz)
- [회수·실행/현재 소스 검증](../../outputs/task10i/robot_filter_live_completion_20261104_v1/verification.json)
- [종료 점검](../../outputs/task10i/robot_filter_live_completion_20261104_v1/remote_postcheck.txt)
