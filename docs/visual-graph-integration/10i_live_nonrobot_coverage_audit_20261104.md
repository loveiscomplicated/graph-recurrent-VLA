# 10-I: live 최적화 연결 전수 점검 — 로봇 필터·손목 gate 제외

점검일:2026-11-04 KST. **주 live 진입점 `graph.plan_a_bc.rollout`에는 이번 핵심
GPU 최적화·512 변경이 모두 연결돼 있다.** 그러나 다른 평가 진입점까지 모두
같이 이식된 상태는 아니다. CUDA 비전 입력, Graph export, Terra memo,512 설정,
좌표계 설정 전달에 관한 별도 평가 진입점의 누락을 확인했다.

이번 작업은 조사다. production 최적화/필터/gate 코드를 수정하지 않았고,
별도의 audit runner와 이 보고서만 추가했다. 로봇 필터와 손목 gate의 구현·성능·
판정 동등성은 검토 대상에서 제외했다.

## 주 live 내부인지 재확인

이 보고서의 별도 평가 관련 네 항목은 모두 **주 live 실행 경로 바깥**의 문제다.
기준 경로는 `graph.plan_a_bc.rollout.evaluate → run_episode`다.

| 지적한 문제 | 주 live에서의 실제 처리 |
| --- | --- |
| 예전 CPU 비전 adapter 호출 | CUDA RGB는 TensorOnlineVisionAdapter.process로 들어감 |
| CUDA Graph의 직접 numpy 호출 | 저장 직전에 numpy_graph로 명시적 CPU export 후 numpy 호출 |
| F1 DetectionMemo와 Terra 불일치 | 주 evaluate는 LiveGrounding을 직접 생성하며 F1 memo를 사용하지 않음 |
| 별도 renderer의512 미상속 | 자체 environment가 두 카메라512 sensor_configs와 공통 CUDA 출력 wrapper를 지정 |

일부 별도 평가가 주 live의 sensors 같은 함수를 가져다 쓰는 의존 방향이다.
주 live가 그 별도 평가 루프를 호출하지는 않는다.
다만 파일 삭제 가능 여부는 별도다. TensorOnlineVisionAdapter는 기존
OnlineVisionAdapter의 초기화/reset/close를 상속하며, 주 live는
rollout_stop_clock 모듈의 audit/helper를 재사용한다. 해당 모듈을 import한다고
그 안의 과거 평가 루프나 CPU process가 실행되는 것은 아니다.

주 live에 실제로 남은 비용인 Graph 직후 영상 mask의 CPU export는 이 네
항목과 별개이며, 원래 보고서의 발견4에 기록한 그대로다.

## 범위와 검증 방식

주 진입점은 [rollout.evaluate](../../graph/plan_a_bc/rollout.py:227) →
`run_episode` → `environment/sensors` → `TensorOnlineVisionAdapter` →
`CurrentGeometry.build` → `PlanA.represent/graph` → `pack_graphs/model.step`이다.

정적 호출 추적만으로 완료 처리하지 않았다. 기존 VastAI58752018, RTX3090에서
실제512 CUDA 카메라와 설치된 SAM2.1 Small checkpoint를 사용해 원래
`rollout.run_episode`를 실행했다. EE/World 각각2프레임, 총4프레임이며 각 프레임은
두 카메라를 처리한다. 실제 SAM2 초기화·추적, 역투영부터 Graph까지의 구현,
출력 저장과 모델 입력 패킹이 실행됐다.

범위를 지키기 위해 로봇 필터는 ready-mask fixture, 손목 gate는 ready
mask/TensorViews fixture로 대체했다. 실제 필터·gate 계산은 실행하지 않았다.
Grounding은 고정 box fixture로 대체해 유료 API 호출을 하지 않았다.
정책은 zero-command stub으로 대체했다. 따라서 학습된 정책/GNN의 성공률이나
고정 box의 의미적 정확성을 평가한 실행이 아니다. 각2프레임 동안 실제 physics
10회가 실행됐고 기존 stop-the-clock 검사를 통과했다.

함수·Triton launch·CUDA Graph replay를 관측하고, CPU point 경로를 호출하면
실패하도록 막았다. 관측한 명시적 `Tensor.cpu`의 shape/byte/호출 문맥을 기록했다.
작은 scalar의 암묵적 동기화까지 전부 계측한 profiler는 아니며 latency도 보고하지 않는다.

## 주 live 경로의 연결 목록

아래는 현재 CUDA가 사용 가능한 주 live 경로 기준이다.

| 변경·최적화 | 실제 연결 | 확인 결과 |
| --- | --- | --- |
| 두 카메라512×512 | `rollout.environment`의 per-camera `sensor_configs` | 연결. 실제 CUDA RGB/Depth와 이전512 smoke로 확인 |
| native GPU 관측 출력 | CUDA renderer → 공통 `ObservationDevice` → `TensorCameraObservation` | 연결. 센서 이미지 NumPy 변환 없음 |
| Terra primary + 인증 fallback | `evaluate`의 `LiveGrounding`, 카메라별 stream, joint initialization | 연결. 코드·기존 mock 계약 테스트로 확인. 이번 실제 API call0회 |
| SAM2 tensor RGB·CUDA mask | `TensorOnlineVisionAdapter` → `TensorFrames` → `SAM2Tracker.start_frames/track` | 연결. 실제 설치 모델로 초기화·다음 프레임 추적 실행 |
| 마스크 픽셀만 역투영 | `make_packet_cuda` → `torch_projection.project` | 연결. compact CUDA XYZ이며 CPU dense XYZ 호출 차단 상태로 통과 |
| PCA 법선·기하 특징 | `features_cuda` → `fused_normals_compact/fused_features_compact` | 연결. 각각 좌표계별4회 호출 관측 |
| GPU sampling 정보 | `build_sampling_info_cuda` | 연결. CPU feature raster/evidence 경로 호출 차단 상태로 통과 |
| GPU Shape-First | `shape_first_cuda` | 연결. CUDA 선택 결과가 packet으로 이어짐 |
| 외곽선 ROI·canonical 개선 | `mask_bounds` → ROI crop → `compress/hook/trace/canonical` | 연결. bounds/canonical 각각 좌표계별6회, compress30회 관측 |
| voxel 대표점 최적화 | `voxel_choices`, PCA source 정렬 생략·flag gather | 연결. voxel kernel 좌표계별11회 관측 |
| FPS 초기 거리·반복 최적화 | 큰 입력의 `_FPSPlan`, `seed_tile_bounds/fps_seed_bounded`, CUDA Graph replay | 연결. `FPSPlan.run` 좌표계별4회 관측 |
| 선택 출처 조회 최적화 | `mark_membership/gather_membership` | 연결. 좌표계별6/4회 관측 |
| 선택 결과 GPU 반환 | `sample_cuda`에서 device 미지정 → 입력 CUDA 장치 유지 | 연결. CPU selection/export 경로 호출 차단 상태로 통과 |
| Peg0/Box1 분리 보존 | mask축 → selected0/1 → `object_id/object_slices` → 관측 slice | 연결. CUDA packet과 final point labels 유지 |
| World/EE 선택·단일 변환 | live config → CurrentGeometry → packet builder → Graph | 연결. 실제 EE/World 둘 다 실행하고 frame label 검사 |
| GPU packet·관측·일치 검사 | `selected_packet`, `observations_from_tensor_packet`, `consistent_views_tensor` | 연결. 각각 좌표계별4회 호출 |
| GPU 기록·3mm 중복 제거 | `current_records_tensor/deduplicate_tensor` | 연결. 각각 좌표계별2회 호출, CSR 출처 CUDA 유지 |
| 물체별 최종128점 GPU FPS | `budget_tensor(...,128)` | 연결. 좌표계별2회, 최종 Peg/Box 각128점 |
| GPU 엣지·Graph | `build_graph_tensor`, CUDA neighbor/free-space, Node21/Edge4 | 연결. CUDA Graph·records 반환 |
| 깊이 배열 반복 복원 제거 | Graph는 동일 TensorViews의 depth를 재사용 | 연결. 기존 CPU free-space builder를 호출하지 않음 |
| 모델 입력 GPU 유지 | `pack_graphs([graph]).to(model_device)` | 연결. 실제 `model.step` 진입 시 node/edge/attrs CUDA 확인 |

주 경로의 GPU 판정에 `triton_fused` 문자열만 믿지 않았다. 실제 fused 함수와
sampling kernels, final FPS, CUDA KD 및 Graph 호출을 기록했다.
CPU `frame_points`, `current_records`, `deduplicate`, `budget`, `build_graph`,
`to_cpu_evidence`, `numpy_observations`, `numpy_records`, `numpy_selection`을
호출하면 실패하도록 막아도 주 live loop가 통과했다.

World 실행에서 새 `seed_tile_bounds` launch 기록이 없는 것은 미연결이 아니다.
EE에서 만든 같은 shape/stream의 CUDA Graph plan을 World 실행이 replay했고,
`FPSPlan.run`4회가 관측됐다. 현재 입력·seed·used buffers는 replay 전에 복사·
초기화하는 원래 구현을 사용한다.

작은 FPS 입력은 eager/pruned 경로를 쓰는 것이 기존 채택 규칙이다.
깊이 중앙값의 `kthvalue` 전환은 당시 느려서 채택하지 않았으므로 현재 full sort는
이식 누락이 아니다. 우선순위 중복 제거의 일부 GPU `isin`도 남도록 채택됐으며,
최종 provenance 조회는 새 mark/gather kernel을 사용한다.
`fused_normals_compact`에는 수치적으로 애매한 소수 covariance를 다시 푸는
Torch `eigh` 보정 분기가 존재하지만 그 텐서도 CUDA다. CPU PCA fallback이 아니다.

## 실제 Graph 결과와 추가 표면 검사

고정 box를 쓴 주 live fixture는 각 프레임에 노드259개, directed edge80개,
점256개를 만들었다. 노드/엣지5개 tensor와 기록·출처는 CUDA였다.
이 fixture는 표면 엣지가 채택되지 않아, 이 결과만으로 비어 있지 않은
free-space 작업까지 검증했다고 하지 않았다.

추가로 기존 저장512 hand-camera Peg mask 입력에서 같은 GPU 경로를 실행했다.
여기에서도 실제 로봇 필터·gate 대신 동일 경계 fixture를 사용했다.
EE/World 모두 노드131개, directed edge1,006개를 만들었고,
free-space 함수에 **1,632개 실제 중간 sample**이 들어갔다.
`node_input/node_type/edge_index/edge_attr/edge_type`5개 전체 배열이
이전 CPU 저장 기준과 비트 단위로 일치했다. 이 저장 입력의 Box는 비어 있고
base는 빈 mask의 복제 카메라다. 두 물체 동시 처리는 주 live fixture에서
실행했지만 의미적 mask 정확성을 평가한 것은 아니다.

## 발견1: 다른 live 평가 파일은 CUDA 인터페이스 이식이 남음

[graph_recurrent10g/eval_runner](../../graph/graph_recurrent10g/eval_runner.py:100),
[geometry_reach10d/evaluate](../../graph/geometry_reach10d/evaluate.py:45),
[corrective_reach10f/recovery_eval](../../graph/corrective_reach10f/recovery_eval.py:66)는
최신 `rollout.sensors`를 사용하면서 여전히 예전 CPU
[OnlineVisionAdapter](../../graph/integration_baseline/online_vision_adapter.py:85)를 사용한다.
이 adapter는 입력을 `np.asarray(rgb)`로 바꾼다. 현재 sensors가 반환하는 CUDA
RGB를 넣으면 `TypeError: can't convert cuda:0 device type tensor to numpy`가 난다.

[handoff10h/runner](../../graph/handoff10h/runner.py:12)와
[predstate10i/rollout](../../graph/predstate10i/rollout.py:12)도
`graph_recurrent10g.eval_runner`에서 그 예전 adapter를 가져온다.
이 다섯 평가 계열은 주 live의 tensor adapter 연결을 자동으로 상속하지 않는다.

같은 두 번째 미이식은 직접 Graph `.numpy()`다.
F1의 [initial_student_input](../../graph/graph_recurrent10g/eval_runner.py:113),
[handoff](../../graph/handoff10h/runner.py:75),
[predstate](../../graph/predstate10i/rollout.py:75),
[reach 평가 저장](../../graph/geometry_reach10d/evaluate.py:112),
[corrective Graph dump](../../graph/corrective_reach10f/recovery_eval.py:114)에 남아 있다.
기본 Graph는 이제 CUDA tensor이므로 이 export도 TypeError다.
주 live의 `numpy_graph`처럼 명시적 host export가 필요한 위치다.

두 실패 계약은 실제 CUDA 이미지와 실제 CUDA Graph를 써서 재현했다.
이 다섯 평가 전체를 각각 완주한 것은 아니며, 실패하는 공통 경계와 호출
파일을 대조한 결과다. 이전 frozen source로 이미 실행된 결과가 변경되거나
그 과거 실행이 실패했다는 뜻은 아니다. **현재 체크아웃으로 이 평가 파일을
다시 실행할 때** 이식이 필요하다. 이번 요청은 점검이므로 고치지 않았다.

## 발견2: 예전 F1 memo는 Terra joint 인터페이스를 전달하지 않음

[eval_runner.vision](../../graph/graph_recurrent10g/eval_runner.py:59)은 현재
`make_vision`에서 반환한 `LiveGrounding`을
[DetectionMemo](../../graph/geometry_reach10d/optimizations.py:55)로 감싼다.
이 memo에는 `initialize/new_stream` 전달이 없고,
`self.detector(rgb,prompt)`라는 과거 DINO callable 계약만 있다.
`LiveGrounding`은 joint `initialize` 계약이며 `__call__`이 없다.
따라서 CUDA 이미지 문제를 먼저 해결해도 이 조합은
`TypeError: 'LiveGrounding' object is not callable`로 실패한다.

메서드 호출 계약만 로컬에서 재현했다. 생성자·credential 읽기·API를 실행하지
않고 `object.__new__(LiveGrounding)`으로 타입의 비호출성 및 memo 호출을 확인했다.
단순히 새 detector import를 공유한다고 F1/handoff/predstate까지 Terra 연결이
끝났다고 볼 수 없는 이유다. 주 `rollout.evaluate`는 이 memo를 사용하지 않는다.

## 발견3: 별도 renderer 분기는512를 지정하지 않음

[geometry_reach10d.runtime.environment](../../graph/geometry_reach10d/runtime.py:21)는
`renderer_gpu_gate.json`의 adopted가 true이면 자체
[gpu_render_environment](../../graph/geometry_reach10d/runtime.py:8)로 간다.
그 분기는 `sensor_configs` 없이 `gym.make`를 호출한다.
따라서 주 live에 넣은512 설정과 관측 출력 wrapper를 상속하지 않는다.
현재 설치 task의 센서 기본값은128이며 이전 native 기본 smoke에서도 확인했다.
이 별도 renderer 분기 자체를 이번 audit에서 새로 실행한 것은 아니다.

adopted가 false일 때만 최신 주 `rollout.environment`를 사용한다.
따라서 primary512 변경을 모든 live 계열에 공통 적용한 것으로 볼 수 없다.
또한 위 다섯 평가의 `CurrentGeometry(...)` 생성은 대부분 config의
`geometry.coordinate_frame`을 전달하지 않아 기본 EE를 사용한다.
주 live의 World/EE 설정 전달 역시 별도 평가 계열에 자동 전파되지 않는다.

## 발견4: 주 live에는 저장용 CPU 복사가 제어 호출 앞에 남음

[rollout.py:131](../../graph/plan_a_bc/rollout.py:131)은 Graph 반환 직후에
`numpy_packet(geometry.last_raw_masks)`를 호출하고, 그 다음 `pack_graphs`와
`model.step`을 호출한다. 실제로 **프레임당1,048,576byte =1MiB**의 마스크
host 복사를 관측했다. 두 카메라×두 물체×512×512 bool이다.
이 복사는 영상 표시용이지만 현재 위치상 모델 호출·action 전송을 기다리게 하고
`inference_seconds` 측정에도 들어간다. 이번에는 복사 시간을 측정하지 않았다.

이는 Graph가 CPU로 되돌아가서 처리되는 경로는 아니다. Graph와 model input은
계속 CUDA였다. 다만 제어 지연을 더 줄이려면 이 artifact export의 위치·조건을
다음 작업에서 검토할 수 있다. 이번 조사에서 순서를 변경하지 않았다.

다른 host 경계도 확인했다.

| 위치 | 실제 동작 | 평가 |
| --- | --- | --- |
| Grounding scan | `HostScanFrames`가 요청 RGB를 CPU로 변환 | API 이미지 인코딩/기존 grounding의 명시적 경계. 이번 fixture에서 t0 두 카메라만 각768KiB 복사; t1 추적에서는 없음 |
| 작은 metadata | 보정행렬·TCP·Robot76의 검증/FK, 로봇3개 좌표 준비 | 기존 host 계약. 선택점 배열 CPU 반환과 다름 |
| scalar 제어 | ROI4값, 수렴 여부, cardinality, 입력 검증 | 여전히 동기화 가능. zero-sync 최적화라고 주장하지 않음 |
| 매 프레임 영상 | mask export 및 `video_frame`의 RGB host 읽기 | 명시적 기록 경계. mask export는 모델 호출 앞, RGB 영상 생성은 action 후 |
| 주기적 Graph dump | `numpy_graph` 후 저장 | 모델에 들어간 Graph는 CUDA 유지; 저장은 명시적 CPU export |

점 처리 내부의 명시적 array readback은 이번 trace에서 없었다.
`geometry.build`의 scope에서 본 이미지 readback은 그 안에서 호출한 vision의
초기 grounding scan뿐이었다. 전체 CUDA scalar readback을 막은 것은 아니다.

## 환경 확인: SAM2 hole filling 확장 누락

실제 SAM2 실행에서 설치 패키지가 `sam2._C`를 import하지 못해
`fill_holes_in_mask_scores` 후처리를 건너뛴다는 경고가 나왔다.
초기화/추적과 CUDA mask 반환은 성공했다. 이는 이번 GPU Graph 연결 누락은
아니지만 현재 환경의 후처리 제약이다. 모든 SAM2 후처리가 정상 실행됐다고
보고할 수 없다. 이번에는 패키지를 설치하거나 이 동작을 바꾸지 않았다.
해당 경고는 원시 실행 로그에 보존했다.

## 판정과 후속 범위

주 live에서 미연결 상태로 남은 핵심 GPU point/Graph 최적화는 발견하지 못했다.
반면 별도 F1/reach/corrective/handoff/predstate 평가를 현재 checkout으로
실행하려면 tensor vision, 명시적 Graph export, joint Terra memo 전달,
512/출력 wrapper/좌표계 설정을 그 진입점에도 연결해야 한다.
주 live의 다음 지연 최적화 후보는 별개로 저장용 mask export의 제어 경로 분리다.

저장 HDF5 기반 `task10i_stage4a2.frozen_inference`와 historical standalone
PlanA/cuda_resident는 다른 입력·실험 계약을 유지하는 경로다. 이들을 최신 주
live와 같다고 세거나, 보존된 DINO/CPU reference 자체를 미이식 오류로 세지 않았다.
데이터셋 재선정·학습·성공률 평가도 이번 조사에 포함하지 않았다.

## 원자료·실험 식별

최종 실행 ID: **`36650fc487e44bf58ace335bdc24f8f1`**.
공통 `ExperimentLogger(team='graph', job_type='diagnostic', seed=42)`를 재사용했고
W&B disabled다. 유료 API call0회, 실제 SAM2 checkpoint hash 및 source SHA는
summary에 기록했다. source1,784개 파일을 실행 전 확인했다.
기존58752018/RTX3090 환경만 사용했다. local은 macOS CPU 계약만 확인했다.
로봇 필터·gate 실행, 새 패키지/드라이버 설치, publishing, commit은 하지 않았다.

로컬 grounding/legacy online 계약18개 중17개 통과, CUDA 전용1개 skip이었다.
초기 audit v1은 KD 함수를 import alias가 아닌 원래 module에서 관측해 호출수를
놓친 harness assertion이 있었다. 관측 위치를 수정한 v2에서 통과했고,
비어 있지 않은 표면 reference 검사를 추가한 최종 v3도 통과했다.
초기 로컬 실행의 모듈명 오타 실패 로그도 최종 성공 로그와 별도로 보존했다.
production 결함과 audit harness 실패를 섞지 않는다.

- [최종 summary](../../outputs/task10i/live_nonrobot_audit_58752018_20261104_v1/remote/verified_v3/summary.json)
- [EE 호출·readback trace](../../outputs/task10i/live_nonrobot_audit_58752018_20261104_v1/remote/verified_v3/ee/events.json)
- [World 호출·readback trace](../../outputs/task10i/live_nonrobot_audit_58752018_20261104_v1/remote/verified_v3/world/events.json)
- [readback 위치·byte 재검사](../../outputs/task10i/live_nonrobot_audit_58752018_20261104_v1/derived_audit.json)
- [Terra memo 계약 재현](../../outputs/task10i/live_nonrobot_audit_58752018_20261104_v1/memo_contract_probe.json)
- [실제 SAM2 경고 포함 실행 로그](../../outputs/task10i/live_nonrobot_audit_58752018_20261104_v1/remote/run_v3.log)
- [로컬 계약 최종 성공 로그](../../outputs/task10i/live_nonrobot_audit_58752018_20261104_v1/local_grounding_tests_v2.log)
- [검증 코드](../../graph/experiments/live_nonrobot_audit_20261104_v1/verify.py)
- [실행 source archive](../../outputs/task10i/live_nonrobot_audit_58752018_20261104_v1/source_final_v3.tar.gz)
- [실행 source manifest](../../outputs/task10i/live_nonrobot_audit_58752018_20261104_v1/source_v3_manifest.json)
- [원시 결과 묶음](../../outputs/task10i/live_nonrobot_audit_58752018_20261104_v1/result_v3.tar.gz)


## 주 live에 남은 부가비용 구체화

로봇 필터·손목 gate를 제외한 현재 주 live의 부가비용 목록이다. 코드 위치를
재확인했으며 새 성능 측정은 하지 않았다. 아래 항목의 존재와 byte 수는 확인했지만,
시간상 병목 순위는 아직 정하지 않았다.

| 항목 | 발생 시점 | 구체적인 작업 |
| --- | --- | --- |
| 영상용 mask export | 매 프레임, 모델 호출 전 | 두 카메라·두 물체512 bool mask를 묶어1MiB GPU→CPU 전송. 포장·임시 배열·host 복사 포함 |
| 영상 RGB·인코딩 | 매 프레임, action 후 | 두512 RGB를 CPU로 복사해 총1.5MiB 원시 영상 전송, mask overlay, resize, OpenCV video encoding |
| 진단·상태 기록과 Graph dump | hidden/command는 action 전, trace/video/주기적Graph dump는 action 후 | hidden·command·raw 출력을 host로 기록. Graph는 t0/매25프레임/최초상태 도달 시 CPU export·NPZ 저장 |
| GPU scalar 동기화 | 비전·점 처리·Graph 검사 중 | SAM2 mask_pixels.sum.tolist, ROI4값, 연결 성분 수렴, bool/any/isfinite/point-count 검사. 작은 값도 GPU 완료를 기다릴 수 있음 |
| 작은 metadata 왕복·반복 계산 | 매 프레임 sensors/packet 준비 | CUDA 관측 qpos/qvel·보정행렬을 CPU 검증으로 읽고, 일부 보정·Robot76을 GPU 소비자에게 다시 전달. host 좌표 행렬 계산도 있음 |
| GPU 복사·포장·임시 할당 | 매 프레임 | sensors의 image snapshot clone, tensor vision의 raw RGB clone·전처리, packet/관측 특징 clone, 단일Graph에도 pack_graphs의 cat/metadata 생성·검사 |
| grounding 요청 준비·대기 | 초기화/물체 미발견 scan | 해당 RGB GPU→CPU, PNG·base64·JSON 준비와 동기 HTTP 대기. 두 물체가 초기화된 추적 프레임에는 이 scan을 하지 않음 |

영상의 mask1MiB + RGB1.5MiB는512 두 카메라 기준 원시 payload만 합친2.5MiB다.
복사량으로 소요시간을 추정하거나 이것이 최대 병목이라고 단정하지 않는다.
CPU PhysX가 실행할 command 자체의 host 전달은 필요한 경계다. command 기록용
추가 복사와 hidden/raw 기록은 이 필요한 전달과 별도로 볼 수 있다.

현재 inference_seconds는 model/codec 직후, hidden/command host 기록 전에
끝난다. 따라서 그 기록과 action 이후 video/Graph dump 비용을 충분히 반영하지
않는다. 매 프레임 command 전송까지의 지연과 다음 프레임 처리를 시작하기까지의
전체 주기를 구분해 측정해야 한다. action 후 영상 작업도 다음 제어 주기에 영향을 준다.

우선 분리 측정할 대상은 영상 mask/RGB export·video encoding·진단 export다.
그 다음 기록용 scalar 읽기와 필수 검증용 동기화를 구분해 측정할 수 있다.
Snapshot 소유권과 데이터 검증 보장을 유지하면서 공유·묶음 처리를 검토해야 한다.
이 목록은 이미 합의한 GPU 연산의 미연결 목록이 아니라 남은 비용의 위치 목록이다.
