# 10-I: 렌더링부터 Graph 반환까지 현재 경로 점검 — 2026-11-03

> 후속 업데이트(2026-11-04): 아래 표는 11월3일 점검 당시의 기록이다. 현재 live GPU 관측 출력은 [새 검증 보고서](10i_live_gpu_observation_output_20261104.md)를 따른다.

Graph 후반부의 GPU 전환은 완료했다. 렌더링→Terra→SAM2→Graph 전체를 하나로
연결해서 실행·시간 검증한 상태는 아니다. 아래는 Terra를 사용할 예정인 흐름에
현재 구현 상태를 함께 표시한 표다. 기본 `CurrentGeometry`의 triton_fused 경로를
기준으로 하며 선택형 cuda_resident 실험 경로와 구분한다.

| 순서 | 단계 | 처리 내용 | 실행·데이터 위치 | 현재 상태 |
| --- | --- | --- | --- | --- |
| 1 | 시뮬레이터 이미지 생성 | 같은 시점의 base/hand RGB·Depth와 카메라 보정값 생성 | 시뮬레이터/렌더러 → 현재 live 어댑터는 CPU 관측 배열 | 기존 경로. 전체512 live 실행은 이번에 검증하지 않음 |
| 2 | 센서·로봇 상태 정리 | episode/time, RGB-D, intrinsic/extrinsic, TCP·로봇 FK 형상 결합·검사 | CPU | 구현됨 |
| 3-A | Terra API 호출·응답 해석 | RGB 인코딩·전송 → Peg/Box 검출 box 해석 → SAM2 prompt 준비 | CPU + 외부 API | 별도 실험 코드 있음. 현재 live 기본 DINO를 대체하는 연결은 남음 |
| 3-B | 로봇 필터 준비 | 현재 FK 형상·깊이·보정 snapshot, 형상 변환·hull 등 준비 | CPU worker; vision 실행과 병렬 | 기본 경로에 연결됨. SAM2 실행 후 필터가 이 결과를 기다림 |
| 4 | SAM2 이미지 준비 | RGB 전처리·GPU 업로드, 초기 prompt/추적 state 준비 | CPU → GPU | 현재 OnlineVisionAdapter에는 임시 JPEG 저장·재읽기가 남음 |
| 5 | SAM2 분할·추적 | box prompt로 초기 mask 생성, 이후 현재 프레임 추적; Peg/Box 분리 | GPU → CUDA bool masks | CUDA mask 반환 연결됨. Terra 호출 주기·초기화 정책은 별도 연결 필요 |
| 6 | 로봇 필터 적용 | SAM2 mask와 준비된 로봇 형상·Depth 비교, 로봇·겹침 픽셀 제거 | GPU | 다른 작업에서 관리하는 구간. 이번 Graph 변경에서 로직 수정 안 함 |
| 7 | 손목 카메라 gate | raw/clean mask로 손목 Peg의 깊이·3D 연결성·국소 지지·타 카메라 일치 검사; 손목 Box 제외 | raw/clean mask GPU → CPU, gate CPU | CPU 경로가 남음. gate 내부 전체 XYZ 생성도 남음 |
| 8 | 공유 GPU 입력 준비 | gate 이후 mask·Depth·카메라 보정 텐서를 준비하고 이후 단계에서 공유 | CPU → GPU | 구현됨. 이후 Graph까지 동일 snapshot 재사용 |
| 9 | 마스크 픽셀 역투영 | 유효 깊이가 있는 비겹침 mask 픽셀만 camera XYZ 생성 | Torch CUDA | GPU 구현 완료 |
| 10 | PCA 법선·기하 특징 | 국소 표면 법선, 법선 변화, 깊이 차이·validity 계산 | Triton GPU, triton_fused 경로 | GPU 구현 완료 |
| 11 | 샘플링 정보 구성 | 인접 픽셀의 깊이·법선 차이와 외곽선으로 형태 후보 준비 | GPU | GPU 구현·범위 축소 최적화 완료 |
| 12 | Shape-First 선택 | 구조점 우선 선택, voxel 대표점·표면 FPS로 보충 | GPU | GPU 구현·최적화 완료 |
| 13 | 공통 좌표계·packet | 선택점·법선을 선택한 World 또는 EE로 직접 변환, Peg/Box·출처를 함께 구성 | GPU; 작은 식별·보정 메타데이터는 CPU | CUDA packet 반환 구현 완료 |
| 14 | 관측·카메라 간 일치 검사 | tensor observation 구성, 다른 카메라의 mask·Depth로 선택점 검증 | GPU | GPU 구현 완료 |
| 15 | 점별 기록·3mm 중복 제거 | 구조점 우선 대표점 유지, 중복 관측의 전체 출처 병합 | GPU TensorRecords | GPU 구현 완료 |
| 16 | 물체별 최종128점 FPS | Peg/Box 각각 최대128점, 특징·출처를 함께 선택 | GPU | GPU 구현 완료 |
| 17 | 엣지 생성 | 표면 이웃·법선·witness·free-space 검사, 로봇–물체 및 Peg–Box 연결 | GPU | GPU 구현 완료 |
| 18 | 최종 Graph 구성·검증·반환 | Node21/Edge4, 타입·edge index·특징 구성 및 current/source 검사 | Graph·기록 배열은 GPU; Python 제어·일부 scalar 검사는 host | GPU 반환 구현 완료 |

3-B의 CPU worker는 현재 `CurrentGeometry.build(..., mask_producer)`가
vision callback을 호출하기 전에 제출한다. 현재 callback은 DINO/SAM2다.
Terra까지 포함한 live callback은 아직 연결되지 않았으므로 Terra와의 실제
overlap 시간을 측정했다고 해석하면 안 된다. 필터 적용은 mask와 준비 결과가
모두 있어야 시작한다. 두 병렬 구간의 시간은 단순히 더하지 않는다.

현재 live entrypoint는 Grounding DINO를 생성한다. Terra API 요청·파싱과
SAM2 연계의 저장 이미지 실험 구현은 존재하지만, live 기본 경로가 Terra로
교체됐다는 의미는 아니다. Terra의 초기화/재탐지 호출 정책도 이 연결에서
확정해야 한다.

손목 gate에는 별도의 `camera.world_points()` 호출이 남아 있다.
이 함수는 전체 Depth의 camera XYZ를 만든 다음 World로 변환하고,
gate가 그 결과에서 clean mask 픽셀을 선택한다. 앞서 최적화한 mask-only GPU
역투영은 **gate 이후 Graph용 점 생성 경로**다. 이를 gate 내부의 기존 CPU
역투영까지 제거한 것으로 확대 해석하면 안 된다. gate는 raw/clean mask를
CPU로 내려받고, 통과 mask는 이후 GPU 입력 준비에서 다시 올라간다.

렌더링부터 SAM2까지도 host 경계가 있다. 현재 live sensors는 관측을 NumPy로
받으며, OnlineVisionAdapter는 임시 JPEG를 쓰고 SAM2 loader로 다시 읽어서
GPU로 올린다. 별도 CUDA 이미지 bridge/선택형 ingress 실험이 존재하더라도
이 기본 live 경로의 JPEG 처리가 자동으로 대체되는 것은 아니다.

현재의 완료 범위는 gate 이후 GPU 입력 준비부터 Graph·TensorRecords CUDA
반환까지다. 작은 identity/calibration 및 로봇3개 좌표 준비는 host에 남고,
Python 분기·동적 크기·검증 scalar의 동기화도 남는다. 파일 저장은 명시적인
numpy_graph/numpy_records export를 사용하며, 모델 입력 pack_graphs는 CUDA
Graph를 바로 받는다.

직전512/Vast58752018의 EE6.081ms는 표의 **17~18번**, 즉 최종 FPS 결과가
준비된 뒤 엣지·Graph·최종 검사 구간이다. 렌더링, Terra, SAM2, 로봇 필터,
손목 gate, 앞선 점 처리를 포함한 전체 시간이 아니다. 측정 표본의 Box는
비어 있고 base는 빈 mask의 복제 입력이었다. 전체 경로의 지연시간은 별도
통합 실행으로 측정해야 한다. 이번 점검에서는 새 API call, simulator/CUDA
실행, 성능 재측정이나 코드 수정을 하지 않았다.

근거 코드와 이전 실행 보고서:

- [live 렌더러 설정·sensors·vision callback](../../graph/plan_a_bc/rollout.py)
- [현재 DINO/SAM2 생성](../../graph/plan_a_bc/cache.py)
- [JPEG 전처리·검출 초기화·CUDA mask 반환](../../graph/integration_baseline/online_vision_adapter.py)
- [Terra 별도 요청 실험](../../graph/experiments/api_terra_effort_grounding_20261103_v1/README.md)
- [로봇 준비의 비동기 worker](../../graph/experiments/robot_filter_gpu_io_10i_20261103_v1/preparation.py)
- [기본 조립 경로·mask readback](../../graph/plan_a_bc/geometry.py)
- [CPU wrist_gate와 GPU represent](../../graph/plan_a_geometry/core.py)
- [전체 Depth camera/world XYZ 구현](../../graph/persistent_geometry/contracts.py)
- [mask-only 역투영·PCA·sampling 연결](../../visual/runtime/graph_perception.py)
- [최종 GPU Graph 구현](../../graph/persistent_geometry/tensor_graph.py)
- [GPU Graph 실행·시간·제약](10i_graph_cuda_58752018_20261103.md)
