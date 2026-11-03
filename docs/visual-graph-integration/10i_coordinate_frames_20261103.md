# Task 10-I — World / EE 좌표계 선택과 경로 통일

2026-11-03. 사용자가 선택한 좌표계를 샘플링 이후 packet, 관측, 중복 제거,
최종 점 선택, Graph 반환까지 유지하도록 구현했다. 속도 재측정은 하지 않았다.

## 설정

```python
from graph.plan_a_bc.geometry import CurrentGeometry

geometry = CurrentGeometry(assets, schema, coordinate_frame='world')  # 절대 World
geometry = CurrentGeometry(assets, schema, coordinate_frame='ee')     # 현재 TCP, 기본값
```

`perception_backend='triton_fused'`, explicit `'cpu'`, 선택적 `'cuda_resident'`
모두 같은 설정을 사용한다. 패킷 생성 API인 `make_packet` / `make_packet_cuda`도
동일한 `coordinate_frame` 인자를 받는다. `world`/`ee` 외의 설정은 거부한다.

새 실험 설정 파일에서는 다음 값을 사용한다. Cache 생성과 live rollout이 같은
값을 읽는다. 기존 모델/데이터에 대해 학습이나 평가를 새로 실행하지 않았다.

```json
{"geometry": {"coordinate_frame": "world"}}
```

## 좌표 처리

원본 RGB-D 역투영, 픽셀 이웃/PCA, 깊이 기반 특징, Shape-First는 기존 camera-local
정의를 유지한다. 이 선택 결과를 **Camera→World 또는 Camera→EE로 직접 한 번**
변환한다. 이전의 Camera→EE→World→EE 점 배열 왕복은 새 공통 경로에서 제거했다.
법선은 같은 변환의 회전 성분만 적용하며, translation은 더하지 않는다.

| 구간 | `world` | `ee` |
| --- | --- | --- |
| 선택 packet의 점·법선 | World | 현재 TCP 기준 EE |
| 두 카메라 관측·중복 제거·최종 FPS | World 유지 | 같은 시점 EE 유지 |
| 최종 물체 노드·법선 | World 유지 | EE 유지 |
| TCP/두 finger-pad 노드 | 원본 World 위치 | 원본 robot 위치를 EE로 변환 |
| 엣지 방향 | World 축 기준 | EE 축 기준 |

관측 생성은 이미 설정한 frame의 packet 좌표를 그대로 사용한다. 카메라 간 일관성
검사와 surface free-space 검사는 `inverse(T_world_from_camera) @ T_world_from_frame`
행렬로 투영에 필요한 임시 camera 좌표만 계산한다. 보관하는 점 배열과 관측을
World로 돌렸다가 다시 EE로 되돌리지 않는다. 거리/반경/법선/tangency/witness
정책과 source 우선순위는 바꾸지 않았다.

원본 camera calibration, source `camera_points`, raw robot 관측은 명시적 입력/
출처 정보로 유지한다. `depth_jump_m`는 원래 카메라 z 측정값의 차이이고,
normal variation은 회전 불변 scalar이므로 다른 의미의 값으로 바꾸지 않는다.
로봇 필터 및 샘플링 전 손목 gate 구현은 이번 변경 대상이 아니다.

## 계약·호환성

- Packet/Graph frame 이름은 `World_metres` / `EE_metres`다. Normal feature의
  frame 표기도 각각 World / EE로 맞췄다.
- `T_frame_from_camera`가 선택한 frame의 변환 행렬이다. EE packet에는 기존
  `T_EE_from_camera` alias도 보존하고 두 값이 일치하는지 검사한다. World
  packet에는 EE 변환이라는 잘못된 이름을 붙이지 않는다. 이전 EE packet도
  receiver에서 계속 읽을 수 있다.
- Current record는 `xyz`, `normal`, `coordinate_frame`을 사용하며, EE에는
  `reference_ee_pose`가 있다. EE 값을 `world_xyz` 아래 저장하지 않는다.
- 서로 다른 frame/EE 기준 포즈가 섞인 관측/records, camera/robot 기준 포즈 불일치,
  frame과 calibration/point 변환이 맞지 않는 packet은 거부한다. World와 EE
  Graph를 한 batch로 pack하는 것도 거부한다.
- Cache identity에 유효 frame을 기록하고 version을
  `graphs_common_coordinate_frame_v2`로 바꿨다. 예전 cache는 재사용하지 않는다.
  새 출력의 diagnostic 좌표는 `point_xyz`와 `coordinate_frame` scalar로 저장하며,
  Graph readback에서도 frame을 복원한다. 학습 loop/모델/Action 정의는 그대로다.
- Historical standalone `PlanA`의 frame 미지정 경로는 이전 진단용 동작을 유지한다.
  Shared persistent map도 World-only이며 EE 관측을 거부한다. 이번에 선택 가능한
  두 경로는 current-only `CurrentGeometry`에 연결했다.

## CPU·CUDA 검증

다음 suite를 실행했다:

```text
visual.runtime.test.test_coordinate_frames
visual.runtime.test.test_object_groups
visual.runtime.test.test_graph_perception
visual.runtime.test.test_vision_features
visual.runtime.test.test_variable_rate_graph_contract
graph.plan_a_geometry.tests
graph.persistent_geometry.tests.test_contracts
graph.plan_a_bc.tests
```

- 로컬 macOS: **95개 중89개 통과, CUDA 전용6개 skip**.
- VastAI **58752018**, RTX3090: **95개 중94개 통과,1개 skip**. Skip은 실제
  preregistration 파일이 없는 source-only 배포의 기존11,487-step schedule 점검이다.
- Python3.12.15, Torch2.14.1+cu130, NumPy2.5.4의 기존 Vast 환경을 사용했다.

새 검증은 비항등 camera/TCP 회전·이동, 서로 다른 두 camera calibration,
물체별 source 보존, 잘못된 frame/행렬/EE reference 거부, World/EE 법선 및
엣지 방향 변환, 3mm 중복 제거/최종 FPS/연결의 물리적 관계, cache frame readback을
포함한다. 직접 packet→관측 경계에는 추가 점 transform이 호출되지 않는 것도
검사했다. CPU public `CurrentGeometry` 반환을 두 frame에서 확인했다.

CUDA NumPy/tensor packet 경로와 resident CUDA의 선택점→관측→compact
dedup/budget→최종 Graph 경로도 두 frame에서 검증했다. 이 resident 검증은
현재 준비된 mask를 주입하고 robot filter/gate를 우회하므로 해당 별도 로직을
검증했다고 주장하지 않는다. GPU 출력 device와 source provenance를 확인했다.

빈 Graph도 두 frame에서 별도 확인했다: TCP/finger 노드3개, 엣지0개이며 로봇
위치가 선택한 기준에 맞았다. Simulator 생성/rollout·정책 학습/평가·성공률은
이번에 실행하지 않았다.

## 실제512 입력

이전과 같은 실제 `10i_s1_0844_t080` hand-camera512×512, Peg mask61,056 pixels,
Box0 pixels를 사용했다. 기록된 TCP/calibration과 실제 URDF FK assets를 검증했다.
준비된 mask/gate를 주입해 `CurrentGeometry.build`를 실행했다. 실제 base-camera
입력이 없으므로 동일 sensor의 빈 base support를 붙였다. 두 실측 카메라 rollout
결과는 아니며, 다른 calibration 간 검사는 위 합성 검증으로 보완했다.

| 검증 결과 | World | EE |
| --- | ---: | ---: |
| 선택 packet 점 | 1,105 | 1,105 |
| 최종 물체 점 | 128 | 128 |
| 최종 Graph 노드 | 131 | 131 |
| 최종 Graph 엣지 | 1,006 | 1,006 |
| Camera→선택 frame 직접 계산 대비 최대 위치 오차 | 3.72e-9m | 3.72e-9m |
| CUDA/NumPy packet 점 좌표 차이 | 0 | 0 |

각 frame에서 packet→관측의 점/F5, records→Graph의 점/법선은 exact 일치했다.
두 frame의 sampling source, 최종128개 source ID와 edge index도 일치했다.
Edge 길이의 두 frame 간 최대 차이는 **8.85e-9m**였다. 방향 벡터는 각 frame의
축으로 표현된다. CPU/NumPy와 CUDA packet의 일반 비교는 기존 float32 허용
오차(point1e-6m, feature2e-5)를 사용했고, 이번 실제 점 비교는 차이0이었다.

이는 일반적인 float32 연산을 bitwise 회전 불변으로 만든다는 의미는 아니다.
정확한 반경 경계나 동일 거리 tie에 가까운 입력에서는 frame별 반올림이 점 선택/
연결에 영향을 줄 수 있다. 거리 정책을 완화하거나 근사 탐색으로 바꾸지는 않았다.
이런 경계와 별개로 frame을 혼용하거나 잘못된 변환을 적용하는 오류는 위 계약으로
검사한다. 기존 checkpoint의 성공률이나 학습 성능이 보존된다는 주장은 하지 않는다.

## 재현성과 기록

공통 `ExperimentLogger` 재사용, W&B disabled, publishing 없음.
실제512 검증 UUID: **`47521056d86d4d90b1f23c0cc8ed4cef`**.
원격 root: `/workspace/task10i_coordinate_frames_20261103_v1`.
성공 source는 `source3`이며,299개 overlay 파일을 실행 전에 hash 검증했다.
기존 dirty 작업은 보존했고 robot-filter 파일은 이번 작업에서 수정하지 않았다.
로컬 pre-change source/identity와 각 실행 snapshot/log를 보존했다.

초기 로컬 새 fixture는 원래 점 집합에도 존재하는 중복을 무시하고 전체 중복 수를1로
가정해 실패했다. 중복 추가 전의 실제 기준 대비1 증가를 확인하도록 fixture를
수정했다. 첫 원격 source 묶음은 root `graph/vision_input.py` 갱신과 resident 경로의
기존 radius-kernel 모듈을 빠뜨려 실패했다. 실행 소스를 덮어쓰지 않고 누락 파일을
포함한 새 `source3`에서 재검증했다. 성공 source299개와 verifier가 현재 파일에
일치함을 다시 확인했다. Package/driver/service 변경이나 다른 작업 중지는 없었다.

- [실제512 검증 결과](../../outputs/task10i/coordinate_frames_58752018_20261103_v1/remote/verified/verification.json)
- [Topology 후속 readback 검사](../../outputs/task10i/coordinate_frames_58752018_20261103_v1/remote/verified/topology_audit.json)
- [World/EE packet·Graph 원본 배열](../../outputs/task10i/coordinate_frames_58752018_20261103_v1/remote/verified/geometry.npz)
- [성공 CUDA 테스트 로그](../../outputs/task10i/coordinate_frames_58752018_20261103_v1/remote/tests3.log)
- [성공 로컬 테스트 로그](../../outputs/task10i/coordinate_frames_58752018_20261103_v1/local_tests_final.log)
- [빈 Graph 검사](../../outputs/task10i/coordinate_frames_58752018_20261103_v1/empty_graph_check.json)
- [실제512 실행 로그](../../outputs/task10i/coordinate_frames_58752018_20261103_v1/remote/verified.log)
- [하드웨어·환경](../../outputs/task10i/coordinate_frames_58752018_20261103_v1/remote/hardware.json)
- [전체 source/input manifest](../../outputs/task10i/coordinate_frames_58752018_20261103_v1/remote/source_manifest.json)
- [검증 source299개 manifest](../../outputs/task10i/coordinate_frames_58752018_20261103_v1/remote/overlay3_manifest.json)
- [다운로드·hash 검증 receipt](../../outputs/task10i/coordinate_frames_58752018_20261103_v1/download_receipt.json)
- [Verifier·실행 방법](../../graph/experiments/coordinate_frames_58752018_20261103_v1/README.md)
- [공통 좌표계 API](../../graph/plan_a_bc/README.md)

Archive SHA256:
`0d803f0b5999e4ed14672e9594b08017da861bd1e5bcc5b84f85b52b6ad6df2e`.
다운로드 후 전체1,471개 source/input manifest 항목과 archive hash를 검증했다.

## 독립 커밋 범위 확인

좌표계 기능을 커밋할 때 별도 진행 중인 CUDA ingress/robot-filter 변경은 제외했다.
겹치는 파일은 기본 `triton_fused`/CPU 경로의 frame 설정·검증만 분리했다.
선택적 resident prototype의 위 실행 결과는 전체 작업트리/snapshot에 대한 기록이며,
그 prototype 자체와 frame 연결 수정은 별도 미커밋 작업으로 남겼다.
CUDA tensor packet의 World/EE API는 이번 좌표계 커밋에 포함한다.

커밋할 tree만 추출한 독립 복사본에서 같은95개 suite를 다시 실행했다.
**88개 통과,7개 skip, 실패0개**였다. Skip은 macOS의 CUDA 전용6개와 실제
preregistration 파일이 없는 source-only 복사본의 기존 schedule 점검1개다.
Prototype이 없는 checkout에서는 해당 optional 통합 테스트를 명시적으로 skip한다.
그 밖의 CPU/CUDA와 실제512 검증 범위는 위 원본 실행 기록을 따른다.
