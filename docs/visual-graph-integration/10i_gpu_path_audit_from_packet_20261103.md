# Task 10-I — packet 단계부터 GPU 연속 처리 여부 재점검

2026-11-03. 사용자 요청에 따라 추가 관측 구성 benchmark 진행을 중단하고,
현재 호출 경로를 정적으로 재검토했다. 이번 점검에서 runtime 수정이나 새 GPU
측정은 하지 않았다. 앞선 CUDA 검사 연결은 이미 기본 경로에 반영되어 있다.

## 기본 경로

대상은 `CurrentGeometry(..., perception_backend='triton_fused')`다. 생성자는
`tensor_packet_builder=partial(make_packet_cuda, ...)`를 전달하고,
`PlanA.represent`는 다음 순서로 실행한다:

```text
make_packet_cuda
  → selected_packet: 선택점 검사·Camera→EE/World 좌표/법선 변환·GPU packet
  → observations_from_tensor_packet: 물체별 GPU 관측
  → consistent_views_tensor: GPU 관측 일관성 검사
  → numpy_observations: selected 관측 GPU→CPU
  → current_records: CPU Python 점 기록
  → deduplicate: CPU 3mm 중복 제거
  → budget(...,128): CPU 최종 FPS
  → build_graph: CPU 특징/엣지 생성·CPU Torch tensor 반환
```

**그 이후 전체가 GPU인 상태는 아니다.** Point cloud/feature 배열이 GPU에
연속해서 남아 있는 구간은 일관성 검사 반환까지다. 첫 명시적 D2H는
`PlanA.represent`의 `numpy_observations(..., include_structural=True)`다.

| 단계 | 주 데이터 처리 | 반환/보관 위치 |
| --- | --- | --- |
| Packet source/depth/mask·feature 값 검사 | Torch CUDA | 주 배열 CUDA |
| Pose/calibration·identity·schema metadata 확인 | Python/NumPy CPU, GPU bool 결과 동기화도 존재 | 작은 metadata CPU |
| Camera→설정 frame 점 변환·법선 회전 | Torch CUDA | CUDA |
| Packet 구성·반환 | Python dict 구성은 CPU, 주 배열은 CUDA | 혼합 metadata + CUDA 배열 |
| Peg/Box GPU 관측 구성 | Torch CUDA, 작은 camera-origin metadata H2D | CUDA 배열 |
| 카메라 간 관측 일관성 검사 | Torch CUDA | CUDA 배열·keep/audit tensor |
| `numpy_observations` | Packed selected-array D2H 및 CPU 검증 | NumPy CPU |
| 점 기록·3mm dedup·최종128점 FPS | Python/NumPy CPU | CPU |
| 표면/robot-object/Peg-Box 엣지 및 Graph 구성 | NumPy/SciPy CPU | CPU Torch tensor |

`torch.from_numpy`로 만드는 최종 Graph tensor는 CPU tensor다. CUDA 연산을
수행했다는 뜻이 아니다. 기존 rollout 호출부는 Graph를 pack한 뒤 모델 device로
`.to(...)`하여 다시 이동시킨다. 모델이 CUDA라면 여기에서 H2D가 다시 발생한다.

좌표계 설정은 device 선택과 별개다. `ee`는 Camera→EE, `world`는 Camera→World를
packet에서 직접 적용하고, 이후 관측/Graph의 frame을 유지한다. 과거0.541ms
EE packet→World 관측 변환은 좌표계 통일 이전의 CPU 진단 경로다.

## 비슷한 API와 선택 경로

- Direct `graph_perception.make_packet(...)`는 NumPy 호환 API이며 여전히 CPU
  sampling readback과 CPU packet 완성을 수행한다. 기본 `CurrentGeometry`의
  Tensor branch는 이 함수 대신 `make_packet_cuda(...)`를 호출한다.
- Explicit `perception_backend='cpu'`는 CPU 경로다.
- 선택적 `cuda_resident`도 완전 GPU 연산 경로는 아니다. GPU view 검사 후
  `numpy_packet(exports)`로 selected 자료를 CPU로 보내며, native CPU dedup/FPS/
  small-graph 처리를 거친다. 끝의 `.to('cuda')`는 최종 tensor의 위치를 옮긴다.
- Sampling 전 robot filter/wrist gate는 이번 점검 범위에서 변경하지 않았다.

따라서 이후 GPU화를 검토할 첫 경계는 `numpy_observations` 이후의 CPU
records/dedup/FPS/Graph 소비자다. 작은 metadata의 host 검증/행렬 준비와 GPU
동기화까지 모두 없앤 경로라고도 말할 수 없다.

## 확인한 코드

- [기본 Tensor packet builder 연결](../../graph/plan_a_bc/geometry.py#L104)
- [GPU packet·관측·검사 뒤 CPU 반환](../../graph/plan_a_geometry/core.py#L191)
- [Packet 변환·주 배열의 CUDA 처리](../../visual/runtime/tensor_packet.py#L48)
- [Packet calibration의 host 검증](../../visual/runtime/tensor_packet.py#L151)
- [Tensor 관측 구성](../../graph/persistent_geometry/tensor_fusion.py#L174)
- [명시적 NumPy 관측 readback](../../graph/persistent_geometry/tensor_fusion.py#L237)
- [실제 packed D2H](../../visual/runtime/tensor_packet.py#L38)
- [CPU records 구성](../../graph/persistent_geometry/map.py#L301)
- [CPU dedup/FPS](../../graph/plan_a_geometry/core.py#L96)
- [CPU Graph 생성·반환](../../graph/persistent_geometry/graph.py#L100)
- [모델 device로 재전송하는 rollout](../../graph/plan_a_bc/rollout.py#L120)
- [Resident prototype의 CPU 경계](../../graph/experiments/cuda_ingress_10i_20261103_v1/pipeline.py#L75)

중단 전에 추가한 observation-chain benchmark는 아직 Vast에서 실행하지 않았다.
해당 수치는 새로 측정되었다고 주장하지 않는다. 이번 점검은 소스 경로 확인이며,
이전 GPU 구현 검증 결과는 [직전 보고서](10i_view_consistency_cuda_20261103.md)에 있다.
