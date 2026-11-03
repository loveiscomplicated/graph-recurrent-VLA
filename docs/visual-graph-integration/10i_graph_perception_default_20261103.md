# Task 10-I — Graph 기본 역투영·기하 특징 경로 연결

2026-11-03. 요청한 기본 경로 연결과 실제 CUDA 출력 검증을 완료했다.
새 속도 측정, 학습·평가, 모델 성공률 측정은 하지 않았다.

## 적용 범위

현재 checkout의 `graph.plan_a_bc.geometry.CurrentGeometry(assets, schema)`는
기본적으로 다음 경로를 사용한다.

1. `visual.runtime.torch_projection.project`: Torch CUDA에서 mask ∩ finite
   positive depth ∩ non-overlap pixel만 역투영한다. 원본 pixel 중심 규약과
   FP64 계산 → FP32 XYZ를 유지한다. 물체별 raster 순서의 `[N,3]` 점과
   `[N]` source pixel/object ID를 반환한다.
2. `visual.runtime.graph_perception.process_cuda`: 위 compact XYZ를 CUDA에
   유지한 채 `triton_fused` PCA5 법선과 기하 특징 계산에 직접 전달한다.
   기존 Triton 커널에 compact 입력 분기를 추가했으며, 기존 dense 분기도
   보존했다. 법선·특징 출력 역시 이 단계에서는 compact CUDA tensor다.
3. `to_cpu_evidence`: 특징 계산 뒤 기존 CPU Shape-First·packet 소비자에
   전달한다. compact XYZ와 source index를 보존하는
   `CompactOrganizedGeometry(xyz=None)`를 사용한다. feature map은 기존
   인터페이스와 호환되도록 CPU에서 raster로 배치한다.

이 두 GPU 단계 사이에는 CPU 반환이나 `[H,W,3]` XYZ 재구성이 없다.
원본 2D depth/mask, GPU raster→compact row 정수 map은 유지한다. 특징 계산
뒤의 sampling·융합·최종 FPS·Graph 조립은 여전히 CPU다. CUDA calibration
검사와 dynamic compaction, 퇴화 PCA의 guarded Torch `eigh` fallback에 따른
동기화는 남아 있다. GPU 경로의 새 latency 수치는 이 작업에서 측정하지 않았다.

`CurrentGeometry(..., perception_backend='cpu')`는 명시적인 CPU 진단 경로다.
CUDA가 없을 때 기본 경로가 조용히 CPU로 전환하지 않고 오류를 낸다.
Standalone `PlanA`는 `packet_builder=None`일 때 과거 경로를 유지한다.
`CurrentGeometry`가 instance별 builder를 주입하므로 전역 monkeypatch가
필요 없다. Stage9의 최적화된 `sf_represent`도 이 instance builder를 따른다.
이미 실행된 frozen source·cache·checkpoint는 소급 변경되지 않는다.

새 Graph 전용 packet builder는 최종 Graph에서 사용하지 않는 중간 Raw EDE를
생성하지 않는다. 최종 Graph 엣지 생성 정책과 출력은 유지된다. 일반
`graph.persistent_geometry.fusion.make_packet` 등 과거 API는 남아 있다.

로봇 필터링, 손목 gate, mask 생성·grounding/tracking 정책은 변경하지 않았다.
따라서 위의 compact XYZ 설명은 새 역투영·법선·특징 경로에 대한 설명이며,
별도 필터/gate 내부에서 기존 dense 좌표 계산이 없어졌다는 뜻이 아니다.

## 검증

### 로컬 macOS

저장소 `.venv`로 관련 기존·신규 계약 테스트를 실행했다.

| 묶음 | 결과 |
| --- | --- |
| `test_graph_perception`, `test_masked_backprojection`, `test_pca_optimized`, `test_sampling_geometry`, `test_vision_features`, `graph.plan_a_geometry.tests` | 47개 중 41 pass, CUDA 6 skip |
| `graph.plan_a_bc.tests`, 기존 역투영 `test_projection` | 13개 중 11 pass, CUDA 2 skip |

로컬에서는 CUDA/Triton을 실행하지 않았다. 아래 실제 원격 실행과 구분한다.
Simulator 생성·rollout은 실행하지 않았다.

### VastAI 58752018 실제 CUDA

`CONTAINER_ID=58752018`, RTX 3090, Torch `2.14.1+cu130`, Triton `3.8.0`의
기존 환경을 사용했다. 패키지 설치·driver/service 변경·다른 job 중지는 없었다.

`visual.runtime.test.test_graph_perception`와 `test_pca_optimized`:
**19개 전부 pass, skip/failure 0**. 빈 mask, overlap, 무효 깊이, 가는 구조,
퇴화 PCA, 깊이 단절, 경계 등을 포함한다. Dense/compact feature 일치,
CPU packet 대비 source/feature 일치, default dispatch와 synthetic 최종
Graph 일치를 검증했다. 새 packet 경로에서는 기존 dense XYZ builder를
오류 발생 함수로 대체해 호출되지 않음도 확인했다.

초기 attempt1은 overlay와 frozen 의존성의 `sys.path` 재삽입 때문에 과거
`pca_triton`을 import하여 신규 CUDA 계약 3개가 실패했다. 실행 구성을
독립적인 전체 source copy로 바꾼 attempt2에서 19개가 통과했다. 이 과정에서
frozen 의존성 원본은 수정하지 않았다. 실패 로그도 보존했다.

### 실제 저장된 512×512 입력

`10i_s1_0844_t080` hand camera 관측과 같은 시각의 robot 상태, 기존 RGB
추론 mask를 사용했다. Peg mask 61,056 pixels, Box 0 pixels다.

| 검증 항목 | 결과 |
| --- | --- |
| Mask-only CUDA 역투영 점 | 61,056개; 기존 CPU 전체 역투영에서 같은 source를 모은 점과 exact |
| 기존 dense `triton_fused` 대비 normals/depth jump/normal variation | 전체 feature raster 최대 절대 차이 0 |
| Feature validity·경계·이웃 개수 등 이산 필드 | exact |
| 기존 CPU packet 대비 선택점·object ID·source index·feature validity | exact |
| CPU packet 대비 floating geometric features | `atol=2e-5, rtol=0` 통과, NaN 위치 일치 |
| 중간 선택 packet | 1,105점 |
| 최종 Graph | 131 nodes, 1,006 directed edges |
| 최종 node/edge type, edge index/attr, provenance | exact |
| 최종 `node_input` 최대 절대 차이 | `1.1920928955078125e-07` |

이 최종 Graph 검증은 실제 hand 관측에 **empty base support**를 놓고 동일한
prepared mask/gate를 양쪽 경로에 주입했다. 원본 robot FK·상태는 검증했지만
로봇 필터/gate 실행은 의도적으로 우회했다. 두 실제 카메라의 end-to-end
rollout, 새 grounding, 새 데이터셋의 학습이나 성공률을 검증한 결과는 아니다.
합성 두 물체/두 카메라 Graph 계약 검증은 별도로 통과했다.

## 실행·산출물

공통 `utils.experiment_logger.ExperimentLogger`를 재사용했다.
W&B disabled, artifact publishing 없음. 실제512 실행 UUID:
`6b2a917d4a884a09a2dc126e2698073d`.

- 원격 실행: `/workspace/task10i_graph_perception_default_20261103_v1/attempt2`
- [실제512 검증 JSON](../../outputs/task10i/graph_perception_default_20261103_v1/attempt2/real512/verification.json)
- [CUDA 계약 로그](../../outputs/task10i/graph_perception_default_20261103_v1/attempt2/contracts.log)
- [초기 import 구성 실패 로그](../../outputs/task10i/graph_perception_default_20261103_v1/attempt1_contracts.log)
- [실행 source SHA256 manifest](../../outputs/task10i/graph_perception_default_20261103_v1/attempt2/executed_source_manifest.json)
- [다운로드·source 일치 receipt](../../outputs/task10i/graph_perception_default_20261103_v1/download_receipt.json)
- [검증 실행 코드](../../graph/experiments/graph_perception_default_20261103_v1/verify.py)
- [로컬 추가 계약 로그](../../outputs/task10i/graph_perception_default_20261103_v1/local_extra_contracts.log)

원격 evidence archive의 SHA256을 로컬에서 확인했고, 업로드한 source 59개와
검증 runner의 hash가 현재 구현과 같음을 확인했다. 나머지 의존성도 실행
manifest에 기록했다. Git dirty checkout에서 실행했으므로 commit만으로 실행
source가 재현된다고 주장하지 않는다.
