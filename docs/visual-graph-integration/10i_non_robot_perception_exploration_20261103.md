# Task 10-I: Terra·Peg 완전 가시·512 조건의 퍼셉션 최적화 사전 탐색

2026-11-03 KST. **로컬 코드·저장 결과 탐색 완료. 구현·새 GPU 측정·학습은 수행하지 않았다.**

## 이번 작업의 전제

사용자의 후속 설명을 반영해 다음 조건을 기준으로 판단한다.

- 초기 발견에는 **GPT Terra**, 이후 Mask 생성·관측 갱신에는 SAM2를 사용할 예정이다.
- 학습·평가 데이터는 **Peg가 완전히 보이는 경우**로 선별할 예정이다.
- 입력은 **512 수준의 native 해상도**를 우선한다.
- **로봇 필터링 로직은 다른 작업의 담당이다. 이 작업에서 구현·변경·최적화하지 않는다.**
- 이번 요청은 탐색이다. 새 데이터 수집, 모델/API 실행, 기존 중단 실험 재개 또는 최적화 배포를 수행하지 않는다.

따라서 과거 DINO·128·부분 가시 조건의 정책 성공률과 기하 오차를 새 구성의 성능 근거로 직접 대입하지 않는다. 과거 실패는 검사할 항목을 찾는 참고이며 새 시스템의 실패 예측이나 성능 상한이 아니다. 새 분포에서 성공률을 판단할 자료는 아직 확보하지 않았다.

최적화의 목적은 새 조건에서 유용한 관측 정보를 유지하면서 처리 비용을 줄이는 것이다. 현재 Graph의 점 수·Feature·연결 정책을 최종 설계로 고정하지 않는다. 계산만 바꾸는 후보와 관측·선택 의미를 바꾸는 후보는 각각 다른 검증이 필요하다.

## 확인한 실제 구현과 중복 작업 방지

출발점은 최초 `fast_v5_tree`보다 최신인 독립 실험의 **Stage 9**다.

| 위치 | 현재 역할 |
| --- | --- |
| [candidates.py](../../graph/experiments/graph_latency_10i_20261103_v2/candidates.py) | Stage별 최적화 연결과 실행 범위 |
| [dense.py](../../graph/experiments/graph_latency_10i_20261103_v2/dense.py) | GPU PCA5·기하 특징·인접 픽셀 endpoint map |
| [selection.py](../../graph/experiments/graph_latency_10i_20261103_v2/selection.py) | 작은 후보 집합의 단일 block FPS, 큰 집합의 기존 GPU FPS |
| [native.py](../../graph/experiments/graph_latency_10i_20261103_v2/native.py) | C++ 중복 제거·voxel 대표점·최종 FPS의 Python 연결 |
| [packing.py](../../graph/experiments/graph_latency_10i_20261103_v2/packing.py) | SF 전용 표현 구성, 패킷 구성, Peg–Box 부분 정렬 |
| [gpu_graph.py](../../graph/experiments/resolution_perception_graph_20261103_v1/gpu_graph.py) | 손목 반경 연결성·local support 가속 |
| [core.py](../../graph/plan_a_geometry/core.py) | 원래 손목 gate·표현·선택 의미 |
| [geometry.py](../../graph/plan_a_bc/geometry.py) | 현재 두 카메라 builder와 최종 Graph 검증 |

이미 반영된 항목은 GPU PCA5/특징, 전체 pair 목록 대신 endpoint map, native 3mm greedy dedup, native voxel 대표점·최종 FPS, 요청하지 않은 Hybrid 작업 생략, 일부 패킷 복사 감소, 작은 최종 엣지의 CPU 일괄 처리다. 이를 새 개선안으로 중복 제안하지 않는다. Stage 9도 CPU 입력에서 CPU Graph를 반환하는 혼합 경로다.

저장된 최종 측정 config와 현재 Stage 9 코드 10개 파일의 SHA가 모두 일치했다. 연결된 핵심 공통 코드 및 진단 adapter까지 포함해 16개 파일의 identity를 대조했다. 이 대조는 로컬 파일과 저장 측정 identity의 비교이며 원격 현재 상태 확인은 아니다.

## 512 기록이 제공하는 근거와 한계

저장된 단일 장면 `10i_s1_0686_t000`의 Stage 9 Graph 시간은 워밍업 3회 후 15회 중앙값 **163.408ms**, 범위 **152.400–170.590ms**였다. 최초 후보의 같은 실행 내 중앙값은 344.045ms였다.

이 값은 **기존 전체 builder의 공학적 참고치**다. 별도 담당인 로봇 필터 비용도 포함되어 있어 이번 범위의 소요 시간으로 보고하지 않는다. 별도 진단 span을 이 중앙값에서 빼서 나머지 시간을 만들 수도 없다. SAM2·입력 파일 읽기·별도 FK·정책 추론도 이 timer에 포함되지 않는다.

또한 해당 실제 Mask의 원본 protocol은 Terra **reasoning low**다. 로컬 Terra 채택 문서의 **high** 설정 및 앞으로 선별할 Peg 완전 가시 데이터와 같은 조건으로 인증된 표본이 아니다. 이미 Terra 기반 Mask라는 점과 앞으로 사용할 정확한 입력 분포가 같다는 주장을 구별한다.

같은 장면의 **512 계측 1회**는 전체 197.246ms였고, 참고 span은 다음과 같았다.

| 이번 탐색 범위의 구간 | 관측 exclusive wall time |
| --- | ---: |
| 기하·PCA5·특징·endpoint 준비 | 33.792ms |
| 손목 gate의 내부 나머지 | 23.085ms |
| Shape-First 준비 / 선택 | 7.110 / 10.854ms |
| 표현 구성의 미분리 나머지 | 17.568ms |
| 최종 128점 budget | 0.864ms |
| 최종 surface / robot–object / Peg–Box 엣지 | 3.390 / 0.212 / 0.598ms |

`robot–object`는 최종 Graph의 연결 종류 이름이며 로봇 영역 필터링이 아니다. 위 표는 일부 구간만 보인 진단이며 항목 합계가 전체의 정확한 시간 예산은 아니다. 새 분포의 병목 순위나 CUDA 커널 자체 소요 시간을 확정하지 않는다.

근거: [원본 반복 결과](../../outputs/task10i/graph_latency_20261103_v2/final_paired_t4/results.json), [512 trace](../../outputs/task10i/graph_latency_20261103_v2/final_paired_t4/trace_10i_s1_0686_t000_512_s9.json), [이번 범위 재정의·readback](../../outputs/task10i/non_robot_perception_exploration_20261103_v1/target_scope_512.json).

## 먼저 보완할 계측 경계

1024 trace의 약191ms를 순수 Python 융합 또는 순수 중복 제거 시간으로 읽을 수 없는 이유를 코드에서 확인했다.

1. Stage 9 `packing.sf_represent`는 `native.deduplicate`를 직접 호출한다. 기존 profiler가 감싼 `core.deduplicate`를 지나지 않아 native dedup 시간이 독립 span으로 나오지 않는다.
2. `Spans.hook`의 `after` callback은 해당 child span이 끝난 뒤 실행된다. callback 비용이 상위 span의 exclusive 나머지에 들어갈 수 있다.
3. 진단의 packet/cross-view callback은 각각 추가로 `current_records`를 만든다. 이때 진단 adapter가 import한 별도 alias를 사용하므로 실제 생산 경로의 `core.current_records` span과 구분되지 않은 잔여 비용이 생긴다.
4. Stage 9 `dense.extract`는 자체 alias의 `build_geometry`, `fused_normals`, `fused_features`를 호출한다. 기존 `sampling` 함수용 hook만으로는 이 내부가 분리되지 않는다.

따라서 첫 후속 작업은 실제 호출 지점의 **배열 생성·변환·native dedup·전송·GPU 계산**을 나누는 가벼운 계측이다. timed 구간 안에서 진단용 point record와 dense stage snapshot을 추가 생산하지 않는 조건을 우선한다. 전체 wall time과 GPU event 시간은 별도로 읽으며 비동기 실행 시간을 단순 합산하지 않는다.

근거: [진단 adapter](../../graph/experiments/resolution_perception_graph_20261103_v1/graph_adapter.py), [SF 실제 호출](../../graph/experiments/graph_latency_10i_20261103_v2/packing.py).

## 512에서 유력한 추가 탐색 대상

### 1. 전체 특징 배열의 GPU→CPU 반환 축소

`dense.extract`는 카메라마다 두 물체를 순회하고, 각 물체의 전체 해상도 특징을 CPU로 돌려준다. 소스의 명시적 배열 크기를 계산하면 다음과 같다.

- Bool 특징 9개 + 후보 map 7개 + threshold 근접 map 1개: 픽셀당17byte.
- Normal3 + depth jump + normal variation, float32 5채널: 픽셀당20byte.
- Neighbor count, uint8 1채널: 픽셀당1byte.
- 합계 **38byte × 512² × 두 물체 × 두 카메라 = 38MiB/frame**.

명시적 `.cpu()` 호출은 물체당3개, 일반적인 두 카메라·두 물체 경로에서12개다. **이는 코드상 payload 계산이며 측정된 PCIe traffic·대역폭·지연은 아니다.** 내부 복사, 다른 단계와 fallback 비용도 포함하지 않는다.

현재 손목 Box가 비활성화된 빈 support여도 해당 물체의 전체 특징 생성·반환 흐름은 남는다. 빈 물체의 의미를 유지하면서 불필요한 계산과 실체 배열을 줄이는 경로, 선택에 필요한 map과 최종 선택점의 Feature만 반환하는 경로를 먼저 검토할 가치가 있다.

단, Normal/방향 변화가 Shape-First 후보 선정에 쓰이므로 **최종 선택점에서만 Normal을 계산하면 동일한 선택이 보장되지 않는다.** 선택에 필요한 증거 계산과 최종 Feature 전달을 구분해야 한다. GPU 내부 계산을 유지하면서 반환량을 줄이는 후보부터 검증한다. 전체 GPU 이동이 CPU contour/작은 Graph 단계보다 유리한지는 별도 측정한다.

### 2. 손목 gate의 역투영·연결성·support·view 검사

현재 gate는 mask의 유효 pixel을 사용하지만 `camera.world_points()`에서 전체 이미지의 camera/world XYZ를 만든 뒤 해당 pixel을 고른다. 512에서도 필요한 source pixel만 원래 수식으로 역투영하는 후보를 검토할 수 있다.

연결성은 이미 cell 기반 GPU 경로를 사용한다. 아직 cell 차원의 host 변환, support 반환, label 수렴의 Python 판정, 최종 label 반환과 일부 CPU 이웃 검색이 남는다. 새 계측은 이 부분과 cross-view projection을 구분해야 한다. 기존 CPU 연결성 시제품을 GPU 후보보다 빠르다고 가정하지 않는다.

이 후보는 원래 반경, 최대 성분 tie 선택, local support 판정, source pixel과 gate 결과를 보존하는 계산 최적화로 먼저 비교한다. Gate 제거·완화는 이번 탐색의 결론이 아니다.

### 3. 점별 Python 레코드와 좌표·패킷 왕복

`current_records(fuse_views=False)`는 선택점마다 dictionary, source 문자열·tuple, 여러 작은 배열 복사와 상태 필드를 만든다. native dedup이 있어도 Python 정렬·array 추출·출처 병합·반환 record 구성은 남는다. 또한 현재 경로에는 camera→EE→world→최종 EE 변환과 패킷 검증·복사가 존재한다.

내부 계산을 연속 배열과 정수 source index로 유지하고 필요한 경계에서 record를 구성하는 후보가 있다. 다만 **현재 source identity·정렬·대표점·병합 출처·반환값을 유지할 수 있는지**가 검증 대상이다. 좌표 변환을 합치면 중간 float32 반올림이 달라질 수 있어, 계산상 같은 수식이라는 이유로 tensor 동일성을 가정하지 않는다. 최종 공통 인터페이스를 이번에 새로 정하지 않는다.

### 4. Shape-First 준비와 중간 선택

Stage 9의 native voxel·GPU FPS 이후에도 contour walk, canonical 순서, `np.unique`/`np.isin`, metadata 추출, 여러 packet용 배열이 남는다. 세부 계측으로 contour·voxel·초기 anchor·surface FPS·Feature join을 구분한다.

512 및 Peg 완전 가시에서는 잘린 mask보다 경계와 유효 support가 더 클 수 있다. 최종128점만 보고 중간 후보 수가 작다고 가정하지 않는다. 현재 metric voxel scale도 `z/focal length`에 의존하므로 해상도가 바뀔 때 물리적 후보 밀도가 그대로 유지되는 구조는 아니다.

현재 silhouette 후보는 `cv2.RETR_EXTERNAL`로 수집한다. 내부 구멍 경계는 이 silhouette 경로에서 직접 수집하지 않는다. 새 데이터의 Hole 주변 정보가 충분한지 검사할 때 PCA 후보와 silhouette 후보를 구별할 필요가 있다. 이것만으로 현재 Hole 정보 손실의 원인을 확정하거나 새 sampling 정책을 채택하지 않는다.

### 5. 최종 FPS·엣지는 후순위

현재 후보는 최종 budget과 작은 Graph 엣지도 이미 개선했다. 512 단일 trace에서 이들은 앞 구간보다 작았다. 먼저 새 분포의 상위 비용을 줄이고, 다시 계측한 결과에 따라 순서를 바꾸는 편이 합리적이다. 엣지의 계산 비용이 작다는 사실은 그 연결이 작업에 충분하다는 증거와 별개다.

## 새 조건에서 품질을 판단할 항목

Peg 완전 가시와 512는 이전 부분 가시·저해상도 실험과 다른 출발점이다. 새 입력에서 다음을 확인해야 한다.

| 확인 대상 | 필요한 구분 |
| --- | --- |
| Terra→SAM2 | 올바른 초기 대상, 실제 Mask 정확도, 시간에 따른 추적 유지 |
| Peg 관측 | 전체 길이·양 끝·축 방향을 지지하는 source가 선택 전후에 남는지 |
| Box/Hole 관측 | Peg 가시성과 별개로 입구·경계·접근 방향을 지지하는 관측이 있는지 |
| 점 선택·중복 제거 | 정상 표면·작업에 필요한 부분의 보존과 오점의 예산 점유를 별도로 평가 |
| 고해상도 기하 | 5×5 pixel patch의 물리적 크기, Normal/threshold 판정이 새 입력에서 안정적인지 |
| Graph 이후 | 관측 정보가 downstream 추정·action에 도움이 되는지 별도 확인 |

단순히 Mask IoU, 전체 surface coverage 또는 노드 수 하나가 좋아졌다는 이유로 통합 성공을 판단하지 않는다. 반대로 과거 고정 추정기의 실패를 새 조건의 정보 부족으로 단정하지 않는다.

데이터 선정 시에는 ‘Peg 완전 가시’가 초기 관측 기준인지 사용 frame/episode 전체 기준인지, 한 카메라 기준인지 결합 관측 기준인지를 실행 전에 고정해야 한다. 에피소드에서 frame을 제거할 경우 시간 간격·action 대응·recurrent sequence의 연속성을 보존하는지도 확인한다. 이번 탐색에서 선정 규칙이나 데이터 구조를 확정하지 않았다.

## 권장 후속 순서와 검증 경계

1. **새 입력 조건 고정:** 실제 사용할 Terra/SAM2 설정, native512, Peg 가시성 선정 조건을 기록하고 대표 입력을 동결한다. 과거 timing scene과 새 분포를 구분한다.
2. **512 계측 보완:** 이번 범위의 실제 호출 지점과 중간 점 수·전송량·반환량을 기록한다. 진단용 중복 record 생성을 timer에서 분리한다. 로봇 필터 구현은 별도 담당으로 둔다.
3. **반환량·빈 support·gate·record 비용부터 후보 비교:** 한 번에 한 계산 변경을 넣어 출력 보존과 시간을 확인한다. 기존 native/GPU 개선은 재사용한다.
4. **품질 변경은 별도 비교:** 점 수·선택 정책·정밀도·연결 의미를 바꾸는 경우에는 새 입력의 작업 관련 source와 downstream 결과를 평가한다. 기존 tensor 동일성 검사만으로 개선을 인정하지 않는다.
5. **채택 판단:** 같은 입력의 비교와 별도 검증 자료를 구분하고, 지연 감소와 품질 변화 양쪽으로 판단한다. 과거 정책 실험의 성공률과 직접 전후 비교하지 않는다.

계산 보존 후보는 source/병합 목록, gate 결과, 선택 위치, 이산 node/edge 구조 및 Feature validity를 확인한다. 기존 GPU Feature 허용치가 있더라도 그 허용치만으로 정책 출력 동등성이나 새 분포의 성공률을 입증한 것은 아니다.

이번 문서는 최적화 순서에 대한 탐색 결과다. 속도 개선율·512 실시간 처리·새 통합 모델의 성공률을 새로 검증했다고 주장하지 않는다.

## 수행한 검증과 변경 범위

- 저장 최종 반복 결과120행, 512/1024 진단 trace 및 source hash를 로컬에서 재독출했다. [전체 readback](../../outputs/task10i/non_robot_perception_exploration_20261103_v1/readback.json)은 과거 자료의 확인 기록이며 새 목표 분포의 평가가 아니다.
- 기존 CPU 계약을 실행했다: `test_optimization`과 `test_radius_components` 합계 **38개 중28 PASS, CUDA10 SKIP, 실패0**. 기존 계약 묶음의 결과이며 새 최적화나 새 데이터의 검증이 아니다.
- 로컬 C++ compiler로 기존 native helper를 임시 경로에 빌드했다. 패키지 설치는 없었다. SAPIEN import 경고는 있었지만 simulator를 생성·실행하지 않았다.
- 이번 호스트에서 CUDA/MPS, SAM2/Terra 추론, simulator/renderer, 새 Graph latency benchmark, 학습·rollout은 실행하지 않았다. 원격에 접속하거나 중단된 프로세스를 재개하지 않았다.
- 런타임·공통 simulator·로봇 필터·모델·데이터·checkpoint 코드는 변경하지 않았다. 이 보고서와 별도 로컬 readback만 생성했다. 기존 dirty 작업은 유지했다.
- 기존 `ExperimentLogger`의 측정 산출물을 읽었고 새로운 학습/평가 run ID는 발급하지 않았다. W&B/HF 게시·Git commit은 수행하지 않았다.
