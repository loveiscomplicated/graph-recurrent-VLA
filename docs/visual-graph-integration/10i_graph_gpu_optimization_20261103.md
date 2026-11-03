# Task 10-I 그래프 생성 최적화 결과

## 범위와 결론

원래 Task 10-I의 실험은 중단된 상태로 보존하고, 이미 저장된 actual mask와 RGB-D 입력을 사용해 Graph 생성만 최적화했다. SAM2 등 다른 모듈은 이번에 측정하지 않았다. 추가 API 호출, 학습, 제어 rollout, 패키지 설치, 게시를 수행하지 않았다. 공통 simulator 및 기존 Graph 구현은 바꾸지 않았고, 실험 전용 프로토타입을 사용했다.

가장 빠른 후보의 단일 장면 중앙값은 128/512/768/1024 입력에 대해 122.52/352.70/597.69/929.21 ms였다. 원래 CPU 구현보다 각각 약 1.82/8.20/15.32/25.52배 빠르다. 그러나 1024에서 Graph만 약 0.93초로, 센서부터 액션 출력까지 50ms라는 시스템 목표에 도달하지 못했다. 완전한 GPU 상주 구현도 아직 아니다. 이 결과는 배포 가능성과 실시간성을 입증하지 않는다.

## 비교 조건과 결과

Vast.ai RTX 3090 24 GiB, PyTorch 2.14.1+cu130, Triton 3.8.0에서 단독으로 실행했다. 입력은 저장된 동일 장면 `10i_s1_0686_t000`, actual mask repeat 0이다. 4 CPU 스레드, 프로세스 메모리 제한 80 GiB, 해상도별 워밍업 1회 후 3회 측정의 중앙값이다. 워밍업은 CUDA/Triton 초기화와 컴파일 비용을 포함할 수 있으므로 아래에 포함하지 않았다. 측정 경계는 CPU 입력을 받은 Graph builder가 CPU Graph tensor를 반환하는 시간이며, 내부 전송을 포함한다. 파일 읽기·디코딩, 별도 FK 및 probe, SAM2, 모델 추론과 제어는 포함하지 않는다. 세 번의 측정과 한 장면만으로 p95, 최악 지연 또는 다른 CPU/GPU에서의 성능을 추정할 수 없다. 이전 대화의 155/564/1110/1903 ms는 별도 수집 조건의 결과이므로 아래 수치와 직접 전후 비교하지 않는다.

| Graph builder 중앙값 (ms) | 128 | 512 | 768 | 1024 |
|---|---:|---:|---:|---:|
| 기존 CPU | 223.18 | 2,893.31 | 9,159.44 | 23,716.08 |
| 1차 CUDA 하이브리드 | 145.91 | 447.60 | 835.71 | 1,273.38 |
| 사용하지 않는 Raw EDE 생략 | 136.76 | 419.63 | 750.00 | 1,165.84 |
| 추가 GPU PCA5 | 125.57 | 357.40 | 635.52 | 926.13 |
| GPU PCA5 + CPU 공간 인덱스 중복 제거 | **122.52** | **352.70** | **597.69** | **929.21** |
| 비교: GPU 중복 제거, CPU PCA5 | 128.64 | 409.98 | 743.66 | 1,346.73 |

1024에서 공간 인덱스 후보와 GPU PCA5 단독의 약 3ms 차이는 3회 측정의 변동에 비해 작으므로 어느 쪽이 더 빠르다고 단정하지 않는다. GPU 중복 제거는 같은 조건의 Raw EDE 생략 경로보다 1024에서 느려서 채택하지 않았다. 최종 후보 1024의 해당 프로세스 최고 RSS는 약 1.61 GiB, PyTorch가 추적한 GPU 최고 할당량은 약 106.5 MiB다. 두 값은 서로 다른 메모리 지표이며, CUDA 컨텍스트 전체 사용량이나 다른 프로세스의 GPU 메모리를 나타내지 않는다.

## 구현과 검증

1차 하이브리드는 로봇 영역 필터, 손목 카메라의 12mm 연결 성분 및 Shape-First FPS에 CUDA/Triton을 사용한다. 이어서 최종 Graph의 관측점 구성에서 소비하지 않는 Raw EDE 중간 계산을 생략하고, 저장소의 기존 GPU PCA5 normal 구현을 연결했다. 최종 후보는 기존 3mm 거리, 안정적인 대표점 순서 및 원본 observation ID 합치기 규칙을 유지하면서 SciPy 공간 인덱스로 중복 후보를 찾는다. 후보 쌍이 과도하게 많으면 기존 중복 제거로 돌아간다. 전체 경로는 여전히 CPU 기반의 입력 준비, 일부 특징/픽셀 인접성, 윤곽선·중복 제거, 패킷 구성 및 CPU↔GPU 전송을 포함한다.

기존 CPU와 1차 하이브리드의 다섯 Graph tensor 해시는 속도 비교 네 해상도 및 별도 실제 입력 12조건에서 일치했다. 기존 OOM 한 조건은 메모리 제한을 완화한 bounded-memory CPU 경로와 비교해 다섯 해시가 일치했다. Raw EDE 생략 경로는 속도 비교 네 해상도에서 다섯 해시가 정확히 일치했다.

GPU PCA5 + 공간 인덱스 후보는 속도 비교 네 해상도와 별도 12개 실제 입력 조건에서 node/edge의 이산 구조, 위치, PCA 외 node 입력, edge 속성, 선택 및 병합된 원본 observation ID가 기준과 정확히 일치했다. PCA 관련 node 입력 열의 최대 절대 차이는 별도 12조건 전체에서 `1.862645149230957e-09`이며, 저장소 기존 PCA 최적화 테스트의 절대 허용치 `2e-5` 이내다. 따라서 이 후보의 부동소수점 결과를 비트 단위 동일하다고 주장하지 않는다. 이 12조건은 정적/추적/별도 seed 입력을 포함하지만 전체 데이터에 대한 증명은 아니다. GPU 및 공간 인덱스 중복 제거는 합성 중복점·경계점·여러 물체 조건을 기존 구현과 비교하는 테스트도 통과했다. 기존 공유 인프라 테스트 42개, 작업 전용 CUDA 테스트 및 기존 GPU PCA 계약 테스트를 실행했다. 현재 Mac 호스트에서는 CUDA 경로를 실행하지 않았다.

이전 1024 trace의 13개 조건에서 PCA5 normals, robot filter, 3mm 중복 제거, 픽셀 pair evidence, wrist gate 등이 큰 비용이었다. 이 trace는 최종 후보의 사후 프로파일이 아니며 항목별 중앙값을 합산할 수 없다. 남은 큰 간격을 줄이려면 GPU에 입력부터 최종 Graph까지 데이터를 유지하는 경로를 설계하고 CPU 기반 dense feature/인접성 및 윤곽선·패킷 작업, 동기화와 전송을 다시 살펴야 한다. 그 구현과 50ms 달성은 이번 결과로 검증되지 않았다.

## 산출물과 중단 상태

- [실험 코드](../../graph/experiments/resolution_perception_graph_20261103_v1/fast_graph.py), [측정 코드](../../graph/experiments/resolution_perception_graph_20261103_v1/optimization_benchmark.py), [다양한 입력 검증](../../graph/experiments/resolution_perception_graph_20261103_v1/validate_fast_graph.py).
- [후보 1024 원자료](../../outputs/task10i/stage4a2/resolution_perception_graph_20261103_v1/graph_optimization_v1/fast_v5_tree_10i_s1_0686_t000_1024.json), [12조건 검증 원자료](../../outputs/task10i/stage4a2/resolution_perception_graph_20261103_v1/graph_optimization_v1/fast_v5_diverse_validation.json), [1차 CUDA 검증](../../outputs/task10i/stage4a2/resolution_perception_graph_20261103_v1/graph_optimization_v1/pre_adoption_validation.json). 동일 디렉터리에 나머지 해상도와 비교 후보별 JSON이 있다.
- 서버 실험 디렉터리: `/workspace/task10i_resolution_perception_graph_20261103_v1/graph_optimization_v1`; `source_versions/fast_v5_tree/`에 측정 당시 소스가 보존되어 있다. 측정 이후 로컬 benchmark 코드에는 실패 결과를 성공으로 표기하지 않도록 상태 처리만 보완했다. 기존 `scope.json`의 원래 실험 재개 문구는 초기 계획의 기록이며 이번 사용자 지시에 따른 실행 범위가 아니다.
- 원래 실험 프로세스와 이전 최적화 실험 프로세스는 정지 상태로 유지했다. 정지된 이전 프로세스에는 시간 제한 알람이 대기 중일 수 있으므로 무작정 재개해서는 안 된다. 최적화 보고 후 원래 태스크를 재개하지 않았다.
