# Task 10-I 128×128 기존 수치와 Graph 생성 시간 비교

## 확인 결과

과거 100ms 미만 기록은 실제 존재한다. 다만 128×128 입력과 관련해 찾은 수치들은 현재 122.52ms와 같은 작업의 전후 기록이 아니다. 기존 결과를 간과한 채 이전 최적화 보고서에서 이 측정 경계의 차이를 충분히 설명하지 않았다.

| 기록 | 시간 | 범위·주의 |
|---|---:|---|
| Task06, `total_offline_graph_ms` | 중앙값 **7.39ms**, 평균 12.14ms, p95 34.66ms, 473건 | 선택된 단일 카메라의 RGB-D/mask에서 중간 Vision packet을 만드는 시간. 128×128 원본을 사용했고 별도의 두 카메라 robot filter/wrist gate/융합/최종 Graph 생성은 포함하지 않는다. 오래된 CPU 실행의 별도 장면·마스크다. |
| Task06, packet→action | 평균 **6.676ms**, p95 6.756ms | 이미 만들어진 packet부터 Graph 수신과 GNN/GRU action까지. RGB-D 수집, 마스크, 앞의 offline packet 생산을 제외한다. 그 당시 구조는 현 Plan A와 다르다. |
| Visual E2 | point cloud 중앙값 **4.1ms**, tracking 포함 합계 중앙값 44.5ms | 별도 GPU visual streaming 파이프라인으로, 최종 Plan A Graph의 robot filter·wrist gate·두 카메라 융합·Graph21 생성을 포함한 수치가 아니다. 이 표 자체는 입력 128×128 성능을 확정하지 않는다. |
| Task10-I, `fast_v5_tree` | 중앙값 **122.52ms** | 장면 `10i_s1_0686_t000`의 두 카메라 native 128×128과 실제 mask, 1회 워밍업+3회 측정. CPU 입력부터 CPU 최종 Graph 반환까지며 전송 포함, 파일 읽기·별도 FK·SAM2·action은 제외. RTX 3090/AMD EPYC 7543 환경. |

Task06에서는 원본 데이터에 두 카메라가 기록됐지만 packet 생성 코드는 선택된 `camera` 하나의 RGB-D를 사용한다. Task10-I의 `CurrentGeometry.build`는 두 카메라를 동기화하고, robot filter, hand-camera gate, 각 카메라 packet, view 간 검증, 3mm 중복 제거, 물체당 128점 budget, 최종 Graph 생성 및 유효성 검사를 수행한다. 그래서 과거 단일 camera packet 7.39ms에 고정 상수를 더해 현재 122.52ms의 원인별 차이를 계산할 수 없다. 구현, 장면, 마스크, 하드웨어 및 계측 조건도 다르다. Mask 면적·유효점 수와 프레임에 따라서도 작업량이 변한다.

유사한 범위의 과거 단계 10-I Stage4A observer-free 실행에서도 Geometry/Graph 평균 218.8ms가 기록돼 있다. 별도 장면 및 경합·측정 조건의 평균이므로 122.52ms와 정확한 속도비를 주장할 수 없으나, 당시에도 전체 Geometry/Graph가 항상 100ms 이내였다고 보기는 어렵다. 과거 Stage4A-2의 별도 Graph timer CSV 11,690행은 중앙값 351.32ms이며 100ms 미만은 22행이었다. 이는 실험 실행 중 관측값이며 단독 latency benchmark는 아니다.

## 현재 128 경로의 진단

최적화 후보와 같은 128 입력·실제 mask로 GPU 워밍업 후 한 번 실행한 **계측 추가 diagnostic trace**다. 다섯 Graph tensor 해시가 비계측 실행과 같음을 확인했다. trace 총 시간은 123.51ms다. 아래는 각 함수 호출들의 exclusive wall time 합계다. 계측 overhead와 CPU/GPU 동기화가 섞일 수 있고, 표본 1개이므로 정밀한 latency 분포나 인과적 기여도로 해석하지 않는다.

| 구간 | 관측된 exclusive wall time |
|---|---:|
| 3mm 중복 제거 | 23.44ms |
| 로봇 영역 필터 | 15.10ms |
| 표현 구성의 나머지 부분 | 12.28ms |
| Shape-First 준비 / 선택 | 10.77 / 9.65ms |
| 최종 128점 budget | 8.77ms |
| 선택 결과 packet 포장 | 7.17ms |
| 최종 surface edge | 4.36ms |
| 손목 카메라 gate | 4.26ms |

이 표는 중첩 함수의 시간을 exclusive로 분리한 것이며, CUDA 비동기 실행 시 CPU wall이 실제 커널 소요와 같다는 뜻이 아니다. 다른 구간 및 진단 호출 오버헤드가 있으므로 일부 항목의 합을 전체의 정확한 분해로 취급하지 않는다. 현 시점에는 중복 제거, 로봇 필터, sampling/포장, GPU와 CPU 간 경계가 유력한 추가 최적화 대상이라는 관찰까지 가능하다. 각각을 제거했을 때의 시간은 아직 측정하지 않았다.

## 근거

- [Task06 원자료: offline packet 통계](../../experiments/c184b85dadf14db5953f11cdcacb6674/data_pipeline_costs.json), [Task06 보고서](06_graph_gnn_gru_baseline.md), [단일 카메라 packet 생성 코드](../../graph/integration_baseline/vision.py).
- [과거 Visual E2 표](../../visual/results/v1/E2/table.md), [이 범위에 대한 기존 설명](00_search.md).
- [현 두 카메라 builder](../../graph/plan_a_bc/geometry.py), [이전 Task10-I latency](10i_stage4a_final_report.md), [현 후보 latency 원자료](../../outputs/task10i/stage4a2/resolution_perception_graph_20261103_v1/graph_optimization_v1/fast_v5_tree_10i_s1_0686_t000_128.json).
- 과거 Stage4A-2 Graph timer는 `outputs/task10i/stage4a2/increment_v9/task10i_stage4a2_newdev_C_KEEP_20261101_v1/run/paired_graph_BC_KEEP/episodes/*/graph_timing.csv`의 `graph_ms` 11,690행을 로컬 읽기 전용으로 집계했다.

추가 모델/SAM2 측정이나 원래 Task10-I 실험 재개는 하지 않았다. 128 trace는 기존 Graph 후보만 원격 CUDA에서 실행했다.
