# Repository cleanup — 현재 helper를 과거 실험에서 분리

Task ID: `repo_cleanup_helpers_20261104_v1` · 2026-11-04 KST.

## 결과

현재 train·offline 평가·live·기존 cache reader/generator와 Graph 뷰어가 사용하는
helper를 현재 모듈로 이관했다. 기본 실행에서 과거 실험·진단·baseline package를 역으로
불러오지 않는다. 이전 경로는 같은 구현의 alias 또는 re-export를 제공하고, 과거 CLI·
실험 설정·보고서·산출물은 유지했다. 새 모델·feature·Graph 정책을 결정하지 않았다.

- 구현 commit: **`835a12c3f801e6eabfbf530005a5241bc1d548e1`**.
- GPU 검사 fixture 보정 commit: **`77d767202690d7bcc9ae6b568f5167c03d53e212`**.
- 변경 전 기준: `12d7cc74e3ee7c8da8480dad67b10186f4b6a51a`.
- Branch: `codex/cleanup-modular-live-20261104`. Push·HF 게시·데이터 수집은 하지 않았다.

## 변경 전 상태와 보존

처음 확인한 추적 소스는 모두 커밋된 상태였다. 기존 backup `d222fe2`와 모듈화/CUDA
검증 이력을 유지한 채 위 기준 commit에서 시작했다. 작업 도중 다른 채팅의 미완성
보고서가 untracked로 추가됐다. 해당 문서는 수정·stage·commit하지 않았고, 최초 관측본은
[별도 로컬 snapshot](../../outputs/repo_cleanup_helpers_20261104_v1/preexisting_changes.json)으로 기록했다.

이번 작업은 활성 코드의 이관이다. Git 밖 데이터·checkpoint·실험 결과·영상의 원격
보존/복원 검증 완료를 선언하거나, 과거 실험 package를 일괄 삭제하는 작업이 아니다.
기존 ignored 산출물이나 원격 frozen source를 덮어쓰지 않았다.

## 실제 소유 위치

| 위치 | 현재 책임 |
| --- | --- |
| `graph/current.py` | 교체 가능한 현재 Graph builder facade·FK·검사 |
| `graph/geometry/` | camera/surface 계약, CPU filtering·gate, records·기하·CPU/CUDA Graph |
| `graph/perception/` | RGB-D projection·PCA·Shape-First·voxel/FPS·packet·비교용 계산 함수 |
| `graph/backends/` | CUDA robot filter와 사전 준비·native C++·선택 resident backend |
| `vla/perception/` | Terra·명시적 DINO 대안·초기화·SAM2·online adapter·관측 프레임 |
| `vla/data/`, `vla/action_codec.py` | robot schema·공통 통계·packet/JSON/NPZ·action 표현 |
| `vla/workflows/` | 기존 조정 코드와 RNG·physics/controller audit·기록 prefix 검사 |

전체 실험 파일을 새 runtime으로 복사하지 않았다. 큰 실험/진단 파일에서는 사용되는
숫자 계산·I/O·audit 함수만 추출하고 launcher·plot·GT 평가·보고서 생성은 이전 파일에
유지했다. 예를 들어 pointcloud의 SOR/FPS/extract만 채택하고 GT 비교·그림·CLI는
`visual/scripts/pointcloud.py`에 남겼다. Native C++의 인접 source 파일도 함께 이관했다.

정규화는 `StateStatistics(mean,std,as_dict)`를 공통부에 두고, 과거 GT feature 전용
`StateNormalizer.fit/apply`는 기존 prototype subclass에 유지했다. 기본 경로가 GT feature
구현을 import하지 않으면서 이전 API의 생성자·통계·fit/apply를 유지한다.

## 의존성 확인

조사 root는 `tools.train`, `tools.rollout`, `tools.evaluate`, `vla.data.graph_cache`,
`visualization.server`다. 조건부 import도 포함하는 정적 상한이며 프레임마다 실행되는
모듈 수나 실행 시간으로 해석하지 않는다.

- 전체 정적 closure: **152 → 120개 module**.
- Legacy 범위: **103 → 0개**. 아래 prefix 범위를 기준으로 재계산했다.
- 금지한 범위: `graph.experiments`, `plan_a_geometry`, `plan_a_bc`, `persistent_geometry`,
  `integration_baseline`, `graph.prototype`, `visual.runtime`, `visual.scripts`, `legacy_fps`.
- 새 검사에서 이 package들의 import를 강제로 차단한 별도 process로 tools/cache/viewer와
  기본 CPU builder를 로드했다. 선택 CUDA backend의 소스도 AST로 검사한다.
- 뷰어의 pointcloud helper를 파일 경로로 동적 로드하던 경로도 canonical import로 바꿨다.
  저장된 Task09 seed30002의 t0·t1·t45 Graph를 현재 loader로 실제 로드했다(목록16개).

[이전 closure](../../outputs/repo_cleanup_helpers_20261104_v1/dependencies_before.json),
[현재 closure](../../outputs/repo_cleanup_helpers_20261104_v1/dependencies_after.json),
[의존성 차단 테스트](../../tests/test_helper_boundaries.py),
[실제 저장 Graph 로드](../../outputs/repo_cleanup_helpers_20261104_v1/real_viewer_verification.json).

## 수치·정책과 호환성

- 추출한 **346개 함수/클래스 AST**를 변경 전 소스와 대조했다. Import와 module-qualified
  patch 경로를 정규화하면 일치한다. 새로운 수치 알고리즘이나 threshold를 넣지 않았다.
- Geometry protocol과 native C++는 원본과 바이트가 같다. 기존 schema ID·node/edge·
  World/EE·point cap·Grounding 정책·cache/checkpoint 형식을 이름 정리 때문에 바꾸지 않았다.
- 현재 Graph builder와 GPU robot 준비/SAM2 overlap, full classification/correction,
  wrist gate, CUDA-resident projection/selection/records/Graph 경계를 유지했다.
- E0/E1의 초기 가중치 SHA와 action·hidden 출력·gradient norm의 기존 CPU golden 검사도 통과했다.
- Whole leaf module은 실제 module alias여서 이전 import와 monkeypatch가 동일 namespace를
  사용한다. 부분 추출 함수의 spy/optimization patch는 새 소유 module을 가리키도록 연결했다.
  이전 연구 package의 `__init__` search path는 유지했다.
- SAM2 기본 외부 weight는 계속 `visual/checkpoints`를 가리킨다. 이는 보존한 데이터 위치이며
  기본 runtime의 `visual/runtime` 코드 의존성과 구분한다.
- 공통 simulator·ExperimentLogger·publisher 구현, GNN/controller 구조·학습 예산·split·
  action convention·live 필수 영상 정책은 변경하지 않았다. 과거 source-freeze 검사를
  끄거나 기존 manifest를 새 hash로 재인증하지 않았다.

[346개 대조와 asset hash](../../outputs/repo_cleanup_helpers_20261104_v1/extraction_verification.json),
[파일별 SHA·작업 범위](../../outputs/repo_cleanup_helpers_20261104_v1/implementation_scope.json).

## 검증

로컬은 Python3.11.13/Torch2.14.1/macOS CPU다. 정상 완료한 suite는 다음과 같다.
서로 중복되는 test가 있으므로 아래 숫자를 단순 합산한 고유 test 수로 해석하지 않는다.

| Suite | 결과 |
| --- | --- |
| 현재 모델·Graph/live/filter/gate·Grounding·좌표·뷰어·새 경계 검사 | 83개: 통과75, CUDA8 skip, 실패0 |
| Shape-First·voxel·perception·경계·정규화·pointcloud | 63개: 통과60, CUDA3 skip, 실패0 |
| PlanA·persistent geometry·native C++/최적화 CPU | 61개 모두 통과 |
| 초기화·SAM2 filter callback | 4개 모두 통과, predictor fixture 사용 |
| 공통 logger/artifact | 별도 process에서42개 모두 통과 |
| tools CLI | train/rollout/evaluate `--help` 성공 |

Python1,267개 AST parse와 새 안내 문서의 로컬 링크 검증도 통과했다.
[최종 정적 검증](../../outputs/repo_cleanup_helpers_20261104_v1/final_static_verification.json).

[CPU 요약](../../outputs/repo_cleanup_helpers_20261104_v1/cpu_verification.json),
[현재 경로 최종 로그](../../outputs/repo_cleanup_helpers_20261104_v1/tests_final.log),
[추출 helper](../../outputs/repo_cleanup_helpers_20261104_v1/extracted_helper_tests_second.log),
[기하/native](../../outputs/repo_cleanup_helpers_20261104_v1/geometry_tests.log),
[초기화/추적](../../outputs/repo_cleanup_helpers_20261104_v1/perception_tests.log),
[공통 인프라](../../outputs/repo_cleanup_helpers_20261104_v1/infrastructure_tests.log).

### CUDA 보충 검증

Vast.ai **58752018 / NVIDIA GeForce RTX 3090**의 기존 Python3.12.15/Torch2.14.1+cu130 환경에서
**23개 모두 통과, 실패0·오류0·skip0**였다. 위 CPU suite의 CUDA skip을 해소하고,
GPU contour/voxel/FPS와 packet의 추가12개 검사도 실행했다.

- Source commit: `77d767202690d7bcc9ae6b568f5167c03d53e212`.
- Git source와 진단 runner **2,159개 파일 hash**를 실행 전 검증.
- 최종 run ID: **`2b16e3aff5c04852becf7f7a61394056`**. 기존 `ExperimentLogger` 재사용, W&B disabled.
- 실행 시간: **3.397초**. 이전 compile cache를 재사용한 test 시간이며 프레임 성능 측정이 아니다.
- Robot ingress/correction3개, gate2개, overlap1개, World/EE2개,
  Graph perception3개, CUDA selection12개를 검사했다.
- 손목 gate는 저장29,775점 입력을 재사용했고 기존 accepted mask·trace를 정확 비교했다.
- 최초 확장 실행은22개 통과·1개 fixture 오류였다. 예전 준비 mask stub에 빠진
  `wrist_gate_backend`를 추가하고, CUDA 결과 보존을 확인한 뒤 기존 명시적
  `numpy_graph/numpy_records` export로 비교하도록 검사 경계를 맞췄다. Edge 배열의
  정확 일치와 float 허용오차 기준은 낮추지 않았다. 전체23개를 다시 실행해 통과했다.
- 원격 source/output은 별도 `source_v2`/`results_v2`에 저장해 첫 실패도 보존했다.
  초기/최종 결과 파일16개를 로컬로 회수하고 archive·개별 SHA를 검증했다.
- 종료 후 GPU compute process가 없었다. 기존 인스턴스나 다른 실험을 중지/변경하지 않았다.
- 유료 API·실제 Terra/SAM2 추론·새 live rollout·GPU 학습은 하지 않았다.
  전체 simulator renderer→모델→영상 포함 성능 비교를 수행한 결과로 확대하지 않는다.

[GPU 검증 receipt](../../outputs/repo_cleanup_helpers_20261104_v1/gpu/verification.json),
[최종23개 로그](../../outputs/repo_cleanup_helpers_20261104_v1/gpu/remote/results_v2/tests.log),
[패키지/개별 결과](../../outputs/repo_cleanup_helpers_20261104_v1/gpu/remote/results_v2/summary.json),
[최종 source](../../outputs/repo_cleanup_helpers_20261104_v1/gpu/source_v2.tar.gz),
[진단 runner](../../outputs/repo_cleanup_helpers_20261104_v1/gpu/run_cuda_tests.py),
[초기/최종 결과 archive](../../outputs/repo_cleanup_helpers_20261104_v1/gpu/results_all.tar.gz).

### 별도로 확인한 과거 테스트의 기존 실패

추가 실행한 과거 v2 `test_live_feature_action_controller_consistency`는 CPU expert-prefix
재생에서 `prefix diverged 0.001734226942062378`로 실패했다. 변경 전 commit의
StateNormalizer 구현을 복원한 별도 process에서도 같은 값으로 재현했다. 해당 과거
물리 재생 문제를 이번 helper 이관에서 임의로 고치거나 허용오차를 완화하지 않았다.
이 과정에서 Mac CPU simulator를 실행했으며 현재 주 live의 CUDA renderer 검증과 구분한다.

[초기 추가 suite 로그](../../outputs/repo_cleanup_helpers_20261104_v1/extracted_helper_tests_first.log),
[변경 전 StateNormalizer 재현](../../outputs/repo_cleanup_helpers_20261104_v1/legacy_v2_baseline_probe.log).
정규화와 직접 관련된 기존 StateTests4개와 새 상속/API 검사는 정상 통과했다.

## 파일별 이관표

아래는 실제 구현의 이동이다. 이전 module 파일은 호환 또는 실험 CLI shell로 남는다.

| 이전 module | 현재 구현 | 방식 |
| --- | --- | --- |
| `graph.experiments.cuda_ingress_10i_20261103_v1.consumers` | `graph.backends.resident.consumers` | module alias |
| `graph.experiments.cuda_ingress_10i_20261103_v1.gate` | `graph.backends.resident.gate` | module alias |
| `graph.experiments.cuda_ingress_10i_20261103_v1.ingress` | `graph.backends.resident.ingress` | module alias |
| `graph.experiments.cuda_ingress_10i_20261103_v1.pipeline` | `graph.backends.resident.pipeline` | module alias |
| `graph.experiments.cuda_ingress_10i_20261103_v1.selected_records` | `graph.backends.resident.selected_records` | module alias |
| `graph.experiments.graph_latency_10i_20261103_v2.edges` | `graph.backends.surface_edges` | module alias |
| `graph.experiments.graph_latency_10i_20261103_v2.native` | `graph.backends.native` | module alias |
| `graph.experiments.graph_latency_10i_20261103_v2.robot` | `graph.backends.robot_reference` | module alias |
| `graph.experiments.resolution_perception_graph_20261103_v1.gpu_graph` | `graph.backends.cuda_primitives` | 계산 함수 추출·CLI 유지 |
| `graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter` | `graph.backends.cuda_robot` | module alias |
| `graph.experiments.robot_filter_gpu_io_10i_20261103_v1.preparation` | `graph.backends.robot_preparation` | module alias |
| `graph.integration_baseline.bc08_codec` | `vla.action_codec` | module alias |
| `graph.integration_baseline.direct_train` | `vla.workflows.randomness` | 계산 함수 추출·CLI 유지 |
| `graph.integration_baseline.graph` | `graph.geometry.interactions` | 계산 함수 추출·CLI 유지 |
| `graph.integration_baseline.io` | `vla.data.legacy_io` | 계산 함수 추출·CLI 유지 |
| `graph.integration_baseline.online_vision_adapter` | `vla.perception.online` | module alias |
| `graph.integration_baseline.robot` | `vla.data.robot` | module alias |
| `graph.integration_baseline.rollout_stop_clock` | `vla.workflows.live_checks` | 계산 함수 추출·CLI 유지 |
| `graph.integration_baseline.tensor_online_vision` | `vla.perception.tensor_online` | module alias |
| `graph.integration_baseline.vision` | `graph.perception.offline` | 계산 함수 추출·CLI 유지 |
| `graph.persistent_geometry.contracts` | `graph.geometry.contracts` | module alias |
| `graph.persistent_geometry.coordinates` | `graph.geometry.coordinates` | module alias |
| `graph.persistent_geometry.fusion` | `graph.geometry.fusion` | module alias |
| `graph.persistent_geometry.graph` | `graph.geometry.graph` | module alias |
| `graph.persistent_geometry.map` | `graph.geometry.map` | module alias |
| `graph.persistent_geometry.registration` | `graph.geometry.registration` | module alias |
| `graph.persistent_geometry.self_filter` | `graph.geometry.self_filter` | module alias |
| `graph.persistent_geometry.tensor_camera` | `graph.geometry.tensor_camera` | module alias |
| `graph.persistent_geometry.tensor_fusion` | `graph.geometry.tensor_fusion` | module alias |
| `graph.persistent_geometry.tensor_gate` | `graph.geometry.tensor_gate` | module alias |
| `graph.persistent_geometry.tensor_gate_kernels` | `graph.geometry.tensor_gate_kernels` | module alias |
| `graph.persistent_geometry.tensor_graph` | `graph.geometry.tensor_graph` | module alias |
| `graph.persistent_geometry.tensor_knn` | `graph.geometry.tensor_knn` | module alias |
| `graph.persistent_geometry.tensor_knn_kernels` | `graph.geometry.tensor_knn_kernels` | module alias |
| `graph.persistent_geometry.tensor_records` | `graph.geometry.tensor_records` | module alias |
| `graph.persistent_geometry.tensor_records_kernels` | `graph.geometry.tensor_records_kernels` | module alias |
| `graph.persistent_geometry.wrist_gate_dispatch` | `graph.geometry.wrist_gate_dispatch` | module alias |
| `graph.plan_a_bc.robot_frame` | `graph.backends.robot_frame` | module alias |
| `graph.plan_a_bc.tensor_ingress` | `graph.backends.tensor_ingress` | module alias |
| `graph.plan_a_geometry.core` | `graph.geometry.core` | module alias |
| `legacy_fps.adapter` | `graph.perception.legacy_points` | module alias |
| `visual.runtime.coordinate_frames` | `graph.perception.coordinate_frames` | module alias |
| `visual.runtime.cuda_sampling` | `graph.perception.cuda_sampling` | module alias |
| `visual.runtime.cuda_sampling_kernels` | `graph.perception.cuda_sampling_kernels` | module alias |
| `visual.runtime.edge_evidence` | `graph.perception.edge_evidence` | module alias |
| `visual.runtime.extended_delaunay_edges` | `graph.perception.extended_delaunay_edges` | module alias |
| `visual.runtime.geometry` | `graph.perception.geometry` | module alias |
| `visual.runtime.graph_perception` | `graph.perception.graph_perception` | module alias |
| `visual.runtime.grounding` | `vla.perception.dino` | module alias |
| `visual.runtime.initialization` | `vla.perception.initialization` | module alias |
| `visual.runtime.intra_object_edges` | `graph.perception.intra_object_edges` | module alias |
| `visual.runtime.live_grounding` | `vla.perception.live_grounding` | module alias |
| `visual.runtime.normal_estimators` | `graph.perception.normal_estimators` | module alias |
| `visual.runtime.object_groups` | `graph.perception.object_groups` | module alias |
| `visual.runtime.pca` | `graph.perception.pca` | module alias |
| `visual.runtime.rgbd` | `vla.perception.rgbd` | module alias |
| `visual.runtime.sampling` | `graph.perception.sampling` | module alias |
| `visual.runtime.tensor_packet` | `graph.perception.tensor_packet` | module alias |
| `visual.runtime.test.pca_triton` | `graph.perception.pca_triton` | module alias |
| `visual.runtime.test.raw_ede_comparison_adapter` | `graph.perception.raw_ede` | module alias |
| `visual.runtime.test.run_variable_rate_shape_first` | `graph.perception.shape_first` | 계산 함수 추출·CLI 유지 |
| `visual.runtime.test.run_vision_feature_edge` | `vla.data.packet_io` | 계산 함수 추출·CLI 유지 |
| `visual.runtime.test.run_voxel_structural_sampling` | `graph.perception.voxel` | 계산 함수 추출·CLI 유지 |
| `visual.runtime.torch_projection` | `graph.perception.torch_projection` | module alias |
| `visual.runtime.tracking` | `vla.perception.tracking` | module alias |
| `visual.runtime.triangle_intra_object_edges` | `graph.perception.triangle_intra_object_edges` | module alias |
| `visual.runtime.vision_packet` | `graph.perception.vision_packet` | module alias |
| `visual.scripts.pointcloud` | `graph.perception.pointcloud` | 계산 함수 추출·CLI 유지 |

추가로 `feature_rows`는 `graph.perception.vision_packet`, packet hash는
`vla.data.packet_io`, 기존 atomic writer는 `vla.data.atomic`에 단일 구현을 둔다.
`StateStatistics` 분리와 위 통계 subclass는 앞 절을 따른다. 기존 `plan_a_bc.geometry`는
`graph.current` module alias로 맞춰 patch namespace도 일치시켰다.

## 남은 범위

현재 helper 분리는 완료했다. 과거 source/설정/결과의 전체 원격 보존·복원 확인과
확인된 옛 코드 제거, 합의한 저장 SAM2 mask 기반2단계 cache, 실제 전체 GPU live의
전후 성능 비교는 별도 남은 작업이다. 이번 완료를 전체 repo cleanup 완료로 확대하지 않는다.

[현재 구조](../architecture.md) · [기본 실행](../usage.md) ·
[기존 정리 결정](repo_cleanup_preflight_20261104.md).
