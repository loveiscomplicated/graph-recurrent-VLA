
# Point Cloud Representation 코드 분석

핵심 확인: 현재 Vision의 E9 출력과 Graph prototype은 연결되어 있지 않습니다. E9는 RGB-D 관측에서 물체별 최대 1,024점을 내보내고, Graph prototype은 GT 형상에서 물체별 64점을 별도로 생성합니다. 따라서 새 관측 geometry를 Graph에 쓰려면 Vision의 점 추출과 Graph의 입력 계약을 각각 다뤄야 합니다. e9_export.py:33 (visual/scripts/e9_export.py:33), maniskill_gt.py:13 (graph/prototype/maniskill_gt.py:13), config.toml:1 (graph/prototype/config.toml:1)

## 코드에서 직접 확인한 사실

### 1. RGB-D와 mask에서 point cloud까지의 호출 흐름

현재 E9 경로는 collect_v1.py → collect.Recorder의 프레임 저장 → track_v1.py의 SAM2 mask 저장 → e9_export.py의 mask 로드와 pointcloud.extract() 호출입니다. Recorder는 카메라별 RGB, metre 단위 depth, segmentation, intrinsic, camera-to-world를 프레임별 배열로 저장합니다. Tracker는 peg·box의 bool mask와 logit을 저장하며, E9는 이를 풀어 base camera의 depth와 결합합니다. collect_v1.py:160 (visual/scripts/collect_v1.py:160), collect.py:155 (visual/scripts/collect.py:155), track_v1.py:314 (visual/scripts/track_v1.py:314), eval_v1.py:28 (visual/scripts/eval_v1.py:28), e9_export.py:53 (visual/scripts/e9_export.py:53)

extract()는 mask erosion을 조건부 적용한 뒤 유효 depth 픽셀을 역투영하고, 최대 2,048점으로 임의 축소, 통계적 이상점 제거(SOR), 최대 1,024점으로 FPS를 수행합니다. 작은 cluster 제거는 기본값에서 꺼져 있습니다. 반환값은 camera 좌표 점, world 좌표 점, 개수 정보입니다. E9는 logit이 양수이고 mask가 비지 않았을 때만 이를 호출하며, 최종 점이 50개 미만이면 해당 물체를 무효·빈 cloud로 저장합니다. pointcloud.py:122 (visual/scripts/pointcloud.py:122), e9_export.py:74 (visual/scripts/e9_export.py:74)

별도로 v0의 run_pipeline.py도 초기 Grounding DINO bbox → SAM2 추적 → pointcloud.process_episode() → extract() 순서로 point cloud를 저장합니다. 이 v0 경로에는 검출 실패 시 GT bbox 대체가 있습니다. v1 초기 검출 코드는 GT 대체가 없다고 명시합니다. 두 경로를 관측 전용 baseline으로 혼동하면 안 됩니다. run_pipeline.py:68 (visual/scripts/run_pipeline.py:68), run_pipeline.py:59 (visual/scripts/run_pipeline.py:59), init_v1.py:1 (visual/scripts/init_v1.py:1)

### 2. Sampling과 난수 제어

“1,024점을 random sampling”이라는 설명은 정확히는 2,048점까지 무작위 사전 축소한 뒤 FPS로 최대 1,024점 선정입니다. FPS도 첫 점을 RNG로 뽑습니다. 점이 상한 이하이면 FPS는 입력을 그대로 반환하므로 출력 개수는 가변입니다. CPU 구현의 상수는 FPS_PRE=2048, MAX_PTS=1024이고, E9는 episode마다 default_rng(0)을 만들어 프레임과 물체에 걸쳐 순차적으로 사용합니다. 같은 입력·처리 순서의 재실행은 재현되지만 프레임마다 동일한 점을 고정하는 방식은 아닙니다. pointcloud.py:51 (visual/scripts/pointcloud.py:51), pointcloud.py:108 (visual/scripts/pointcloud.py:108), pointcloud.py:132 (visual/scripts/pointcloud.py:132), e9_export.py:60 (visual/scripts/e9_export.py:60)

GPU 대체 구현도 2,048점 사전 축소와 FPS를 사용합니다. CUDA torch.Generator를 seed로 초기화하고 randperm과 FPS 시작점에 사용합니다. use_fps=False일 때만 마지막 단계도 randperm입니다. CPU와 GPU는 난수열이 달라 선택 점이 같지는 않다고 코드가 명시합니다. pc_gpu.py:1 (visual/scripts/pc_gpu.py:1), pc_gpu.py:47 (visual/scripts/pc_gpu.py:47), pc_gpu.py:88 (visual/scripts/pc_gpu.py:88), pc_gpu.py:116 (visual/scripts/pc_gpu.py:116)

### 3. Organized 영상과 원본 pixel 좌표의 보존 범위

수집 NPZ와 v1data.load() 단계에는 RGB [T,H,W,3], depth·segmentation [T,H,W]가 organized 형태로 남습니다. E9가 extract()를 호출하는 순간에도 해당 프레임의 RGB, depth, mask, K가 모두 있습니다. backproject()는 np.nonzero로 (v,u)를 만들지만 점 좌표만 반환합니다. 따라서 이후 사전 축소·SOR·FPS를 통과한 점의 원본 (u,v)는 point cloud 반환값이나 E9 출력에 명시적으로 남지 않습니다. collect.py:7 (visual/scripts/collect.py:7), v1data.py:64 (visual/scripts/v1data.py:64), e9_export.py:55 (visual/scripts/e9_export.py:55), pointcloud.py:71 (visual/scripts/pointcloud.py:71)

기존 pose fitting의 별도 backproject_px()는 (points,v,u)를 반환해 해당 픽셀의 RGB를 사용합니다. GPU point cloud 구현도 전체 pixel ray 격자를 보유하지만 선택 mask sel과 점의 대응은 반환하지 않습니다. 선택 점의 픽셀 출처와 RGB-D 경계 정보를 Graph로 넘기는 현행 필드는 없습니다. pose_v1.py:32 (visual/scripts/pose_v1.py:32), e9_export.py:91 (visual/scripts/e9_export.py:91), pc_gpu.py:92 (visual/scripts/pc_gpu.py:92), pc_gpu.py:113 (visual/scripts/pc_gpu.py:113)

### 4. 좌표 변환

Recorder는 ManiSkill의 extrinsic_cv를 world→camera 4×4 행렬로 만들고 역행렬을 camera(CV)→world로 저장합니다. 역투영은 pixel centre (u+0.5,v+0.5), depth z, K를 사용합니다. to_world()는 회전·평행이동을 적용합니다. E9는 기록된 TCP pose의 역변환을 world 점에 적용해 TCP 원점과 축에 정렬된 {o}_points_ee를 만듭니다. collect.py:163 (visual/scripts/collect.py:163), pointcloud.py:71 (visual/scripts/pointcloud.py:71), pointcloud.py:79 (visual/scripts/pointcloud.py:79), e9_export.py:69 (visual/scripts/e9_export.py:69), e9_export.py:83 (visual/scripts/e9_export.py:83)

좌표계 호환성에 중요한 차이가 있습니다. graph.gt_nodes에는 실제 TCP 축으로 회전시키는 world→TCP 함수가 있지만, 현재 Graph prototype의 node_features()는 world 위치에서 TCP 위치만 뺍니다. 축은 world 축 그대로입니다. E9의 points_ee를 현재 Graph feature와 동일한 좌표라고 취급할 수 없습니다. gt_nodes.py:31 (graph/gt_nodes.py:31), graph_builder.py:38 (graph/prototype/graph_builder.py:38), graph_builder.py:53 (graph/prototype/graph_builder.py:53)

### 5. Object ID, TCP, fingertip의 경로

Recorder의 actor_ids()는 simulator의 segmentation_id_map에서 peg와 box의 numeric ID를 찾아 수집 NPZ에 기록합니다. 이 ID는 GT mask 평가에 쓰입니다. SAM2 쪽의 1→peg, 2→box는 tracker 내부 물체 ID이며 simulator segmentation ID와 별개입니다. E9 출력에서는 물체 소속을 numeric ID 대신 peg_·box_ 키 이름으로 구분합니다. collect.py:189 (visual/scripts/collect.py:189), collect_v1.py:177 (visual/scripts/collect_v1.py:177), track_v1.py:36 (visual/scripts/track_v1.py:36), e9_export.py:74 (visual/scripts/e9_export.py:74)

TCP world pose는 Recorder가 로봇 agent.tcp.pose에서 저장하고 E9가 전달합니다. Fingertip은 E9에 없습니다. 공유 GT accessor는 각 finger link pose에 고정된 rubber pad 중심 offset을 적용해 world 좌표의 left/right fingertip을 생성합니다. Graph의 GTAdapter가 이를 GeometryState로 전달하고, GraphBuilder는 TCP와 양쪽 fingertip을 첫 세 robot node로 놓습니다. 한편 graph.gt_nodes.get_gt_node_positions()는 robot node와 GT peg/hole corners까지 함께 만드므로 새 관측 Graph의 입력원으로 그대로 쓰기에는 맞지 않습니다. collect.py:176 (visual/scripts/collect.py:176), gt_state.py:43 (envs/maniskill/gt_state.py:43), gt_state.py:63 (envs/maniskill/gt_state.py:63), maniskill_gt.py:27 (graph/prototype/maniskill_gt.py:27), graph_builder.py:40 (graph/prototype/graph_builder.py:40), gt_nodes.py:38 (graph/gt_nodes.py:38)

### 6. 현재 Vision 출력과 Graph 입력 계약

| 경로 | 현재 구조와 자료형 |
|---|---|
| E9 Vision | 물체별 points_world, points_ee: [1024,3] float32; valid_pts: [1024] bool; relevance: [1024] float32; 물체별 valid·age·confidence. 최대 2,048개 물체 점이며 각각 padding합니다. TCP pose, 추정 peg/hole pose, 평가용 GT pose도 같은 NPZ에 있습니다. Fingertip·pixel 좌표·구조점 유형·표면 경계 필드는 없습니다. e9_export.py:3 (visual/scripts/e9_export.py:3), e9_export.py:36 (visual/scripts/e9_export.py:36), e9_export.py:86 (visual/scripts/e9_export.py:86) |
| Graph prototype | GeometryState는 world 위치·물체 local 점·normal·GT 물체 pose/크기 등을 가진 batched float32 Tensor 구조입니다. 기본값은 peg 64점과 hole surface 64점입니다. GraphBatch.x는 [B,131,14], context는 [B,81], node_type은 [131] 정수, edge_index는 [2,E] 정수, edge_attr는 [E,8] float, edge_type은 [E] 정수입니다. geometry.py:25 (graph/prototype/geometry.py:25), config.toml:1 (graph/prototype/config.toml:1), graph_builder.py:63 (graph/prototype/graph_builder.py:63) |

Graph의 14차원 node feature는 유형 one-hot 5, 위치 3, 물체 local 위치 3, normal 3입니다. 내부 surface edge는 물체 local 좌표의 k-NN과 연결 보장용 tree를 사용하며, 현재 RGB-D의 표면 불연속성을 보지 않습니다. context에는 GT 물체 pose·크기·hole offset도 포함됩니다. 따라서 기존 GeometryState에 관측 점만 대입하는 것은 요청하신 관측 전용 Graph 입력을 만들지 못합니다. graph_builder.py:28 (graph/prototype/graph_builder.py:28), graph_builder.py:46 (graph/prototype/graph_builder.py:46), graph_builder.py:105 (graph/prototype/graph_builder.py:105)

### 7. 테스트, 벤치마크, dependency, 전송

기존 검증에는 공유 simulator CPU 테스트, TCP 상대 GT node 테스트, Graph의 analytic geometry·edge·모델 테스트가 있습니다. Vision의 test_pose_v1.py는 pose fitting 자체 검사입니다. pointcloud.py의 평가는 GT mask cloud와의 centroid 차이·peg 내부 비율 등을 계산하지만, corner/edge 보존, 픽셀 provenance, 가림 경계 분류, surface edge filtering의 계약 테스트는 확인되지 않았습니다. tests/test_maniskill_env.py:20 (tests/test_maniskill_env.py:20), tests/test_graph_gt_nodes.py:44 (tests/test_graph_gt_nodes.py:44), test_models.py:18 (graph/prototype/tests/test_models.py:18), test_pose_v1.py:21 (visual/scripts/test_pose_v1.py:21), pointcloud.py:157 (visual/scripts/pointcloud.py:157)

기록된 E2 벤치마크에서는 두 물체의 point cloud 처리 중앙값이 CPU 경로 39.0 ms, GPU 경로 4.1 ms이고, streaming·compile·GPU 전처리·prune 조합의 tracking 포함 p95는 45.1 ms입니다. 이는 기존 코드의 저장된 결과이며 이번 호스트에서 재측정한 수치가 아닙니다. Graph의 latency 측정은 batch 1 CPU 기준입니다. E2 table.md:3 (visual/results/v1/E2/table.md:3), E2 table.md:7 (visual/results/v1/E2/table.md:7), compare.py:40 (graph/prototype/compare.py:40)

CPU 추출은 NumPy·OpenCV·SciPy cKDTree를 사용합니다. GPU 대체는 PyTorch·Triton을 사용하며 torch.cdist로 SOR를 계산합니다. PCL 또는 OrganizedEdgeFromRGBNormals 의존성과 호출은 현재 Vision dependency·코드에서 확인되지 않았습니다. Recorder의 np0()가 simulator Torch Tensor를 CPU NumPy로 옮기고, offline tracker도 mask를 .cpu().numpy()로 저장합니다. E2 GPU 경로는 depth와 camera transform을 CUDA Tensor로 옮겨 point cloud를 GPU에 유지하지만, 평가용 mask는 다시 CPU로 복사합니다. pointcloud.py:35 (visual/scripts/pointcloud.py:35), pc_gpu.py:11 (visual/scripts/pc_gpu.py:11), pc_gpu.py:103 (visual/scripts/pc_gpu.py:103), pyproject.toml:5 (visual/pyproject.toml:5), collect.py:131 (visual/scripts/collect.py:131), e2_stream.py:78 (visual/scripts/e2_stream.py:78)

### 8. 변경을 최소화할 연결 위치

Vision 쪽 지점은 e9_export.py가 pcl.extract()를 호출하기 직전입니다. 이때 organized RGB-D, 예측 mask, K, camera-to-world가 함께 있으므로 관측 픽셀의 geometry와 경계 후보를 평가할 수 있습니다. extract()의 2,048점 무작위 축소나 erosion 이후에 특징점을 찾으면 필요한 경계가 이미 사라졌을 수 있습니다. 기존 pose fitting 경로는 독립적으로 유지할 수 있습니다. e9_export.py:55 (visual/scripts/e9_export.py:55), e9_export.py:78 (visual/scripts/e9_export.py:78), pointcloud.py:125 (visual/scripts/pointcloud.py:125), pointcloud.py:132 (visual/scripts/pointcloud.py:132)

Graph 쪽 지점은 GeometryState→node_features()→GraphBuilder 경계입니다. 이 경로는 고정 수, local 점·normal, GT context를 전제하므로 새 관측 입력용 변환 계약과 가변 노드 처리 방식을 Graph 팀에서 정해야 합니다. 현재 모델 readout은 물체 유형별 평균을 내고, flat 모델은 입력 폭을 점 개수로 고정합니다. 이는 Vision 출력 필드만 바꿔 해결할 수 있는 부분이 아닙니다. geometry.py:25 (graph/prototype/geometry.py:25), graph_builder.py:38 (graph/prototype/graph_builder.py:38), models.py:45 (graph/prototype/models.py:45), models.py:75 (graph/prototype/models.py:75)

## 확인되지 않은 가정과 질문

- 관측 가능한 hole rim·peg edge가 현재 카메라 해상도와 mask 품질에서 얼마나 검출 가능한지는 코드·기존 지표만으로 확인되지 않습니다. 특히 erosion과 SOR가 그 후보를 얼마나 제거하는지 측정이 필요합니다.
- OrganizedEdgeFromRGBNormals를 현재 지원 환경인 Linux x86_64/CUDA Vision 환경에 어떤 binding으로 연결할지, RGB 경계와 가림 경계를 요구한 정확도로 구별할 수 있을지는 확인되지 않았습니다. visual/README.md:53 (visual/README.md:53)
- 새 표현의 물체별 최소 점 수, Structural 후보 판정, 경계 정보를 edge filtering에 전달할 자료형, 두 물체 중 하나가 완전히 안 보일 때 Graph readout 동작은 아직 구현 계약이 없습니다.
- 동일 조건의 관측 cloud 기반 GNN BC 대조 실험은 현 Graph 경로에서 확인되지 않았습니다. 현재 Graph 학습 데이터는 GTAdapter가 생성합니다. prepare_dataset.py:87 (graph/prototype/prepare_dataset.py:87)

## 수정 시 호환성·성능 문제

- E9는 물체별 [1024,3] padding과 valid_pts 계약입니다. 제안한 합계 상한·가변 점 수·padding 없음으로 바꾸면 E9를 읽는 E6의 mask 인덱싱과 기존 NPZ 소비 코드의 호환성을 확인해야 합니다. e6_relevance.py:90 (visual/scripts/e6_relevance.py:90)
- E9의 TCP 축 좌표와 Graph의 world 축 상대 좌표를 섞으면 위치와 edge 벡터가 잘못 해석됩니다. 기존 Graph의 hole_points는 관측 Box cloud가 아니라 GT로 만든 hole surface이며, context에도 물체 GT가 들어갑니다.
- 현재 GraphBuilder는 물체 내부 torch.cdist와 k-NN, 물체 간 torch.cdist, 중복 edge 정리를 수행합니다. 점 수와 가변성이 바뀌면 구성 시간·메모리·batch 방식이 달라집니다. 물체 점이 0개일 때의 topk와 유형별 평균도 별도 처리가 필요합니다. graph_builder.py:105 (graph/prototype/graph_builder.py:105), graph_builder.py:118 (graph/prototype/graph_builder.py:118), models.py:50 (graph/prototype/models.py:50)
- organized 경계 검출을 CPU에서 수행한다면 GPU streaming mask·depth의 readback과 동기화 비용을 포함해 재측정해야 합니다. 현재 E2의 4.1 ms point cloud 수치에 새 특징 검출 비용은 포함되지 않습니다.

## 가장 먼저 할 수 있는 독립 작업 1개

관측 점의 pixel 출처를 보존하는 CPU 후보 추출기를 별도 함수로 추가하는 작업을 권합니다. 입력은 한 프레임의 RGB·depth·예측 mask·K이고, 출력은 sampling 전 유효 관측점과 이에 정렬된 (v,u) 및 RGB입니다. 기존 extract(), E9 포맷, pose fitting, rule-based policy는 이 작업에서 건드리지 않는 범위가 적절합니다.

Acceptance criteria: 합성 RGB-D·mask에서 모든 반환 점이 해당 (v,u)의 depth와 K로 재계산한 3D 점과 일치하고, RGB도 같은 픽셀에서 오며, 빈 mask·무효 depth·경계 픽셀을 검사합니다. GT 물체 pose·크기를 입력받지 않습니다. 기존 extract() 출력은 동일 seed에서 그대로 재현됩니다. 이 출처 정보가 확보되어야 Structural 후보와 표면 불연속성의 검출·평가를 같은 관측 픽셀에 묶을 수 있습니다.

이번 조사 범위: 파일 수정·설치·리팩터링은 하지 않았고 git status --short는 비어 있습니다. 설치된 루트 환경의 버전은 ManiSkill 3.0.1, SAPIEN 3.0.3, PyTorch 2.14.1이며 CUDA와 MPS는 사용 불가입니다. CPU RGB-D 실행 예제는 프로세스가 exit 134로 종료되어 관측 출력까지 확인하지 못했습니다. 따라서 이번 호스트에서 simulator RGB-D 실행이나 GPU/Triton 경로가 검증됐다고 주장하지 않습니다. 설치된 ManiSkill task 소스와 저장소의 실행 예제·기존 테스트를 대조한 코드 분석 결과입니다. check_env.py:23 (examples/check_env.py:23), peg_insertion_side.py (.venv/lib/python3.11/site-packages/mani_skill/envs/tasks/tabletop/peg_insertion_side.py:67)