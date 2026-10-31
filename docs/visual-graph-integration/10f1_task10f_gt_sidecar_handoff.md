# Task10-F → Task10-F1 GT Sidecar Handoff

2026-10-30. 기존 Task10-F1을 유지하면서 Task10-F의 **승인된 완료 데이터**를 우선 재사용하기 위한 최소 기록 요구사항이다. Task10-F1 담당 실행은 Task10-F의 실행 중 수집 코드·프로세스를 변경하지 않는다. 이 문서는 Task10-F 담당 작업이 다음 안전한 수집 경계에서 적용할 계약이다.

## 현재 확인한 차이

57640710 `/workspace/task10f_recovery_v2/pilot`의 완료·승인 receipt40개(P/N 합계)를 읽기 전용으로 조사했다. 당시 manifest는20개 P branch, planned40개였고 cache manifest는 engineering2개뿐이다. 승인 HDF5에는 RGB-D/보정/Robot76/Timestep/Segment/Action Supervision Mask가 있으나, 매 시점 Peg/Hole pose·형상이 없다. 정확한 직렬화 simulator/controller snapshot도 발견되지 않았다. In-memory snapshot의 replay 검사 통과는 디스크에 해당 snapshot이 보존되어 있다는 뜻이 아니다.

따라서 **GT-supervised DATA_READY가 아니다.** Actor centroid를 GT로 대체하거나 초기 pose·이동 norm으로 각 시점 pose를 채우지 않는다. 기존 Safety Gate 실패 Pilot은 승인 대상에서 제외한다. 상세 근거는 [읽기 전용 감사](../../outputs/task10f1/task10f_dataset_audit.json)에 있다. 이 수치는 감사 시점의 상태이며 이후 수집 진행을 막지 않는다.

## 추가로 보존해야 할 최소 GT Sidecar

원본 RGB-D/Robot76 capture와 **동일 pre-action timestep t**에서 별도 NPZ/HDF5에 다음을 기록한다. Sidecar는 Graph/Student 입력에 전달하지 않는다.

| 필드 | Shape / 규칙 | 원본 Source |
|---|---|---|
| `raw_peg_pose` | `[T+1,7]`, world xyz metres + unit wxyz | `get_gt_state(env).peg_pose` |
| `raw_hole_pose` | `[T+1,7]`, world xyz + unit wxyz | `get_gt_state(env).hole_pose`, box actor origin과 구분 |
| `raw_peg_half` | `[T+1,3]`, peg-local half extents metres | `get_gt_state(env).peg_half_sizes` |
| `raw_hole_radius` | `[T+1,1]`, metres | `get_gt_state(env).hole_radius`, square-hole half opening |
| `timestep`, `timestamp` | `[T+1]`, original H5와 일치 | actual elapsed control clock /20Hz |
| `parent_id`, `branch_id`, `segment` | action/observation indexing 명시 | 원본 branch manifest |
| `physical_valid` | pre-action 관측별 명시된 판정 | 승인된 physical safety audit; prefix도 근거 연결 |
| `gnn_supervision_mask` | 물리 유효성과 GT 정합성을 통과한 관측 | Action Supervision Mask와 별도 |

TCP·두 pad center·gripper는 기존 Robot76의 measured TCP/finger pose, finger joint 및 검증된 pad local offset으로 계산할 수 있다. 수집 시 `raw_gt`의 로봇 필드와 비교해 최대 오차를 기록한다. 가능하면 원본 `recurrent.common.raw_gt` 전체8개 raw 필드와 원본 `build_features(raw,fresh=1,age=0)` 결과38D도 sidecar에 보존하여 독립 검증한다.

각 append 전후 elapsed_steps가 같아야 한다. T actions에는 관측 t=0..T-1만 학습에 사용한다. 마지막 t=T는 terminal observation으로 보존하되 action label과 혼동하지 않는다. Disturbance의 Action Mask가 OFF라도 별도 GNN Mask가 ON이고 물리·GT 계약이 통과하면 GNN 학습에 사용할 수 있다. 실패 Pilot을 GNN Mask만 바꾸어 자동 승인하지 않는다.

이미 수집한 자료는 **정확한 저장 simulator/controller/RNG snapshot과 실행 action replay**, 그리고 timestep별 robot·RGB-D·보정/pose 정합성 검증이 모두 가능한 경우에만 새 GT sidecar를 생성한다. 현재 snapshot 부재 자료를 seed-only 근사 replay나 정적 초기 GT로 보충하지 않는다. 실행 중 코드는 변경하지 않고, 필요하면 담당 작업이 독립된 새 승인 수집을 수행한다.

## 완료 Marker: `gnn_dataset_ready.json`

Task10-F 전체 실험이나 BC 학습 완료를 기다릴 필요는 없다. 승인한 수집 범위·Graph Cache·GT Label·물리 판정이 모두 완료된 뒤 root에 이 marker를 atomic rename으로 작성한다. 아직 작성 중인 파일을 포함하지 않는다.

```json
{
  "schema_version": 1,
  "status": "DATA_READY",
  "task": "10-F",
  "source_instance": 57640710,
  "dataset_id": "immutable_collection_identity",
  "collection_complete": true,
  "graph_cache_readback": "PASS",
  "physical_validity": "PASS",
  "gt38_alignment": "PASS",
  "parent_branch_provenance": "PASS",
  "files_sealed": true,
  "runtime_contract": "relative/path/runtime_contract.json",
  "runtime_contract_sha256": "sha256",
  "dataset_manifest": "relative/path/gnn_dataset_manifest.json",
  "dataset_manifest_sha256": "sha256",
  "files": [{"path": "relative/path/file", "sha256": "sha256", "bytes": 123}]
}
```

`files`에 dataset manifest/runtime 계약, 각 episode의 원본 H5, GT sidecar, graph NPZ·receipt, 물리 판정·GNN Mask·branch provenance를 모두 명시한다. 파일 경로는 Task10-F root 안의 안전한 상대경로로 제한한다. 디렉터리 symlink나 mutable 상위 폴더 전체를 복사하지 않는다. 원본의 외부 의존 파일은 provenance를 남겨 별도 sealed export에 복사한다.

Episode manifest는 각 row에 `episode_id`, `parent_id`, `branch_id`, `split`, `distribution`(nominal_reach/disturbed_robot/expert_recovery/grasp_lift/alignment_insertion), `actions`, `raw_path`, `gt_path`, `graph_path`, `graph_receipt_path`, `physical_audit_path`, `gnn_mask_path` 및 각 SHA256을 제공한다. Raw·label·graph의 observation indexing과 prefix 복제 범위를 명시한다. 동일 parent의 prefix·P/N branch를 train/validation에 나누지 않는다. Frozen Test/과거 Final/new F1 Final seeds를 fitting에 넣지 않는다.

Runtime 계약은 simulator/package/controller/frequency/backend, camera metadata·mount/units, Robot76 schema, frozen09-D protocol/Graph configuration, DINO/SAM2 weight·preprocessor 및 source SHA를 포함한다. Task10-F1 Full-task 자료와 runtime/camera/geometry 일치 또는 별도의 엄격한 동등성 증거가 없으면 자동 혼합하지 않는다.

## F1 인계 경계

F1 Watcher는 marker와 모든 파일의 SHA/크기/Readback·승인·GT 정합성을 확인한 뒤 한 번의 `DATA_READY` 이벤트를 기록한다. 같은 content Event ID에는 하나의 Invocation Journal claim만 허용한다. 완료 파일만 57987772의 독립 import 경로로 복사하고 전송 후 다시 검증한다. Codex 재개 시 기존 F1 캐시·상태를 확인하여 아직 수행하지 않은 GNN 단계만 진행한다.

검증된 실제 데이터 구성과 학습 예산은 본 학습 전에 새로운 final freeze 파일로 기록한다. 기존96/12/12 Full-task 계획/설정 파일은 원본 기록으로 보존한다. 이 인계는 새 GRU·Action 학습·Git commit·HF publishing을 허용하지 않는다.
