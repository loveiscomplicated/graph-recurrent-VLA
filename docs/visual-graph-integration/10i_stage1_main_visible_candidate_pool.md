# Task10-I / Stage 1 — Main-visible Initial State Candidate Pool

상태: **COMPLETED**. ExperimentLogger UUID: `098bd09a011b4c9d99ca881d6a83b760`.
실행 Root: `/Users/jeong-yunseong/Documents/programming/graph-recurrent-VLA/outputs/task10i/stage1/verified_pool_v2`. 이 결과는 최종 학습 데이터셋이 아닌 초기 상태 후보 풀이다.

## Q1. 기존 감사 재현

동일 geometry.py/protocol.json으로 320/320 Primary ID가 일치했다. Train FAIL 190/272, Validation FAIL 25/48; 고유 Parent FAIL 63/96, 9/16. 기존 112 Parent를 실제 CPU PhysX + CUDA Renderer로 다시 Reset하여 RGB/Depth/Camera/Peg/TCP exact 일치를 검사했다. [회귀 검증](../../outputs/task10i/stage1/verified_pool_v2/visibility_regression.json), [Source](../../outputs/task10i/stage1/verified_pool_v2/source_identity.json), [Protocol](../../outputs/task10i/stage1/verified_pool_v2/visibility_protocol_identity.json).

## Q2–Q4. 신규 시도·고유 통과·실패 이유

신규 Random Reset 926회, Visibility PASS 301개 (32.51%), 최종 유효 고유 Parent **300개**. 부족량 0개. 최대 1,200개 사전 동결 Seed를 순서대로 사용하며 300개 도달 시 중단한다. 성능·공간 Descriptor로 Seed/배치를 수정하지 않았다.

실패 사유(중복 집계): `{"white_end_out_of_frame": 617, "white_end_unresolved": 462, "grasp_out_of_frame": 536, "grasp_unresolved": 363, "white_end_depth_unconfirmed": 4, "red_end_out_of_frame": 99, "red_end_occluded": 1, "red_end_depth_unconfirmed": 8, "red_end_unresolved": 5, "grasp_depth_unconfirmed": 2, "grasp_occluded": 1}`.

## Q5. 물리 유효성

Physical valid 923/926. Finite pose/qpos/qvel/shape, joint limit, native reset workspace support, zero clock, calibration/depth/controller, actual collision mesh convex SAT 검사를 적용했다. 심각한 침투는 최소 분리 이동량 >2mm로 수집 전 고정했다. Table 정상 접촉은 허용하며 10µm 수치 오차만 허용한다. Reset 직후 PhysX contact 목록은 비어 있으므로 이를 무충돌 증거로 쓰지 않았다. Robot self-collision의 exhaustive 동역학 검증이나 Expert 성공을 주장하지 않는다. **EXPERT_SUCCESS=NOT_TESTED**, action 실행 0회.

## Q6. Replay

모든 Eligible은 다른 Seed로 Reset한 뒤 원래 Seed로 재초기화하여 모든 저장 geometry/robot/camera/RGB/depth/segmentation과 등록 simulator state, 초기 controller를 exact 비교한다. 별도 프로세스에서 전체 Eligible을 다시 재생하고 Visibility도 비교했다. [독립 Replay](../../outputs/task10i/stage1/verified_pool_v2/candidate_replay_verification.json).

Snapshot에는 등록 Actor/Articulation state와 Controller, RNG, Clock을 저장했다. 지원 API의 get_state_dict에는 shape/randomized task assets와 PhysX hidden solver cache가 포함되지 않으므로 Snapshot 단독 완전 복원을 주장하지 않는다. 검증한 후속 경로는 **동일 Runtime/Source에서 reset(seed)로 shape를 재구성하고 initial.npz 및 snapshot을 readback 대조**하는 경로다. 미래 trajectory의 bitwise identity는 검증하지 않았다. 상태 미세 차이를 허용하는 tolerance는 Replay에 사용하지 않았다.

## Q7. Seed/중복

기존 /workspace 연구 JSON manifest, 기존 Source literal/range 및 감사 Parent를 조사하여 보수적으로 기존 정수·Seed namespace와 분리했다. 기존 Train/Validation/Final/10H/10D 파일의 실제 경로와 SHA는 [Inventory](../../outputs/task10i/stage1/verified_pool_v2/existing_parent_inventory.json)에 있다. 후보 Seed·ID 중복 없음; 기존 조사 Seed 충돌 없음. Exact duplicate 0, near duplicate 0. Near duplicate는 1mm translation, 1degree rotation, 0.1mm half-size 기준의 진단이며 제거하지 않았다. [동결 Seed](../../outputs/task10i/stage1/verified_pool_v2/candidate_seed_manifest.json), [중복 감사](../../outputs/task10i/stage1/verified_pool_v2/duplicate_audit.json).

## Q8. 상태 범위

Eligible 범위 (m, rad; world frame):

```json
{
  "peg_xyz": {
    "min": [
      -0.09956424683332443,
      -0.12499396502971649,
      0.015007692389190197
    ],
    "max": [
      0.09849109500646591,
      -0.0001029670238494873,
      0.024907700717449188
    ]
  },
  "peg_yaw_rad": {
    "min": 0.5301763403992533,
    "max": 2.6074938638654914
  },
  "peg_length_m": {
    "min": 0.17017856240272522,
    "max": 0.24915850162506104
  },
  "peg_thickness_m": {
    "min": 0.030015384778380394,
    "max": 0.049815401434898376
  },
  "peg_minus_hole_world": {
    "min": [
      -0.14452721178531647,
      -0.4928427040576935,
      -0.1502104550600052
    ],
    "max": [
      0.1756582260131836,
      -0.20746426284313202,
      -0.03214408457279205
    ]
  },
  "tcp_peg_distance_m": {
    "min": 0.12095356732606888,
    "max": 0.2306574583053589
  },
  "boundary_margin_px": {
    "min": 0.08465203433557633,
    "max": 36.628653319231404
  }
}
```

Pose quaternion은 ManiSkill raw/GT38D와 동일한 `[x,y,z,qw,qx,qy,qz]`, descriptor의 명시적 `_xyzw` relative quaternion만 SciPy 순서다. RGB uint8, Depth signed int16 mm; nonpositive depth는 invalid로 보존한다. `depth_supported_expected_surface_fraction`은 프레임 내 예상 표면에 대한 depth 비율이며 `gt_peg_image_fraction`은 전체 128×128 중 GT peg 픽셀 비율이다. 전체 3D 표면 관측률과 혼동하지 않는다. GT pixel-perfect 진단은 Primary Gate에 사용하지 않았다.

![PASS/FAIL 분포](../../outputs/task10i/stage1/verified_pool_v2/distributions.png)
![Reset 순서별 누적 통과율](../../outputs/task10i/stage1/verified_pool_v2/pass_rate.png)

## Q9. 다음 단계 사용 가능성

[Eligible export](../../outputs/task10i/stage1/verified_pool_v2/eligible_candidate_pool.json)는 300개 상태·Descriptor·RGB-D·GT 증거·Snapshot·Source/Runtime hash를 연결한다. 모든 split은 UNASSIGNED다. 72개 보완 선택 대비 4.17배 후보이며 Random/Coverage-aware 비교는 아직 수행하지 않았다. [전체 Gallery](../../outputs/task10i/stage1/verified_pool_v2/gallery.html), [모든 시도](../../outputs/task10i/stage1/verified_pool_v2/candidate_attempts.csv), [적격성](../../outputs/task10i/stage1/verified_pool_v2/candidate_eligibility.csv), [State](../../outputs/task10i/stage1/verified_pool_v2/candidate_initial_states.npz), [Descriptor](../../outputs/task10i/stage1/verified_pool_v2/candidate_state_descriptors.csv), [Visibility](../../outputs/task10i/stage1/verified_pool_v2/candidate_visibility_metrics.csv).

## Q10. 기존 통과 Parent와 결합 전 필요한 검증

다음 단계에서는 기존 33 Train/7 Validation 통과 Parent와 신규 후보 사이 물리 상태 중복·near duplicate를 비교하고, Parent 단위 split 누출을 방지해야 한다. 선택 방법·예산·Seed와 평가 분포를 먼저 동결하고, 선택 이후 Expert 수집 단계에서 실제 full-task 성공 및 RGB-D/GT/clock 정합성을 다시 검증해야 한다. 현재 Candidate 적격성은 Expert 성공 보증이 아니다. 새로운 Blind Test seed는 이 분석 Pool에 포함하지 않았다.

## 실행·검증·제한

기존 Task10-H 프로세스는 중단/변경하지 않았다. GPU 경합을 고려해 단일 Worker로 실행했다. CPU PhysX, CUDA SAPIEN Renderer를 지정 RTX3090/57987772에서 실행했다. macOS simulator, GPU PhysX, vision/policy inference는 실행하지 않았다. 장시간 실행은 tmux `task10i-stage1`의 독립 Python runner가 관리한다. 원자적 Candidate commit, SHA/readback, lock, frozen source/config, 유한 retry(자동 retry 0), Crash-resume fixture와 별도 Replay를 적용했다. [Integration](../../outputs/task10i/stage1/verified_pool_v2/integration_test.json), [Artifact 검증](../../outputs/task10i/stage1/verified_pool_v2/candidate_pool_verification.json), [Runtime](../../outputs/task10i/stage1/verified_pool_v2/runtime_identity.json), [Timing](../../outputs/task10i/stage1/verified_pool_v2/timing_summary.json).

공통 ExperimentLogger를 W&B disabled로 재사용했다. 기존 Dataset/Checkpoint/10-H는 읽기 전용이며 Git commit·외부 Publishing은 하지 않았다. Stage2/Stage3를 자동 시작하지 않는다. 관측되지 않은 실패 유형의 예시 이미지는 합성하지 않고 [example index](../../outputs/task10i/stage1/verified_pool_v2/example_index.json)에 NOT_OBSERVED로 기록했다.

## 독립 Calibration 재검증과 보존된 BLOCKED Run

원래 수집 Run `run_v3`은 GL 행렬 기반 추가 검증에서 1픽셀 차이로 BLOCKED로 종료했으며 원본 상태와 실패 기록은 그대로 보존했다. 이 보고서의 `verified_pool_v2`은 별도 ExperimentLogger UUID의 새 검증 Run이다. Candidate/Seed/Primary label/Physical/Replay 결과는 한 건도 수정하지 않았다.

기존 Primary가 사용하는 CV Calibration을 독립 OBB Slab 알고리즘에 입력했을 때 **926개 전체의 Mask·필수 영역·Primary Verdict가 exact 일치**했다. 최대 공통 Depth 수치 차이는 0.001391mm로 기존 독립 수치 검사 0.1mm 이내다. Primary 가시성 Depth tolerance는 기존 1.5mm 그대로다.

별도 저장된 float32 GL Calibration에서는 `10i_s1_0364`의 (u=113,v=116) 경계 1픽셀만 추가 hit가 생겼다. 해당 ray의 Peg 내부 구간은 약 83nm이고 실제 GT ID는 Peg가 아니다. GL 기반 Primary 재계산은 이 한 후보에서 FAIL이며, 그 차이를 [Calibration 진단](../../outputs/task10i/stage1/verified_pool_v2/calibration_diagnostic.json)에 보존했다. GL 결과를 exact/PASS로 수정하거나 1픽셀을 가시성 Gate에서 임의로 허용한 것이 아니다. 승인된 기존 CV Primary를 그대로 검증했다. 기존 프로토콜에서 Pixel-perfect raster 조건이 별도 진단인 것과 같이, 보정 행렬의 추가 래스터 조건을 새로운 Primary Gate로 승격하지 않았다.

수집 중간의 RNG 직렬화 오류와 Gallery null projection 오류는 각각 이전 `run`/`run_v2`에 보존했다. 두 Engineering 실패는 본 수집 전에 발생했으며 수정 후 새 Run에서 회귀·Integration부터 다시 실행했다. 기존 합성 가시성 테스트 6개와 새 Collision/Hash/Atomic Manifest 계약 테스트 3개가 통과했다.

물리적으로 무효한 초기화 3개는 [실제 진단](../../outputs/task10i/stage1/verified_pool_v2/physical_invalid_diagnostic.json)에 있으며, 그중 Visibility PASS도 적격 수에 포함하지 않았다. 최종 Eligible은 300개 모두 Physical Valid 및 독립 Reset Replay PASS다. 단일 Worker를 사용했고 다중 Worker 동등성은 실행하지 않았다. Timing의 Reset 비용은 Rendering을 포함한 합산값이며 순수 Render 비용을 별도로 분리 측정하지 않았다.

원본 RGB-D는 Export의 `raw_rgb_npz`/`raw_depth_npz`에 지정한 NPZ key에서 읽는다. `annotated_rgb_preview`는 Gallery 전용이다. 후보 분포는 native Random Reset을 기존 가시성·명시적 물리 유효성·Replay 조건으로 제한한 분포다. macOS에서는 저장 Artifact의 수치 교차 검증만 수행했다. 실제 Simulator와 300개 Replay는 원래 지정 Linux Runtime에서 실행했으며 GPU PhysX나 로컬 Simulator를 실행한 것이 아니다.

첫 500회 Visibility PASS는 149/500 (29.80%)다. [기존 Parent와 신규 Eligible의 exact Peg Geometry 중복 검사](../../outputs/task10i/stage1/existing_vs_candidate_geometry_audit.json)도 통과했다.

후보분포는 native Random Reset 분포를 기존 Main 가시성·명시한 물리 유효성·Replay 조건으로 제한한 분포다. Export의 `annotated_rgb_preview` 경로는 주석이 있는 Gallery Preview이며, 후속 분석의 실제 원본 영상은 `initial_npz` 안의 `rgb`/`depth_raw_mm` 키를 사용한다.

## 인계 검증 및 Engineering 이력

첫 Engineering Run은 BatchedRNG 직접 직렬화 오류, 두 번째 Run은 카메라 뒤 투영의 null 좌표를 Gallery에서 처리하지 못해 BLOCKED로 보존했다. 두 Run 모두 본 Candidate 수집 전 중단됐으며, 수정 후 run_v3에서 회귀·Integration부터 재실행했다. run_v3은 마지막 GL 진단 1픽셀 차이로 BLOCKED를 보존했고, 별도 UUID의 verified_pool_v2에서 기존 CV 계약을 독립 전수 검증했다. 기존 Primary Protocol/Threshold 변경은 없었다.

기존 합성 가시성 테스트 6개와 신규 물리/Hash/원자적 Manifest 계약 테스트 3개가 통과했다. [독립 신규 가시성 검증](../../outputs/task10i/stage1/verified_pool_v2/independent_visibility_verification.json), [로컬 인계 SHA·CSV·NPZ 전수 Readback](../../outputs/task10i/stage1/delivery_verification.json). 로컬 검증은 macOS NumPy에서 실행했으며 로컬 Simulator/CUDA를 실행한 것은 아니다. 단일 Worker를 사용했으며 다중 Worker 동등성 및 분리된 순수 Render 비용은 별도로 측정하지 않았다. Timing의 reset_including_render_s는 Reset과 Render의 합산 시간이다.

## 원격 미러 인계 제한

최종 Canonical 후보 풀은 로컬 `outputs/task10i/stage1/verified_pool_v2/`에 완전히 보존돼 있고 8,506개 파일 SHA 및 926행 CSV/NPZ Readback을 통과했다. 지정 서버의 별도 `verified_pool_v2` 미러는 SSH 전송 중 연결이 종료돼 **원격 Readback 미완료**다. 원격의 부분 미러를 검증 완료 인계본으로 사용하면 안 된다. 원래 원격 `run_v3` 수집·Replay 증거 및 BLOCKED 상태는 그대로 보존했다. Vast 읽기 전용 API는 인스턴스를 running으로 보고하며, 인스턴스/Task10-H 프로세스 재시작이나 자원 회수는 하지 않았다. [미러 상태](../../outputs/task10i/stage1/mirror_status.json).
