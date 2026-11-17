# Task 03-D Phase 2 — Priority FPS Selection Trace 최소 계획

**상태: 계획에 따라 실행 완료.** 사용자 후속 요청으로 아래의 제한된 Phase 2 범위를 실행했습니다. 실제 결과는 [완료 보고서](03d_priority_fps_trace_phase2.md)에 기록했습니다. 인터페이스나 sampling 정책은 변경하지 않았습니다.

## 질문과 고정 범위

[Phase 1](03d_box_rim_bottleneck_phase1.md)은 완료로 인정하며, 기존 coverage·ROI·depth 집계를 반복하지 않습니다. 이번 질문은 frame 0 Box의 전체 후보에서 Priority 32점으로 좁혀지는 실제 선택 과정입니다.

| 조건 | S2 | S3 |
|---|---|---|
| Estimator | FD2 | PCA5 |
| 전체 후보 node 수 | 297 | 599 |
| 후보 unique reference coverage | 57/103 | 91/103 |
| Priority node 수 | 32 | 32 |
| Priority unique reference coverage | 13/103 | 3/103 |

고정값은 `num_points=256`, `structural_ratio=.25`, `orientation_threshold_deg=20`, `seed=0`입니다. 위 수치는 기존 결과이며, Phase 2에서 새로 실행한 결과가 아닙니다.

재사용할 자료는 다음과 같습니다.

- Task03 원본 run: `2551448b0fae4c6bbce3caf48774275f`
- 조건: `S2_n256_r0.25_t20_s0`, `S3_n256_r0.25_t20_s0`
- 원본 `f0000_samples.npz`, frame metrics, `comparison.json`, 저장된 evaluated source
- Phase 1 run: `d3f14ee494894daabb8ba2fab8e1b00c`의 결과와 source geometry 재구성 경로
- 저장 후보: `evidence/fd2/1/20/orientation_flat`, `evidence/pca5/1/20/orientation_flat`

후보와 feature를 재추출하지 않습니다. Frame 120, Surface FPS trace, 새로운 sampling 정책, outlier filtering, GT hole quota, threshold/seed sweep, PCL, Graph 및 BC 실험은 범위 밖입니다.

## A. 기존 코드 재사용 위치와 trace 필요성

**우선안은 runtime hook 없이 저장된 선택 순서와 기존 `fps_indices()`를 이용하는 사후 trace입니다.**

| 재사용 위치 | 소스에서 확인한 동작 |
|---|---|
| `visual/runtime/sampling.py::extract_candidates()` | 관측점은 `np.flatnonzero()`에 따른 원본 raster 순서이며, orientation 후보도 이 순서를 유지함 |
| `sample_points()`의 orientation 선택 | `oi = np.flatnonzero(m['orientation_candidate'])`와 `obj.points[oi]`를 사용하므로 FPS 입력 후보는 오름차순 `flat_index` 순서 |
| `sample_points()`의 물체별 seed | `SeedSequence([config.seed, object_id]).generate_state(1)[0]`를 정수로 변환. 내부 `BOX=1`이므로 이번 입력은 `[0, 1]` |
| `fps_indices()` | XYZ를 float64로 변환. 최초점은 로컬 `default_rng(seed).integers(N)` 한 번으로 선택 |
| `fps_indices()`의 거리 갱신 | 좌표별 squared distance를 계산하고 `np.minimum(..., out=distance)`로 기존 최소거리와 결합. 선택한 행은 `-inf`로 설정 |
| `fps_indices()`의 다음 선택 | `distance.argmax()` 사용. 정확한 동률에서는 가장 앞선 candidate-local index 선택 |
| `sample_points()`의 결과 수집 | Priority 순서를 `initial` 앞부분에 보존하고, 동일한 선택 행으로 좌표·역할·metadata를 수집 |
| `visual/runtime/test/run_sampling.py`의 NPZ 저장 | 결과 배열과 metadata를 재정렬하지 않고 그대로 저장 |

따라서 저장 배열에서 Box를 필터링한 뒤 `priority_selected=True`를 취하면 실제 Priority 선택 순서를 얻을 수 있습니다. 승인 후 저장 후보 XYZ에 **기존** `fps_indices(..., count=32, seed=파생값)`를 호출하고, 반환된 local index를 원본 `flat_index`로 매핑해 저장 순서와 원소별로 대조합니다. 별도 FPS 알고리즘이나 sampler 복제는 필요하지 않습니다.

내부 hook이 필요해진다면 최소 위치는 `fps_indices()`의 최초 RNG 선택 직후·거리 갱신 전과 반복문의 `argmax()` 직후·거리 갱신 전입니다. 현재 계획에는 이 hook 추가를 포함하지 않습니다. 저장 순서나 거리 재구성에 불일치가 있으면 원인을 보고하고, runtime 변경은 별도 검토 대상으로 남깁니다.

## B. 최소 trace 항목과 사후 분석 방법

각 방법의 32개 step에 다음 항목만 기록합니다.

| 항목 | 기록 내용 |
|---|---|
| 순서 | Step과 최초 RNG 선택 여부 |
| 식별 | Candidate-local index와 원본 `flat_index` |
| Geometry | XYZ와 CV-camera depth, metre 단위 |
| 거리 | 선택 직전 최소 squared distance, m² |
| Argmax/tie | 전체 최대값, 정확한 동률 수, 동률 중 첫 local index, 선택 규칙 충족 여부 |

최초 RNG step은 선택 전 거리가 모두 `+inf`이며 argmax 선택이 아닙니다. JSON에는 초기 상태를 명시하고, 유한 거리와 tie 통계는 `null` 및 사유로 기록합니다.

이후 거리 상태는 저장된 선택 prefix로 사후 계산합니다. 각 후보와 이미 선택된 점 사이의 거리를 **기존 코드와 같은 float64 좌표별 산술식**으로 계산하고 최솟값을 취합니다. 이미 선택된 행은 비교에서 제외합니다. 저장된 순서는 입력으로만 사용하며, 사후 계산으로 다음 선택점을 생성하지 않습니다.

전체 candidate별 거리 이력은 저장하지 않습니다. 후보 XYZ와 선택 순서로 다시 계산할 수 있으므로, step별 선택점 정보와 거리 경쟁 요약만 저장합니다.

**FPS 호출과 순서 검증이 끝난 뒤에만** rim reference matching과 negative ROI annotation을 수행합니다. FPS 호출은 후보 XYZ·개수·seed만 받으며, GT와 ROI는 선택·seed·거리 갱신에 접근하지 않습니다.

사후 분석은 다음으로 제한합니다.

- 기존 joint `2px AND 3mm` 조건으로 reference와 하나 이상 매칭되는 rim 후보와 나머지 후보를 구분
- 각 step 직전 두 후보 집합의 잔여 개수와 최대 최소거리 비교
- 전체 승자와 rim 후보 최대값의 거리 차이 및 정확한 동률 여부 확인
- Step별 새로 덮인 reference ID와 누적 unique reference coverage 기록
- 아직 덮이지 않은 reference에 매칭되는 잔여 후보의 거리 경쟁 확인
- 선택점에 기존 negative ROI 소속을 사후 부착하고, 겹치는 ROI를 합산해 인과 손실률로 해석하지 않음

잔여 rim 후보가 없는 등 비교 집합이 비어 있으면 해당 통계는 N/A로 기록합니다. 임의 depth cutoff나 새로운 rim corridor는 도입하지 않습니다.

## C. 정확한 변경 파일과 변경 범위

승인 후 다음 두 파일만 추가하는 계획입니다.

| 파일 | 변경 범위 |
|---|---|
| `visual/runtime/test/run_priority_fps_trace.py` | Frame 0 S2/S3 저장 자료 로딩, 기존 FPS 호출, 저장 순서 검증, 사후 거리·annotation, 로컬 결과 저장 |
| `visual/runtime/test/test_priority_fps_trace.py` | 순서 보존, 거리 재구성, tie/RNG, annotation 독립성 테스트 |

`sampling.py`, estimator, 기존 runner, Phase 1 구현은 변경하지 않습니다. RGB-D adapter와 organized geometry, Phase 1의 source/selected alignment helper, 기존 `CoverageMatcher`를 재사용합니다. Phase 1 전체 entrypoint는 재실행하지 않습니다.

실행 결과는 기존 `ExperimentLogger`가 생성하는 **새 UUID 폴더**에 trace JSON, 검증 결과, 간결한 분석으로 저장합니다. 기존 Artifact 계약으로 등록하고 Task03·Phase 1 원본은 보존합니다. 외부 업로드, 새로운 설치, commit은 수행하지 않습니다.

## D. 결과 불변 테스트·실제 실행 범위·완료 기준

아래 완료 기준으로 검증했습니다.

**테스트 범위**

- 기존 `visual.runtime.test.test_sampling`과 신규 `visual.runtime.test.test_priority_fps_trace` 실행
- 합성 입력에서 최초 RNG 선택, 정확한 tie, 중복 XYZ 및 float64 거리 재구성 확인
- 사후 분석 활성화 전후 기존 FPS 반환값 동일성 확인
- RNG 생성과 `integers()` 호출을 감시하여, 비어 있지 않은 이번 Priority 호출당 초기 draw 한 번 및 분석 측 추가 RNG 소비 없음 확인
- Annotation을 제거하거나 변경해도 선택 순서와 거리 결과가 동일함을 확인

**실제 실행 범위**

Frame 0 Box의 S2/S3 각각 32점만 대상으로 합니다. 분석 전후 비교를 위해 방법별 두 번, 총 네 번의 기존 Priority FPS 호출을 계획합니다. 독립적인 로컬 RNG를 사용하는 동일 입력 재현이며, Surface FPS와 전체 `sample_points()`는 실행하지 않습니다.

**완료 기준**

1. 현재 FPS와 Task03 저장 소스의 관련 정의가 일치하고, 입력 hash와 source geometry가 보존됨.
2. 저장 후보 순서와 재구성 XYZ가 실제 FPS 입력 순서·geometry에 대응함.
3. 재현한 32개 `flat_index`가 저장 Priority 순서와 정확히 일치하고, Priority role이 유지됨.
4. 모든 비초기 step이 재구성한 거리의 argmax 및 첫 행 tie 규칙을 만족함.
5. 저장 최종 선택점의 XYZ·pixel·물체·role·feature 행 정렬이 유지됨. Feature는 재추출하지 않음.
6. 기존 allocation 함수와 저장 counts를 대조하여 물체별 budget과 Priority quota가 일치함. 이를 위해 다른 물체의 FPS를 실행하지 않음.
7. 사후 누적 Priority coverage가 S2 **13/103**, S3 **3/103**이며, Phase 1 reference ID 집합과 정확히 일치함.
8. 원본 artifact를 덮어쓰지 않고 새 UUID 결과에 실행 범위와 PASS/FAIL을 기록함.

불일치하면 원인을 보고하고 중단합니다. GT 이산화, tolerance, ROI 또는 assertion을 완화하지 않습니다. Runtime hook을 추가하는 방식으로 자동 전환하지 않습니다.

선택 순서와 거리 경쟁이 확인되어도 원거리 후보의 **인과 기여율**이나 새로운 sampling 정책의 성능이 입증되는 것은 아닙니다. Phase 2는 완료 보고서에 기록한 범위에서 종료합니다.
