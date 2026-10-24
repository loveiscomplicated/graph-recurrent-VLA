# Task 03-D Phase 3 — FPS Initialization vs Candidate Distribution 최소 계획

**상태: 계획 제출 / 구현·진단 승인 대기.** 저장 artifact와 코드를 읽어 재사용 가능 범위를 확인했다. Phase 3 코드 변경, FPS·matcher 진단, 테스트 및 결과 HTML 생성은 실행하지 않았다. 이 문서는 계획이며 Phase 3 결과 보고서가 아니다.

질문은 **S3의 낮은 rim Priority coverage가 초기 FPS 선택점에 민감한 현상인지, 동일 초기점에서도 후보 집합의 공간적 분포 때문에 지속되는 현상인지**다. 완료된 [Phase 2](03d_priority_fps_trace_phase2.md)의 step별 선택·거리 경쟁 분석은 반복하지 않는다.

## A. 기존 Task03 seed 1/2 artifact 재사용 가능 여부

Task03 run `2551448b0fae4c6bbce3caf48774275f`의 [Frame 0 NPZ](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0000_samples.npz), [frame metrics](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0000_metrics.json), [comparison config](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.json)에 요청한 여섯 조건이 모두 존재한다. 고정 조건은 `num_points=256`, `structural_ratio=.25`, `orientation_threshold_deg=20`, Frame 0 Box다.

| 방법 | Seed | 저장된 첫 Priority flat index | 저장된 최종 rim coverage |
| --- | ---: | ---: | ---: |
| S2 FD2 | 0 | 101598 | 16/103 |
| S2 FD2 | 1 | 119061 | 10/103 |
| S2 FD2 | 2 | 112919 | 10/103 |
| S3 PCA5 | 0 | 96553 | 3/103 |
| S3 PCA5 | 1 | 117010 | 5/103 |
| S3 PCA5 | 2 | 102621 | 3/103 |

위 coverage는 저장 metrics의 최종 Sampling 결과를 읽은 값이다. 새로 평가한 Priority coverage가 아니다.

- 모든 조건에 flat index, XYZ, pixel, object ID, Priority role과 23개 metadata 필드가 저장돼 있다. Box Priority 32점은 원래 선택 순서대로 앞부분에 보존된다.
- 모든 seed에서 Peg quota는 전체 129점/Priority 32점, Box quota는 전체 127점/Priority 32점으로 같다.
- 같은 estimator의 후보는 seed별 별도 배열이 아니라 공유 evidence 배열 하나로 저장된다. 후보 생성 코드도 seed를 읽지 않는다. S2의 297개와 S3의 599개 후보를 seed 간 동일하게 재사용할 수 있다.
- estimator나 기존 seed FPS를 다시 실행할 필요가 없다.

## B. 기존 결과로 답할 수 있는 질문과 부족한 정보

승인 후 seed 1/2의 저장 Priority를 기존 `CoverageMatcher`로 평가하면 다음을 비교할 수 있다. Seed 0은 Phase 2 결과를 재사용한다.

- 최초 Priority 선택점의 flat index와 XYZ
- rim에 매칭되는 실제 Priority node 수와 unique reference ID 수
- Priority depth의 min/median/p95/max
- 기존 negative ROI union 점유
- 저장된 최종 Sampling coverage와 Priority coverage의 관계

저장된 최종 coverage만으로도 S3의 낮은 coverage가 seed 0에서만 나타난 현상은 아니라는 점은 확인된다. Priority는 최종 선택점의 부분집합이므로 seed 1/2의 Priority coverage도 각각 5/103, 3/103을 넘을 수 없다.

다만 저장된 S2/S3 초기점은 여섯 조건 모두 다르다. 기존 seed 비교는 관측된 초기 선택 민감성을 보여주지만, 동일 초기점에서도 S3의 낮은 coverage가 지속되는지는 직접 답하지 못한다. 최종 coverage에는 Surface 선택 결과도 포함되므로 Priority와 구분해야 한다. 모든 seed 차이를 초기점 하나의 인과 효과로 단정하지 않는다.

## C. 필요한 최소 offline control 설계 및 변경 파일

두 후보 집합의 실제 교집합은 **252개**다. 기존 seed 요약을 먼저 확인한 뒤, 동일 초기점 비교가 필요하면 다음 한 쌍만 실행하는 계획이다.

1. 교집합을 flat index 오름차순으로 정렬하고 `intersection[len(intersection)//2]`를 공통 초기점으로 선택한다. 현재 이 값은 **114957**이다. 이 규칙은 GT reference와 rim ROI를 사용하지 않는다.
2. 동일 source geometry에서 초기 XYZ를 복원하고 S2/S3 입력 좌표가 정확히 같은지 검증한다.
3. 공통 flat index를 각 후보 배열의 local index로 변환해 기존 `fps_indices(points, 32, initial=[local_index])`를 호출한다.
4. 기존 API는 초기점 1개를 포함해 총 32개를 반환하며 나머지 31개를 기존 거리·tie 규칙으로 선택한다. `initial`이 있으므로 최초 RNG 선택은 생략된다.
5. 두 호출이 끝난 뒤 같은 103개 reference로 coverage, depth, negative ROI 요약만 평가한다. 전체 step trace는 생성하지 않는다.

차이가 유지되면 이 공통 초기점과 고정 FPS 조건에서 후보 집합 차이에 따른 결과 차이가 남는다고 해석한다. 차이가 줄거나 역전되면 초기점에 대한 민감성을 뒷받침한다. 한 초기점 결과를 모든 초기점으로 일반화하거나 후보 수·위치 분포 각각의 인과 효과로 분해하지 않는다.

추가할 코드는 다음 두 파일이다.

- `visual/runtime/test/run_priority_fps_seed_control.py`
- `visual/runtime/test/test_priority_fps_seed_control.py`

Phase 1 geometry/matching helper와 Phase 2 source 검증 helper를 재사용한다. 기존 `fps_indices()`의 초기 선택 지원으로 수행할 수 있으며 production FPS 구현을 복제할 필요가 없다. 교집합·XYZ·API 검증에서 불일치가 발견되면 보고하고 중단한다.

## D. 검증 항목, 완료 기준 및 산출물 계획

승인 후 다음을 확인한다.

- 원본 hash와 evaluated FPS·후보·allocation 정의 일치, 입력 보존
- 여섯 조건의 source XYZ·pixel·object·role·선택 순서 정렬과 quota 일치
- Seed 0의 Phase 2 reference ID 및 요약 재사용 일치
- Seed 1/2 평가가 동일한 저장 reference와 기존 동일 pair `2px AND 3mm` matcher를 사용함
- Control 실행 시 공통 초기 flat index·XYZ의 정확한 일치, 후보 순서 보존, 고유한 32점 반환 및 GT annotation 독립성

신규 테스트는 저장 결과 분석 경로의 FPS 미호출, `initial` 사용 시 RNG 미호출, 초기점 포함 총 32점, source/local index 대응, annotation 독립성 및 잘못된 입력의 명시적 실패를 확인한다. 관련 기존 sampling·matcher 테스트도 실행할 계획이다. 불일치하면 assertion이나 tolerance를 완화하지 않고 중단한다.

완료 산출물은 다음 위치에 작성한다.

- Markdown 결과 보고서: `docs/visual-graph-integration/03d_priority_fps_seed_control_phase3.md`
- 새 ExperimentLogger UUID 디렉터리의 HTML: `priority_fps_seed_control.html`
- 같은 UUID에 seed별 요약과 선택적 control 결과 JSON, source/input hash, 검증·테스트 로그, PASS/FAIL/SKIP, 초기점 비교 overlay 및 seed별 Priority/final coverage 비교 PNG

HTML에는 초기점·coverage 표, 최소 그림 및 결과 JSON 링크를 넣는다. Markdown에는 HTML과 원본 artifact 상대경로를 연결한다. 기존 `ExperimentLogger`와 artifact 등록 계약으로 로컬에 보존한다.

새 sampling 정책, candidate 제거, depth cutoff, GT 기반 quota, normal 재계산, 다른 frame sweep, PCL, Graph, BC, GPU 최적화 및 외부 업로드는 이 계획에 포함하지 않는다. 계획 제출 후 중단하고 구현 승인을 기다린다.
