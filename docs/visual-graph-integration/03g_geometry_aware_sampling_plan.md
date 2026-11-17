# Task03-G — Geometry-Aware Structural Sampling 최소 검토 계획

## A. 기존 결과에서 이미 답할 수 있는 내용

이번 제출은 기존 문서와 현재 source를 읽은 **계획**이다. 코드 변경, 새 진단 계산,
테스트, FPS, simulator 실행은 하지 않았다. 결과 Markdown·HTML과 ExperimentLogger
UUID도 생성하지 않는다. 기존 dirty/untracked 파일은 보존한다.

연구 대상은 물체·task를 모르는 선택 규칙이다. 관측된 후보의 공간 범위와 국소 구조를
제한된 예산으로 대표하고, 일반 표면 및 향후 Task-Relevance Scoring에 필요한 후보
정보를 함께 남길 수 있는지를 본다. 특정 Peg/Box 구조의 coverage 최대화가 목적은 아니다.

| 기존 근거 | 이미 확인된 내용 | 이번 계획에 주는 제약 |
| --- | --- | --- |
| [03-D Phase1](03d_box_rim_bottleneck_phase1.md) | Frame0 S3 Box 후보는 관측 rim reference 91/103을 덮지만 Priority와 Final은 3/103이다. 후보 599개, Priority 32개이며 관측률은 103/103이다. | 후보 수 증가와 후보 coverage는 선택 후 보존을 보장하지 않는다. 후보 재검출로 원인을 다시 조사하지 않는다. |
| [03-D Phase2](03d_priority_fps_trace_phase2.md), [Phase3](03d_priority_fps_seed_control_phase3.md) | 저장 FPS에서 rim 후보가 전체 거리 경쟁에서 밀린 경로가 확인됐다. 공통 초기점 control에서도 S3의 낮은 coverage가 남았다. | 초기점 하나나 후보 밀도 하나의 독립 인과 효과로 손실을 설명하지 않는다. 새 seed·trace 실험은 하지 않는다. |
| [03-E](03e_priority_fps_metric_control.md) | S3 λ=1→0에서 Box Priority refs는 3→9, Final은 3→12/103이다. Peg Final은 151→160/462이지만 구간별 손익이 다르다. Box Surface mean/p95는 11.253/18.751→11.580/19.077mm다. | Z 경쟁 완화만으로 국소 구조 누락이 해결되지 않는다. 저장 Global λ=0을 직접 비교 기준으로 삼고 λ=1은 맥락상 기준선으로 남긴다. |
| [03-F](03f_local_structural_sampling.md) | Global λ=0→UV control에서 Peg C→P p95/max는 6.662/8.198→8.904/15.126mm, Box는 29.779/46.052→35.093/96.389mm다. Final refs는 Peg 160→140, Box 12→3이다. | UV 후보 개수 기반 분할·균등 quota는 채택하지 않는다. UV group 수·quota를 다시 튜닝하지 않는다. |
| 03-F synthetic | 길이 1:2:4 clean 구조의 Global/UV 선택 수는 [5,8,19]/[8,12,12]다. 조밀 직선에서 UV control의 reference p95가 5.625→7.500mm로 악화됐다. Noise 선택 수는 두 방식 모두 1/32였다. | 후보 개수는 물리적 범위를 대신하지 못한다. 이 noise fixture는 UV가 noise quota를 과도하게 받았다는 증거도, 일반적인 noise 강건성 증거도 아니다. |
| [03-AB](03ab_adaptive_node_budget.md) | 평균 N −62.93%, sampling 시간 −45.55%와 함께 Peg retention 61.28→28.16%, Box 11.43→7.77%의 손실이 발생했다. 총 N과 Priority cap이 함께 줄었다. | 작은 mask를 단순 geometry로 취급하지 않는다. 연산량 절감으로 보존 성능을 대신 판단하거나 N 감소와 Priority 감소의 효과를 분해했다고 주장하지 않는다. |

이번 기준은 S3 PCA5, ep000 Frame0, 전체 N=256, seed=0, threshold=20°다.
Peg는 X=11,780/C=703/P=32/S=97/F=129, Box는 X=11,665/C=599/P=32/S=95/F=127이다.
X는 원본 유효 관측 support, C는 기존 Structural 후보, P/S/F는 Priority/Surface/Final이다.
같은 source index·raster 순서·원본 CV-camera XYZ(m)·intrinsic·mask·feature provenance를 유지한다.
객체 ID는 소속과 예산 분리에만 쓰며 객체 이름에 따른 규칙은 두지 않는다.

03-F의 Global 행은 03-E의 저장 λ=0을 읽은 결과다. 03-F 최종 UUID
`1f7ad262dd2e4b2fb0ac805a4eda5278`은 최초 수치 실행
`c4709984427d4283815a3971a415b4af`의 문서 후처리다. 이를 독립 반복 실험으로 세지 않는다.
이미 확인된 관측 소실·거리 경쟁·밀도/UV 실패·면적 budget 손실은 재진단하지 않는다.

## B. 추천 후보 하나: 관측 연속성 patch의 3D 범위 배분 + 기존 FPS

**검토 후보는 하나만 둔다.** 관측 연속성으로 분리한 patch마다 그 안의 후보가 차지하는
3D 범위에 따라 Priority를 배분하고, patch 내부 선택과 후속 Surface에 기존 FPS를 쓴다.
이는 다음 feasibility 비교를 위한 조건부 후보이며, 최종 정책 채택이나 grouping의
신뢰성이 확보됐다는 결론이 아니다. 특히 아래 synthetic evidence 공백을 숨기지 않는다.

### 재사용 가능한 evidence와 그 의미

| 현재 source/저장 자료 | 재사용 정보 | 해석 한계 |
| --- | --- | --- |
| `geometry.py:OrganizedGeometry`, Task01 저장 `f0000.npz` | 원본 dense XYZ, valid depth, 물체 support, overlap 제외, flat index 대응 | 관측 support는 물리적 표면 완성본이 아니다. tracker 오염과 positive depth outlier도 남는다. |
| `edge_evidence.py:PairEvidence`, Task02-C 저장 `f0000_pairs.npz`의 `pca5/{peg,box}/*` | right/down pixel pair의 `pixels`, `supported`, `overlap`, `depth_jump`, `normal_angle_deg` 등 | 인접 pixel의 독립 측정 evidence다. 물리 edge·구조 label·GNN edge가 아니다. |
| `GeometryParameters`, 저장 normal/pair 설정 | 기존 depth-jump gate: abs 0.01m + rel 0.02×min(z1,z2), 기존 normal angle 20° | 표면 분리의 충분조건이 아니다. 해당 값은 기존 관측 진단 기준이며 이번 결과에 맞춰 조정하지 않는다. |
| Task03 저장 orientation 후보와 03-E의 C/X | 후보 membership, 원본 XYZ·순서, 고정 anchor·seed·예산 | 후보에는 false response가 있다. 후보 연결성이나 물리적 중심선 순서를 보장하지 않는다. |
| Task02-B 저장 PCA5 normal/validity 및 pair angle | normal 유효성과 국소 변화 | 새 normal estimator·candidate detector 없이 쓸 수 있다. 단, Task03의 optimized PCA5와 저장 evidence의 source/config 대응을 먼저 확인해야 한다. |

저장 schema는 기존 writer에서 확인했다. 이번에는 NPZ를 새로 분석하거나 patch 수·분포를
계산하지 않았다. 현재까지 이 patch가 안정적인 sampling 단위라는 실험 근거는 없다.
기존 source/hash 계약이 맞지 않거나 필요한 저장 evidence가 없으면 재추출·대체 추정으로
진행하지 않고 그 공백을 보고한다.

### 승인 후 검토할 단일 규칙

1. **관측 patch 구성:** 물체별 X의 4-neighbour pair 중 양 끝이 같은 물체의 유효 support이고,
   overlap이 없으며, 저장 `depth_jump=False`, finite normal angle ≤20°인 pair만 임시 연결에 쓴다.
   그 연결의 성분을 patch로 삼는다. C만 연결하지 않고 X를 통과하는 관측 경로를 허용해
   후보 사이의 빈 pixel을 곧바로 구조 단절로 취급하지 않는다. 유효한 연결이 없는 관측점도
   singleton으로 보존한다. Normal/depth discontinuity pair를 물리적 선분으로 잇지 않는다.
2. **후보 귀속과 물리 범위:** 각 후보는 원본 source index로 해당 patch에 귀속한다.
   C가 없는 patch에는 Priority를 배분하지 않지만 X와 Surface eligibility는 그대로 둔다.
   patch g의 후보 집합 Cg에서 `s_g = ||max(Cg.xyz)-min(Cg.xyz)||₂`를 계산한다.
   이 3D bbox 대각선은 관측된 범위의 proxy이며 구조 길이·곡선 길이·표면적은 아니다.
   후보 개수는 capacity에만 쓰고 weight에는 쓰지 않는다. Singleton/동일 XYZ는 s=0이다.
3. **32점 배분:** 기존 전체-C seeded anchor를 첫 Priority로 유지한다. 이 1점을 해당 patch의
   quota에 먼저 계상하고 남은 31점을 s에 비례해 capacity-capped largest remainder로 배분한다.
   각 patch 최소 1점이나 동일 quota를 강제하지 않는다. Patch/동률 순서는 최소 flat index다.
   capacity 포화분은 같은 규칙으로 재배분한다. 양의 span capacity를 모두 썼거나 전부 s=0이면
   남은 slot만 기존 Global λ=0 순서에서 중복 없이 채운다. 실제 데이터에서는 그 저장 순서를
   읽으며 새 Global FPS를 실행하지 않는다. 이 보충은 outlier 제거가 아니다.
4. **기존 FPS 재사용:** patch 내부에서는 원본 XYZ의 3D Euclidean `fps_indices()`를 쓴다.
   Anchor patch는 원래 anchor를 `initial`로 사용하고, 나머지는 동일 object seed와 flat 순서를
   사용한다. Anchor를 F의 첫 행에 두고 나머지 patch 결과를 최소 flat 순으로 모은다.
   모든 P를 `initial`로 넣어 원본 X에서 기존 3D Surface FPS를 수행한다. P는 각 32점,
   F는 Peg 129/Box 127, 전체 256점이다. 좌표는 변환하거나 feature로 대체하지 않는다.
5. **원본 후보 보존:** 선택 NPZ와 별도로 전체 C/X 및 patch membership·span·quota·보충 출처를
   남긴다. 선택되지 않았다는 이유로 noisy label을 붙이거나 후보를 삭제하지 않는다.
   이는 후속 Task-Relevance 실험에서 같은 후보를 다시 비교하기 위한 provenance이며,
   32점 P에 모든 유의미한 후보 정보를 보존한다는 보장은 아니다.

새로 정할 group 개수는 없다. Patch 수는 고정된 관측 gate의 결과다. Normal angle을 낮추거나
연결 범위를 늘리는 sweep, 작은 성분 삭제, 깊이 cutoff, GT 기반 병합·분할은 하지 않는다.
임시 연결은 sampling용 성분 label을 계산하는 데만 쓰고 Graph edge로 저장·노출하지 않는다.

### 일반성 근거와 예상 실패

일반성의 근거는 **형상·task label 대신 같은 관측 규칙과 실제 3D 범위를 사용한다는 설계상 성질**이다.
성능 일반화의 실증 근거는 아직 없다. 고정 patch의 bbox 극값이 같으면 내부 후보 밀도가
높아져도 weight는 늘지 않는다. 길게 퍼진 후보는 compact 후보보다 큰 weight를 받을 수 있고,
작고 조밀한 noise cluster에는 별도 최소 quota를 주지 않는다. Pixel이 가까워도 관측 depth/normal
불연속을 통과하지 않게 해 다른 표면을 분리할 가능성이 있다. 이들은 검증할 가설이다.

| 다루려는 문제 | 규칙의 기대 효과 | 남는 실패 |
| --- | --- | --- |
| 조밀 후보에 예산 집중 | 후보 수 대신 고정 patch의 3D span으로 배분 | 밀도·해상도 변화가 normal validity나 patch 연결 자체를 바꾸면 배분도 달라진다. Density invariance를 보장하지 않는다. |
| 긴 구조의 범위 누락 | 넓은 Cg span에 더 많은 quota와 patch 내 FPS | 곡선·분기·면적은 bbox로 표현되지 않는다. 두께가 다른 patch 간 공정한 배분도 보장하지 않는다. |
| 가까운 pixel의 다른 표면 | 저장 depth jump와 normal angle을 연결 veto로 사용 | gate 미만의 깊이 차이, 비슷한 normal의 별도 표면, tracker bridge는 병합될 수 있다. |
| 작은 noisy cluster의 과다 quota | 동일 quota·patch별 최소 예약을 없앤다 | 멀리 퍼진 noise, 가는 noise bridge, anchor가 noise에 놓인 경우는 억제하지 못한다. Compact한 실제 중요 구조도 적게 받을 수 있다. |
| 좁은 구조의 Global 경쟁 소실 | 분리된 patch 내부에서 Priority를 선택 | 일반 표면 경로로 같은 patch에 연결되거나 quota가 0이면 다시 누락된다. 같은 seam의 양쪽 관측 띠가 별도 patch가 되어 중복 예산을 받을 수도 있다. |

BBox span은 camera 축 회전에도 민감하고 단일 극단점에 크게 변한다. Normal-invalid 영역과
급한 곡률은 patch를 잘게 나눌 수 있다. 오염 제거 없이 이 실패를 공개한다. 일반 Surface는
같은 X와 FPS를 사용하더라도 P 초기 집합이 바뀌므로 mean/p95가 악화될 수 있다.

**Evidence 부족에 따른 중단 경계:** 기존 03-F synthetic은 sparse C·XYZ·pixel과 평가 annotation만
제공한다. Dense X, normal, adjacent pair evidence가 없으므로 위 관측 연속성 patch를 검증할 수 없다.
구조 ID·선분 endpoint·noise label로 patch를 대신 만들거나 점 사이를 보간해 support를 만들지 않는다.
이 입력에서는 evidence-unavailable을 명시하고 Global λ=0 보존 경로만 비교한다. 동률을 새로운
규칙의 성공으로 세지 않는다. 따라서 기존 자료만으로 full-rule의 길이·밀도·시점 강건성을
입증하는 검증 관문은 충족할 수 없다. 이를 해소하려고 이번 범위를 늘리지 않는다.
구현 승인이 있더라도 이 공백을 받아들인 제한적 실제-frame feasibility인지 먼저 명시해야 하며,
full-rule 검증이 필수라면 추가 관측 evidence가 승인·확보될 때까지 구현을 보류한다.

## C. 재사용 위치와 정확한 최소 변경 파일

현재 제출에서 추가하는 파일은 **이 계획 Markdown 하나**다. 승인 후에도 production sampler나
공통 evaluator를 고치지 않는 별도 offline runner 구성을 우선한다.

| 읽기 전용 재사용 위치 | 실제 재사용 대상 |
| --- | --- |
| [sampling.py](../../visual/runtime/sampling.py) | `fps_indices(points,count,seed,initial)`의 거리·초기화·동률·중복 계약. `sample_points()` 전체 production 경로를 복제하거나 변경하지 않는다. |
| [geometry.py](../../visual/runtime/geometry.py), [edge_evidence.py](../../visual/runtime/edge_evidence.py) | `OrganizedGeometry.observed_mask`, `GeometryParameters`, `PairEvidence` 필드 해석. `pair_evidence()` 및 normal 추출을 새로 호출하지 않는다. |
| [03-F runner](../../visual/runtime/test/run_local_structural_sampling.py) | `load_actual()`의 저장 E λ=0·C/X·anchor 검증, `spatial_metrics`, `summary`, `nearest`, `grid_definition/grid_ids`, `evaluate_synthetic`, `FIXTURES`. UV grouping/`choose_priority`/`select_local`은 재사용하지 않는다. |
| [03-E runner](../../visual/runtime/test/run_priority_fps_metric_control.py) | `feature_rows`, `evaluation_inputs`, `score_case`, `peg_labels`, `rim_labels`, `segment_spacing`. `select`/`execute_controls`/실제 baseline 재실행은 사용하지 않는다. |
| [sampling_evaluation.py](../../visual/runtime/test/sampling_evaluation.py) | `CoverageMatcher`, `staged_reference_coverage`, 기존 same-pair matching·분모·중복 제거 계약 |
| [03-D helper](../../visual/runtime/test/run_rim_bottleneck.py) 및 [trace helper](../../visual/runtime/test/run_priority_fps_trace.py) | `source_contract`, `require_equal`, 기존 source/좌표/ROI 검증과 보존 helper. 새 진단 trace는 만들지 않는다. |
| [ExperimentLogger](../../utils/experiment_logger.py), [experiment_artifacts](../../utils/experiment_artifacts.py) | 새 평가 UUID, `record`, `log`, `link_output`, `digests`, `register_existing_output`. 팀별 logger/publisher를 추가하지 않는다. |

`sampling.allocate_budget()`는 입력 shape `(2,)`인 **두 물체 전용 API**다. 임의 patch 수에
그대로 쓸 수 없다. 공유 allocator 확장 필요성을 검토했으나, 이번 test-only 가설 때문에
production 계약을 넓히지 않는다. Runner 안에서 patch quota만 처리하는 작은 helper를 두고
물체 예산 129/127은 저장값 그대로 쓴다. 범용 allocator를 별도 모듈로 병렬 도입하지 않는다.

| 승인 후 최소 파일 | 예정 변경 |
| --- | --- |
| `visual/runtime/test/run_geometry_aware_sampling.py` — 신규 | 저장 evidence adapter, patch labeling·span quota helper, 기존 FPS 호출, E/F helper 조합, 저장/비교/그림/정적 HTML. 기존 runner의 전체 실행 함수를 호출하지 않는다. |
| `visual/runtime/test/test_geometry_aware_sampling.py` — 신규 | 연결 veto·quota·fallback·GT 독립성·source/feature 정렬에 필요한 계약만 검증한다. 성능 향상을 assertion으로 만들지 않는다. |
| `docs/visual-graph-integration/03g_geometry_aware_sampling.md` — 승인 후 신규 | 실제 결과와 미실행 범위, trade-off, evidence 공백, 채택 보류 판단. 이번에는 생성하지 않는다. |

기존 D/E/F/AB 파일·baseline·source snapshot·sidecar, `sampling.py`, normal/candidate detector,
`sampling_evaluation.py`, 공통 simulator/logger/publisher의 변경은 계획에 포함하지 않는다.
기존 helper 인터페이스가 부족하면 부족한 지점을 보고하고 최소 계획을 갱신한다.

## D. 고정-budget offline 비교·검증 기준과 산출물 계획

### 조건과 실행 순서 — 별도 구현 승인 이후에만 적용

1. 원본 source/config/input/artifact hash와 Git dirty 상태를 보존한다. 기존 source 계약,
   E/F의 저장 Global λ=0 선택/역할/XYZ/예산, Task03 원본 λ=1과 pair/normal evidence의 대응을
   확인한다. 기준선 재현은 저장 배열·지표 대조이며 FPS 재실행이 아니다. 불일치하면 중단한다.
2. 단일 patch/배분/초기화 규칙과 평가 정의를 config에 고정하고 새 계약 테스트를 수행한다.
   저장 same-pair matcher 및 빈 집합/N/A 계약을 완화하지 않는다. 표면 단절 veto, singleton·0 span,
   capacity 포화·정수 동률·anchor 1점, 32점 충족/부족, annotation 독립성을 확인한다.
   정렬 오류나 baseline 불일치를 우회하는 임의 fallback은 금지한다.
3. 실제 데이터는 Frame0의 새 규칙만 선택한다. P 저장 후 Surface를 선택하고 전체 선택을
   `selection_before_gt.npz`로 고정한다. 이후에만 GT reference·구간·ROI 평가를 연결한다.
   선택 helper의 입력에는 pose/hole/rim/face-pair/reference/ROI/task score를 넣지 않는다.
4. 동일 C/X/N/P와 seed를 유지한 새 규칙 대 **03-E 저장 Global λ=0**을 주 비교로 둔다.
   저장 Global λ=1, 실패한 03-F UV control도 같은 표·그림에 표시한다. λ=.25, UV sweep,
   Adaptive 및 추가 frame/episode는 실행하지 않는다. Grouping·quota·patch내 3D metric 변경이
   묶인 비교이므로 개선을 단일 요인의 인과 효과로 귀속하지 않는다.
5. 03-F 저장 synthetic NPZ/manifest와 Global/UV 선택을 우선 재사용한다. 7개 fixture의 좌표,
   충돌 제거, 길이·밀도·시점·가림 설정을 바꾸지 않는다. 새 규칙의 evidence-unavailable 경로는
   저장 Global 선택을 재사용하며 새 FPS를 돌려 동일 baseline을 만들지 않는다. Full-rule의
   synthetic 비교는 N/A로 남긴다. 기존 가림 사례의 경계 후보 0과 한 선분만 보이는 한계를 유지한다.

Synthetic은 `line_uniform`, `line_dense`, `separated_clean`, `separated_noise`, `camera_near`,
`camera_far`, `camera_occluded`를 그대로 제시한다. 관측 불가능한 geometry 복원을 요구하지 않는다.
추가 synthetic evidence나 fixture 생성은 이 최소 계획의 자동 후속 작업이 아니다.

### 평가 정의와 판단 기준

| 우선순위 | 기록할 내용 | 정의 보존 및 해석 |
| --- | --- | --- |
| Primary: C→P | 원본 3D 거리의 mean/p50/p95/max, ECDF, worst source/UV/XYZ, raw 거리 배열 | 03-F `spatial_metrics`의 Euclidean 최근접 정의를 유지한다. 선택에 쓴 patch 내부 거리로 primary를 바꾸지 않는다. |
| Primary: 공간 범위·국소 빈 영역 | 기존 독립 UV 4×4, XYZ 최대 2×2×2 구역의 후보/선택 count, P=0 구역, 구역별 p95/max와 동등 구역 평균 | 모든 조건에 동일 X bbox·경계를 쓴다. 선택 patch 충족률은 설명용으로만 별도 기록하고 primary 성공 지표로 쓰지 않는다. |
| Primary: 밀도 민감도 | 기존 구역별 candidate density·선택률·점유 share, uniform/dense 직선과 길이 1:2:4의 기존 비교 | 전체 C→P 분위수는 후보 밀도에 가중된다. 동등 구역 지표와 함께 읽으며, full-rule synthetic 미검증을 명시한다. |
| Primary: 일반 Surface | 전체 X→F 3D mean/p95와 원시 거리, 보조 max | 03-E `score_case`의 정의다. S만의 거리나 깨끗한 부분만의 거리로 교체하지 않는다. |
| Primary: synthetic 길이·밀도·시점 | 03-F `evaluate_synthetic`의 visible reference→P 거리, 구조별 실제 선택 수, noise 점유 및 raw/정규화 거리 | annotation은 평가 전용이다. 이번 자료로 full-rule 항목은 N/A다. Fallback 동률을 강건성 통과로 판정하지 않는다. |
| Secondary: Peg/Box 구조 | C/P/S/F 단계별 unique reference, 실제 matching node 수, P/S 교집합·Surface 추가분, 구간별 count와 source | 같은 point/reference pair가 2px AND 3mm를 만족해야 한다. 관측 가능한 reference를 retention 분모로 쓰고 GT-visible 분모와 구분한다. |
| Secondary: 간격/누락 | 기존 Box verified line 투영 간격·missing-reference run span, Peg 및 ambiguous N/A | reference 배열 순서를 물리 순서로 간주하지 않는다. 실제 연속 미보존 길이로 보간하지 않는다. |
| Secondary: 오염·depth | 기존 known-negative ROI별 count/union, X/C/P/S/F depth min/median/p95/max와 X 분위수 구간 점유 | ROI·depth를 선택으로 피드백하지 않는다. 일반 Surface의 평면/negative 점유를 자동 오선택으로 취급하지 않는다. |

Box X의 max depth 30.726m 때문에 기존 XYZ 격자에서 C가 한 구역에 들어가는 한계가 있다.
이를 고치려고 bbox·격자·정규화 분모·outlier를 변경하지 않는다. Raw 3D tail, worst-point 확대,
기존 UV 구역과 동일 축척의 overlay를 함께 제시한다. 후보에서 먼 위치는 관측 C 위의
미피복 위치이며 연속 빈 공간의 크기를 복원한 값이 아니다.

실행의 계약 통과와 알고리즘의 유용성을 분리한다. 계약 관문은 같은 C/X, source/pixel/XYZ 정렬,
각 P=32/F=129·127, 중복·padding 없음, P⊆C⊆X와 F⊆X, GT annotation 변경 시 선택 불변이다.
원본 feature 23 field는 03-E `feature_rows()`처럼 동일 S3 baseline에 exact available인 행만
gather하고 availability mask/compact positions를 기록한다. 나머지 feature를 채우거나 다시
추출하지 않는다. 완전한 Vision Tensor 출력 검증을 주장하지 않는다.

유용성은 λ=0 대비 **물체별·지표별 delta를 모두 공개**해 판단한다. C→P tail·국소 미피복,
일반 Surface mean/p95, noise/negative 점유의 손익을 한 표에 놓는다. Tail 개선이 Surface나
noise의 악화를 동반하면 trade-off로 기록하고 일괄 개선이라 부르지 않는다. 임의의 종합 점수,
사후 허용 오차, Box 특정 rim의 0→양수 같은 성공 조건은 만들지 않는다. Secondary GT 개선만으로
primary 손실을 상쇄하지 않는다. Patch 근거·full-rule synthetic 검증이 부족하면 feasibility의
한계 또는 미검증으로 종료한다. 한 frame과 기존 fixture만으로 일반화 성공을 선언하지 않는다.

### 결과 산출물 — 이번에는 생성하지 않음

별도 승인된 실행에는 기존 `ExperimentLogger(team='visual', job_type='evaluate', seed=0)`의
새 UUID를 사용한다. Project/group/UUID 규칙은 그대로 따르고 `WANDB_MODE=disabled`로 로컬
기록한다. 원본 run ID/hash, 실제 적용 config, source snapshot·dirty diff, 입력·출력 보존 검증을
남긴다. 원본 파일에 새 sidecar를 쓰지 않고 새 산출물만 기존 artifact API로 연결한다.

| 예정 경로 | 내용 |
| --- | --- |
| `experiments/<run_id>/geometry_aware_sampling.json` | 고정 설정, 원본 run 연결, baseline/새 규칙의 지표와 delta, patch·quota·fallback 근거, evidence 공백/N/A |
| `experiments/<run_id>/geometry_aware_sampling.npz` | 전체 C/X, 원본 XYZ/source, P/S/F·역할, patch membership/span/quota, 원시 거리/매칭, exact available feature와 mask. Evaluation annotation은 별도 namespace |
| `experiments/<run_id>/selection_before_gt.npz` | GT 평가를 연결하기 전 선택과 입력 hash |
| `experiments/<run_id>/geometry_aware_sampling.html` | 외부 CDN·JS fetch 없는 간단한 정적 비교 페이지 |
| `experiments/<run_id>/plots/*.png` | 후보/patch 분포와 선택점, 동일 조건의 3D 국소 coverage와 tail, 기존 synthetic 비교/미검증, Peg/Box 구조, Surface·noise·depth trade-off |
| `experiments/<run_id>/verification.json`, `tests.log`, `source_hashes.json`, config/source snapshot | 계약 PASS/FAIL/N/A, 원본 불변성, provenance·실행 범위·재현 정보, 상대 링크/PNG 확인 |
| `docs/visual-graph-integration/03g_geometry_aware_sampling.md` | 승인 후 결과 보고서. 모든 trade-off·중단·미실행 항목 포함 |

HTML에는 후보 분포, 선택점, 3D 국소 coverage, synthetic 비교, Peg seam/Box rim 보존,
Surface trade-off 및 원본 JSON/NPZ/PNG의 상대 링크를 포함한다. 미실행 비교도 N/A 이유와
기존 자료 링크를 보여 준다. Markdown과 HTML은 아래 상대경로로 양방향 연결한다.

```text
결과 Markdown → ../../experiments/<run_id>/geometry_aware_sampling.html
HTML → ../../docs/visual-graph-integration/03g_geometry_aware_sampling.md
```

실행 대상은 기존 호스트의 CPU NumPy/SciPy offline 경로만이다. 이번 계획에서는 CPU를 포함해
어떤 계산 backend/device도 실행하지 않았다. 승인 후에도 simulator·renderer·새 episode,
GPU/CUDA/MPS/PCL, FPS latency benchmark, Graph/GNN/BC·encoder·stale cache 구현은 범위 밖이다.
Dependency 설치·외부 업로드·Git commit도 하지 않는다.

03-AB는 제약 근거로만 남긴다. Adaptive와 이 규칙의 결합은 각 구성요소를 따로 검증한 뒤의
독립 실험이다. 총 N 감소와 Priority 감소를 분리해 다룰 필요도 그때 검토한다.
[기존 stale 계약](00_interface.md)에 따른 마지막 유효 geometry·내부 연결·관측 EE pose 유지,
valid/age 표시를 바꾸지 않는다. 현재 관측의 빈 출력과 stale 유지 정책을 혼동하지 않는다.

**이번 제출은 계획 Markdown 하나로 끝내며 별도 구현 승인을 기다린다.**
