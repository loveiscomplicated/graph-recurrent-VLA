# Task05-E — Raw Extended Delaunay vs Strict EDE 비교 계획

상태: **계획 제출·구현 및 실험 승인 대기**. 이번 변경은 이 Markdown 하나다. 코드 수정, Raw Graph 생성, 테스트, benchmark, GPU·simulator 실행, 결과 HTML 생성은 하지 않았다. 아래 구현·측정·검증은 모두 승인 후 수행할 계획이다. 새 ExperimentLogger UUID도 아직 생성하지 않았다.

목표는 Strict EDE가 제거한 연결의 관측상 문제와 검증 비용을 함께 확인하여, Raw EDE를 후속 GNN ablation 후보로 검토할 근거를 만드는 것이다. Mesh 복원, 물리적 연결의 정답 판정, GNN 성능 확정, 최종 정책 채택은 이번 비교의 범위가 아니다.

## A. 기존 후보를 재사용하는 최소 Raw EDE 구현

### A1. 확인한 구현과 재사용 지점

루트 지침, `README.md`, `envs/maniskill/README.md`, `docs/PROJECT_CONTEXT.md`, Visual README와 runtime README, [engineering rules](../engineering_rules.md), [tracking](../experiment_tracking.md), [artifact 운영 규칙](../huggingface_artifacts.md)을 읽었다. 다음은 현재 branch의 파일과 저장 자료를 읽어 확인한 사실이며 새 실행 결과가 아니다.

| 확인 대상 | 현재 동작과 계획에 주는 제약 |
|---|---|
| [`extended_delaunay_edges.py`](../../visual/runtime/extended_delaunay_edges.py) | `candidate_edges(object_id, source_index, component_id, image_shape, extended=True)`가 검증 없이 canonical 후보·origins·triangles·component 상태를 반환한다. Raw의 직접 재사용 지점이다. |
| 같은 파일의 builder | `build_extended_delaunay_edges`는 후보 생성 후 `prepare_validation`, Edge별 raster와 `validate_cover`를 실행한다. Raw에서 이 builder를 호출하고 accepted를 덮어쓰는 방식은 사용하지 않는다. |
| 같은 파일의 정렬 | `collinear_order`는 Python integer exact cross/projection과 UV/source/row tie-break를 사용한다. `directed_edges`는 양방향 변환 후 `np.unique(..., axis=0).T.copy()` 순서를 반환한다. |
| [`run_extended_delaunay_edges.py`](../../visual/runtime/test/run_extended_delaunay_edges.py) | 저장 입력·packet IO·불변성 검사·지표 helper를 선택적으로 재사용할 수 있다. `load_inputs`와 `execute` 전체는 validation·후보 생성·파일 쓰기를 포함하므로 순수 입력 reader로 호출하지 않는다. |
| [`graph/vision_input.py`](../../graph/vision_input.py) | v2는 k-NN/TMD/DE/Strict EDE 정책만 허용한다. 기존 DE/EDE metadata 검사는 accepted pair 집합·origin/기하를 검사하지만 전체 후보 재현이나 directed 배열의 exact 순서까지 보장하는 Raw branch는 없다. |
| Task05-B/C 저장 자료 | B NPZ에 `EDE_candidates`, `EDE_origins`, `EDE_accepted`, `EDE_triangles`, `EDE_edges`가 있고 JSON에 candidate별 복수 reasons/counts/witnesses가 있다. C에도 optimized audit·packet·상세 cover/pair 자료가 있다. 새 Raw 진단을 생성하지 않고 기존 Strict 진단을 읽을 수 있다. |

### A2. 고정 입력과 기준선

| 실행 | UUID | 재사용 범위 |
|---|---|---|
| Task03-K | `9185355a32b8433abc70f1738f983377` | `selection_before_gt.json/.npz`, condition별 선택 row/source/provenance |
| Task04 | `9f90293b227e4dbfaaa6290afad9cc3b` | `packets/<key>.npz`, `runtime_evidence/f<frame>.npz`: 저장 Point/Feature/좌표·observed component·관측 evidence |
| Task05 | `91ecd0413a87455cabc5e813cddaea8d` | 기존 입력 hash·component/fallback 정합 기준. TMD를 새 비교 정책으로 실행하지 않는다. |
| Task05-B | `9f36f7c89dbd4097a8be9358138d1537` | `extended_delaunay_edgewise.json/.npz`, `records/<key>.json`, `packets/<key>_EDE.npz`: 후보와 Strict topology의 기준 |
| Task05-C | `185bc034567d429daf3f1d09f94dc56c` | `ede_exact_optimization.json/.npz`, `records/<key>_optimized.json`, `packets/<key>_optimized.npz`: Strict exact 교차 확인·기존 상세 진단 |
| Task05-D | `be206b04f7b245c4a9436317b8f87987` | CPU/MPS 실행 범위·기존 timing·equivalence 참고. 이번에 GPU runner를 호출하지 않는다. |

조건은 frame `0,18,27,45,56,66,80,120` × keep_ratio `0.5,1.0` × phase `0`, 총 16개다. key는 기존 `f0000_r0.5`, `f0000_r1` 형식을 따른다. object `0=Peg`, `1=Box`의 32개 slice를 빈 object까지 포함한다.

원본 RGB-D는 `visual/data/step2/ep000.npz`의 필요한 관측 배열만, raw mask는 `visual/results/runtime_tracking/ep000/masks.npz`를 읽는다. `source_frame`으로 정렬하고 기존 input hash와 비교한다. GT mesh/pose/segmentation 배열을 읽거나 GT로 연결을 분류·제거하지 않는다. 기존 artifact의 카메라/TCP 변환 metadata는 불변성 확인 용도로만 사용한다.

저장 Point, Camera/EE XYZ, F=5와 compact feature, availability/validity/diagnostics, provenance, object/component/source-index/row ordering을 dtype·shape·bytes 및 NaN 위치까지 보존한다. Sampling, PCA5, feature extraction, EE transformation, observed component 계산·재분할은 재실행하지 않는다. Receiver의 기존 좌표 정합 검사는 저장 좌표 검증이며 새 좌표 추출이 아니다.

[Task05-B 기록](05b_extended_delaunay_edgewise.md)의 후보 U=30,143, accepted U=25,046, rejected U=5,097, Strict isolated=302, singleton=270은 **승인 후 입력·저장 결과를 대조할 기준**이다. 이를 Task05-E의 측정값으로 기재하지 않는다. 기존 11,095 Point-row 집계 역시 조건별 관측 수이며 고유 물리 Point 수나 독립 episode 수가 아니다.

### A3. Raw 실행 경로와 변경 범위

새 정책 이름은 `raw_extended_delaunay`, `version=1`, `evaluation_only=True`, `candidate_mode='extended_delaunay'`, `edge_validation='none'`을 제안한다. 실제 metadata 상수는 작은 evaluation adapter에 둔다. 여기서 `edge_validation='none'`은 픽셀 경로 연속성 검증이 없다는 뜻이며 receiver의 후보·배열 계약 검사는 유지한다. 기존 `EDE_POLICY`를 복사해 검증을 통과한 것처럼 표시하지 않는다. 정책에 기존 component gate·Qhull 옵션·정렬 계약은 명시하되 Strict의 raster/rejection/witness 계약은 넣지 않는다.

Raw 정상 실행은 다음 순서만 수행한다.

1. 저장된 유효 Point row와 동일 object·observed component ID를 받는다.
2. 기존 `candidate_edges(..., extended=True)`를 그대로 호출한다. correctness 비교에서는 B에 저장된 후보 배열도 기준으로 읽는다.
3. 반환된 모든 canonical undirected 후보를 최종 변으로 사용하고, 기존 `directed_edges(candidates)`로 양방향 Edge를 만든다.
4. Raw 정책·전체 candidate pair/origin metadata를 packet에 넣고 Raw 전용 receiver 계약을 통과시킨다.

`Final Raw edges = All valid Extended Delaunay candidate edges`다. 동일 object·저장 observed component grouping, n=0/1 무변, n=2 직접 연결, n≥3 exact collinear chain, 비일직선 2D Delaunay triangle 변을 유지한다. Qhull `Qbb Qc Qz Q12`, exact collinearity, tie-break, canonical/directed 순서를 바꾸지 않는다. `stored_triangles` 인자는 EDE에서 금지되어 있으므로 Raw를 DE의 frozen T0 경로로 우회하지 않는다.

QhullError·vertex omission·coplanar 정보를 기존 helper 상태대로 보존한다. 자동 k-NN/fallback, jitter/QJ, 새 Delaunay 방식으로 보완하지 않는다. 기존 `check_execution`과 같이 QhullError/omission이 있으면 완전한 계약 PASS로 처리하지 않고 해당 조건·row/source·원인을 기록하여 원인 확인 전 후속 benchmark/채택 판단을 중단한다. 후보가 없는 singleton은 정상 상태다.

Raw producer와 receiver는 Supercover, Mask support, finite-positive 경로 depth, overlap 경로 검사, right/down PairEvidence 열거, depth-jump 판정, rejection reason·pixel witness 생성을 호출하지 않는다. 저장 Point의 유효성 및 component 제약은 유지한다. 추가 length/depth/normal threshold, outlier removal, GT filter는 없다.

최소 변경안은 다음과 같다. 이번에는 어느 파일도 구현하지 않는다.

| 승인 후 파일 | 범위 |
|---|---|
| 신규 `visual/runtime/test/run_raw_ede_comparison.py` | 입력 전용 reader, 저장 Strict와 비교, CPU benchmark, 분석·그림·보고서 생성. 기존 helper와 아래 adapter를 import한다. |
| 신규 `visual/runtime/test/test_raw_ede_comparison.py` | Raw 및 receiver 계약·회귀 테스트. |
| 신규 `visual/runtime/test/raw_ede_comparison_adapter.py` | Raw 정책 상수와 후보→directed/metadata 포장만 둔다. Runner와 receiver가 이 작은 모듈을 참조하며 adapter는 receiver/runner를 import하지 않는다. |
| `graph/vision_input.py` | 새 정책을 인식하는 명시적 평가 전용 branch와 최소 candidate/edge 검사만 추가한다. 기존 함수 signature·기존 정책 branch·기본값은 유지한다. |
| `docs/visual-graph-integration/00_interface.md` | 새 evaluation-only packet metadata와 receiver의 허용 범위만 문서화한다. |
| 신규 결과 Markdown | `docs/visual-graph-integration/05e_raw_ede_comparison.md`. |

작은 adapter 분리는 receiver에서 평가 runner 전체를 import하거나 `__main__`을 중복 적재하는 문제를 피하기 위한 범위다. 공통 Graph API를 재설계하지 않는다. 기존 CPU EDE/Task05-C/GPU B1 및 그 runner는 수정하지 않고 source hash로 확인한다. Raw를 최종 Graph/GNN production 기본값으로 연결하지 않는다.

## B. 연결성·시각적 비교 설계

### B1. 먼저 확인할 topology 계약

평가 topology의 Strict 쪽은 B/C 저장 결과를 그대로 사용한다. 비교 분석을 위해 Strict를 새로 계산한 결과로 대체하지 않는다. 승인 후 아래 gate를 16조건 각각, object별 및 ratio별로 확인한다.

- B/C의 저장 candidate/origin/accepted/final Edge 및 공통 Point metadata가 exact 일치하는지 확인한다. 저장 source snapshot/hash와 현재 사용 helper binding도 확인한다.
- Raw correctness adapter가 반환한 후보는 B의 `EDE_candidates`와 dtype/shape/bytes/order가 정확히 같아야 한다. origins·triangles·component 상태도 의미 필드별로 대조한다. timing은 exact 비교 대상에서 제외한다.
- `Strict U ⊆ Raw U`, `Raw U = saved candidate U`, `Raw U − Strict U = saved candidates[~accepted]`, `Strict U − Raw U = ∅`를 검사한다. Canonical row pair와 `(object_id, source_i, source_j)` identity 양쪽으로 검사한다.
- `Strict edge_index = directed_edges(candidates[accepted])`, `Raw edge_index = directed_edges(candidates)`의 exact 배열 순서를 검사한다. Raw에서 저장 거부 Edge 복원 이외의 topology 변경은 계약 실패다.
- 전체 U/accepted/rejected/Strict isolated/singleton을 위 기록과 대조한다. Singleton 270개의 row/source 집합이 그대로 isolated인지 검사한다. Raw isolated의 전체 수를 270이라고 미리 가정하지 않는다. 비-singleton isolated가 있으면 Qhull 누락·오류·후보 incident 여부와 기존 component 상태로 설명한다.

Gate 실패 시 실패 자료를 새 UUID에 보존하고 benchmark 및 정책 결론으로 진행하지 않는다. 기존 artifact를 수정하여 숫자를 맞추지 않는다.

### B2. 모든 조건의 기본 시각화

모든 16조건·32 object slice에 Strict / Raw / Difference의 세 열을 동일 Point·동일 축·동일 crop·동일 camera view로 만든다. 빈 Box도 `N=0` panel과 JSON 행을 남긴다. 우선 상세 검토 frame은 `0,27,45,80,120`의 양 ratio이며 `18,56,66`도 기본 그림·원본 링크·집계에서 빠지지 않는다.

| 배경/좌표 | 세 정책 열에 공통으로 보여줄 내용 |
|---|---|
| 원본 RGB + UV | 전체 장면, 선택 Point, Edge. UV는 저장 source index와 image width에서만 복원한다. |
| Object raw mask | 관측 mask의 hole·경계·overlap을 구분하고 같은 Edge를 겹친다. Depth-gated support와 raw mask를 혼동하지 않도록 범례를 둔다. |
| Depth map | metre 단위·invalid 표시·공통 color scale을 사용한다. 원거리 plateau가 보이는 전체 범위와 근거리 확대 범위를 함께 제시하고 clipping 범위를 표기한다. |
| Camera XYZ 3D | m 단위 축·동일 aspect와 관점, 전체 Point extent 및 물체별 확대. 8.431m 구간을 축에서 숨기지 않는다. |

따라서 기본 자료는 object slice마다 네 배경의 3열 비교를 포함한다. 전체 물체와 전체 장면의 context 그림도 연결한다. PNG의 Edge 표시는 undirected 한 번씩이며 packet의 directed 수와 구분한다. 해상도가 부족하면 파일을 분리하여 가독성을 확보하고 모든 panel 경로를 coverage manifest에 기록한다.

Difference에서 Strict 유지 Edge는 파랑, Raw 복원 Edge는 주황으로 고정한다. Point는 두 정책에서 같은 위치·크기로 표시한다. 사유별 보조 panel/HTML 보기에서는 저장된 Strict 진단만 읽어 색·선 모양·label로 구분한다. 여러 사유를 가진 Edge는 모든 label과 reason bitset을 보존하고 조합별 보기 또는 겹친 표시를 제공한다. 한 가지 색을 대표 사유로 보이더라도 나머지를 지우거나 단일 원인으로 집계하지 않는다.

대상 사유는 `mask_outside`, `invalid_depth`, `overlap`, `pair_unsupported`, `pair_missing`, `delta_invalid`, `depth_jump` 모두다. 빈 사유도 0으로 표시한다. 사유별 Edge 수(중복 가능), 정확한 사유 조합별 unique Edge 수, 저장 pixel/pair count를 구분한다. 모든 reason 수의 합을 복원 Edge 수로 해석하지 않는다. Stored witness가 top-k이면 전체 결함 위치 목록이라고 보고하지 않는다. 더 많은 경로 위치가 필요하면 C/D에 **이미 저장된** 상세 cover/pair 자료를 읽고 Raw validator를 재실행하지 않는다.

### B3. 사례 선택과 정성 판정

전체 기본 그림을 먼저 보존하고, 문제와 자연스러워 보이는 사례를 같은 기준으로 확대한다. Frame/ratio/object별 복원 Edge의 UV 길이·3D 길이·endpoint depth 차이 상위 5개, 사유 조합별 canonical 순서 첫 사례를 우선 검토 대상으로 고정한다. 중복은 합치고 해당 strata가 비면 빈 상태를 남긴다. 이 순위는 진단·그림 선택용이며 Edge를 제거하는 threshold가 아니다. 실제 검토 중 발견한 추가 사례도 선택 이유와 row/source를 남긴다.

관측 공백 가로지름, mask boundary/hole, 큰 depth 차이의 연결, 서로 다른 면이 만나는 정상 모서리로 보이는 연결, Strict가 과하게 끊은 것으로 보이는 연결, 과도하게 길거나 부자연스러워 보이는 연결을 각각 검토한다. 원본 RGB·raw mask·depth·선택 endpoint와 3D를 나란히 보며 `문제가 있어 보임 / 자연스러워 보임 / 판단 불가`를 기록한다. 물리적 정답 label로 사용하지 않는다. 자연스러운 추가 연결이 발견되지 않으면 사례를 억지로 채우지 않는다.

전수 Edge 자동 진단의 분모는 saved rejected unique U, 실제 수동 검토의 분모는 검토한 unique U로 분리한다. 전체 기본 panel 검토 여부도 condition/object별로 기록한다. 읽지 않은 Edge를 정상으로 세거나, 일부 확대 그림의 비율을 전체 물리 오류율로 일반화하지 않는다.

Frame0 Box의 약 8.431m plateau는 양 ratio에서 별도 상세 panel을 만든다. [기존 기록](05b_extended_delaunay_edgewise.md)에 따르면 `(251,206)…(262,206)`의 12 Point chain과 인접 11 Edge는 Strict에서도 유지됐다. 원래 `(251,206)↔(259,206)` 약 263.469mm 관계는 직접 Edge와 chain을 통한 간접 연결을 구분한다. 저장 `remote`의 row/source/UV/XYZ/chain을 기준으로 그대로 남는 Edge, Raw에서 추가되는 incident Edge, 다른 Graph component와의 병합 유무를 따로 확인한다. Strict에서도 통과한 오염 plateau를 Raw만의 신규 실패 또는 정상 물리 복원으로 설명하지 않는다.

### B4. 연결성 및 기하 통계

모든 수치는 frame × ratio × object와 condition 전체에 저장하고, ratio별·object별 집계도 제공한다. 전체 분포 집계와 condition 평균 집계의 가중 방식을 명시한다. 서로 다른 ratio의 row나 같은 frame의 Point를 독립 episode로 취급하지 않는다.

| 지표 | 정의/보존 단위 |
|---|---|
| N, candidate U, final U/E | 모든 Point와 unique canonical undirected U, directed E=2U. 빈 object도 포함한다. |
| Isolated / Graph CC / size | degree=0 row/source, singleton을 포함한 Graph connected component 수·node label·size 전체 분포. 저장 observed component ID와 Graph CC label은 별개다. |
| Degree | 전체 row별 degree와 mean/median/p95/max, object별 분포. |
| 복원 연결 | added U, added-incident row, `degree_strict=0 & degree_raw>0`인 새 연결 Point를 구분한다. |
| CC 병합 | Raw CC마다 포함하는 Strict CC ID·size·교차 연결 added Edge 목록. 단순 component 내부 Edge 증가와 서로 다른 Strict CC 병합을 구분한다. object/observed component 간 병합은 허용하지 않는다. |
| Edge geometry | Strict 유지·Raw 전체·added 세 집합 각각의 Camera XYZ 3D length(m/mm), UV Euclidean length(pixel), signed/absolute endpoint camera z 차이(m). 전체 배열·mean/median/p95/max·histogram/ECDF를 저장한다. |
| Tail 사례 | 위 상위 길이·depth 차이 사례의 row/source/UV/Camera XYZ/EE XYZ, 원본 배경 및 Strict 사유를 연결한다. |

Node 수·불변 필드·singleton identity는 정책 간 같아야 한다. Edge 증가·고립 감소·CC 병합만으로 품질 향상을 판정하지 않는다. Endpoint z 차이는 진단값이며 경로 연속성의 정답이나 필터가 아니다. 정상적으로 같은 observed component 안에 있어도 직선 후보가 관측 공백을 가로지를 수 있다는 점을 정성 평가와 함께 설명한다.

## C. Edge별 검증 제거의 실제 CPU 비용 비교

### C1. 측정 정책과 경계

주 비교는 **현재 host의 같은 repository-root `.venv/bin/python`·입력·CPU 설정**에서 새로 측정한 Strict와 Raw다. Strict는 기존 `build_extended_delaunay_edges(..., extended=True, optimized=True)`의 Task05-C CPU 경로를 사용한다. Task05-D CPU A와 같은 single-normal SAT를 선택하여 느린 구형 raster만 분모로 삼지 않는다. Supercover·Mask·Depth·기존 reason/count/top-k witness와 metadata/receiver 작업을 그대로 수행하고, 반환 topology는 저장 B/C와 exact 대조한다. 기존 구현은 수정하지 않는다.

시각 비교의 Strict topology 재사용과 비용 측정의 Strict 독립 재실행은 목적을 분리한다. Raw도 매 invocation 후보를 새로 생성한다. 저장 candidate를 읽고 포장하는 비용을 fresh 후보 생성 시간으로 보고하지 않는다. 양 정책 간 candidate/validator cache를 공유하지 않는다.

| 비용 구간 | Strict | Raw |
|---|---|---|
| 입력 로드·hash gate | 저장 Point/component/RGB-D/evidence 로드, 무결성 확인. 별도 IO/setup 시간 | 같은 고정 입력·공통 무결성 gate. Graph 실행은 dense RGB-D/evidence를 요구하지 않음 |
| 공통 입력 준비 | packet copy, 저장 component/row 입력 준비 | 동일 packet copy와 저장 row/component 준비 |
| 정책별 준비 | `prepare_validation(evidence)`를 매번 fresh 실행. 필요한 selected-source map을 쓰면 그 비용도 포함 | validator lookup/selected-source dense map을 만들지 않음 |
| Candidate | 기존 EDE helper를 매번 독립 실행 | 같은 helper를 매번 독립 실행 |
| Edge validation | 기존 CPU Supercover, support/Mask·Depth/PairEvidence·reasons/witness | 호출 없음. 해당 시간은 0, 기능은 `NOT_APPLICABLE` |
| Packaging | accepted 변·Strict metadata·packet | 전체 후보·Raw metadata·packet |
| Receiver | 기존 필수 v2 검증과 Strict branch·deepcopy·출력 | 공통 v2 검증과 D1 Raw branch·deepcopy·출력 |
| 평가 분석 | 저장 Strict 진단 join, metric, hash/assertion, 그림·HTML·JSON/NPZ IO | 양쪽 Core 밖에서 별도 측정 |

주 Core wall은 **메모리에 로드된 저장 입력에서 공통 준비를 시작해 정책별 준비·후보 생성·검증·포장·receiver 반환까지**다. 관측 component는 이미 저장된 값을 사용하며 새 component 계산 비용은 포함하지 않는다. 별도로 input IO/setup, 공통 준비, Strict 전용 lookup 준비, producer candidate, Strict validation 세부 단계, packaging, receiver, unclassified overhead를 기록한다. Receiver 내부 Raw candidate 재확인은 producer candidate와 구분한다.

기존 C의 lookup 준비 밖 Core와 혼동하지 않도록 `Core 전체(준비 포함)`와 `준비 제외 build→receiver`를 함께 표기한다. 정책별 준비를 제외한 값만 주 성능으로 제시하지 않는다. 평가용 원본 RGB-D 읽기·시각화·통계·saved Strict reason 조회는 Core 밖이다. Strict 정상 builder가 수행하는 endpoint/observed 확인 비용은 포함하고 숨기지 않는다. Raw는 기존 Point 유효성 gate와 공통 receiver 검증을 유지한다.

이 비교는 정책이 요구하는 실제 작업량 차이를 측정한다. Raw에 Strict 사유·witness metadata 생성을 강요하거나 Strict에서 해당 연산을 빼지 않는다. Raw에서는 필수 receiver를 생략하지 않는다. 저장 observed component의 upstream 생성까지 포함하는 end-to-end latency라고 부르지 않는다.

### C2. 반복·검증·보고

- 승인 후 correctness와 receiver 테스트가 통과한 뒤 측정한다. NumPy import 전에 OMP/OpenBLAS/VECLIB thread 환경을 1로 고정하고 host/OS/CPU, interpreter 경로·버전, NumPy/SciPy/Qhull extension hash·옵션, GC와 실제 thread pool 확인 가능 여부를 기록한다. Dependency를 설치하지 않는다.
- 16조건 각각 정책당 warmup 10회, 측정 30쌍을 사용한다. AB/BA 각 15회, condition 시작 순서를 교대한다. 동일 반환값 lifetime·GC 설정을 유지하고 outlier를 삭제하지 않는다.
- `perf_counter` Core wall과 `process_time` CPU time을 보존한다. 기존 stage timer를 재사용하고 추가 세부 계측은 별도 pass로 수행한다. nested stage를 중복 합산하거나 별도 pass 시간 합을 실제 Core wall로 보고하지 않는다.
- 매 반환값은 timer 종료 후 해당 정책의 frozen correctness 결과와 exact 비교한다. Raw가 Strict와 topology가 달라야 하는 것은 허용하되, 각 정책 내부 후보/Edge 순서·불변 필드는 매번 같아야 한다. Assertion/hash·검토 비용은 별도 기록한다.
- condition별 mean/median/p95/max와 모든 raw sample, ratio별 집계, 최대 N·최대 candidate 조건 및 최악 sample을 보고한다. 전체 speedup은 같은 조건 가중치의 wall 합 비율로 계산하며 paired 차이도 보존한다. 30표본 p95는 기술통계다.
- object별 candidate/validation 시간과 작업량은 가능한 기존 timer로 분리한다. shared packet/receiver/Core wall을 object별 독립 실행 시간처럼 분배하지 않는다. 필요하면 object timing이 포함하는 stage 범위를 표시한다.

과거 [Task05-D](05d_gpu_edge_benchmark.md)의 CPU S1 평균 **101.941ms**, MPS B1 S1 평균 **69.254ms**, B1 Supercover **3.900ms**, CPU Mask **11.436ms**, CPU Depth **29.569ms**는 참고 표에만 둔다. B1은 GPU cover + CPU Mask/Depth인 hybrid이며 3.900ms는 pipeline stage wall이다. GPU event/kernel latency는 당시 `TIMING_UNSUPPORTED`였으므로 CPU Raw Core와 kernel 비교를 만들지 않는다. 과거 평균에서 Mask/Depth 시간을 빼 Raw 성능을 추정하지 않는다.

이번 주 비교에는 MPS/CUDA 재실행을 넣지 않는다. 과거 B1과 전체 비용을 나란히 제시할 때만 fresh 준비→receiver 종료 경계 및 host/interpreter·동기화·metadata 범위 차이를 명시하고, 새 paired speedup으로 해석하지 않는다. CPU Raw가 충분히 빠르면 GPU Raw 개발을 추가하지 않는다. 50ms/20Hz 통합 성능은 vision·sampling·feature·component·GNN·controller를 포함하지 않으므로 PASS 판정하지 않는다.

## D. Receiver 계약·테스트·정적 HTML·artifact 계획

### D1. 명시적 Raw receiver 계약

기존 `validate_packet`/`receive_vision_packet`의 공통 identity·Point·F=5·availability/validity·source·좌표/transform 정합·deepcopy/+3 동작을 재사용한다. `_validate_v2_edges`에 Raw 정책만을 위한 branch를 추가하고 k-NN/TMD/DE/Strict EDE 조건과 동작은 유지한다. 별도 production 기본값이나 허위 Strict metadata를 만들지 않는다.

Raw packet은 새 정책과 `candidate_edge_pairs`, `candidate_edge_origins`를 공급한다. 관측 component는 기존 `observed_component_id`를 사용한다. Strict의 `accepted_edge_pairs`, `accepted_edge_origins`, `accepted_triangles`를 Raw의 근거로 재사용하지 않고, Raw packet의 정책별 metadata로 명확히 교체한다. Point/Feature/provenance 등 나머지는 바꾸지 않는다.

최소 구현에서 receiver는 기존 `candidate_edges(..., extended=True)`를 **픽셀 검증 없이 한 번 다시 호출**하여 공급 pair/origin 및 `edge_index`를 expected와 exact 비교한다. 이는 noncollinear origin label만 검사하여 임의 chord나 누락 candidate가 통과하는 것을 막기 위한 선택이다. Producer metadata를 자기 자신과 비교하거나 hash만 믿고 필수 계약을 생략하지 않는다. API에 trusted cache/context를 새로 추가하는 설계는 이번 최소안에 넣지 않는다.

이 선택은 Raw의 Qhull/candidate가 producer와 receiver에서 각각 한 번 실행되는 비용을 갖는다. 주 benchmark에 둘 다 포함하고 `receiver_candidate_recheck`를 분리하여 보고한다. 이를 숨긴 최적화 Raw 수치를 만들지 않는다. 향후 이 비용이 병목이면 별도 증거와 승인으로 계약을 보존하는 공유 방법을 검토한다.

검사 항목은 다음과 같다.

- `int64 [U,2]` canonical pair, `int64 [2,E]` directed Edge, index 범위, a<b, unique/lexicographic 순서, self-loop 없음, E=2U.
- 모든 변의 동일 object·동일 저장 observed component, source identity·image 범위, candidate grouping과 origin 규칙 일치.
- n=0/1/2·exact collinear·noncollinear 규칙에 따른 **전체 후보와 exact 일치**. QhullError/omission은 명시 상태로 남기고 정상 수신 PASS로 숨기지 않는다.
- `edge_index == directed_edges(expected_candidates)`의 dtype/shape/bytes/order. 단순 set equality만으로 순서 검사를 대신하지 않는다.
- receiver 전후 Point/Camera/EE XYZ, F=5, validity·provenance·object/component/source/row 불변, copy 독립성.
- robot 입력 없음: offset 0. 실제 same-frame robot 3행을 공급할 때 기존 point_node_index·edge_index의 `+3`, node kind·robot mask·source `-1` 규칙 유지. Offline mapping fixture는 실제 robot 실행 검증과 구분한다.

Raw receiver에는 dense Mask·Depth 경로 검증이나 PairEvidence, rejected reason/witness를 전달하지 않는다. 공통 v2의 저장 endpoint finite/positive camera depth 검사는 유지하지만 Edge 경로 depth를 다시 검사하지 않는다. 기존 Strict receiver를 Raw metadata 검사로 대체하지 않는다.

### D2. 승인 후 테스트와 중단 기준

테스트는 신규 모듈에서 아래 계약을 검증하고 관련 기존 suites를 그대로 실행한다. 현재 계획 작성 중에는 테스트를 실행하지 않았다.

| 범위 | 확인할 내용 |
|---|---|
| Raw 후보 fixture | 빈 object, singleton, two-point, 수평/수직/대각·불균등 간격 exact chain, 비일직선 Delaunay, tie/source/row 순서, Qhull error/omitted vertex의 기존 처리. 기대 후보는 기존 helper·기존 고정 artifact와 대조한다. |
| 검증 비호출 | Raw producer와 receiver에서 기존 raster/`prepare_validation`/`validate_cover`/pair 열거 진입점을 호출하면 실패하는 spy fixture. Mask hole·invalid depth·overlap·jump가 있는 Edge도 **이미 고정된 같은 component 후보라면** Raw가 저장 Strict의 거부 때문에 제거하지 않는지 확인한다. 실제 component를 재분할하지 않는다. |
| Receiver 음성 검사 | 잘못된 row/object/component, 중복/역순 pair, self-loop, 한쪽 방향 누락, directed 순서 변경, 후보 누락/새 chord, origin/policy 위장, 불변 필드 훼손을 거부하는지 확인한다. |
| 보존·수신 | F=5/NaN·validity·provenance 원본 bytes, deepcopy, robot 없음/+3 mapping, 기존 정책 수신 동작을 확인한다. |
| 고정 16조건 | B1절의 topology 계약, 전체 candidate exact, subset/difference, singleton identity, B/C 저장 Strict 일치, benchmark 매 반환값 재현을 확인한다. |
| 산출물 | JSON/NPZ packet readback, ragged offset·reason bitset·row/source join, metric 재계산, 16조건/32 slice coverage, PNG decode·실제 시각 검토·HTML 상대 링크 검증을 확인한다. |

관련 기존 테스트는 `test_extended_delaunay_edges`, `test_triangle_intra_object_edges`, `test_intra_object_edges`, `test_vision_features`, `test_variable_rate_graph_contract`다. GPU suite/새 MPS probe는 실행 계획에 넣지 않는다. 공통 Logger/Publisher를 변경하지 않으므로 신규 외부 서비스 점검도 하지 않는다.

입력/source hash 불일치, 저장 Strict 불일치, Raw candidate 불일치, pixel validator 호출, 비허용 topology 변화, receiver 필수 계약 실패, 예상 밖 Qhull 오류/누락은 원인을 기록하고 benchmark 전에 중단한다. 이미 측정 중 발견되면 유효한 성능 결론을 내리지 않고 실패 run 자료를 남긴다. isolated의 설명되지 않은 변화와 artifact readback/coverage 누락도 완료 PASS를 막는다.

### D3. 정적 HTML과 결과 저장

승인 후 공통 `ExperimentLogger(team='visual', job_type='evaluate', seed=0)`를 사용하고 `WANDB_MODE=disabled`로 로컬/offline 기록을 유지한다. Logger가 생성한 새 `uuid4().hex`를 유일한 실행 identity로 사용한다. 기존 run ID/project/group 규칙을 변경하지 않는다. `record()`에 여섯 source run·실제 고정 조건·source/input hash·scope·host 정보를 추가하며 identity 필드를 덮어쓰지 않는다.

결과 Markdown은 `docs/visual-graph-integration/05e_raw_ede_comparison.md`, HTML은 `experiments/<run_id>/raw_ede_comparison.html`에 둔다. Markdown→HTML은 `../../experiments/<run_id>/raw_ede_comparison.html`, HTML→Markdown은 `../../docs/visual-graph-integration/05e_raw_ede_comparison.md`로 양방향 연결한다. 이번 계획도 결과 Markdown에서 연결한다. 실제 UUID가 생기기 전 가짜 결과 링크나 빈 결과 파일을 만들지 않는다.

HTML은 외부 CDN·외부 font·원격 JS가 없는 정적 문서로, 기본 PNG·표·local anchor만으로 `file://`에서 읽을 수 있게 한다. 필터/확대 보기를 넣으면 JS/CSS와 필요한 데이터는 로컬/inline으로 묶어 network/fetch에 의존하지 않는다. 실제 browser 확인을 실행했는지와 HTML parse/link 검사만 했는지를 구분하여 기록한다.

필수 HTML 내용은 다음과 같다.

- Strict/Raw 정의와 같은 Point의 Strict/Raw/Difference overlay, 모든 condition/object로 가는 원본 gallery.
- Object raw mask·RGB-D 배경, Camera XYZ 3D 전체 및 확대, 복원 Edge·다중 rejection reason 분포와 사례 연결.
- 자연스러워 보이는 추가 연결·문제가 있어 보이는 연결·판단 불가 사례와 검토 분모/선택 기준.
- Frame0 원거리 plateau, Strict에서도 남던 연결과 새 추가 연결의 구분.
- Graph CC·isolated·병합·degree·UV/3D length·endpoint depth 차이 분포와 tail 사례.
- CPU Core 계산 비용·공통/정책별 준비·receiver 재확인·평가 분석 비용, 과거 MPS 참고 범위와 미실행 장치·한계.
- 아래 Q1–Q5의 관측 근거와 결론 상태, 계약 PASS와 품질 판단의 구분.

새 UUID 안에는 다음을 보존한다. 원본 여섯 run과 그 sidecar는 수정·재등록하지 않는다.

| 경로 | 내용 |
|---|---|
| `config.yaml`, `metadata.json`, `metrics.json`, `manifest.json` | 기존 Logger의 실행 identity·조건·로컬 기록 |
| `raw_ede_comparison.json/.npz` | 16조건 전체 원본 불변 배열, strict/raw/candidate edges, added mapping·origin·reason bitset, row별 degree/CC, 길이·depth 배열, 요약 통계 |
| `records/<key>.json`, `packets/<key>_raw.npz` | object별 상세 결과·검토 및 저장 Raw packet. 필요한 Strict 사본은 새 run 내 dependency로 식별 |
| `plots/`, `coverage.json`, `review.json` | 전 slice의 PNG 경로, 전체/확대 그림, 실제 검토 상태·사례와 label·미검토 범위 |
| `timings_raw.json`, `environment.json` | 정책별 전 반복 sample·순서·stage/전체 wall/CPU·측정 범위·실제 환경 |
| `input_gate.json`, `verification.json`, `tests.log` | subset/difference·순서·비호출·수신·불변성·readback·통계·링크·coverage 및 테스트 실제 결과 |
| `source_hashes.json`, `source_snapshot/`, `implementation.diff`, `preexisting_status.txt` | 실행 소스 snapshot·binding/hash, 시작 dirty 상태, 계획 후 승인된 변경 diff |
| `verification_scripts/`, `inputs/` | 실제 검증에 사용한 script, 필요한 관측 RGB/mask/depth·저장 Strict 진단의 명시적 dependency 사본과 원본 run/path/hash 연결 |

NPZ는 `allow_pickle=False`로 읽을 수 있는 numeric/string 배열과 ragged offsets를 사용한다. JSON과 NPZ key/dtype/shape/bytes/order·통계·packet receiver 재수신을 대조한다. 사유가 여러 개인 Edge의 모든 reason과 원본 witness/count를 평가 자료에 보존하되 Raw packet/Core metadata에는 넣지 않는다. RGB-D dependency는 필요한 관측만 명시적으로 복사하고 원본 hash·source frame을 남긴다. Copy를 기존 run의 신규 결과로 등록하거나 원본 sidecar 소유권을 바꾸지 않는다.

새로 만든 결과만 `logger.link_output`으로 등록한다. JSON은 metrics, HTML/PNG/Markdown은 명시적 `kind='reports'`, NPZ는 기존 inventory 규칙에 맞게 등록한다. 입력 dependency와 source snapshot의 목록/hash도 새 run manifest 보조 자료로 보존한다. 기존 artifact에 `link_output`을 호출하지 않는다. HF publisher·dry-run·W&B sync·업로드는 이번 후속 비교에도 포함하지 않으며 별도 명시 요청이 있을 때만 기존 `tools.publish_experiment`를 사용한다.

작업 시작 시 현재 dirty/untracked 파일, 여섯 source run의 입력/artifact/sidecar를 hash 목록으로 보존한다. 승인된 신규 파일과 명시적 receiver/docs 변경 이외에는 끝에서 같은 hash여야 한다. 기존 dirty receiver 변경은 시작 snapshot에 남기고 그 위의 Task05-E diff만 분리한다. 기존 실험 artifact는 덮어쓰지 않는다. 이번 계획 작성에서도 기존 dirty/untracked 파일을 보존한다.

### D4. 판단할 질문과 완료 경계

| 질문 | 판단 근거와 답변의 한계 |
|---|---|
| Q1. Strict 거부 Edge가 실제 시각 문제를 일으키는가? | 저장 사유·원본 배경·복원 Edge 확대를 대조한다. 사유 존재와 실제 정성 문제를 동일시하지 않는다. |
| Q2. Raw가 표면 구조를 더 자연스럽게 연결하는가? | 정상 모서리/연속 표면으로 보이는 사례와 CC 병합을 함께 제시한다. 물리 정확성 입증으로 보고하지 않는다. |
| Q3. Mask 공백·Depth 불연속 연결이 얼마나 나타나는가? | 저장 사유별 unique/중복 수, 다중 사유 조합, 실제 검토 label·검토 분모를 분리한다. 전체 물리 오류율은 확정하지 않는다. |
| Q4. Strict 품질 이득이 비용을 정당화하는가? | 같은 CPU 경계의 실측 절감량과 관측상 예방한 문제·과도하게 끊은 사례를 함께 비교한다. 사전 채택 점수나 새로운 filtering threshold를 만들지 않는다. |
| Q5. 후속 GNN ablation 후보로 충분히 합리적인가? | 계약·전 조건 관측 진단·품질/비용 trade-off로 후보 유지/보류/추가 검토를 제안한다. 학습 성능과 최종 정책은 미확정이다. |

후속 구현·평가가 승인되면 입력 gate → 최소 adapter/receiver 구현 → 계약·회귀 테스트 → 16조건 exact 비교 → CPU paired benchmark → 전 조건 통계·시각 검토 → JSON/NPZ/PNG/정적 HTML·Markdown 검증 순서로 수행한다. 실행하지 않은 항목은 `NOT_EXECUTED`, 근거가 부족한 품질 판단은 `판단 불가`로 남긴다.

금지 범위는 sampling/keep_ratio/voxel·PCA5/feature·component 방식 변경, 새 GT/depth/length/normal filter·outlier 제거, 새 Delaunay/k-NN fallback, Strict 삭제, GPU B2/Grid Traversal 개발, GNN/GRU/BC 학습·rollout, 추가 episode/simulator, dependency 설치, 외부 업로드, Git commit이다. **이번 제출은 이 계획 파일에서 종료하며, 구현·실험 승인 전까지 중단한다.**
