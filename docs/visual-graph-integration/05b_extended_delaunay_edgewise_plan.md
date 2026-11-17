# Task05-B — Extended Delaunay + Edge-wise Observation Validation 계획

상태: **계획만 작성 / 구현·실험 승인 대기**. 새 정책 후보는 **EDE 하나**이며 DE는 변경 효과를 분리하는 분석 대조 조건이다. 목표는 기존 Shape-First Point 사이의 관측상 타당한 국소 연결이다. 물리적 Triangle Mesh 복원이나 최종 GNN 인터페이스를 확정하지 않는다.

이번 작업은 기존 코드·저장 JSON·파일 구조를 읽고 이 계획만 작성했다. 신규 Edge 생성, Delaunay 실행, 테스트, CPU benchmark, ExperimentLogger 생성은 하지 않았다. 아래의 구현·검증·산출물은 모두 승인 후 작업이다.

## 0. 확인한 현재 구현과 입력

루트 지침, `README.md`, `envs/maniskill/README.md`, `docs/PROJECT_CONTEXT.md`, Visual README 및 runtime README, engineering/tracking/HF 문서를 읽었다. 현재 branch의 dirty/untracked Task04/05 구현을 기준으로 검토했으며 기존 파일을 수정하지 않는다.

| 현재 소스 | 확인한 재사용 범위 / 제약 |
| --- | --- |
| [triangle_intra_object_edges.py](../../visual/runtime/triangle_intra_object_edges.py) | `QHULL_OPTIONS='Qbb Qc Qz Q12'`, integer collinearity 판정, `supercover`, `adjacent_pairs`, `prepare_validation`, `validate_cover`가 존재한다. `supercover`는 이미 2점 선분을 지원한다. 기존 builder는 triangle 내부와 세 변을 검사하고 T0/TM/TMD를 함께 만든다. |
| [run_triangle_intra_object_edges.py](../../visual/runtime/test/run_triangle_intra_object_edges.py) | 입력 정합 gate, 저장 triangle/stage Edge, `graph_metrics`, `diagnose`, 원거리 pair 추적, 계측·보고서 사례가 있다. 기존 `main/load_inputs/run_numerics`를 그대로 호출하면 K8/TMD 재생성과 benchmark까지 실행되므로 새 runner에서 호출하지 않는다. |
| [intra_object_edges.py](../../visual/runtime/intra_object_edges.py) | full support의 right/down depth-gated component와 K8가 존재한다. 저장된 row별 component를 재사용하며 새로운 분할 규칙을 만들지 않는다. |
| [vision_input.py](../../graph/vision_input.py) | v1/v2 수신, row 보존·deep copy, 실제 robot 공급 시 +3을 지원한다. v2는 현재 K8/TMD 정책만 허용한다. TMD에는 accepted-triangle 변 합집합 검증이 있어 DE/EDE를 TMD로 위장해 전달할 수 없다. |
| [run_vision_feature_edge.py](../../visual/runtime/test/run_vision_feature_edge.py), [run_adaptive_node_budget.py](../../visual/runtime/test/run_adaptive_node_budget.py) | `read_packet/save_packet`, `load_evidence`가 존재한다. 저장 배열을 복원하는 IO만 재사용하고 sampling, feature extraction, 좌표 변환 경로는 호출하지 않는다. |

현재 설치 파일의 NumPy `2.4.6`, SciPy `1.17.1` metadata를 읽었다. Task05 저장 환경은 Python `3.11.13`, macOS arm64, 같은 NumPy/SciPy, Qhull `8.0.2`, 단일 thread 환경이다. 이번에는 SciPy/Qhull 계산을 실행하지 않았으며 현재 backend/device 동작을 검증한 것으로 해석하지 않는다.

### 고정 입력 및 출처

| source run | 사용할 저장 자료 |
| --- | --- |
| Task03-K `9185355a32b8433abc70f1738f983377` | `selection_before_gt.json/.npz`, `selection/f*_r*.npz`: 원래 선택·source/provenance 정합 근거 |
| Task04 `9f90293b227e4dbfaaa6290afad9cc3b` | `packets/f*_r*.npz`: camera/EE XYZ, F=5, availability/validity, provenance, object/component/source row mapping, 기존 K8. `runtime_evidence/f*.npz`: 저장 depth/XYZ, mask/overlap/valid_depth, PCA5 PairEvidence |
| Task05 `91ecd0413a87455cabc5e813cddaea8d` | `triangle_intra_object_edges.json/.npz`, `records/`, `packets/`, `input_gate.json`, `source_hashes.json`, `verification.json`: T0 삼각형과 TMD Edge, 단계별 degree/labels, 관측 감사, component 상태, 계측과 원거리 추적 |

상위 보고서: [Task04](04_vision_feature_edge.md), [Task05](05_triangle_intra_object_edges.md). 저장 JSON의 [Task05 수치](../../experiments/91ecd0413a87455cabc5e813cddaea8d/triangle_intra_object_edges.json)를 우선 사용한다. Task04/05에는 수치 완료 후 보고서 복구 이력이 있으므로 computation source와 report source를 구분해 hash를 기록한다.

- Frame `0,18,27,45,56,66,80,120` × ratio `.5` Primary / `1.0` Reference × phase `0`: 16개 packet, object별 32개 slice다. 독립 episode 16개가 아니며 ratio를 합해 고유 물리 Point 수로 해석하지 않는다.
- 기존 가변 N과 모든 Point row를 유지한다. Frame18/56의 빈 Box도 유지한다. Point/Feature/validity/provenance/transform metadata를 gather·복사만 하며 재계산하거나 수정하지 않는다.
- 승인 후 새 후보를 만들기 전에 source frame/camera/shape/object/source-index와 저장 component ID를 Task03-K/04/05 사이에서 exact 대조한다. Float/NaN 배열은 원본 dtype·shape·값·NaN 위치 및 byte digest를 보존한다. 기존 입력/소스/artifact와 sidecar의 SHA-256을 시작·종료에 대조한다.
- 저장 component는 object와 함께 식별한다. 후보 생성은 저장 `observed_component_id`로 grouping한다. 전체 observed support component 수·support 크기는 저장 common 통계를 사용한다. 선택점이 없는 component도 분모에서 누락하지 않는다. 저장 자료가 부족하면 그 항목을 N/A와 사유로 남기고 임의 분할·feature 재추출을 하지 않는다.

## A. 후보 생성 규칙

### A1. component별 최소 분기

| 최종 선택점 n / integer UV | DE 분석 대조 | EDE 후보 | 출처 |
| --- | --- | --- | --- |
| 0 또는 1 | 없음 | 없음; Point 유지 | component 상태만 기록 |
| 2 | 없음 | 두 점 사이 undirected 후보 1개 | `two_point_fallback` |
| n≥3, 전체 UV exact collinear | 없음 | 직선 순서의 이웃만 최대 n−1개 | `collinear_chain` |
| n≥3, 비일직선 | 기존 Delaunay 변 | 동일 Delaunay 변 | `delaunay_edge` |

모든 후보는 동일 object·동일 저장 observed depth-gated component 내부다. 전체 component가 비일직선이면 일부 국소 점의 collinearity를 이유로 추가 Edge를 만들지 않는다. Complete graph, k-NN/radius fallback, component 강제 병합은 없다. Candidate는 연결 확정이 아니며 B의 검증을 모두 통과해야 한다.

### A2. exact collinearity와 deterministic chain

원본 source-index와 image width로 integer `(u,v)=(source%W,source//W)`를 얻는다. 중복 `(object,source)`는 원래 계약대로 입력 오류다. Integer UV를 float로 반올림하거나 jitter하지 않는다.

1. component 내부 임시 index만 `(u,v,source_index,final_row)`의 사전식 순서로 정렬한다. 최솟값을 a, 최댓값을 b로 잡아 `d=b−a`를 정의한다. 수평/음의 기울기는 +u, 수직은 +v 방향이다.
2. 모든 p에 대해 `d_u*(p_v−a_v)−d_v*(p_u−a_u)==0`인지를 exact integer로 확인한다. Python integer 또는 overflow 검사를 거친 int64를 사용한다. n≥3에서만 전체 collinear 분기를 결정한다.
3. collinear이면 integer projection `t=(p−a)·d` 오름차순, 동률은 source-index, final row 순으로 정렬한다. 유효한 unique UV에서는 projection 동률이 없지만 tie rule을 명시한다. 인접 두 row만 canonical undirected `(min(i,j),max(i,j))`로 만든다.
4. 원본 Point/Feature row와 배열 자체는 정렬하지 않는다. Chain의 임시 순서 및 source↔row mapping을 진단 자료에 보존한다. 입력 순서가 달라도 source pair 집합이 같은지 확인한다.

### A3. 비일직선 Delaunay와 예외 상태

기존 builder와 동일한 component 순회, final-row 입력 순서, float64 UV, `Qbb Qc Qz Q12`로 삼각형을 얻는다. 임의 UV 정렬이나 `QJ`를 추가하지 않는다. 각 삼각형의 세 변을 canonical unique undirected 후보로 추출한다. 삼각형 통과 여부는 이후 DE/EDE의 조건이 아니다.

정확한 ablation에서는 **저장 T0 삼각형의 변을 DE의 기준 후보로 사용**하고 EDE의 비퇴화 후보와 exact 일치를 요구한다. 승인 후 필요한 재현은 고정 16개 packet의 후보 일치 확인에 한정한다. 동일 Qhull 입력/환경에서 다른 diagonal이 나오면 source/순서/버전부터 조사하며 후보를 맞추기 위해 jitter·재선택하지 않는다.

Component 상태에는 `empty/singleton/two_point/collinear/ok/qhull_error/omitted_vertices`와 diagnostic flags를 분리한다. `tri.coplanar`, 누락 final-row/source, Qhull exception 종류를 기록한다. Nearly-collinear는 exact collinear와 별개다. A2의 baseline에 대한 `max(abs(cross(d,p−a)))/dot(d,d)`를 기록하고, nonzero이면서 ≤`1e-8`이면 diagnostic flag로만 표시한다. Cocircular는 인접 Delaunay 삼각형의 convex 4점에 대한 exact integer incircle determinant=0을 local tie flag로 기록한다. 이는 모든 4점 조합의 전수 검사나 새 연결 규칙이 아니다. 진단은 core 계측 밖에 둔다.

예상하지 못한 QhullError/누락 vertex는 collinear chain으로 바꾸지 않는다. 실패 상태와 영향을 저장하고 해당 조건의 완전한 비교는 FAIL/INCOMPLETE로 표시한다. 성공한 simplex나 누락 node를 조용히 버리고 성공으로 보고하지 않는다.

## B. Edge-wise Mask/Depth 검증 정의

### B1. 기존 helper를 선분 단위로 재사용

각 canonical 후보 `(i,j)`에 대해 `supercover(uv[[i,j]]) → validate_cover(cover, context[object])`를 한 번 적용한다. `prepare_validation`의 기존 PairEvidence lookup을 재사용한다. DE와 EDE의 공통 후보에는 동일 validator와 동일 입력을 사용한다. 삼각형 내부, 다른 변, bbox 전체 pair는 검사하지 않는다.

- Pixel centre는 integer UV, cell은 닫힌 ±0.5 사각형이다. 선분과 cell의 모서리·변만 접촉해도 cover에 포함한다. 기존 doubled-coordinate exact SAT 정의와 같다.
- 모든 접촉 cell이 image 안, 같은 물체 raw mask 안, finite-positive depth, overlap 아님을 만족해야 한다. `mask_outside`, `invalid_depth`, `overlap`, `image_outside`를 독립 사유로 기록한다.
- 필요한 pair 집합은 **두 endpoint cell이 모두 이 Edge cover에 포함된 원본 right/down 직교 인접 pair**다. 경로에서 가까운 pair 하나만 고르거나 diagonal pair를 새로 만들지 않는다.
- 예: `(0,0)→(1,1)` corner-touch에서는 `(0,0),(1,0),(0,1),(1,1)` 네 cell이 포함된다. 따라서 두 수평 pair `(0,0)↔(1,0)`, `(0,1)↔(1,1)`와 두 수직 pair `(0,0)↔(0,1)`, `(1,0)↔(1,1)`를 모두 검사한다. 인접 대각 구간에서도 이 규칙을 그대로 적용한다.
- 해당 pair가 unsupported, missing, signed delta nonfinite, 기존 depth-jump=true이면 거부한다. 원인 이름은 `pair_unsupported`, `pair_missing`, `delta_invalid`, `depth_jump`를 유지한다. 입력 gate에서 저장 gate가 `abs(z_b−z_a)>0.01+0.02*min(z_a,z_b)`와 일치하는지 확인하되 PairEvidence를 재추출하거나 덮어쓰지 않는다.
- Endpoint 전체 depth 차이, 3D 길이, normal angle, GT face/mesh/hole/pose는 제거 조건이 아니다. 연속 ramp와 실제 모서리의 normal 변화도 관측 경로가 유효하면 허용한다.

지원하지 않는 경로를 pair missing과 혼동하지 않는다. 기존 validator처럼 support 실패 Edge도 depth 사유를 진단해 다중 원인을 남긴다. rejected Edge 수는 unique Edge 수이며 여러 사유 count를 단순 합산하지 않는다. 각 사유의 pair/cell count와 witness를 별도로 저장한다.

### B2. 독립 유지와 packaging

ABC에서 AB/AC가 통과하고 BC가 실패하면 AB/AC는 유지한다. 같은 Edge가 여러 triangle에 있어도 원래 triangle의 다른 실패를 물려받지 않는다. 최종 후보별 판정 완료 후 accepted undirected 집합을 unique 처리하고 양방향 directed `[2,E] int64`로 포장한다. 후보 중복 제거는 검증 재사용을 위한 것이며 accepted 집합 확정 이후의 최종 dedup도 확인한다. `E=2U`, self-loop=0, directed duplicate=0, 역방향 누락=0을 요구한다. 연결을 잃은 Point도 유지한다.

후보 출처·검증 사유·witness는 별도 평가 metadata다. 기존 node provenance와 F=5를 수정하거나 GNN Feature에 자동 추가하지 않는다. 저장 raw mask/XYZ와 finite-positive 정의가 모순되면 입력 오류로 중단하며 재추정으로 보정하지 않는다.

## C. 공정한 비교·ablation·테스트

### C1. 비교 행렬과 포함 관계

| 조건 | 후보 / 원래 채택 의미 | 이번 비교에서의 역할 |
| --- | --- | --- |
| K8 | 기존 same-component k=8 / 기존 방식 | 저장 Task04/05 연결성 기준선. 공통 경로 감사로 Edge를 다시 제거하지 않는다. |
| TMD | 저장 T0 Delaunay / triangle 전체 통과 | 저장 Task05 기준선. 원래 정책과 단계별 구성을 보존한다. |
| DE | 동일 T0 Delaunay 변 / Edge별 검사 | TMD→DE: triangle 전체 제거 폐지 효과 |
| EDE | DE 후보 + 2점/전체 collinear chain / 동일 Edge별 검사 | DE→EDE: 퇴화 보완 효과. 유일한 새 정책 후보 |

K8/TMD topology·metric은 저장 자료를 우선 재사용한다. 저장 T0/accepted triangle/Edge/index의 정합으로 충분하면 기존 정책을 재실행하지 않는다. 정확 재현이 필요할 때만 고정 입력의 해당 후보·Edge 구성을 확인하며 production default를 변경하지 않는다.

동일 Point/component와 유효성 정의에서 source pair와 final-row pair 두 기준으로 `TMD ⊆ DE ⊆ EDE`를 검사한다. 기존 TMD는 각 변도 이미 검사하므로 이 전제가 맞으면 TMD Edge는 DE에 남아야 한다. 차집합을 조건/object별 원본 source, row, 후보 출처, 관련 triangle, cover/pair witness와 함께 저장한다. 위반은 개선으로 해석하지 않고 source/index, component, supercover/pair 정의, T0/TM/TMD 구성과 Qhull tie부터 조사한다. 해결 전 포함 관계 PASS나 정책 우월성을 보고하지 않는다.

K8/TMD/DE/EDE의 최종 unique Edge에 가능한 한 동일한 B의 경로 감사를 적용한다. Task05의 `edge_audit`는 이미 이 선분 helper를 사용하므로 저장 사유/witness를 우선 재사용하고 부족한 항목만 승인 후 감사한다. Normal 차이는 유효한 endpoint normal 둘이 있는 Edge에서만 계산하고 미측정 분모도 기록한다.

### C2. 72+249=321의 저장 자료 재확인

이번 계획에서 Task05 JSON의 `records[*].component_status.N/status`와 저장 `stages.K8/T0.degree`만 읽어 합산했다. 새 Edge나 validator를 실행하지 않았다.

| 분모 | 전체 Point-row 관측 수 | K8 고립 | T0 후보 없는 Point | n=2 component / Point | n≥3 전체 collinear component / Point | K8 비고립→T0 고립 |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| Primary .5, 8 frame·두 object | 4174 | 129 | 253 | 16 / 32 | 16 / 92 | 124 |
| Reference 1.0, 8 frame·두 object | 6921 | 141 | 338 | 20 / 40 | 28 / 157 | 197 |
| 두 ratio 합계 | 11095 | 270 | 591 | 36 / **72** | 44 / **249** | **321** |

즉 321은 두 ratio를 합한 퇴화 component의 Point-row 수이며 episode 고유 node 수나 복구 보장이 아니다. 이 저장 집계에서 T0 후보가 없는 591개는 singleton 270개 + 2점 72개 + collinear 249개다. 승인 후 source-index별 component membership과 저장 NPZ degree로 재확인하고 frame/object별 상세표를 만든다. 이것은 EDE accepted 결과와 별도다.

후속 결과는 아래 일곱 수량을 같은 분모와 source 목록으로 분리한다.

1. 저장 T0 degree=0인 Point.
2. n=2 component의 Point.
3. n≥3 전체 exact-collinear component의 Point.
4. 실제 신규 fallback 후보에 incident한 unique Point와 후보 Edge 수.
5. fallback 후보 중 Mask/Depth 사유로 거부된 unique Edge와 관련 Point.
6. TMD degree=0이고 EDE degree>0인 실제 복구 Point. DE에서 복구된 부분과 EDE fallback에서 추가 복구된 부분을 따로 표시한다.
7. EDE에서도 degree=0인 Point와 남은 원인(singleton/후보 없음/모든 후보 거부/계산 실패 등).

전체 TMD 대비 EDE isolated/Graph component 변화도 반드시 함께 보고한다. 부분집합 321만 분모로 삼아 전체 연결성 개선으로 설명하지 않는다.

### C3. Frame0 Box 원거리 trade-off

저장 `suspect`에서 `(251,206)↔(259,206)`, 263.468742mm pair는 `.5` rows `[856,861]`, `1.0` rows `[1600,1602]`다. 두 조건 모두 object=Box, observed component=20, 선택점 12개, `status=collinear`, T0 후보 없음, incident triangle=0, TMD Edge 없음이다. 기존 triangle 검증은 N/A다. 약 8.431m의 동일 depth·유효 관측 경로라는 기록은 [Task05 보고서](05_triangle_intra_object_edges.md)의 한계이며 실제 새 chain 검증 결과는 아직 없다.

승인 후 두 ratio에서 다음을 따로 추적한다.

- 12개 Point 전체의 원래 row/source/UV/camera XYZ와 deterministic chain 순서. 기존 263mm pair가 chain의 **인접 후보인지**를 먼저 확인한다. 양 끝이 선택됐다는 사실만으로 해당 긴 pair가 후보라고 가정하지 않는다.
- 모든 chain 후보의 UV 거리·실제 camera XYZ 길이, cover/cell/pair 감사 및 accepted/rejected 결과. 원래 pair의 직접 연결과 여러 짧은 Edge를 통한 간접 연결을 구분한다.
- 기존 고립 Point 중 연결된 수·잔존 고립 수, 전체 원거리 component의 degree/Graph component/길이 분포와 최대 Edge. Full extent와 확대 그림을 함께 제공한다.
- 전체 frame/object의 endpoint depth 분포, 큰 depth tail과 Edge 점유를 고정한 동일 진단 구간으로 보고한다. 8.431m 추적 component는 진단용 태그이며 정책 입력이나 negative ROI가 아니다.

EDE가 고립을 줄이는 동시에 오염된 plateau의 연결을 늘릴 가능성을 필수 trade-off로 평가한다. 새 긴 Edge도 그대로 공개한다. Targeted depth cutoff, 물체별 예외, known-hole/negative ROI, radius/length cutoff 또는 새 outlier 제거를 도입하지 않는다.

### C4. frame/object/ratio별 기록

| 범주 | 기록할 지표와 분모 |
| --- | --- |
| Connectivity | 전체 N, full observed support component 수 및 선택점이 있는 component 수, 실제 Graph connected component 수, singleton 포함 component-size 분포, isolated 수/비율, degree mean/median/p95/max. 빈 object의 N=0 비율·분위수는 N/A, 개수는 0. |
| 변화 | TMD→DE와 DE→EDE의 새 비고립 Point/source, TMD→EDE 전체 복구·잔존 고립, 추가/제거 undirected Edge 집합, component split/merge 대응. Isolated node도 Graph component에 포함. |
| Geometry | 후보/accepted/rejected unique U, directed E, UV 거리와 camera XYZ 길이 mean/p95/max 및 raw 분포/top-longest source/UV/XYZ. EE 좌표는 원본을 표시만 한다. |
| Observation | mask 밖, invalid depth, overlap, image 밖, depth jump, unsupported/missing pair, invalid delta 및 검증 불가를 분리. Edge 단위 사유 count와 witness cell/pair count를 구분. 원거리 depth 점유, normal >20° 유지 수와 normal 측정 가능 분모. |
| 후보 상태 | n=0/1/2, 전체 collinear, noncollinear, nearly-collinear/cocircular flags, QhullError, 누락 vertex/coplanar 및 fallback별 후보·통과·거부/incident Point. |

분포는 unique undirected Edge 기준이며 directed 복제로 길이 표본을 늘리지 않는다. Normal 20°는 Structural 후보 검출의 기존 의미를 유지하고 veto로 쓰지 않는다. 실제 관측의 모서리 횡단 예시는 RGB/UV·depth 경로·normal 화살표로 보여주되 물리 face 정답이 없으면 `NOT_ESTABLISHED`로 표시한다.

### C5. CPU 및 메모리 계측 설계 — 승인 후에만 실행

기존 **277.977ms**는 component 비용과 분석용 triangle 전체/변 검증, 모든 사유, T0/TM/TMD 포장을 포함한 Task05 조건별 평균이다. 이를 새 EDE와 직접 나누어 속도 개선율로 쓰지 않는다. Triangle raster를 없애도 현재 `supercover`는 선분 bbox grid를 할당하므로 비용 감소를 보장하지 않는다.

- 동일 interpreter/host/package/Qhull/thread 설정, 동일 16개 고정 입력에서 3 warmup + 5 측정을 계획한다. 방법 순서를 교대하고 raw sample·mean/median/p95/max를 보존한다. 5회 p95는 기술통계다.
- 정확성 비교는 저장 baseline을 우선한다. **속도 비교용으로만** K8/TMD를 고정 입력에서 제한 재실행하며 기존 helper 의미를 바꾸지 않는다. 같은 실행에서 DE/EDE도 독립적으로 후보 생성부터 receiver까지 측정한다. 저장 triangle 로드 또는 DE/EDE 사이 후보/검증 cache 공유 시간을 후보 생성 시간으로 보고하지 않는다.
- 모든 방법은 저장 row별 component를 그대로 사용한다. 기존 TMD builder가 요구하는 map은 선택 source 위치에 저장 ID를 넣는 lookup adapter로 공급할 수 있다. 이는 미선택 pixel의 component를 추정하는 full-support map이 아니며 외부에 그런 map으로 저장하지 않는다. K8도 같은 저장 ID를 사용한다. Component 재분할·sampling·feature extraction·EE 변환은 없다.
- 공통 입력/label lookup 준비 및 `prepare_validation` 시간을 명시한다. 계측 범위는 저장 evidence 이후로 통일하고 I/O/hash audit/그림/압축/Logger/전체 공통 Edge 감사는 별도로 기록한다. Packet copy, method별 필수 validation·사유 기록, dedup/packaging, receiver는 포함한다. 기존 TMD helper의 추가 stage 포장·triangle 진단 비용도 분리해 공개하며 최적화된 production TMD라고 부르지 않는다.
- 단계: 후보 생성 전체, Delaunay 단독, 2점 fallback, collinear 판정/정렬/chain, Edge supercover, Mask 검사, depth pair 열거/lookup/검사, Edge dedup/packaging, packet copy/packaging, Graph receiver, 실제 CPU 전체 wall 및 process CPU time. 상위 subtotal과 하위 단계는 중복 합산하지 않는다. TMD의 triangle raster/검증은 별도 단계로 표시한다.
- Object별 후보/검증 loop 시간은 분리하고 전체 packet receiver/공통 준비는 shared로 표시한다. Shared 비용을 object별 실측값으로 꾸미지 않는다. Standalone method subtotal과 cache 공유 전체 실행 wall도 구분한다. 전체 Vision latency·20Hz 통합 성능으로 해석하지 않는다.
- Fresh worker별 import baseline, input-load high-water, core peak RSS와 차이, packet/evidence/lookup/후보/cover 배열 bytes, 별도 분석 포함 전체 peak를 기록한다. Darwin/Linux RSS 단위를 통일하고 native Qhull/allocator overhead는 RSS 포함·개별 분리 불가로 표시한다.

최소 ablation은 DE/EDE 하나의 동일 validator를 사용한다. 별도 fast path, 새 raster 알고리즘, cutoff sweep을 추가하지 않는다. 비용 이득은 위와 같은 범위에서 얻은 측정으로만 판단한다.

### C6. 최소 계약 테스트 — 지금은 실행하지 않음

기존 [triangle 테스트](../../visual/runtime/test/test_triangle_intra_object_edges.py)의 Fraction raster oracle, synthetic fixture, observation-only 접근 제한과 receiver 사례를 우선 재사용한다. Real 입력의 PCA5/Feature 생성은 금지한다. Synthetic에서만 직접 정의한 관측/evidence를 구성한다.

| 계약 | 최소 사례 / 확인 내용 |
| --- | --- |
| 소속·작은 n | 빈 전체/빈 object/선택점 없는 support component, n=1, n=2; 동일 object/component만 후보 생성, 두 점도 경로 실패 시 0 Edge |
| 직선 chain | n≥3 horizontal/vertical/positive·negative slope, uneven spacing, 역순·행 permutation; exact 판정, source pair 불변, 이웃 n−1개 이하, 원본 행 불변, complete graph 금지 |
| 퇴화와 비퇴화 구분 | 정상 직선 vs Mask 단절/Depth jump 직선. Endpoint가 우회 관측 경로로 같은 component여도 직선 segment 실패는 거부. 비일직선 component의 국소 직선 subset에는 추가 chain 금지 |
| Delaunay 일치·예외 | 저장 T0 후보 일치, cocircular repeat, nonzero nearly-collinear, mocked QhullError/omitted vertex; 상태 분리와 자동 fallback 부재 |
| 선분 cover/pair | Fraction oracle, corner-touch 네 직교 pair, horizontal/vertical/negative slope, image boundary; bbox 내부라도 cover 밖 pair의 실패는 무관 |
| 관측 사유 | Mask 밖/invalid·nonpositive depth/overlap/outside, unsupported/missing/nonfinite delta/depth jump를 독립 검증. Gate의 `>` 경계도 확인 |
| Edge 독립성 | ABC에서 BC만 실패 시 AB/AC 유지, triangle 내부만 실패해도 유효 변 유지, 공유 변 dedup |
| Ramp·corner·GT | Endpoint 총 dz가 커도 국소 연속이면 허용. 동일 관측에서 normal을 90°로 바꿔도 topology 불변. GT/known-hole/mesh/pose 접근 시 실패시키는 guard |
| Packaging·보존 | Accepted Edge만 양방향, E=2U, self-loop/duplicate/asymmetry/range 오류 거부. Point/camera/EE/F=5/availability/validity/provenance/row order 불변과 입력 artifact hash |
| Receiver 회귀 | v1 기존 수용 의미, v2 K8/TMD 기존 검증 유지, DE/EDE 명시 정책과 잘못된 출처·accepted-edge 집합·component/index 거부. Deep copy 및 실제 synthetic robot 3개 공급 시만 +3 |
| 집합·저장 계약 | TMD⊆DE⊆EDE, 비퇴화 DE/EDE 일치, fallback 외 추가 없음. JSON/NPZ readback, PNG decode/실제 시각 검토, 상대 링크와 protected hash |

신규 테스트와 기존 `test_triangle_intra_object_edges`, `test_intra_object_edges`, `test_vision_features`, `test_variable_rate_graph_contract`를 승인 후 관련 회귀 범위로 삼는다. 공통 Logger/Publisher 코드를 바꾸지 않으므로 이번 계획을 위해 인프라 테스트나 online 점검을 실행하지 않는다. Contract PASS는 물리 mesh 정확도·GNN 성능·최종 정책 채택을 뜻하지 않는다.

## D. 최소 변경 파일·산출물 계획

### D1. 예상 구현 파일

| 파일 | 승인 후 최소 역할 |
| --- | --- |
| `visual/runtime/extended_delaunay_edges.py` — 신규 | 저장 row/component 입력 검증, A의 후보 분기, 동일 Edge validator, DE/EDE 명시 metadata, accepted Edge 포장·audit. 기존 `QHULL_OPTIONS/supercover/prepare_validation/validate_cover`를 import하는 좁은 adapter. |
| `visual/runtime/test/test_extended_delaunay_edges.py` — 신규 | C6의 신규·receiver 회귀 계약. 기존 fixture/oracle 재사용. |
| `visual/runtime/test/run_extended_delaunay_edges.py` — 신규 | 저장 입력 gate, baseline 재사용, DE/EDE 고정 입력 비교, 공통 감사·비용 계측, 산출물과 보고서·HTML. 기존 packet/evidence IO와 가능한 metric helper를 import하고 기존 runner 전체 workflow는 호출하지 않음. |
| `graph/vision_input.py` — 좁은 수정 | 새 DE/EDE evaluation-only 정책 allowlist와 accepted-edge algebra 검증. 기존 K8 E≤16N·TMD accepted-triangle union·v1/Feature/robot 의미를 그대로 유지. |
| `docs/visual-graph-integration/00_interface.md` — 좁은 수정 | 승인 후 실제 구현 상태, edge policy metadata/수신 검증 범위만 추가. 현재 문서는 이번 계획에서 수정하지 않음. |

기존 triangle builder는 후보 생성과 triangle 검증이 결합돼 있다. DE/EDE 후보를 얻기 위해 이 builder를 전부 실행하지 않는다. 새 adapter는 동일 옵션의 짧은 Delaunay 호출/상태 수집만 수행하고 기존 validation helper를 재사용한다. 기존 TMD 모듈 refactor, 병렬 validator, 새 공통 geometry 계층을 만들지 않는다. 저장 후보와 exact 일치 계약으로 이 좁은 호출부의 일탈을 막는다.

새 receiver metadata는 evaluation-only의 명시적 policy name/version, candidate mode(`delaunay`/`extended_delaunay`), 기존 support/depth 규칙을 기술한다. 최소한 canonical accepted undirected row pair와 정렬된 출처 metadata를 갖고 `intra_object_edge_index`가 그 정확한 양방향 집합인지 검사한다. 상세 rejected 후보/cell/pair witness는 packet에 강제하지 않고 별도 audit에 둔다. 출처 값과 2점/전체 collinear chain의 소속·기하 계약을 검사하며 알 수 없는 policy는 거부한다. Dense 관측 검증은 helper 책임으로 유지하고 receiver가 RGB-D 검증까지 끝냈다고 주장하지 않는다. F=5/node provenance는 바꾸지 않는다.

이 다섯 구현 파일 외에 본 계획과 아래 결과 문서가 산출물이다. 기존 K8/TMD 소스·runner·원본 artifact는 변경하지 않는다. 새로운 파일 분리가 실제로 더 필요하면 먼저 중복 회피 근거와 변경 범위를 보고하며 자동으로 공통 API를 확대하지 않는다.

### D2. 승인 후 결과·정적 HTML

공통 `utils.experiment_logger.ExperimentLogger(team='visual', job_type='evaluation', ...)`가 만든 **새 UUID 하나**를 사용한다. 기존 run ID/project/group 규칙을 따르고 `WANDB_MODE=disabled`로 로컬 실행한다. `record/log/link_output`을 재사용하고 source run 세 개, 실제 설정·환경·Git dirty/source snapshot/hash를 보존한다. 원본 입력의 sidecar 소유권을 새 run으로 바꾸지 않는다. 게시·동기화는 실행하지 않는다.

필수 경로는 다음과 같다. `<run_id>`는 계획의 placeholder이며 지금 생성하지 않는다.

- 결과 Markdown: `docs/visual-graph-integration/05b_extended_delaunay_edgewise.md`
- 정적 HTML: `experiments/<run_id>/extended_delaunay_edgewise.html`
- 동일 run: `extended_delaunay_edgewise.json/.npz`, `records/`, `packets/`, `plots/*.png`, `input_gate.json`, `source_hashes.json`, `verification.json`, `tests.log`, source snapshot/implementation diff 및 raw timing/memory samples.

HTML은 외부 CDN 없이 로컬 PNG/표/링크로 읽을 수 있게 만든다. 동일 Point/동일 축의 K8/TMD/DE/EDE overlay, Delaunay와 2점/직선 후보 출처, triangle 실패와 Edge별 유지의 차이, Mask/Depth 경로·사유 witness, 고립/Graph component 변화, 복구/잔존 고립 Point, 8.431m component의 전체 연결 결과와 길이, 정상 모서리 횡단의 관측 예시, length/degree 분포, CPU 단계·메모리를 포함한다. 모든 조건을 표/JSON/NPZ로 공개하고 성공 사례만 선택해 요약하지 않는다.

Markdown→HTML은 `../../experiments/<run_id>/extended_delaunay_edgewise.html`, HTML→Markdown은 `../../docs/visual-graph-integration/05b_extended_delaunay_edgewise.md`로 양방향 연결한다. HTML에서 JSON/NPZ는 같은 directory 상대경로, PNG는 `plots/...`로 연결하고 계획 링크도 제공한다. JSON/NPZ readback, 상대 링크 존재, PNG decode와 실제로 본 PNG, 실제 브라우저 실행 여부를 분리 기록한다. 미실행 검사는 `NOT_EXECUTED`다. 보고서 렌더링 실패 시 저장 수치와 같은 UUID로 report-only 복구하며 Edge/benchmark를 다시 실행하지 않는다.

HTML/PNG/Markdown은 `link_output(..., kind='reports')`, JSON은 `kind='metrics'`, NPZ는 기존 trajectory 분류로 명시 등록한다. HF ZIP/publish와 W&B sync는 이번 workflow에 포함하지 않는다. Training 실패와 publishing 실패를 결합할 새 로직도 만들지 않는다.

### D3. 승인 후 작업 순서와 종료 기준

1. 기존 dirty/untracked 및 세 source run을 hash로 보호하고 고정 입력·저장 baseline gate를 완료한다. 누락/불일치는 보고하고 관련 비교를 중단한다.
2. 신규 adapter·계약 테스트·명시 receiver 검증만 구현하고 C6의 synthetic/기존 회귀를 실행한다. Sampling/Feature/변환 경로가 호출되지 않도록 guard한다.
3. 고정 16개 입력에 DE/EDE를 적용하고 포함 관계, 일곱 복구 수량, 원거리 trade-off 및 공통 경로 감사를 기록한다. 이 단계에서 정책을 production default로 전환하지 않는다.
4. C5의 같은 환경·범위로 한정한 CPU/메모리 비교를 수행한다. 계산·해석과 보고서 렌더링을 분리하고 원본 baseline을 덮어쓰지 않는다.
5. 결과 Markdown/정적 HTML/상대 asset과 hash·계약 검증을 마친 뒤 실제 실행/미실행 범위와 채택 보류 판단을 보고한다. 모든 원본 hash 불변을 확인한다.

## 이번 계획의 중단선

현재 변경은 **이 계획 Markdown 한 파일**뿐이다. 결과 Markdown/HTML/실험 UUID는 만들지 않는다. 승인 전까지 신규 정책 구현, Edge 생성, 테스트·benchmark·Graph/GNN 학습을 시작하지 않는다.

Sampling/keep_ratio/phase/voxel/PCA5/Feature estimator, EE 변환, normal veto, strict face segmentation, GT/known-hole 필터, outlier 제거, radius/length cutoff, k-NN fallback/강제 연결, robot/inter-object Edge, GNN encoder, stale geometry는 변경하지 않는다. 새 episode/simulator/renderer/GPU/MPS/PCL, dependency 설치, 외부 업로드, Git commit도 실행하지 않는다. CPU 자료를 읽은 사실과 simulator/device를 실행한 사실을 혼동하지 않는다.
