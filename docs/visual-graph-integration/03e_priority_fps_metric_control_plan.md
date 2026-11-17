# Task 03-E — Depth-Weighted Structural FPS & Local Geometry Preservation: 구현 전 계획

**상태: 계획 검토 대기. 구현·실험·테스트는 미승인·미실행이다.**

기존 Task 03-E의 단순 3D FPS vs image-space 2D FPS 비교 계획을 이 계획으로 대체한다.
현재 코드와 저장 artifact를 읽어 재사용 범위를 확인했으며, 기존 FPS 병목 분석을 반복하지 않았다.
이 문서 저장은 구현·실험·테스트의 승인을 의미하지 않는다.

## A. 기존 코드·artifact 재사용

### A.1 기준 자료

| 구분 | Run ID | 재사용 자료 |
| --- | --- | --- |
| Task03 | `2551448b0fae4c6bbce3caf48774275f` | [comparison.json](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.json), [Frame 0 NPZ](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0000_samples.npz), [Frame 0 metrics](../../visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/f0000_metrics.json) |
| Task03-D Phase1 | `d3f14ee494894daabb8ba2fab8e1b00c` | [단계별 coverage·ROI·depth 결과](../../visual/results/diagnostic_previews/03d_phase1/diagnostic.json), [보고서](03d_box_rim_bottleneck_phase1.md) |
| Task03-D Phase2 | `aa5d71078e0841fd886b5026d6030ebc` | [기존 선택 순서·거리 경쟁 결과](../../visual/results/diagnostic_previews/03d_phase2/priority_fps_trace.json), [보고서](03d_priority_fps_trace_phase2.md) |
| Task03-D Phase3 | `f36f1e9e2bb64866b8d6c792db158635` | [seed·공통 초기점 통제 결과](../../visual/results/diagnostic_previews/03d_phase3/priority_fps_seed_control.json), [보고서](03d_priority_fps_seed_control_phase3.md) |

저장된 Phase1–3 결과는 근거로 인용한다. 기존 병목 trace와 seed 실험은 반복하지 않는다.
위 대용량 artifact와 `experiments/`의 링크는 현재 호스트의 로컬 자료를 가리키며 별도 checkout에는 없을 수 있다.

### A.2 기존 FPS와 저장 후보 복원

[기존 `fps_indices()`](../../visual/runtime/sampling.py)를 그대로 사용한다.

- float64 제곱거리와 source 순서에 따른 argmax tie 처리를 유지한다.
- `initial=[후보 내 local index]`, `count=32`는 초기점을 포함한 총 32점을 반환한다.
- `initial` 경로에서는 초기 RNG draw를 수행하지 않는다.
- 새 FPS 선택 알고리즘을 구현하거나 복제하지 않는다.

Task03 NPZ의 `evidence/{fd2|pca5}/{object_id}/20/orientation_flat`에서 후보 source index와 순서를 읽는다.
원래 Priority 선택 순서는 조건별 `flat_index`와 `priority_selected`에서 읽는다.

전체 XYZ와 관측 support는 기존 [Task01 Frame 0 geometry NPZ](../../visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/f0000.npz)를 사용한다.
여기에 dense XYZ, validity, object mask, overlap mask가 저장돼 있다.
저장 배열로 기존 `OrganizedGeometry`를 구성하고 후보 flat index를 gather한다.
Backprojection이나 frame geometry를 다시 계산하지 않는다.
저장 metadata의 episode·mask hash와 camera는 Task03과 같으며, 배열 단위 동일성은 구현 승인 후 검증한다.

저장된 seed 0의 첫 Priority source index와 후보 수는 다음과 같다.

| Object | Estimator | 후보 node 수 | 고정 첫 flat index |
| --- | --- | ---: | ---: |
| Peg | S2 / FD2 | 870 | `213708` |
| Peg | S3 / PCA5 | 703 | `214225` |
| Box | S2 / FD2 | 297 | `101598` |
| Box | S3 / PCA5 | 599 | `96553` |

### A.3 Priority → Surface 재현

변경한 Priority source index를 전체 기존 관측 support의 local row로 매핑한다.
그다음 원본 XYZ에 다음 기존 함수를 호출한다.

```python
fps_indices(
    support_xyz,
    total_budget,
    seed=object_seed,
    initial=priority_rows,
)
```

Priority는 결과 앞부분에 유지되고 나머지가 Surface다.
Surface 후보는 해당 물체의 전체 기존 관측 support이며, Surface 거리는 기존 3D Euclidean distance다.
Allocation 재계산이나 전체 sampler 복제는 필요 없다.

### A.4 평가 helper와 국소 정보

[기존 sampling 평가 helper](../../visual/runtime/test/sampling_evaluation.py)의
`CoverageMatcher`, `staged_reference_coverage`, `count_ratio`와 기존 거리 통계를 사용한다.
기존 Phase1–3 runner의 geometry/context 재구성 경로는 호출하지 않는다.

기존 [normal-comparison Frame 0 artifact](../../visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/f0000.npz)에는 face map과 Peg `seam_endpoints`가 있다.
기존 [structural-edges Frame 0 artifact](../../visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/f0000_pairs.npz)에는 평가 ROI가 있다.
Task03의 저장 reference와 일치함을 확인한 뒤 평가에만 사용한다.
Negative ROI union은 기존 ROI 제외 규칙을 적용해 복원한다.

### A.5 확인된 제약과 승인 후 필요한 확인

1. **Reference의 물리적 순서는 저장돼 있지 않다.** UV/XYZ만 있고 선분 ID·물리적 순서·arc length는 없다.
   Peg reference는 raster 순서이며 Box rim도 물리적 선분 순서가 아니다.
2. **신규 선택점의 전체 원본 feature 복원에는 gap이 있다.** Task03 NPZ는 선택점의 23 metadata field만 저장한다.
   과거 dense PCA5 normal cache는 reference 구현이며 Task03의 optimized PCA와 bit-exact하다고 보장할 수 없다.
   이를 원본 S3 feature로 대체하지 않는다.
   신규 선택점은 source correspondence를 보존하고 exact feature가 없는 항목은 명시적으로 미저장 처리한다.
   전체 feature tensor의 exact 복원이 필수라면 현재 저장 자료의 gap으로 보고한다.
3. 승인 후 추가 확인은 저장 geometry/support·reference·ROI provenance, feature 복원 범위, λ=1 재현 검증으로 한정한다.
   Normal·후보·tracking을 다시 계산하지 않는다.

## B. 정확한 최소 실험 설계

### B.1 연구 질문과 고정 조건

- **가설 A — Depth distance domination:** Z 차이의 가중치를 낮추면 기존 rim/seam 후보를 더 많이 선택하는가?
- **가설 B — Global FPS의 local structure 보존 한계:** Z 영향을 줄여도 좁은 rim·긴 seam의 국소 coverage와 간격 문제가 남는가?

실험 범위는 ep000 Frame 0 × S2/S3 × Peg/Box × 세 λ이며 총 12개 object별 결과다.

| 고정 항목 | 값 |
| --- | --- |
| Frame / episode | ep000 Frame 0 |
| Estimator | S2 / FD2, S3 / PCA5 |
| `num_points` | 256 |
| `structural_ratio` | 0.25 |
| `orientation_threshold` | 20° |
| Seed | 0 |
| 후보·관측 support | 기존 집합과 source 순서 유지 |
| GT 사용 | 선택 완료 후 평가에만 사용 |

| Object | Total budget | Priority | Surface |
| --- | ---: | ---: | ---: |
| Peg | 129 | 32 | 97 |
| Box | 127 | 32 | 95 |

### B.2 변경 변수와 실행 순서

Priority FPS의 거리 metric만 변경한다. Camera CV frame의 metre 단위 좌표를 사용한다.

\[
d_\lambda^2(p_i,p_j)
=(x_i-x_j)^2+(y_i-y_j)^2+\lambda(z_i-z_j)^2
\]

| 조건 | λ | Priority 거리 계산용 좌표 |
| --- | ---: | --- |
| Baseline | 1.0 | `[x, y, z]` |
| Control A | 0.25 | `[x, y, 0.5z]` |
| Control B | 0.0 | `[x, y, 0]` |

`metric_xyz = [x, y, sqrt(lambda) * z]`는 별도 거리 계산용 배열이다.
실제 selected XYZ·feature·pixel provenance는 변환하지 않는다.
λ=0은 Camera XY-only이며 normalized image-space UV FPS와 같지 않다.
X/Y도 depth에 의존하므로 원근 효과 전체가 제거됐다고 해석하지 않는다.
Global scaling을 점 간격 개선안으로 제안하지 않는다.

실행 순서는 다음과 같다.

1. 네 object/estimator 조합의 λ=1 Priority 순서와 Final 순서를 원래 저장 선택과 정확히 대조한다.
   **하나라도 실패하면 λ=0.25/0을 실행하지 않고 불일치 원인을 보고한다.**
2. 통과 후 각 조합에서 A의 동일한 최초 source point를 후보 local index로 넣어 각 control의 Priority 32점을 선택한다.
3. Priority 선택이 모두 확정된 후 원본 XYZ의 기존 3D Surface FPS로 Peg 97점, Box 95점을 추가한다.
4. 선택 완료 후 GT·ROI를 사용해 평가한다.

물체별 seed는 기존 `SeedSequence([0, object_id])` 계약을 유지한다.
같은 estimator·물체 안에서 후보 집합·순서·최초 source point·quota·support·source correspondence는 동일하다.
효과는 각 estimator·물체 내부의 λ paired 차이로 먼저 해석한다.
추가 λ/seed/frame/episode sweep은 수행하지 않는다.

### B.3 Structural geometry retention

집합을 다음과 같이 정의한다.

| 기호 | 정의 |
| --- | --- |
| X | 전체 기존 관측 support |
| C | 저장 후보 |
| P | Priority 선택 |
| S | 추가 Surface 선택 |
| F | P ∪ S |
| R(A) | 같은 point/reference pair가 2px AND 3mm를 만족하는 unique reference 집합 |

각 조건에서 다음을 기록한다.

- C/P/S/F의 실제 node 수, matching node 수, unique reference 수와 ID.
- Candidate·Priority·Final coverage.
- Surface 추가 unique coverage: `|R(S) \ R(P)|`.
- Priority/Surface reference overlap: `|R(P) ∩ R(S)|`.
- Raw numerator/denominator와 비율.

Retention 분모는 같은 관측 reference인 Box 103 / Peg 462로 고정한다.
GT visible Box 103 / Peg 464는 별도 observation 지표로 보존한다.
한 node가 여러 reference를 덮는 경우와 여러 node가 같은 reference를 덮는 경우를 구분한다.

### B.4 국소 Structural 분포와 간격

Priority-only와 Final 분석을 분리한다.

1. P/F의 실제 Camera CV XYZ·pixel 위치를 표시하고 joint matcher에 매칭된 rim/seam node를 별도로 표시한다.
   Final의 Surface 경로에서 structural reference를 덮은 점도 포함한다.
2. 각 reference에서 선택점까지의 최근접 pixel 거리와 실제 3D 거리(mm)를 각각 저장한다.
   전체 선택점 기준 기존 지표와 matching structural node만의 지표를 구분한다.
   두 최근접점이 다르더라도 이를 joint match로 합치지 않는다.
3. 관측 reference를 주분모로 mean/median/p95/max와 covered/missing 위치를 비교한다.
   Peg의 미관측 reference 2개는 별도로 표시한다.
4. 최소 공통 구간 집계는 기존 `reference_quadrants`를 재사용하되 영상 공간 구역으로 명명한다.
5. Peg는 저장 face map의 기존 인접 face 전이와 seam endpoint를 이용해 평가용 face-pair별 구간을 구분한다.
   모서리 접점·복수 귀속·구간 판별 불가 reference는 별도로 남긴다.
   Box는 기존 rim 정의의 네 선분에 저장 reference XYZ를 대응시키되 reference 좌표·개수·ID는 바꾸지 않는다.
6. 구간별 coverage와 matching node 수, 3D/pixel 최근접 node 간격을 기록한다.
   물리적 순서가 확인된 단일 선분에서만 선분을 따른 간격과 끝부분 미보존 길이를 계산한다.
   순서가 불명확한 구간은 해당 간격을 N/A로 두고 공간 overlay·거리 분포로 보고한다.

Reference 배열 순서나 구간을 가로지르는 Euclidean 거리를 구조상 간격으로 해석하지 않는다.
전체 seam retention 개선만으로 모든 개별 모서리가 충분히 표현됐다고 판단하지 않는다.
새 허용 간격·필수 node 수·구조별 quota는 도입하지 않으며 평가 구간 정보는 sampler에 전달하지 않는다.

### B.5 일반 Surface representation

전체 동일 물체 관측 support X에서 Final F까지의 최근접 실제 3D 거리 mean/p95(mm)를 기존 방식으로 계산한다.

- 최종 sampled point의 공간 분포.
- Priority/Surface source 중복 여부.
- 기존 negative ROI union의 Priority 점유.
- Priority depth min/median/p95/max.
- 기존 관측 depth median/p95 구간별 후보·Priority node 수, 비율 및 선택률.

Depth 구간은 설명용이며 filtering이나 오염 label로 사용하지 않는다.
일반 평면의 Surface point를 자동으로 잘못된 선택으로 분류하지 않는다.

### B.6 재사용 결과와 새 계산의 경계

| 재사용 | 승인 후 새 계산 |
| --- | --- |
| 기존 후보·reference·저장 선택·allocation | λ별 Priority/Final 선택과 baseline reproduction |
| Phase1–3의 수치·선택 순서·해석 근거 | Peg의 단계별 retention 및 λ별 paired 차이 |
| 저장 geometry/support·평가 ROI·face map | 국소 분포·간격과 동일 support의 Surface trade-off |

## C. 변경 파일 및 산출물 계획

### C.1 변경 파일

| 경로 | 계획 |
| --- | --- |
| `visual/runtime/test/run_priority_fps_metric_control.py` | 신규 평가 전용 runner: 저장 입력 로딩, 기존 FPS 호출, 평가·그림·보고서 생성 |
| `visual/runtime/test/test_priority_fps_metric_control.py` | 신규 최소 계약 assertion |
| `docs/visual-graph-integration/03e_priority_fps_metric_control.md` | 구현 승인 후 반드시 작성할 최종 결과 보고서 |

Production `sampling.py`, normal estimator, allocation, feature extraction, tracking과 기존 Phase1–3 파일은 변경하지 않는다.
현재 Git의 기존 변경도 보존한다.

### C.2 실행 identity와 결과 파일

승인 후 기존 `ExperimentLogger(team="visual", ..., job_type="evaluate")`로 새 UUID를 생성한다.
`WANDB_MODE=disabled`로 로컬 기록하며 기존 UUID/project/group 규칙은 변경하지 않는다.

새 `experiments/<new_uuid>/`에 다음을 저장한다.

| 산출물 | 내용 |
| --- | --- |
| `priority_fps_metric_control.html` | 간단한 로컬 HTML 결과 비교 |
| `priority_fps_metric_control.json` | 선택 순서·source index·집계·paired 차이·제약 |
| `priority_fps_metric_control.npz` | 원본 선택 XYZ/pixel/object 대응, reference별 거리·coverage 배열, exact feature 가용 여부 |
| PNG | 선택 위치, Box rim 확대, Peg seam/모서리 구간 확대, coverage 비교, 국소 간격/Surface trade-off |
| 검증·재현 자료 | `verification.json`, 입력·source hash, source snapshot, 변경 diff, 테스트 로그, 보고서 snapshot |

HTML에는 다음을 포함한다.

- S2/S3의 기존 3D FPS와 depth-weighted FPS의 Priority/Final 선택 위치.
- Box rim 확대 overlay.
- Peg seam 및 모서리 구간 확대 overlay.
- λ별 Priority/Final coverage 비교.
- 국소 간격과 일반 Surface coverage trade-off.
- JSON/NPZ/PNG 상대 링크.

외부 자원 없이 HTML을 표시하고 결과 파일을 직접 열 수 있게 한다.
최종 Markdown에서는 `../../experiments/<new_uuid>/priority_fps_metric_control.html`과 결과 파일,
기존 네 run 및 원본 Task03 artifact를 상대경로로 연결한다.
기존 artifact를 새 run 소유로 재등록하거나 기존 sidecar를 변경하지 않는다.

이번 문서는 계획 저장만 수행한다. Runner·test·최종 결과 Markdown·HTML·새 실험 UUID는 생성하지 않는다.
외부 업로드·dependency 설치·commit은 하지 않는다.

## D. 검증 및 완료 기준

### D.1 승인 후 검증

| 항목 | 기준 |
| --- | --- |
| λ=1 재현 gate | 네 조합 모두 Priority 32점과 Final 전체 source 순서·XYZ·coverage가 원래 저장 결과와 일치. 실패하면 controls 중단 |
| 초기 선택점 | 같은 estimator·물체에서 A의 첫 flat index를 λ 간 동일하게 유지 |
| 거리 수치 | 합성 입력에서 변환 좌표 제곱거리가 `dx² + dy² + λdz²`와 일치 |
| λ=0 | Z만 변경해도 metric 거리와 선택이 동일. XY 중복 tie는 기존 source 순서로 처리 |
| 원본 배열 | metric 좌표는 별도 배열. 선택 XYZ·pixel·object ID·가용 원본 feature는 source index로 gather하며 변환하지 않음 |
| Feature gap | 미저장 feature를 근사 cache로 채우지 않으며 exact 복원 범위를 보고 |
| 통제 | 후보·순서·quota·threshold·support hash가 λ 간 동일 |
| Budget·중복 | P는 물체별 32점. P/S source 중복 없음. Final Peg 129 / Box 127, 전체 256점 |
| Joint matching | 같은 point/reference pair의 2px AND 3mm 조건 사용 |
| 집계 | Node 수와 unique reference 수 구분. `R(F) = R(P) ∪ R(S)`와 Surface 추가·overlap 확인 |
| Surface 거리 | Peg/Box 모두 같은 support·거리 방향·단위로 mean/p95 계산 |
| GT 독립성 | 합성 GT annotation을 변경해도 Priority/Final 선택 동일. 평가 구간 label도 sampler 입력에서 제외 |
| 원본 보존 | 원본 artifact/input/source 및 기존 작업 파일 hash를 실행 전후 비교 |
| 보고서 | HTML·Markdown 상대 링크와 PNG 표시 확인 |

### D.2 테스트 재사용

다음 기존 테스트를 재사용한다.

- `visual.runtime.test.test_sampling`
- `visual.runtime.test.test_sampling_geometry`
- `visual.runtime.test.test_sampling_evaluation`
- `visual.runtime.test.test_priority_fps_trace`
- `visual.runtime.test.test_priority_fps_seed_control`

신규 테스트는 metric·고정 초기점·원본 불변·budget 계약에 집중한다.
실제 데이터 replay는 runner에서 수행하며 테스트에 반복하지 않는다.

### D.3 해석과 완료 기준

완료 여부는 개선 여부와 독립적이다.
같은 입력에서 실제 선택 위치, Box rim/Peg seam retention, 국소 누락·집중 및 Surface 손익을 재현 가능하게 제출하면 완료다.

| 결과 | 해석 범위 |
| --- | --- |
| λ 감소로 rim/seam 보존 개선 | 현재 Priority 선택에 depth 거리 경쟁이 영향을 준다는 근거. 원거리 점의 오염 여부나 전체 episode 우수성은 확정하지 않음 |
| λ 감소에도 국소 누락 유지 | Z 조정만으로 부족하다는 결과. Local/component-aware sampling은 별도 후속 작업으로만 검토 |
| Box 개선과 Peg 또는 Surface 악화 동시 발생 | 물체·구조별 trade-off. Box 한 지표만으로 기본 metric을 채택하지 않음 |

이번 비교는 ep000 Frame 0에 한정한다.
BC/GNN 성능, 독립 episode 일반화, 기본 sampling 정책 채택을 결론 내리지 않는다.

### D.4 범위 밖 작업과 중단 상태

다음은 향후 구현 범위에도 포함하지 않는다.

- GT 기반 hole quota·구조별 강제 배분, 새 candidate detector.
- Normal 재계산, Structural threshold 변경, depth filtering·outlier 제거.
- Global budget·ratio 변경, local/component-aware sampling 신규 구현, FPS 알고리즘 복제.
- 추가 λ/seed/frame/episode sweep, PCL 설치, Graph/GNN/BC 구현, GPU 최적화.
- 외부 업로드·dependency 설치·commit.

현재는 읽기 확인과 이 계획 문서 저장만 완료했다.
구현·실험·테스트 및 simulator·renderer·GPU/backend 실행은 모두 미실행이다.
계획 검토 후 별도 승인을 받을 때까지 여기서 중단한다.
