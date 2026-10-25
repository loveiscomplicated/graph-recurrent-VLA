# Task05 — Observation-Constrained Triangular Graph Edges

Run UUID: `91ecd0413a87455cabc5e813cddaea8d`. 계약 **PASS**, 고정 입력 16개 조건의 CPU offline 비교 완료. 최종 정책은 **채택 보류**다.

[정적 HTML](../../experiments/91ecd0413a87455cabc5e813cddaea8d/triangle_intra_object_edges.html) · [JSON](../../experiments/91ecd0413a87455cabc5e813cddaea8d/triangle_intra_object_edges.json) · [NPZ](../../experiments/91ecd0413a87455cabc5e813cddaea8d/triangle_intra_object_edges.npz) · [검증](../../experiments/91ecd0413a87455cabc5e813cddaea8d/verification.json) · [테스트 로그](../../experiments/91ecd0413a87455cabc5e813cddaea8d/tests.log) · [Source/input hashes](../../experiments/91ecd0413a87455cabc5e813cddaea8d/source_hashes.json) · [Implementation diff](../../experiments/91ecd0413a87455cabc5e813cddaea8d/implementation.diff)

## 결과와 판단

TMD는 관측 규칙상 invalid unique Edge를 7722→0개로 줄였다. 동시에 K8에서 비고립이던 1337개 node가 고립되고, 기존 Graph component 132개가 추가로 분리됐다. 이는 16개 조건을 합한 수이며 물리적 동일 node의 episode 통계가 아니다.

공통 component 비용을 포함한 조건별 CPU mean의 평균은 K8 20.915ms, TMD 277.977ms다. 이번 구현에서 속도 개선은 없었다. 검증 비용을 포함한 실제 값이며 전체 Vision→Graph latency나 20Hz 결과가 아니다.

| stage | Triangle T | unique U | directed E | isolated | Graph components | 관측 감사 invalid U | normal >20° U |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K8 | N/A | 49191 | 98382 | 270 | 408 | 7722 | 13673 |
| T0 | 19456 | 29902 | 59804 | 591 | 649 | 5095 | 6247 |
| TM | 15105 | 24709 | 49418 | 1429 | 1491 | 723 | 5865 |
| TMD | 14385 | 23814 | 47628 | 1607 | 1668 | 0 | 5724 |

Support 단계에서 triangle 4351개, 이후 depth 단계에서 720개를 거부했다. T0/TM은 동일 Delaunay의 중간 진단이며 새 후보는 TMD 하나다.

## 입력·규칙·수신 계약

Task04 `9f90293b227e4dbfaaa6290afad9cc3b`와 Task03-K `9185355a32b8433abc70f1738f983377`를 사용했다. Frame 0/18/27/45/56/66/80/120 × keep_ratio .5/1, phase 0이다. 전체 입력의 k-NN(k=8)과 observed component exact 재현을 신규 비교 전에 완료했다. Frame18/56의 빈 Box도 유지했다.

저장 Point/camera XYZ/EE XYZ/F=5/validity/provenance/행 순서와 변환 metadata를 그대로 읽었다. Sampling, PCA5/Feature extraction, EE 변환을 재실행하지 않았다. Receiver의 기존 좌표 일관성 검사는 유지했으며 새로운 좌표를 생성하거나 덮어쓰지 않았다.

동일 object/observed component의 UV에 2D Delaunay (`1.17.1`, `Qbb Qc Qz Q12`)를 적용했다. Integer pixel centre의 닫힌 ±.5 cell과 접촉하는 모든 cell을 int64 SAT로 포함했다. Boundary 세 경로와 triangle 전체를 각각 검사하고, **해당 cover에 양 끝 cell이 모두 속하는 right/down pair만** 검사했다. Bbox 외부 pair는 검사하지 않는다. Corner-touch는 모든 접촉 cell을 포함하므로 보수적인 경계 거부 가능성이 있다.

Mask 밖/invalid depth/overlap/image 밖과 pair missing/unsupported/invalid delta/depth jump를 구분했다. 기존 adjacent gate `abs(delta_z)>0.01+0.02*min(z)`를 유지했다. Endpoint 전체 depth 차이, normal angle, face label을 veto로 사용하지 않았다. 모든 거부 사유를 기록하기 위해 support에서 이미 거부된 triangle도 depth 진단을 수행했으며 그 비용은 core에 포함했다.

Triangle 하나가 실패하면 그 세 변을 모두 제거하고 다른 accepted triangle의 공유 변만 유지했다. Receiver는 accepted triangle index/소속/양의 UV 면적과 Edge 합집합의 정확한 일치를 검증한다. Dense RGB-D 검증은 helper의 책임이다. 기존 k-NN/v1 의미와 실제 공급된 robot에서만 +3 규칙을 보존했다.

## Frame 0 원거리 연결

- `f0000_r0.5`: 약 263.469mm pair의 Delaunay 후보 포함 **False**, incident triangle 0개, 최종 Edge 유지 **False**. Observed component 20의 선택점 12개가 UV collinear여서 triangle을 만들지 못했다. 따라서 이 pair의 support/depth triangle 검증은 N/A다. 상세 상태와 다른 top-10 원거리 Edge는 HTML/JSON에 있다.
- `f0000_r1`: 약 263.469mm pair의 Delaunay 후보 포함 **False**, incident triangle 0개, 최종 Edge 유지 **False**. Observed component 20의 선택점 12개가 UV collinear여서 triangle을 만들지 못했다. 따라서 이 pair의 support/depth triangle 검증은 N/A다. 상세 상태와 다른 top-10 원거리 Edge는 HTML/JSON에 있다.

해당 pixel 경로는 약 8.431m의 동일 depth와 jump=false를 갖는 기존 관측이다. Mask/depth가 함께 오염된 plateau를 이 규칙으로 물리적 outlier라고 판정할 수 없다. 대상 pair가 없어져도 물리적 오류 해결을 의미하지 않는다. 좌표/Box hole/GT에 따른 선택 예외는 없다.

Frame 0 Box의 최종 최대 Edge는 ratio .5에서 104.559mm, ratio 1에서 136.428mm다. 각각 endpoint camera-z가 약 1.290↔1.389m, 1.260↔1.389m인 다른 연결이 남았다. 전체 원거리 Point는 삭제하지 않았으며 모든 길이 통계와 top-10에 포함했다.

## 연결성·모서리 진단

Normal 차이가 20°를 넘는 TMD Edge 5724개가 유지됐다. 조건별 실제 RGB-D 경로와 normal 화살표 PNG를 제공한다. 관측의 굽은 영역을 잇는 사례와 normal-only veto가 없다는 계약 근거이며, 물리적 face/corner 정답 라벨은 **NOT_ESTABLISHED**다. Synthetic corner 테스트는 동일 support/depth에서 normal angle을 90°로 바꿔도 연결이 불변임을 확인했다.

새 고립 row/source, K8 Graph component별 TMD 분할 크기, 모든 degree와 최종 Graph component label, triangle·edge 거부 원인/witness를 JSON/NPZ에 보존했다. 모든 Point를 유지하고 2점/collinear 성분에 fallback Edge를 넣지 않았다. 정상 연결 손실과 관측 공백 제거 사이의 trade-off가 있어 채택을 자동화하지 않는다.

## 조건별 비용·크기

| condition | N | T0→TM→TMD | K8→TMD E | 새 고립 N | K8 core mean ms | TMD core mean ms | K8/TMD peak RSS MiB |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| f0000_r0.5 | 950 | 1670→1273→1168 | 8486→3966 | 109 | 26.163 | 404.059 | 152.1/162.8 |
| f0000_r1 | 1694 | 3054→2255→2065 | 15138→6958 | 252 | 64.464 | 698.236 | 154.0/170.3 |
| f0018_r0.5 | 453 | 853→600→595 | 4112→1980 | 51 | 12.065 | 200.708 | 149.2/159.1 |
| f0018_r1 | 807 | 1546→1046→1036 | 7304→3376 | 146 | 33.235 | 350.495 | 150.4/161.2 |
| f0027_r0.5 | 539 | 972→753→728 | 4888→2404 | 56 | 19.294 | 226.394 | 149.2/158.8 |
| f0027_r1 | 883 | 1633→1224→1196 | 7952→3892 | 122 | 26.147 | 365.262 | 150.6/161.3 |
| f0045_r0.5 | 514 | 879→713→695 | 4490→2272 | 47 | 19.006 | 198.515 | 149.2/158.3 |
| f0045_r1 | 830 | 1455→1145→1119 | 7318→3656 | 95 | 21.280 | 331.888 | 150.2/160.8 |
| f0056_r0.5 | 371 | 650→536→513 | 3220→1662 | 27 | 8.316 | 148.503 | 148.8/157.3 |
| f0056_r1 | 636 | 1143→902→855 | 5518→2760 | 72 | 17.947 | 257.210 | 149.3/159.9 |
| f0066_r0.5 | 424 | 712→624→599 | 3666→1948 | 28 | 18.334 | 156.683 | 148.5/157.1 |
| f0066_r1 | 630 | 1059→910→872 | 5516→2844 | 60 | 14.355 | 234.922 | 152.6/158.4 |
| f0080_r0.5 | 477 | 747→634→601 | 4128→2002 | 56 | 19.550 | 164.696 | 152.6/157.2 |
| f0080_r1 | 747 | 1204→981→921 | 6704→3090 | 102 | 13.599 | 259.551 | 150.8/159.1 |
| f0120_r0.5 | 446 | 735→606→571 | 3774→1914 | 40 | 7.719 | 183.837 | 148.6/157.4 |
| f0120_r1 | 694 | 1144→903→851 | 6168→2904 | 74 | 13.172 | 266.668 | 149.3/159.5 |

같은 CPU/interpreter에서 3회 warmup + 5회 측정하고 반복마다 방법 순서를 교대했다.
공통 component는 frame-repeat마다 한 번 생성해 두 ratio/방법이 공유한다. 조건별 core에는 그 공통 비용을 한 번 더한 standalone subtotal을 표시하며 조건 합산에서 중복 없는 전체 wall과 구분한다. 양쪽 모두 packet copy/receiver를 포함하고 TMD lookup·rasterization·support/depth·dedup도 계측했다. Hash audit/진단/NPZ/그림/logging은 core 밖이다. 5개 표본의 p95는 기술통계다.

계측 대상은 현재 evaluation helper다. 이 helper는 T0/TM/TMD의 변 합집합 세 개와 모든 triangle 거부 사유를 함께 만든다. `edge_packaging_ms`에도 세 단계 포장이 포함된다. 따라서 이 수치를 분석 출력을 생략한 production TMD-only 구현의 최소 latency로 해석하지 않는다. 이번 구현의 실제 비용을 비교하며 별도 fast path/최적화는 실행하지 않았다.

메모리는 fresh worker의 import baseline/input-load high-water/전체 peak와 배열 byte, lookup·scratch 추정치를 기록했다. Native Qhull/Python allocator overhead는 RSS에 포함되며 별도로 정확히 분리 측정하지 못했다. K8의 기존 Task04 시간은 재사용하지 않고 같은 실행에서 재계측했다.

![길이](../../experiments/91ecd0413a87455cabc5e813cddaea8d/plots/edge_lengths.png)

![연결성](../../experiments/91ecd0413a87455cabc5e813cddaea8d/plots/degree_components.png)

![CPU](../../experiments/91ecd0413a87455cabc5e813cddaea8d/plots/latency.png)

![메모리](../../experiments/91ecd0413a87455cabc5e813cddaea8d/plots/memory.png)

## 검증과 범위

계약 테스트 33개 PASS. Fraction 기반 독립 raster oracle, 퇴화/누락/QhullError, shared edge, ramp/interior seam, GT 독립성, normal veto 없음, accepted-triangle union, deep copy/offset을 검사했다. JSON/NPZ readback, PNG decode·상대 링크·원본 보호 hash는 verification.json에 별도 기록한다.

실제 브라우저: **NOT_EXECUTED**. PNG 시각 검토 상태는 verification.json에 실제 검사한 파일만 기록한다. 계약 PASS와 TMD validation 위반 0은 물리적 mesh 정확성·최종 정책 채택·GNN 성능을 증명하지 않는다.

공통 ExperimentLogger와 로컬 artifact 연결을 재사용했다. Dependency 설치·외부 업로드·Git commit, 추가 episode/simulator/renderer, CUDA/MPS/PCL, GNN/GRU/BC 학습·rollout, feature/sampling/threshold 변경, k-NN fallback은 수행하지 않았다. 결과 검토 후 다음 정책을 결정한다.

최초 수치 계산·계측 완료 후 normal PNG 렌더러의 row 인덱싱 오류가 발생했다. `failure.json`과 최초 계산 소스·수치를 보존하고 **동일 UUID의 저장 수치만으로 보고서를 복구**했다. 새 Edge/benchmark/extraction은 실행하지 않았다. `report_recovery.diff`, `numerical_results_before_report_recovery.json` 및 JSON의 전후 numerical hash를 제공한다. 최초 failed 이력은 metadata의 `initial_report_failure`에 보존하고 `report_recovery_completed=true`, `numerical_evaluation_completed=true`, `status=finished`로 완료 범위를 구분했다.
