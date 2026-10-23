# Organized geometry 진단 baseline — Task 01

설계 기준은 [`00_interface.md`](../../docs/visual-graph-integration/00_interface.md)다.
이 모듈은 그 앞단의 관측 geometry·국소 기하 진단만 구현한다. 최종 Tensor,
Corner/Edge/Surface 분류, sampling, graph edge, stale cache는 포함하지 않는다.
실제 실행 결과·한계·Git 보존 기록은 [Task 01 완료 보고](../../docs/visual-graph-integration/01_organized_geometry.md)에 있다.

## 사용

Repository root에서 실행한다. 새 dependency는 없다. Runtime 계산에는 NumPy만
필요하며, 진단 runner는 기존 Matplotlib과 pointcloud 모듈의 설치된 의존성을 사용한다.

```python
import numpy as np
from visual.runtime.rgbd import PEG, iter_saved_rgbd
from visual.runtime.geometry import process_geometry

for frame in iter_saved_rgbd(
    "visual/data/step2/ep000.npz",
    "visual/results/runtime_tracking/ep000/masks.npz",
):
    result = process_geometry(frame)
    geometry = result.geometry
    peg = result.objects[PEG]
    # 이 index는 원본 H*W raster의 C-order index다. Sampling 정책은 구현하지 않았다.
    indices = np.flatnonzero(peg.valid_points)
    xyz = geometry.xyz.reshape(-1, 3)[indices]
    normals = peg.normals.reshape(-1, 3)[indices]
    normal_available = peg.normal_valid.ravel()[indices]
    variation = peg.normal_variation_rad.ravel()[indices]
```

`normal_available=False`인 point도 관측 geometry로는 유효할 수 있다. 그런 normal은
NaN이고, variation도 별도의 validity를 확인해야 한다. Float feature의 NaN을 0으로
채워 관측된 평면으로 해석하면 안 된다. `valid_points`는 최종 인터페이스의 물체별
`valid`나 sampling 기준을 확정하는 필드가 아니다.

## 함수와 결과

| 함수 | 역할 |
| --- | --- |
| `build_organized_geometry(frame)` | 입력 shape/K/mask 확인, finite positive depth 역투영, 원본 mask 복사와 중첩 표시 |
| `OrganizedGeometry.observed_mask(object_id)` | 원본 mask ∩ valid depth ∩ 중첩 아님 |
| `estimate_surface_normals(geometry, object_id, parameters)` | 같은 물체의 완전한 5-pixel stencil로 camera-facing unit normal과 validity 계산 |
| `compute_geometric_features(...)` | 4-neighbour depth 변화·경계 원인·normal variation 계산; 같은 object의 normal 결과를 전달 |
| `process_geometry(frame, parameters=None)` | 위 단계를 실행하고 source metadata·단계별 CPU 시간 반환 |

`OrganizedGeometry.xyz`는 `[H,W,3]` float32 CV camera 좌표(m)다. 원본 depth의 모든
유효 pixel을 유지하므로 배경도 포함한다. 무효 depth 위치는 XYZ 세 성분 모두 NaN이다.
`valid_depth`는 `[H,W]` bool이며 `object_masks`는 기존 `PEG=0`, `BOX=1` mapping을
그대로 사용한다. `overlap_mask`는 tracking mask 중첩이다. 중첩 pixel의 XYZ·원본 mask는
남기지만 소속을 추정하지 않고 양쪽 물체의 `valid_points`와 normal 계산에서 제외한다.

| 물체별 `GeometricFeatures` | Shape / 의미 |
| --- | --- |
| `object_mask`, `valid_points` | `[H,W]` bool; 원본 mask / 모호하지 않은 관측점 |
| `normals`, `normal_valid` | `[H,W,3]` float32 / `[H,W]` bool; 계산 불가 normal은 NaN |
| `depth_jump_m`, `depth_pair_valid` | `[H,W]` float32 / bool; 유효한 4-neighbour와의 최대 절대 z 차이, pair 없으면 NaN |
| `depth_discontinuity` | `[H,W]` bool; threshold를 넘는 유효 depth pair의 해당 물체 쪽 endpoint |
| `mask_boundary` | `[H,W]` bool; mask 안에서 in-image 이웃 mask가 끝나는 위치 |
| `depth_validity_boundary` | `[H,W]` bool; 유효한 물체 관측점 옆의 depth가 무효인 위치 |
| `image_boundary` | `[H,W]` bool; 원본 mask가 영상 가장자리에 닿는 위치 |
| `normal_variation_rad`, `normal_variation_valid` | `[H,W]` float32 / bool; 같은 물체의 이웃 유효 normal과 최대 각도(rad) |
| `normal_neighbor_count` | `[H,W]` uint8; variation에 실제 사용한 이웃 수 0–4 |

모든 배열은 원본 `(v,u)`와 직접 대응한다. `flat_index = v*W+u`를 그대로 공유한다.
별도의 pixel provenance 유틸리티나 point reordering은 없다. 입력은 수정하지 않는다.
반환 배열도 소비자가 in-place 수정하지 않는 것을 전제로 한다.

## 계산 규약

기존 `visual/scripts/pointcloud.py::backproject`의 식을 organized 배열에 적용한다.
Runtime에서 legacy plotting/SciPy 의존성을 import하지 않도록 식만 옮겼으며, 테스트에서
기존 함수와 bitwise 동일성을 확인한다. K는 zero-skew CV pinhole 형태를 요구한다.

```text
z = depth[v,u]                         # metres, finite and > 0
x = (u + 0.5 - cx) * z / fx
y = (v + 0.5 - cy) * z / fy
```

Normal은 중심과 좌·우·상·하가 모두 같은 물체의 유효·비중첩 point일 때
`cross(Xright-Xleft, Xdown-Xup)`을 정규화한다. 중심–이웃 depth jump가 하나라도
기준을 넘거나 stencil이 퇴화하면 계산하지 않는다. 방향은 `normal·XYZ <= 0`으로
카메라 쪽을 향한다. 영상 가장자리·mask 경계·hole에서는 한쪽 미분이나 복원을 하지 않는다.

Discontinuity baseline 기준은 `abs(z1-z2) > 0.01m + 0.02*min(z1,z2)`다.
`GeometryParameters`로 명시적으로 바꿀 수 있지만 이번 검증에서는 기본값을 유지했다.
Depth 변화는 다른 물체·배경의 **실제 유효 관측 depth**도 비교한다. Normal과 variation은
그 경계를 넘어 계산하지 않는다. 무효 depth를 0m point로 비교하지 않는다.

Mask 경계는 무효·중첩 pixel에도 표시될 수 있는 관측 진단이다. 단독 point selector로
사용하면 안 된다. Depth validity 경계는 유효한 endpoint에만 표시한다. 영상 밖은
가상의 mask/depth 이웃으로 채우지 않고 `image_boundary`로 구분한다.

Variation은 `atan2(||n1×n2||, n1·n2)`의 최대값이다. Normal은 존재하지만 이웃 normal이
하나도 없으면 variation은 NaN이다. RGB나 물체의 색은 계산에 사용하지 않는다.

## 저장된 실제 관측 진단

```bash
WANDB_MODE=disabled .venv/bin/python visual/runtime/test/run_geometry.py \
  --input visual/data/step2/ep000.npz \
  --masks visual/results/runtime_tracking/ep000/masks.npz \
  --camera base_camera --out visual/results/runtime_geometry \
  --frames 0 15 18 45 80 120 --repeats 3

.venv/bin/python -m unittest discover -s visual/runtime/test -p 'test_*.py' -v
```

Runner는 `iter_saved_rgbd`로 전체 source row를 연결하고 selected row만 PNG·NPZ로
저장한다. `--frames`는 tracker의 local row가 아니라 원본 `source_frame`이다.
빈 mask와 중첩·무효 depth를 각각 기록한다. GT가 있으면 runner에서만 mask 진단에
사용하며 `geometry.py`에는 전달하지 않는다. GT가 없는 episode도 처리한다.

`ExperimentLogger(team="visual", job_type="evaluate")`의 UUID, project/group,
로컬 metadata·scalar·artifact sidecar를 재사용한다. 입력 hash도 보존한다.
기존 `pointcloud.to_world`는 3D 그림에만 사용한다. Geometry NPZ에는 RGB를 저장하지
않는다. RGB overlay의 20° cutoff는 표시용이며 runtime의 feature나 분류 기준이 아니다.
3D 표시 범위는 1–99 percentile과 여백으로 제한하고 범위 밖 개수를 표시한다.
Point를 제거하지 않으며 모든 원본 관측점과 feature가 NPZ에 남는다.

CPU 시간은 전체 영상 XYZ 생성·검증, 두 물체 normal, 두 물체 discontinuity/variation,
그 합계로 구분한다. 세 번 warmup 후 각 frame을 `--repeats`만큼 순차 실행하며
loading/그림/압축 저장/logger 시간은 제외한다. GPU 구현·실행, simulator 재실행,
tracking 재실행과 HF 게시를 수행하지 않는다.
