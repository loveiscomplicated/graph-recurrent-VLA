# Tracking과 RGB-D 연결

`rgbd.py`는 Visual tracker의 출력에 같은 시점의 원본 관측을 연결하는
Visual runtime 모듈이다. RGB-D·카메라 정보·TCP pose 연결과 downstream용
물체 ID 변환을 모두 Visual 출력 단계에서 처리한다.

저장된 mask는 다음처럼 읽는다. `episode_path`와 `camera`는 tracking에 사용한
원본으로 지정한다. 기존 runner의 `tracking.json`에 두 값이 기록되어 있다.

```python
from visual.runtime.rgbd import PEG, BOX, iter_saved_rgbd

for frame in iter_saved_rgbd(
    "visual/data/step2/ep000.npz",
    "visual/results/runtime_tracking/ep000/masks.npz",
    camera="base_camera",
):
    peg_mask = frame.masks[PEG]
    box_mask = frame.masks[BOX]
    # frame.rgb, frame.depth, frame.intrinsic, frame.cam2world, frame.tcp_pose
```

`masks.npz`의 row `i`는 원본 episode의 `source_frame[i]` row와 연결된다.
RGB, depth, intrinsic, cam2world, TCP pose, 선택적 timestamp/frame_idx에 모두
동일한 source row를 적용한다. `frame_idx`는 simulator의 기록된 step이고,
`source_frame`은 NPZ 배열의 row이므로 서로 다른 값이어도 된다.
stride나 늦은 초기화가 있어도 mask row를 원본 row로 간주하지 않는다.

현재 tracker iterator에도 직접 연결할 수 있다. 아래 `selected`는 SAM2에 준
RGB 영상의 각 프레임이 원본의 어느 row인지 나타낸다. 전체 원본 영상을 그대로
추적했다면 `source_frames`를 생략한다.

```python
import numpy as np
from visual.runtime.rgbd import attach_rgbd

selected = [0, 3, 6, 9]  # 실제 tracker 입력 영상과 동일한 source row 목록
with np.load("visual/data/step2/ep000.npz", allow_pickle=False) as episode:
    for frame in attach_rgbd(
        tracker.track(state), episode,
        camera="base_camera", source_frames=selected,
    ):
        pass  # 동일 source_frame의 mask와 RGB-D/kinematics 사용
```

| 입력 | 연결 결과 |
| --- | --- |
| SAM2 Tracker ID `1` / 저장 key `peg` | Visual 출력 물체 ID `PEG=0` (Graph에서 사용) |
| SAM2 Tracker ID `2` / 저장 key `box` | Visual 출력 물체 ID `BOX=1` (Graph에서 사용) |
| `{camera}/rgb` | `rgb`: `[H,W,3]` |
| `{camera}/depth` | `depth`: `[H,W]`, 이미 metre 단위인 CV camera z |
| `{camera}/intrinsic` | `intrinsic`: `[3,3]`, CV camera intrinsic |
| `{camera}/cam2world` | `cam2world`: `[4,4]`, CV camera → world |
| `tcp_pose` | `tcp_pose`: `[7]`, world `[x,y,z,qw,qx,qy,qz]` |

출력 물체 ID는 기존 Graph prototype의 node type ID나 simulator segmentation
ID와 별개다. Graph 코드를 import하지 않는다.
`peg_id`/`box_id`, GT 물체 pose/크기, segmentation은 읽지 않는다.
원본 depth를 다시 mm→m로 변환하거나 `cam2world_gl`로 대체하지 않는다.

출력은 CPU NumPy 배열이며 Torch mask는 detach 후 CPU로 옮긴다. 기존
`SAM2Tracker.track()` 반환 형식과 `masks.npz` 형식은 유지한다. 미검출 물체는
빈 boolean mask로 남고, 원본 depth의 무효값도 유지한다. 관측 누락, frame 수나
mask 해상도 불일치, 범위 밖 source row, 알 수 없는 tracker ID는 오류로 처리한다.
현재 mask 파일 자체에는 episode/camera 식별자가 없으므로 같은 크기의 다른
episode/camera를 잘못 지정했는지는 자동 검출할 수 없다.

검증:

```bash
.venv/bin/python -m unittest discover -s visual/runtime/test -p 'test_*.py' -v
```

## GT 비교 리포트와 depth 영상

저장된 tracking mask를 평가하는 실행기:

```bash
WANDB_MODE=disabled .venv/bin/python visual/runtime/test/run_rgbd.py \
  --input visual/data/step2/ep000.npz \
  --masks visual/results/runtime_tracking/ep000/masks.npz \
  --camera base_camera \
  --out visual/results/runtime_rgbd
```

매 실행마다 `--out/<run_id>/`에 주 산출물 두 개를 만든다.

| 산출물 | 내용 |
| --- | --- |
| `gt_report.html` | 물체별 GT 비교 수치, 프레임별 변화 그래프·전체 수치표, 입력 hash와 설정, JSON 원자료 내장 |
| `depth.mp4` | RGB+예측/GT 윤곽, RGB 위 depth overlay, 추적 물체의 depth를 나란히 표시한 영상 |

GT segmentation과 `peg_id`/`box_id`는 평가 실행기에서만 읽는다. 원본의
`source_frame`에 맞춰 비교하므로 stride에서도 GT 시점이 일치한다. ID `18/19`
같은 simulator GT 값은 episode에서 읽으며 Visual 출력 ID `0/1`과 구분한다.

- Mask: IoU, precision, recall, TP/FP/FN. 평균 IoU는 GT가 보이는 모든
  프레임을 포함하고 미검출은 0이다. Micro 지표는 전체 프레임의 pixel 수를
  합산하므로 물체가 안 보이는 프레임의 오검출도 반영한다.
- 3D: 예측 mask와 GT mask를 동일한 기록 depth로 역투영한 가시 표면의
  centroid 거리와 양방향 평균 최근접 거리(Chamfer)를 mm로 표시한다.
  `Chamfer = 0.5 × (mean(pred→GT) + mean(GT→pred))`이며 제곱 거리가 아니다.
  기존 `visual/scripts/pointcloud.py`의 `backproject`와 `to_world`를 재사용한다.
- 유효 depth는 finite이고 양수인 값이다. Sampling·erosion·필터링 없이 모든
  유효 관측점을 사용한다. 양쪽 중 하나라도 점이 없으면 3D 오차는 `n/a`로
  남기고 계산 가능한 프레임 수와 실패 프레임 수를 함께 보여 준다.

Depth는 simulator의 기록된 관측값이다. 이 평가는 segmentation이 가시 표면에
만드는 차이를 측정하며, 독립적인 depth 추정 정확도나 물체 pose 오차를 평가하지
않는다. Intrinsic, camera-to-world, TCP pose도 기록된 입력으로 사용한다.

영상은 metre 단위 CV camera z를 사용하고 `--depth-min` / `--depth-max`로
지정한 색상 범위를 전체 프레임에 고정한다(기본 0.05–1.0 m). 무효 depth는
검게, 범위 밖 값은 끝 색상으로 표시한다. Color clipping은 수치 평가에는
영향을 주지 않는다. 영상에 source row·simulator step·timestamp·범례와
물체별 IoU·3D 거리·median depth를 표시한다.

기본 fps는 선택된 timestamp 간격의 median으로 정한다. 불규칙한 간격은
영상에서 일정 fps로 근사되며 각 프레임에 실제 timestamp를 표시한다.
Timestamp가 없거나 1프레임뿐이면 `--fps`를 명시한다.

`ExperimentLogger(team="visual", job_type="evaluate")`로 scalar·입력 hash와
실행 identity를 기록하고 두 산출물의 tracking sidecar를 연결한다. 위 명령은
W&B를 비활성화한 로컬 실행이다. HF publishing은 수행하지 않는다.
Simulator·SAM2 추론·CUDA 실행 없이 저장 데이터만 CPU에서 평가한다.
