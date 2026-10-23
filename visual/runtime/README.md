### Continuous Grounding DINO camera video

The runtime visualization runner uses every RGB frame from an episode NPZ. It
derives playback fps from the recorded `timestamp` array. The checked-in Step 2
contact sheet contains only six representative frames; collect an episode for
continuous motion. From the repository root, the following macOS CPU path was
run with the root `.venv` and Homebrew MoltenVK:

```bash
VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json \
  .venv/bin/python visual/scripts/collect.py \
  --seeds 0 --out visual/data/step2 --res 512 --cams base_camera \
  --sim-backend physx_cpu --render-backend cpu

.venv/bin/python visual/runtime/test/run_grounding.py \
  --input visual/data/step2/ep000.npz \
  --out visual/results/runtime_grounding/ep000_full --device cpu
```

The observed `ep000` had 121 consecutive 512×512 camera frames at 20 Hz. The
runner writes annotated PNGs, `detections.json`, and a 121-frame MP4 at 20 fps.
Use `--camera hand_camera` only with an NPZ collected with that camera. Grounding
DINO runs independently on each frame, so its boxes can move between frames.

To initialize SAM2 from Grounding DINO and save object masks, run:

```bash
.venv/bin/python visual/runtime/test/run_tracking.py \
  --input visual/data/step2/ep000.npz \
  --out visual/results/runtime_tracking/ep000 --device cuda
```

The runtime runner applies the Visual v1 RGB-only initialization rules: reject
boxes covering at least one quarter of the frame, jointly assign Peg and Box
without overlapping boxes, and allow an object to remain undetected. It scans
every five processed frames for a missing object and initializes it after two
consistent detections. SAM2 then propagates the assigned object ID. No GT mask,
pose, or simulator segmentation is read by this path. With `--stride`, scan
intervals refer to processed frames. Re-anchoring after tracking loss is not
part of this runner.

This runner needs the SAM2.1 Small checkpoint at
`visual/checkpoints/sam2.1_hiera_small.pt`; pass `--checkpoint` to use another
location. It writes the video, annotated frames, `tracking.json`, and
`masks.npz` under `--out`. The NPZ contains boolean `peg` and `box` arrays
with shape `(frames, height, width)` plus `source_frame` indices. Frames before
initialization and objects never detected have empty masks.

To attach the original RGB-D, intrinsic, CV camera-to-world transform, and TCP
pose to each mask frame, use Visual runtime's
[`iter_saved_rgbd` or `attach_rgbd`](rgbd.md) in `rgbd.py`.
The Visual output aligns observations through `source_frame` and explicitly
converts tracker IDs `1/2` to object membership IDs `PEG=0` / `BOX=1` for Graph.

To evaluate those masks against GT and generate a depth visualization video:

```bash
WANDB_MODE=disabled .venv/bin/python visual/runtime/test/run_rgbd.py \
  --input visual/data/step2/ep000.npz \
  --masks visual/results/runtime_tracking/ep000/masks.npz \
  --camera base_camera --out visual/results/runtime_rgbd
```

Each run writes `gt_report.html` and `depth.mp4` under `--out/<run_id>/`.
The report includes mask IoU/precision/recall, visible-surface distances in mm,
per-frame values, and explicit missing-observation counts. The video shows RGB,
depth overlay, object depth, a fixed metre scale, and per-frame error metrics.
See [metric definitions and limitations](rgbd.md#gt-비교-리포트와-depth-영상).
