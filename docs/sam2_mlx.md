# SAM2 on Apple Silicon

`MLXObjectStream` tracks peg and box from consecutive RGB frames using
SAM2.1 Small on Metal. Supply both initial XYXY boxes on frame zero. Each
frame shares one image encoding between the objects. The tracker uses a
512-pixel internal image, three spatial memories and FP16 for the Hiera
trunk and memory attention. It returns completed MLX boolean masks of
shape `[2, H, W]`, in peg and box order, at the input camera resolution.
Tracking is causal and does not need future frames. The predictor retains
tracking history until reset, so memory is not bounded for long sessions.

```python
from visual.sam2_mlx.object_stream import MLXObjectStream

tracker = MLXObjectStream("visual/checkpoints/sam2.1_hiera_small_mlx.safetensors")
masks = tracker.start(first_rgb, {1: peg_box, 2: fixture_box})
masks = tracker.track(next_rgb, timestep=1)
tracker.reset()
```

Pass uint8 RGB in HWC order. The caller provides initial detections and
owns its policy for visibility, camera capture, and downstream control.

## Setup and replay

On Apple Silicon with Python 3.11 and `uv`, run
`bash visual/sam2_mlx/setup_macos.sh` from a checkout without a root
`.venv`. The script checks out pinned SAM2 and sam2-mlx sources. The
pinned sam2-mlx project declares Python 3.14 or newer; this repository's
Python 3.11 setup uses the imported source and was exercised on M4 Pro.

Place the official SAM2.1 Small weights in
`visual/checkpoints/sam2.1_hiera_small.pt`, then convert them locally:

```bash
.venv/bin/python -m mlx_sam.convert \
  --checkpoint visual/checkpoints/sam2.1_hiera_small.pt \
  --model-id facebook/sam2.1-hiera-small \
  --output visual/checkpoints/sam2.1_hiera_small_mlx.safetensors
shasum -a 256 visual/checkpoints/sam2.1_hiera_small*
```

See [provenance](../visual/sam2_mlx/provenance.json) for the checkpoint
hashes and the MLX source revision. Supply a local NPZ with `rgb`
(uint8 `[T,H,W,3]`) and `boxes` (two initial XYXY boxes, peg then box).
Set its path in the [replay config](../evaluation/configs/sam2_mlx_replay.yaml).

```bash
WANDB_MODE=disabled .venv/bin/python main.py --config evaluation/configs/sam2_mlx_replay.yaml
```

The replay records input hashes, per-frame latency and warmed percentiles
through ExperimentLogger. The timer includes JPEG preprocessing, tracking
and completed GPU masks. It ends before any host mask readback.

On the M4 Pro, two passes of a 31-frame PegInsertionSide base-camera
recording gave medians of 41.15 and 41.39 ms after dropping the first
16 frames (local run `083bbcff3fde4f8b805667b2ec7c834c`). The clip has no
per-frame ground-truth masks. This measures one camera with fixed initial
boxes and no appearance guard. Camera capture, Graph, robot control and
other devices were outside this replay.

In earlier same-process experiments, adding the appearance rule with
CPU-based rejection took 43.74–44.10 ms on the two-object clip. The
experimental GPU rule and masked memory took 44.70–44.91 ms. The latter
saved about 1.1 ms on the rule and memory handling but added about 0.5 ms
in measured memory attention. Component timings used a separate
synchronized replay. Both variants suppressed all ten frames of a
synthetic black occlusion and recovered at the first visible frame
(GT mask IoU 0.987). Neither guard variant is part of this tracker.
