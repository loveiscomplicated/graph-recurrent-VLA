"""Detect objects with Grounding DINO, track them with SAM2, and save a movie.

Run from the repository root::

    .venv/bin/python visual/runtime/test/run_tracking.py \
        --input visual/data/step2/ep000.npz --device cuda

Grounding DINO jointly assigns peg and box boxes, including late initialization.
SAM2 propagates their masks. Outputs include an MP4, PNGs, masks, and metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = SCRIPT_DIR.parent
VISUAL_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = VISUAL_ROOT / "data/step2/ep000.npz"
sys.path.insert(0, str(RUNTIME_DIR))

from tracking import SAM2Tracker  # noqa: E402
from initialization import detect_initial  # noqa: E402


OBJECTS = {
    "peg": (1, (255, 70, 70), "red and white block"),
    "box": (2, (60, 130, 255), "cardboard box"),
}


def load_input(path: Path, camera: str, parser: argparse.ArgumentParser):
    """Return RGB frames, input kind, timestamps, and a useful default fps."""
    if path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            key = f"{camera}/rgb"
            if key not in archive:
                parser.error(f"{path} has no '{key}' array")
            frames = np.asarray(archive[key], dtype=np.uint8)
            timestamps = np.asarray(archive["timestamp"]) if "timestamp" in archive else None
        fps = 20.0
        if timestamps is not None:
            if len(timestamps) != len(frames):
                parser.error(f"timestamp length {len(timestamps)} differs from frame count {len(frames)}")
            gaps = np.diff(timestamps.astype(np.float64))
            gaps = gaps[gaps > 0]
            if len(gaps):
                fps = float(1.0 / np.median(gaps))
        return frames, "episode", timestamps, fps

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        parser.error(f"could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        parser.error(f"video contains no readable frames: {path}")
    if not np.isfinite(fps) or fps <= 0:
        fps = 20.0
    return np.stack(frames), "video", None, fps


def initial_boxes(frames: np.ndarray, args) -> dict:
    """Run the RGB-only v1 joint assignment and late initialization."""
    from grounding import GDino

    detector = GDino(device=args.device, threshold=args.threshold)
    prompts = {name: getattr(args, f"{name}_prompt") or default_prompt
               for name, (_, _, default_prompt) in OBJECTS.items()}
    return detect_initial(frames, detector, prompts)


def render_frame(rgb: np.ndarray, masks: dict[int, object], initial: dict,
                 frame_idx: int) -> np.ndarray:
    """Blend predicted masks and draw initial Grounding DINO boxes."""
    out = rgb.copy()
    for name, (object_id, color, _) in OBJECTS.items():
        mask = masks.get(object_id)
        if mask is None:
            continue
        if hasattr(mask, "detach"):
            mask = mask.detach().to("cpu").numpy()
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != rgb.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        out[mask] = (0.45 * out[mask] + 0.55 * np.asarray(color)).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, max(1, min(rgb.shape[:2]) // 256))
        ys, xs = np.nonzero(mask)
        if len(xs):
            cv2.putText(out, name, (int(xs.min()), max(18, int(ys.min()) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    for name, (object_id, color, _) in OBJECTS.items():
        item = initial[name]
        if item["box"] is None or frame_idx != item["frame"]:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in item["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.putText(out, f"GDINO {name}", (x1 + 3, max(16, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.putText(out, f"SAM2 mask + GDINO init box | frame {frame_idx}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Episode NPZ with camera RGB arrays or a video file")
    parser.add_argument("--out", type=Path, default=VISUAL_ROOT / "results/runtime_tracking")
    parser.add_argument("--camera", default="base_camera", help="Camera name for episode NPZ input")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--checkpoint", type=Path,
                        default=VISUAL_ROOT / "checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Grounding DINO threshold used to initialize boxes")
    parser.add_argument("--stride", type=int, default=1,
                        help="Track every Nth source frame and write those frames")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None, help="Output fps; defaults to source fps / stride")
    parser.add_argument("--peg-prompt", default=None)
    parser.add_argument("--box-prompt", default=None)
    parser.add_argument("--hydra-override", action="append", default=[],
                        help="SAM2 config override; may be passed more than once")
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    if not args.checkpoint.is_file():
        parser.error(f"SAM2 checkpoint does not exist: {args.checkpoint}")

    frames, input_kind, timestamps, source_fps = load_input(args.input, args.camera, parser)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        parser.error(f"expected RGB frames with shape (T,H,W,3), got {frames.shape}")
    if timestamps is not None and len(timestamps) != len(frames):
        parser.error("timestamp and RGB frame counts differ")
    selected = list(range(0, len(frames), args.stride))
    if args.max_frames is not None:
        selected = selected[:max(0, args.max_frames)]
    if not selected:
        parser.error("input contains no frames to track")

    sampled = frames[selected]
    initial = initial_boxes(sampled, args)
    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.out / "tracking.mp4" if len(selected) > 1 else None
    output_fps = args.fps or source_fps / args.stride

    tracker = SAM2Tracker(device=args.device, config=args.config,
                          checkpoint=args.checkpoint, hydra_overrides=args.hydra_override)
    records = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sam2_runtime_") as frame_dir:
        for local_idx, source_idx in enumerate(selected):
            frame_path = Path(frame_dir) / f"{local_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frames[source_idx], cv2.COLOR_RGB2BGR))

        height, width = sampled.shape[1:3]
        masks_by_name = {name: np.zeros((len(selected), height, width), dtype=bool)
                         for name in OBJECTS}
        first_group = [name for name in OBJECTS if initial[name]["frame"] == 0]
        groups = [(first_group, 0)] if first_group else []
        groups += [([name], initial[name]["frame"]) for name in OBJECTS
                   if initial[name]["frame"] not in (0, None)]
        for names, start in groups:
            boxes = {OBJECTS[name][0]: initial[name]["box"] for name in names}
            state = tracker.start(frame_dir, boxes, frame_idx=start)
            try:
                for local_idx, masks in tracker.track(state, start_frame=start):
                    for name in names:
                        mask = masks.get(OBJECTS[name][0])
                        if mask is None:
                            continue
                        if hasattr(mask, "detach"):
                            mask = mask.detach().cpu().numpy()
                        mask = np.asarray(mask, dtype=bool)
                        if mask.shape != (height, width):
                            mask = cv2.resize(mask.astype(np.uint8), (width, height),
                                              interpolation=cv2.INTER_NEAREST).astype(bool)
                        masks_by_name[name][local_idx] = mask
            finally:
                tracker.reset(state)

        np.savez_compressed(args.out / "masks.npz", **masks_by_name,
                            source_frame=np.asarray(selected, dtype=np.int32))
        writer_context = (imageio.get_writer(video_path, fps=output_fps, macro_block_size=1)
                          if video_path else nullcontext(None))
        with writer_context as writer:
            for local_idx, source_idx in enumerate(selected):
                masks = {OBJECTS[name][0]: values[local_idx] for name, values in masks_by_name.items()}
                rendered = render_frame(frames[source_idx], masks, initial, local_idx)
                cv2.imwrite(str(frames_dir / f"frame_{source_idx:05d}.png"),
                            cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
                if writer is not None:
                    writer.append_data(rendered)
                records.append({"frame": int(source_idx), "objects": {
                    name: {"object_id": object_id, "mask_pixels": int(masks_by_name[name][local_idx].sum())}
                    for name, (object_id, _, _) in OBJECTS.items()}})
                print(f"tracked frame {source_idx} ({local_idx + 1}/{len(selected)})", flush=True)

    result = {
        "input": str(args.input.resolve()),
        "input_kind": input_kind,
        "camera": args.camera if input_kind == "episode" else None,
        "device": args.device,
        "config": args.config,
        "checkpoint": str(args.checkpoint.resolve()),
        "source_fps": source_fps,
        "output_fps": output_fps,
        "stride": args.stride,
        "frames_tracked": len(records),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "initial_boxes": initial,
        "mask_file": str(args.out / "masks.npz"),
        "video": str(video_path) if video_path else None,
        "frames_dir": str(frames_dir),
        "results": records,
    }
    (args.out / "tracking.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if video_path:
        print(f"saved video: {video_path}")
    print(f"saved frames: {frames_dir}")
    print(f"saved metadata: {args.out / 'tracking.json'}")


if __name__ == "__main__":
    main()
