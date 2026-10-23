"""Temporary Grounding DINO visualization runner.

Run from the repository root::

    .venv/bin/python visual/runtime/test/run_grounding.py --device cpu

The default input is ``visual/data/step2/ep000.npz`` when it exists; otherwise
the checked-in Step 2 contact sheet supplies six representative RGB frames.
Use ``--input`` for another episode NPZ, a contact sheet, or a PNG/JPEG.
Episode NPZ files must contain
``base_camera/rgb`` with shape (T,H,W,3), uint8 RGB. Annotated PNGs, detections,
and an MP4 when multiple frames are available are written under ``--out``.
Episode videos use every recorded camera frame and derive playback fps from the
recorded timestamps when available.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = SCRIPT_DIR.parent
VISUAL_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_EPISODE = VISUAL_ROOT / "data/step2/ep000.npz"
DEFAULT_SHEET = VISUAL_ROOT / "results/step2/ep000_base_camera_frames.jpg"
sys.path.insert(0, str(RUNTIME_DIR))

from grounding import GDino  # noqa: E402


PROMPTS = {
    "peg": "red and white block",
    "box": "cardboard box",
}
COLORS = {
    "peg": (255, 70, 70),
    "box": (60, 130, 255),
}


def annotate(
    rgb: np.ndarray,
    detections: dict[str, tuple[list, list]],
    frame_idx: int,
    top_k: int = 1,
    scale: int = 3,
) -> np.ndarray:
    """Draw detections on an enlarged canvas so boxes and labels are readable."""
    height, width = rgb.shape[:2]
    enlarged = cv2.resize(rgb, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    header_height = 40
    image = cv2.copyMakeBorder(enlarged, header_height, 0, 0, 0, cv2.BORDER_CONSTANT,
                               value=(24, 24, 24))
    for name, (boxes, scores) in detections.items():
        color = COLORS[name]
        for rank, (box, score) in enumerate(zip(boxes[:top_k], scores[:top_k])):
            x1, y1, x2, y2 = (int(round(value * scale)) for value in box)
            y1 += header_height
            y2 += header_height
            thickness = max(2, scale) if rank == 0 else 1
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            label = f"{name} {score:.2f}" if rank == 0 else f"{name} #{rank + 1} {score:.2f}"
            text_y = max(header_height + 20, y1 - 6)
            cv2.putText(image, label, (x1 + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, label, (x1 + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 1, cv2.LINE_AA)

    title = f"sample {frame_idx} | Grounding DINO"
    cv2.putText(image, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def load_rgb(path: Path, camera: str, parser: argparse.ArgumentParser) -> tuple[np.ndarray, str, np.ndarray | None]:
    """Load an episode, a Step 2 contact sheet, or one RGB image."""
    if path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            rgb_key = f"{camera}/rgb"
            if rgb_key not in archive:
                parser.error(f"{path} has no '{rgb_key}' array")
            timestamps = archive["timestamp"] if "timestamp" in archive else None
            return archive[rgb_key], "episode", timestamps

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        parser.error(f"could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if path.name.endswith(f"_{camera}_frames.jpg"):
        height, width = rgb.shape[:2]
        if width % 3 or height % (width // 3):
            parser.error(f"unexpected Step 2 contact sheet dimensions: {rgb.shape}")
        tile = width // 3
        frames = np.stack([rgb[y:y + tile, :tile] for y in range(0, height, tile)])
        return frames, "contact_sheet", None
    return rgb[None], "image", None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_EPISODE if DEFAULT_EPISODE.is_file() else DEFAULT_SHEET,
        help="Episode NPZ, Step 2 contact sheet, or RGB image (uses ep000 NPZ when present)",
    )
    parser.add_argument("--out", type=Path, default=VISUAL_ROOT / "results/runtime_grounding")
    parser.add_argument("--camera", default="base_camera")
    parser.add_argument("--stride", type=int, default=1,
                        help="Run inference on every Nth frame (default: every frame)")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on sampled frames")
    parser.add_argument("--fps", type=float, default=None,
                        help="Output video fps (default: timestamps for episodes, 2 for sheets)")
    parser.add_argument("--display-scale", type=int, default=None,
                        help="Visualization scale (default: 1 at 512px or larger, otherwise 3)")
    parser.add_argument("--top-k", type=int, default=1, help="Draw this many boxes per prompt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--peg-prompt", default=PROMPTS["peg"])
    parser.add_argument("--box-prompt", default=PROMPTS["box"])
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if args.display_scale is not None and args.display_scale < 1:
        parser.error("--display-scale must be at least 1")
    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rgb, input_kind, timestamps = load_rgb(args.input, args.camera, parser)

    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        parser.error(f"expected {args.camera}/rgb with shape (T,H,W,3), got {rgb.shape}")

    stride = args.stride
    frame_indices = list(range(0, len(rgb), stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[:max(0, args.max_frames)]
    if not frame_indices:
        parser.error("input episode contains no frames to process")
    if timestamps is not None and len(timestamps) != len(rgb):
        parser.error(f"timestamp length {len(timestamps)} differs from RGB frame count {len(rgb)}")
    if args.fps is not None:
        fps = args.fps
    elif timestamps is not None and len(frame_indices) > 1:
        gaps = np.diff(np.asarray(timestamps)[frame_indices])
        positive_gaps = gaps[gaps > 0]
        fps = round(float(1 / np.median(positive_gaps)), 3) if len(positive_gaps) else 10.0
    else:
        fps = 2.0 if input_kind == "contact_sheet" else 10.0
    display_scale = args.display_scale or (1 if min(rgb.shape[1:3]) >= 512 else 3)

    prompts = {"peg": args.peg_prompt, "box": args.box_prompt}
    detector = GDino(device=args.device, threshold=args.threshold)
    records = []
    video_path = args.out / "grounding.mp4" if len(frame_indices) > 1 else None
    started = time.perf_counter()

    video_context = imageio.get_writer(video_path, fps=fps, macro_block_size=1) if video_path else nullcontext()
    with video_context as writer:
        for frame_idx in frame_indices:
            image = np.asarray(rgb[frame_idx], dtype=np.uint8)
            detections = {}
            record = {"frame": int(frame_idx), "detections": {}}
            for name, prompt in prompts.items():
                boxes, scores = detector(image, prompt)
                detections[name] = (boxes, scores)
                record["detections"][name] = {
                    "prompt": prompt,
                    "boxes": boxes,
                    "scores": scores,
                    "timing_ms": dict(detector.last_ms),
                }

            rendered = annotate(image, detections, frame_idx,
                                top_k=max(1, args.top_k), scale=display_scale)
            frame_name = f"frame_{frame_idx:05d}.png"
            cv2.imwrite(str(frames_dir / frame_name), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
            if writer is not None:
                writer.append_data(rendered)
            records.append(record)
            print(f"frame {frame_idx}: " + ", ".join(
                f"{name}={len(detections[name][0])}" for name in prompts
            ), flush=True)

    result = {
        "input": str(args.input.resolve()),
        "camera": args.camera,
        "input_kind": input_kind,
        "device": args.device,
        "threshold": args.threshold,
        "stride": stride,
        "fps": fps,
        "display_scale": display_scale,
        "frames_processed": len(frame_indices),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "video": str(video_path) if video_path else None,
        "frames_dir": str(frames_dir),
        "results": records,
    }
    (args.out / "detections.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if video_path:
        print(f"saved video: {video_path}")
    print(f"saved images: {frames_dir}")


if __name__ == "__main__":
    main()
