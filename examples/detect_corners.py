"""Detect corners in a single saved native-512 RGB-D frame."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from visual.corner_detection import detect_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ: rgb, depth, intrinsic_cv, extrinsic_cv; optional masks",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new NPZ file for tensor results"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda[:index], mps")
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as frame:
        result = detect_tensors(
            frame["rgb"],
            frame["depth"],
            frame["intrinsic_cv"],
            frame["extrinsic_cv"],
            masks=frame["masks"] if "masks" in frame else None,
            device=args.device,
        )
    arrays = {
        name: getattr(result, name).cpu().numpy()
        for name in ("object_id", "uv", "xyz", "camera_xyz", "masks")
    }
    arrays["metadata"] = np.array(json.dumps(result.metadata))
    arrays["parameters"] = np.array(json.dumps(result.parameters))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(f"Saved {len(result.xyz)} corners from {result.xyz.device} to {args.output}")


if __name__ == "__main__":
    main()
