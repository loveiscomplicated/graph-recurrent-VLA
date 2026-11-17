"""Compare a functional extraction with a frozen source package on saved RGB-D.

This opt-in migration check requires external capture/reference inputs. It does
not load object GT, run the simulator, or estimate held-out detection accuracy.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection import TorchCornerDetector, detect_cpu
from visual.corner_detection.backends.torch_ops import resolve_device


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*.py"))}


def load_reference(folder):
    """Load an explicitly provided trusted local Python package in isolation."""
    name = "_corner_reference"
    spec = importlib.util.spec_from_file_location(
        name, folder / "__init__.py", submodule_search_locations=[str(folder)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return (
        importlib.import_module(name + ".detector_torch"),
        importlib.import_module(name + ".detector_v4"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-source",
        type=Path,
        required=True,
        help="trusted source package containing the original detectors",
    )
    parser.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="saved capture manifest.json and identity/camera.npz files",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-numpy-reference", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    # Match the original verification's CPU math configuration.
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parents[1] / "visual/corner_detection"
    frozen = source_hashes(args.reference_source)
    extracted = source_hashes(package)
    original_torch, original_numpy = load_reference(args.reference_source)
    reference = original_torch.TorchCornerDetector(device)
    detector = TorchCornerDetector(device)
    manifest = json.loads((args.capture / "manifest.json").read_text())
    config = dict(
        device=str(device),
        torch=torch.__version__,
        numpy=np.__version__,
        opencv=cv2.__version__,
        capture_manifest_sha256=sha(args.capture / "manifest.json"),
        reference_source_hashes=frozen,
        extracted_source_hashes=extracted,
        include_numpy_reference=args.include_numpy_reference,
        scope="same-input extraction regression; no GT or simulator",
    )
    records = []
    with ExperimentLogger(
        team="visual",
        model="corner_detector",
        seed=42,
        config=config,
        root=args.output / "experiments",
        job_type="verify",
    ) as log:
        log.record(dataset_hash=config["capture_manifest_sha256"])
        for entry in manifest["entries"]:
            for camera in ("base_camera", "hand_camera"):
                path = args.capture / entry["identity"] / (camera + ".npz")
                if sha(path) != entry["files"][camera + ".npz"]:
                    raise ValueError(f"capture hash mismatch: {path}")
                with np.load(path, allow_pickle=False) as frame:
                    inputs = [
                        frame[key].copy()
                        for key in ("rgb", "depth", "intrinsic_cv", "extrinsic_cv")
                    ]
                expected = reference(*inputs)
                actual = detector(*inputs)
                for field in ("xyz", "uv", "camera_xyz", "object_id", "masks"):
                    np.testing.assert_array_equal(
                        getattr(actual, field).cpu().numpy(),
                        getattr(expected, field).cpu().numpy(),
                    )
                assert actual.metadata == expected.metadata, (
                    "candidate metadata changed"
                )
                assert actual.diagnostics == expected.diagnostics, "diagnostics changed"
                assert actual.parameters == expected.parameters, "parameters changed"
                if args.include_numpy_reference:
                    cpu = detect_cpu(*inputs)
                    old_cpu = original_numpy.detect(*inputs)
                    assert cpu["corners"] == old_cpu["corners"], (
                        "NumPy reference corners changed"
                    )
                    assert cpu["diagnostics"] == old_cpu["diagnostics"], (
                        "NumPy diagnostics changed"
                    )
                    np.testing.assert_array_equal(cpu["masks"], old_cpu["masks"])
                records.append(
                    dict(
                        identity=entry["identity"],
                        camera=camera,
                        corners=len(actual.xyz),
                        exact=True,
                    )
                )
            print(f"Verified {entry['identity']}: {len(records)} frames", flush=True)
        assert frozen == source_hashes(args.reference_source), (
            "reference source changed"
        )
        assert extracted == source_hashes(package), "extracted source changed"
        summary = dict(
            status="PASS",
            run_id=log.run_id,
            frames=len(records),
            exact_tensor_masks_metadata=True,
            **config,
            records=records,
        )
        destination = args.output / "summary.json"
        destination.write_text(json.dumps(summary, indent=2) + "\n")
        log.log(dict(frames=len(records), exact_match=1), step=0)
        log.link_output(destination, kind="metrics")
    print(f"PASS: {len(records)} frames on {device}")


if __name__ == "__main__":
    main()
