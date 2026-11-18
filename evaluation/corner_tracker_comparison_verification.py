"""Independently rescore saved common-test XYZ and causal masks in float64."""

import hashlib
import json
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import file_sha256
from utils.experiment_logger import write


def verify(config):
    """Rebuild per-episode eligibility and recompute every reported scalar."""
    root = Path(config["results_root"])
    result = json.loads((root / "results.json").read_text())
    path = root / "predictions.npz"
    if file_sha256(path) != result["predictions_sha256"]:
        raise ValueError("Saved comparison prediction checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    frame_count = len(arrays["sample_ids"])
    if frame_count != result["frames"] or len(set(arrays["sample_ids"].tolist())) != frame_count:
        raise ValueError("Frame IDs must be unique and cover the complete evaluated inventory")
    observed = arrays["observed"]
    reconstructed = np.zeros_like(observed)
    for start, stop in zip(arrays["episode_offsets"][:-1], arrays["episode_offsets"][1:]):
        seen = np.zeros(20, dtype=bool)
        for frame in range(start, stop):
            reconstructed[frame] = seen & ~observed[frame]
            seen |= observed[frame]
    np.testing.assert_array_equal(reconstructed, arrays["eligible"])
    mask_hash = hashlib.sha256(reconstructed.tobytes()).hexdigest()
    if mask_hash != result["mask_sha256"]:
        raise ValueError("Saved comparison mask checksum mismatch")
    initialized = reconstructed & arrays["surface_initialized"] & arrays["online_available"]
    masks = {
        "common_reference": reconstructed,
        "both_initialized": initialized,
        "both_native_missing": initialized & ~arrays["surface_observed"],
    }
    checked = 0
    for cohort, mask in masks.items():
        for model in ("surface", "online"):
            delta = arrays[f"{model}_prediction"].astype(np.float64) - arrays["gt"].astype(np.float64)
            distances = np.sqrt(np.sum(delta * delta, axis=2)) * 1000
            for body, first, last in (("peg", 0, 8), ("box", 8, 16), ("hole", 16, 20), ("all", 0, 20)):
                eligible = mask[:, first:last]
                values = distances[:, first:last][eligible]
                finite = values[np.isfinite(values)]
                expected = result["metrics"][cohort][model][body]
                assert expected["eligible_count"] == len(values)
                assert expected["scored_count"] == len(finite)
                assert expected["missing_prediction_count"] == len(values) - len(finite)
                if len(values):
                    assert expected["coverage"] == len(finite) / len(values)
                if len(finite):
                    actual = {"mean_mm": np.sum(finite) / len(finite),
                              "rmse_mm": np.sqrt(np.dot(finite, finite) / len(finite)),
                              "p90_mm": np.quantile(finite, .90), "p95_mm": np.quantile(finite, .95)}
                    for key, value in actual.items():
                        np.testing.assert_allclose(expected[key], value, rtol=0, atol=1e-10)
                        checked += 1
    assert result["common_eligible_point_frames"] == int(reconstructed.sum())
    assert result["metrics"]["surface_uninitialized_on_reference"] == int(
        (reconstructed & ~arrays["surface_initialized"]).sum()
    )
    assert result["metrics"]["surface_observed_on_reference_missing"] == int(
        (reconstructed & arrays["surface_observed"]).sum()
    )
    verification = {
        "status": "passed", "frames": frame_count, "episodes": len(arrays["episode_offsets"]) - 1,
        "eligible_point_frames": int(reconstructed.sum()), "mask_sha256": mask_hash,
        "prediction_sha256": file_sha256(path), "float64_metric_checks": checked,
        "unique_frame_ids": True, "causal_masks_reconstructed": True,
        "weights_unchanged": result["weights_unchanged"], "test_used_for_selection": False,
    }
    write(root / "independent_verification.json", verification)
    print(json.dumps(verification, indent=2), flush=True)
    return verification
