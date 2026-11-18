"""Prepare the untouched Five-mode test split and audit detector-only association."""

import json
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import monotonic

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.five_mode_corner_tracking import prepare_batch, read_member
from graph.corner_association import CornerMatcher
from utils.experiment_logger import write


def audit_association(corners, episode):
    """Use GT only to score raw candidates and the unchanged causal matcher."""
    length = len(episode["gt"])
    audit = {
        "raw_nearest_mm": np.full((length, 20), np.inf, dtype=np.float32),
        "hint_nearest_mm": np.full((length, 4), np.inf, dtype=np.float32),
        "hole_initialized": np.zeros(length, dtype=bool),
        "box_initialized": np.zeros(length, dtype=bool),
        "hole_fit_center_error_mm": np.full(length, np.nan, dtype=np.float32),
        "hole_fit_radius_error_mm": np.full(length, np.nan, dtype=np.float32),
        "hole_radius_mm": np.zeros(length, dtype=np.float32),
    }
    matcher = CornerMatcher()
    for index, (start, stop) in enumerate(
        zip(corners["corner_offsets"][:-1], corners["corner_offsets"][1:], strict=True)
    ):
        points = corners["corner_world_xyz"][start:stop]
        objects = corners["corner_object_id"][start:stop]
        hints = corners["corner_contour_hole"][start:stop]
        hints = hints | (corners["corner_method"][start:stop] == "box_opening")
        family = np.where(objects == 0, 0, np.where(hints, 2, 1))
        assignment = matcher.step(points, family, episode["tcp"][index])
        observed = assignment >= 0
        np.testing.assert_array_equal(observed, episode["observed"][index])
        np.testing.assert_array_equal(
            points[assignment[observed]], episode["detected"][index, observed]
        )
        truth = episode["gt"][index]
        for slots, candidates in (
            (slice(0, 8), points[objects == 0]),
            (slice(8, 20), points[objects != 0]),
        ):
            if len(candidates):
                distances = np.linalg.norm(
                    truth[slots, None] - candidates[None], axis=-1
                )
                audit["raw_nearest_mm"][index, slots] = distances.min(axis=1) * 1000
        candidates = points[(objects != 0) & hints]
        if len(candidates):
            distances = np.linalg.norm(truth[16:20, None] - candidates[None], axis=-1)
            audit["hint_nearest_mm"][index] = distances.min(axis=1) * 1000
        audit["box_initialized"][index] = matcher.box.pose is not None
        audit["hole_initialized"][index] = matcher.hole.shape is not None
        radius = np.linalg.norm(truth[17] - truth[16]) / 2
        audit["hole_radius_mm"][index] = radius * 1000
        if matcher.hole.shape is not None:
            box_center = truth[8:16].mean(axis=0)
            y_axis = truth[10] - truth[8]
            z_axis = truth[9] - truth[8]
            y_axis /= np.linalg.norm(y_axis)
            z_axis /= np.linalg.norm(z_axis)
            offset = truth[16:20].mean(axis=0) - box_center
            center = np.array([offset @ y_axis, offset @ z_axis])
            fitted_center, fitted_radius = matcher.hole.shape
            audit["hole_fit_center_error_mm"][index] = (
                np.linalg.norm(fitted_center - center) * 1000
            )
            audit["hole_fit_radius_error_mm"][index] = (
                abs(fitted_radius - radius) * 1000
            )
    return audit


def prepare_test_batch(job):
    """Reuse frozen preparation and add score-only diagnostics to separate test files."""
    summaries = prepare_batch(job)
    release, output = Path(job["release"]), Path(job["output"])
    records = {(row["seed"], row["policy_mode"]): row for row in job["records"]}
    for summary in summaries:
        path = output / summary["path"]
        with np.load(path, allow_pickle=False) as source:
            episode = dict(source)
        record = records[summary["seed"], summary["mode"]]
        receipt = record["files"]["corners"]
        with tarfile.open(release / receipt["archive"]) as source:
            corners = read_member(source, receipt)
        audit = audit_association(corners, episode)
        np.savez_compressed(path, **episode, **audit)
        summary["sha256"] = file_sha256(path)
    return summaries


def prepare(config):
    """Read exactly the pre-existing 450 test episodes, without changing any split."""
    release = Path(config["release"])
    manifest_path = release / "manifest.json"
    if file_sha256(manifest_path) != config["manifest_sha256"]:
        raise ValueError("Source release manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())
    records = [row for row in manifest["records"] if row["split"] == "test"]
    test_seeds = {row["seed"] for row in records}
    development_seeds = {
        row["seed"] for row in manifest["records"] if row["split"] != "test"
    }
    if len(records) != 450 or len(test_seeds) != 90 or test_seeds & development_seeds:
        raise ValueError("Expected 450 held-out episodes from 90 disjoint reset seeds")
    output = Path(config["output"])
    (output / "test").mkdir(parents=True, exist_ok=True)
    batches = {}
    for record in records:
        batches.setdefault(record["batch"], []).append(record)
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("Install zstd before preparing original raw archives")
    started = monotonic()
    summaries = []
    with ProcessPoolExecutor(max_workers=config.get("workers", 8)) as pool:
        futures = [
            pool.submit(
                prepare_test_batch,
                {
                    "release": str(release),
                    "raw_root": config["raw_root"],
                    "output": str(output),
                    "records": rows,
                    "zstd": zstd,
                },
            )
            for rows in batches.values()
        ]
        for index, future in enumerate(as_completed(futures), 1):
            summaries.extend(future.result())
            print(
                f"Prepared test batch {index}/{len(futures)}, {monotonic() - started:.1f}s",
                flush=True,
            )
    summaries.sort(key=lambda row: (row["seed"], row["mode"]))
    write(
        output / "manifest.json",
        {
            "format": "five_mode_corner_tracker_test_v1",
            "episodes": len(summaries),
            "frames": sum(row["frames"] for row in summaries),
            "seeds": len(test_seeds),
            "source_manifest_sha256": config["manifest_sha256"],
            "preparation_config": config,
            "records": summaries,
            "gt_used_for_association": False,
        },
    )
