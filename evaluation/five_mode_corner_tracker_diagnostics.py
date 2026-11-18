"""Independently rescore frozen predictions and diagnose large Hole errors."""

import json
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import file_sha256
from evaluation.five_mode_corner_tracker_comparison import summarize
from utils.experiment_logger import ExperimentLogger, write


def analyze(config):
    """Quantify first-observation failures without changing models or test inputs."""
    root = Path(config["comparison"])
    result = json.loads((root / "results.json").read_text())
    predictions = root / "predictions.npz"
    if file_sha256(predictions) != result["predictions_sha256"]:
        raise ValueError("Prediction artifact checksum mismatch")
    with np.load(predictions, allow_pickle=False) as source:
        data = dict(source)
    missing = ~data["observed"][:, 16:20]
    unseen = ~data["ever"][:, 16:20]
    diagnostics = {"models": {}, "independent_metric_recomputation": True}
    for name, model in result["models"].items():
        prediction = data[f"prediction_{name}"]
        error = np.linalg.norm(prediction - data["gt"], axis=-1) * 1000
        np.testing.assert_allclose(error, data[f"error_{name}_mm"], rtol=0, atol=1e-5)
        for body, slots in (
            ("peg", slice(8)),
            ("box", slice(8, 16)),
            ("hole", slice(16, 20)),
        ):
            values = error[:, slots][~data["observed"][:, slots]]
            np.testing.assert_allclose(
                np.mean(values), model["body"][body]["mean_mm"], rtol=0, atol=1e-5
            )
            np.testing.assert_allclose(
                np.percentile(values, 90),
                model["body"][body]["p90_mm"],
                rtol=0,
                atol=1e-5,
            )
        hole_error = error[:, 16:20]
        large = missing & (hole_error > 20)
        all_unseen = unseen.all(axis=1)
        residual = prediction[:, 16:20] - data["gt"][:, 16:20]
        center = residual.mean(axis=1)
        box_x = data["gt"][:, 12] - data["gt"][:, 8]
        box_x /= np.linalg.norm(box_x, axis=-1, keepdims=True)
        normal_error = np.abs((center * box_x).sum(axis=-1))
        in_plane_error = np.sqrt(
            np.maximum((center**2).sum(axis=-1) - normal_error**2, 0)
        )
        total_squared = (residual[all_unseen] ** 2).sum(dtype=np.float64)
        center_squared = 4 * (center[all_unseen] ** 2).sum(dtype=np.float64)
        diagnostics["models"][name] = {
            "hole_missing_point_frames": int(missing.sum()),
            "hole_never_seen_yet_point_frames": int((missing & unseen).sum()),
            "hole_over_20mm_point_frames": int(large.sum()),
            "hole_over_20mm_before_first_observation": int((large & unseen).sum()),
            "hole_absolute_error_sum_from_unseen_fraction": float(
                hole_error[missing & unseen].sum(dtype=np.float64)
                / hole_error[missing].sum(dtype=np.float64)
            ),
            "all_unseen_hole_frames": int(all_unseen.sum()),
            "all_unseen_hole_center_squared_error_fraction": float(
                center_squared / total_squared
            ),
            "all_unseen_hole_center_normal_error": summarize(
                normal_error[all_unseen] * 1000
            ),
            "all_unseen_hole_center_in_plane_error": summarize(
                in_plane_error[all_unseen] * 1000
            ),
        }
    distances = np.linalg.norm(
        data["detected"][:, 16:20, None] - data["gt"][:, None, 16:20], axis=-1
    )
    nearest = distances.argmin(axis=-1)
    wrong_id = data["observed"][:, 16:20] & (nearest != np.arange(4)[None])
    diagnostics["hole_assigned_nearest_gt_id_mismatches"] = int(wrong_id.sum())
    uninitialized = ~data["hole_initialized"]
    diagnostics["hole_uninitialized"] = {
        "frames": int(uninitialized.sum()),
        "box_also_uninitialized_frames": int(
            (uninitialized & ~data["box_initialized"]).sum()
        ),
        "raw_candidate_support_histograms": {},
    }
    for threshold in (5, 10, 15):
        count = (data["raw_nearest_mm"][uninitialized, 16:20] <= threshold).sum(axis=1)
        values, counts = np.unique(count, return_counts=True)
        diagnostics["hole_uninitialized"]["raw_candidate_support_histograms"][
            f"{threshold}mm"
        ] = {
            str(int(value)): int(number)
            for value, number in zip(values, counts, strict=True)
        }
    manifest = json.loads((Path(config["data_root"]) / "manifest.json").read_text())
    camera = np.asarray(config["camera_world_position_m"], dtype=float)
    angles, never, missing_seeds = [], [], set()
    for record in manifest["records"]:
        frames = data["episode_index"] == len(angles)
        hole = data["gt"][frames][0, 16:20]
        normal = -np.cross(hole[2] - hole[0], hole[1] - hole[0])
        normal /= np.linalg.norm(normal)
        view = camera - hole.mean(axis=0)
        view /= np.linalg.norm(view)
        angle = float(np.degrees(np.arccos(np.clip(normal @ view, -1, 1))))
        absent = not data["observed"][frames, 16:20].any()
        angles.append(angle)
        never.append(absent)
        if absent:
            missing_seeds.add(record["seed"])
    angles, never = np.asarray(angles), np.asarray(never)
    diagnostics["initial_hole_view_angle_degrees"] = {
        "definition": "0 degrees is face-on; 90 degrees is edge-on. Protocol-pinned fixed camera, GT plane; score-only.",
        "camera_world_position_m": camera.tolist(),
        "zero_hole_observation_episodes": summarize(angles[never]),
        "at_least_one_hole_observation_episodes": summarize(angles[~never]),
        "zero_hole_observation_distinct_seeds": len(missing_seeds),
    }
    # The summarizer's numeric fields are converted to unit-appropriate keys.
    for name in (
        "zero_hole_observation_episodes",
        "at_least_one_hole_observation_episodes",
    ):
        row = diagnostics["initial_hole_view_angle_degrees"][name]
        diagnostics["initial_hole_view_angle_degrees"][name] = {
            key.replace("_mm", "_degrees"): value for key, value in row.items()
        }
    with ExperimentLogger(
        team="graph",
        model="five_mode_hole_error_diagnostics",
        job_type="evaluate",
        config=config,
        root=root / "diagnostic_experiments",
    ) as logger:
        logger.record(
            source_results_sha256=file_sha256(root / "results.json"),
            predictions_sha256=result["predictions_sha256"],
            script_sha256=file_sha256(__file__),
        )
        diagnostics["run_id"] = logger.run_id
        write(root / "diagnostics.json", diagnostics)
        logger.log(diagnostics)
        logger.link_output(root / "diagnostics.json")
        print(json.dumps(diagnostics, indent=2), flush=True)
