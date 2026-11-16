"""Score saved detector corners against raw observations and supported GT."""

from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
from time import sleep

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from utils.experiment_logger import ExperimentLogger
from visual.corner_detection.evaluation import score, visible_reference


THRESHOLDS_M = {"3mm": 0.003, "5mm": 0.005}
OBJECTS = ("peg", "box")
FAMILIES = ("peg", "box_outer", "hole_rim")
GT_FIELDS = ("peg_pose", "box_pose", "hole_pose", "peg_half_sizes", "hole_radius")
SENSOR_FIELDS = {
    "depth": "observation/sensor_data/base_camera/depth",
    "segmentation": "observation/sensor_data/base_camera/segmentation",
    "intrinsic_cv": "observation/sensor_param/base_camera/intrinsic_cv",
    "extrinsic_cv": "observation/sensor_param/base_camera/extrinsic_cv",
}


def wait_for_file(path, producer_exit):
    """Read immutable published files as the CUDA producer finishes each episode."""
    while not path.is_file():
        if producer_exit is None or producer_exit.exists():
            raise FileNotFoundError(f"Detector cache producer did not publish: {path}")
        sleep(2)


def score_episode(raw_root, row, cache_path, actor_ids):
    """Return additive one-to-one match counts and the exact scored cache hash."""
    contents = cache_path.read_bytes()
    with np.load(io.BytesIO(contents), allow_pickle=False) as archive:
        indices, offsets = archive["frame_indices"], archive["corner_offsets"]
        world, kinds = archive["corner_world_xyz"], archive["corner_object_id"]
    count = row["observations"]
    if (not np.array_equal(indices, np.arange(count)) or offsets.shape != (count + 1,)
            or offsets[0] != 0 or offsets[-1] != len(kinds) or np.any(np.diff(offsets) < 0)
            or world.shape != (len(kinds), 3)):
        raise ValueError(f"Invalid cache frame alignment: {cache_path}")
    counts = Counter(frames=count)
    for index, expected_hash in enumerate(row["frame_sha256"]):
        frame = raw_root / row["path"] / f"frame_{index:06d}.npz"
        raw = frame.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"Raw frame SHA-256 mismatch: {frame}")
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            sensor = {name: data[key][0] for name, key in SENSOR_FIELDS.items()}
            gt = {name: data[f"ground_truth/{name}"][0] for name in GT_FIELDS}
        supported = visible_reference(gt, sensor, actor_ids)
        start, stop = offsets[index:index + 2]
        predictions = [dict(object_id=int(kind), xyz=point)
                       for kind, point in zip(kinds[start:stop], world[start:stop], strict=True)]
        for label, threshold in THRESHOLDS_M.items():
            matches = score(predictions, supported, [], threshold)
            for name, result in zip(OBJECTS, matches, strict=True):
                counts[label, name, "predicted"] += result["predictions"]
                counts[label, name, "supported"] += result["visible_reference"]
                counts[label, name, "matched"] += result["matched_3mm"]
                for family, metrics in result["by_kind"].items():
                    counts[label, family, "family_supported"] += metrics["visible"]
                    counts[label, family, "family_matched"] += metrics["matched"]
    return counts, hashlib.sha256(contents).hexdigest()


def summarize(counts):
    """Compute micro-averaged precision and recall from frame-level counts."""
    result = {"frames": counts["frames"]}
    for label in THRESHOLDS_M:
        objects, families = {}, {}
        for name in OBJECTS:
            predicted = counts[label, name, "predicted"]
            supported = counts[label, name, "supported"]
            matched = counts[label, name, "matched"]
            objects[name] = dict(predictions=predicted, supported=supported, matched=matched,
                                 precision=matched / predicted if predicted else None,
                                 recall=matched / supported if supported else None)
        for name in FAMILIES:
            supported = counts[label, name, "family_supported"]
            matched = counts[label, name, "family_matched"]
            families[name] = dict(supported=supported, matched=matched,
                                  recall=matched / supported if supported else None)
        result[label] = dict(objects=objects, families=families)
    return result


def evaluate(config):
    """Score ready episode archives and seal the report against the cache manifest."""
    source_path = Path(config["source_manifest"])
    source_hash = file_sha256(source_path)
    actor_path = Path(config["actor_ids"])
    actors = json.loads(actor_path.read_text())
    if actors["source_manifest_sha256"] != source_hash:
        raise ValueError("Actor IDs and raw episode manifest must identify the same collection")
    source = json.loads(source_path.read_text())
    episodes = source["episodes"]
    requested = config.get("episodes")
    if requested is not None:
        episodes = [row for row in episodes if row["path"] in requested]
        if {row["path"] for row in episodes} != set(requested):
            raise ValueError("episodes must be paths from the raw manifest")
    cache = Path(config["cache"]).resolve()
    producer_exit = Path(config["producer_exit"]) if "producer_exit" in config else None
    totals, by_mode, records = Counter(), defaultdict(Counter), []
    with ExperimentLogger(team="visual", model="corner_detection_cache_accuracy", config=config,
                          job_type="evaluate") as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True)
        print(f"Output directory: {output}", flush=True)
        for index, row in enumerate(episodes, start=1):
            path = cache / f"{row['path']}.npz"
            wait_for_file(path, producer_exit)
            ids = {int(key): value for key, value in actors["actor_ids"][str(row["seed"])].items()}
            counts, digest = score_episode(source_path.parent, row, path, ids)
            totals.update(counts)
            by_mode[row["mode"]].update(counts)
            records.append(dict(source=row["path"], seed=row["seed"], mode=row["mode"],
                                sha256=digest, **summarize(counts)))
            logger.log({"frames": totals["frames"]}, step=index)
            if index == 1 or index % 10 == 0 or index == len(episodes):
                print(f"Scored {index}/{len(episodes)} episodes ({totals['frames']} frames)", flush=True)

        manifest_path = cache / "manifest.json"
        wait_for_file(manifest_path, producer_exit)
        manifest = json.loads(manifest_path.read_text())
        scored = {row["source"]: row["sha256"] for row in records}
        indexed = {row["source"]: row["sha256"] for row in manifest["episodes"]}
        if (manifest["status"] != "COMPLETE" or manifest["source_manifest_sha256"] != source_hash
                or scored != indexed):
            raise ValueError("Scored episodes must match the final CUDA cache manifest exactly")
        provenance = dict(source_manifest_sha256=source_hash, actor_ids_sha256=file_sha256(actor_path),
                          cache_sha256=file_sha256(manifest_path), cache_run_id=manifest["run_id"])
        logger.record(**provenance)
        results = output / "results.json"
        write_json(results, dict(status="COMPLETE", **provenance, all=summarize(totals), records=records,
                                 by_mode={mode: summarize(counts) for mode, counts in by_mode.items()}))
        logger.link_output(results, kind="metrics")
        print(f"Score report: {results}", flush=True)
