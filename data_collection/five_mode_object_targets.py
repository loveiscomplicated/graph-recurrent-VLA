"""Export checksum-verified Object21 labels for explicitly selected release splits."""

from contextlib import ExitStack
import json
from pathlib import Path
import tarfile

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.five_mode_corner_tracking import read_member
from utils.experiment_logger import write


def prepare(config):
    """Preserve release identities and per-frame TCP for later alignment checks."""
    release = Path(config["release"])
    manifest_path = release / "manifest.json"
    if file_sha256(manifest_path) != config["manifest_sha256"]:
        raise ValueError("Object21 source manifest checksum differs")
    manifest = json.loads(manifest_path.read_text())
    splits = tuple(config.get("splits", ("train", "validation")))
    if splits not in (("train", "validation"), ("test",)):
        raise ValueError("Object21 preparation requires train/validation or test only")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with ExitStack() as stack:
        archives = {}
        for record in manifest["records"]:
            if record["split"] not in splits:
                continue
            receipt = record["files"]["targets"]
            name = receipt["archive"]
            if name not in archives:
                archives[name] = stack.enter_context(tarfile.open(release / name))
            arrays = read_member(archives[name], receipt)
            length = record["frames"]
            if (not np.array_equal(arrays["frame_indices"], np.arange(length))
                    or arrays["object21"].shape != (length, 21)
                    or arrays["tcp"].shape != (length, 3)
                    or not np.isfinite(arrays["object21"]).all()
                    or not np.isfinite(arrays["tcp"]).all()):
                raise ValueError(f"Invalid Object21 frame alignment: {receipt['member']}")
            relative = Path(record["split"]) / f"{record['seed']}_{record['policy_mode']}.npz"
            destination = output / relative
            destination.parent.mkdir(exist_ok=True)
            np.savez_compressed(destination, object21=arrays["object21"], tcp=arrays["tcp"])
            records.append(dict(seed=record["seed"], mode=record["policy_mode"],
                                split=record["split"], frames=length, path=str(relative),
                                sha256=file_sha256(destination), source_sha256=receipt["sha256"]))
    expected = {"train": 3600, "validation": 450, "test": 450}
    if any(sum(r["split"] == split for r in records) != expected[split] for split in splits):
        raise ValueError("Object21 episode counts differ from the fixed release split")
    write(output / "manifest.json", dict(format="five_mode_object21_targets_v1",
          source_manifest_sha256=config["manifest_sha256"], records=records,
          test_payloads_read="test" in splits, config=config))
