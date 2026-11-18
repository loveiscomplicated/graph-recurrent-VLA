"""Check that the recovered data yields aligned batches for both graph variants."""

import hashlib
import io
import json
import tarfile

import numpy as np
import torch

from training.gnn.five_mode_supervised_data import build_data
from training.gnn.packed_five_mode_data import build_data as build_packed_data


def test_loaders_read_recovered_targets(tmp_path):
    mode = "Fixed_Hole_expert"
    records = []
    selections = {}
    shard = "shard_000.tar"
    with tarfile.open(tmp_path / shard, "w") as archive:
        for seed, split in enumerate(("train", "validation"), start=1):
            tables = {
                "corners": dict(corner_offsets=[0, 2],
                                corner_world_xyz=np.zeros((2, 3), dtype=np.float32),
                                corner_object_id=np.asarray([0, 1], dtype=np.int8)),
                "surface": dict(surface_offsets=[0, 1],
                                surface_world_xyz=np.zeros((1, 3), dtype=np.float32)),
                "targets": dict(tcp=np.zeros((1, 3), dtype=np.float32),
                                object21=np.full((1, 21), seed, dtype=np.float32)),
            }
            if split == "validation":
                tables["surface"]["surface_object_id"] = np.asarray([1], dtype=np.int8)
            files = {}
            for kind, arrays in tables.items():
                member = f"batch_{seed}/{kind}/{seed}_{mode}.npz"
                buffer = io.BytesIO()
                np.savez(buffer, frame_indices=[0], **arrays)
                content = buffer.getvalue()
                info = tarfile.TarInfo(member)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
                files[kind] = dict(archive=shard, member=member,
                                   sha256=hashlib.sha256(content).hexdigest())
            records.append(dict(seed=seed, policy_mode=mode, split=split, frames=1, files=files))
            selections[split] = [dict(seed=seed, policy_mode=mode, frames=[0])]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(format="five_mode_supervised_v1", episodes=2,
                                        archives=[{"file": shard}], records=records)))
    index = tmp_path / "condition_index.json"
    index.write_text(json.dumps({"condition_1": selections}))

    for include_surface in (False, True):
        config = dict(data=dict(manifest=manifest, condition_index=index, include_surface=include_surface),
                      training=dict(batch_size=1))
        data = build_data(config, torch.Generator().manual_seed(0))
        try:
            for expected, loader in enumerate((data.train, data.validation), start=1):
                batch = next(iter(loader))
                torch.testing.assert_close(batch.targets["object21"], torch.full((1, 21), float(expected)))
                expected_kinds = [0, 1]
                if include_surface:
                    expected_kinds.append(0 if expected == 1 else 1)
                assert batch.graph.object_type[0].tolist() == expected_kinds
        finally:
            data.train.dataset.close()
            data.validation.dataset.close()

        config["device"] = "cpu"
        packed = build_packed_data(config, torch.Generator().manual_seed(0))
        assert "condition_index_sha256" in packed.identity
        for expected, loader in enumerate((packed.train, packed.validation), start=1):
            batch = next(iter(loader))
            torch.testing.assert_close(batch.targets["object21"], torch.full((1, 21), float(expected)))
            assert batch.sample_ids == [f"{expected}:{mode}:0"]
            expected_kinds = [0, 1, 0] if include_surface and expected == 1 else [0, 1]
            assert batch.graph.object_type[0].tolist() == expected_kinds
