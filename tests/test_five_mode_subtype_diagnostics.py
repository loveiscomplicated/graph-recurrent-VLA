"""Protect subtype boundaries, aggregation and validation-only sample access."""

from functools import partial
import hashlib
import io
import json
import tarfile

import numpy as np
import pytest
import torch
from torch import nn

from data_collection.episode_dataset import canonical_hash, file_sha256
from evaluation.gnn import five_mode_subtype_diagnostics as diagnostics
from utils.experiment_logger import ExperimentLogger


def test_subtypes_are_disjoint_and_respect_plane_tolerance_boundary():
    measurements = np.asarray([
        [2, 1, np.nan], [3, 3, np.nan], [4, 2, 0.001],
        [4, 3, 0.001], [4, 2, 0.001001],
    ])
    groups = diagnostics.subtype_masks(measurements, 0.001)
    for row, name in enumerate(diagnostics.SUBTYPES):
        np.testing.assert_array_equal(groups[name], np.arange(5) == row)


@pytest.mark.parametrize("measurement", [
    [1, 3, np.nan], [4, 0, 0.01], [4, 3, 0.01], [4, 2, np.nan],
    [4, 2, -0.01], [2.5, 1, np.nan], [2, np.inf, np.nan],
])
def test_invalid_or_non_c2_support_is_rejected(measurement):
    with pytest.raises(ValueError):
        diagnostics.subtype_masks([measurement], 0.001)


@pytest.mark.parametrize("split", [None, "train", "test"])
def test_non_validation_request_is_rejected_before_reading_inputs(split, monkeypatch):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("No inputs may be read before rejecting the split")

    monkeypatch.setattr(diagnostics, "pinned_path", unexpected_read)
    with pytest.raises(ValueError, match="validation only"):
        diagnostics.evaluate({"split": split})


def test_physical_errors_use_mm_and_sign_invariant_rotations():
    target = torch.zeros((1, 21))
    target[:, 3] = target[:, 13] = 1
    prediction = target.clone()
    prediction[:, 3:7] *= -1
    prediction[:, 13:17] *= -1
    prediction[:, 0] = 0.003
    prediction[:, 7] = 0.004
    prediction[:, 10] = 0.005

    errors = diagnostics.physical_errors(prediction, target)

    assert errors["peg_position_error_mm"][0] == pytest.approx(3)
    assert errors["endpoint_error_mm"][0] == pytest.approx(4)
    assert errors["hole_position_error_mm"][0] == pytest.approx(5)
    assert errors["peg_rotation_error_deg"][0] == 0
    assert errors["hole_rotation_error_deg"][0] == 0


def test_report_distinguishes_worst_mean_from_largest_error_contribution():
    frame_ids = [
        "1:Moving_Hole_expert:0", "2:Fixed_Hole_random:0",
        "2:Fixed_Hole_random:1", "3:Fixed_Hole_expert:0",
    ]
    groups = {
        "C2-A": np.asarray([True, False, False, False]),
        "C2-B": np.asarray([False, True, True, True]),
        "C2-C": np.zeros(4, dtype=bool),
    }
    errors = {"endpoint_error_mm": np.asarray([10, 6, 6, 6], dtype=float)}

    report, _, _ = diagnostics.build_report(groups, errors, frame_ids)

    assert report["worst_groups"]["highest_mean_endpoint"] == "C2-A"
    assert report["worst_groups"]["largest_endpoint_error_contribution"] == "C2-B"
    assert report["overall"]["metrics"]["endpoint_error_mm"]["mean"] == 7
    assert report["subtypes"]["C2-A"]["by_motion"]["Moving_Hole"]["frames"] == 1
    assert report["subtypes"]["C2-B"]["by_policy"]["Fixed_Hole_random"]["frames"] == 2
    assert report["subtypes"]["C2-B"]["episodes"] == 2
    assert report["subtypes"]["C2-C"]["metrics"]["endpoint_error_mm"] is None
    assert sum(row["endpoint_error_fraction"] for row in report["subtypes"].values()) == 1
    assert sum(row["endpoint_mean_contribution_mm"] for row in report["subtypes"].values()) == 7


def write_validation_fixture(tmp_path):
    """Make one five-subtype episode; other splits point at deliberately absent archives."""
    square = np.asarray([[0, 0, 0], [0.03, 0, 0], [0, 0.03, 0], [0.03, 0.03, 0]])
    tetrahedron = square.copy()
    tetrahedron[-1, 2] = 0.02
    cases = [(square[:2], 1), (square[:3], 3), (square, 2), (square, 3), (tetrahedron, 2)]
    points, kinds, rim, offsets = [], [], [], [0]
    for peg, rim_count in cases:
        points.extend(peg)
        points.extend(np.zeros((rim_count + 1, 3)))
        kinds.extend([0] * len(peg) + [1] * (rim_count + 1))
        rim.extend([False] * len(peg) + [True] * rim_count + [False])
        offsets.append(len(points))
    target = np.zeros((5, 21), dtype=np.float32)
    target[:, 3] = target[:, 13] = 1
    target[:, 7] = np.arange(5) * 0.001
    tables = {
        "corners": dict(corner_offsets=np.asarray(offsets),
                        corner_world_xyz=np.asarray(points, dtype=np.float32),
                        corner_object_id=np.asarray(kinds, dtype=np.int8),
                        corner_contour_hole=np.asarray(rim)),
        "surface": dict(surface_offsets=np.arange(6),
                        surface_world_xyz=np.zeros((5, 3), dtype=np.float32),
                        surface_object_id=np.zeros(5, dtype=np.int8)),
        "targets": dict(tcp=np.zeros((5, 3), dtype=np.float32), object21=target),
    }
    files = {}
    with tarfile.open(tmp_path / "validation.tar", "w") as archive:
        for kind, arrays in tables.items():
            buffer = io.BytesIO()
            np.savez(buffer, frame_indices=np.arange(5), **arrays)
            content = buffer.getvalue()
            member = f"{kind}/1_Fixed_Hole_expert.npz"
            info = tarfile.TarInfo(member)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
            files[kind] = dict(archive="validation.tar", member=member,
                               sha256=hashlib.sha256(content).hexdigest())
    records = [dict(seed=1, policy_mode="Fixed_Hole_expert", split="validation", frames=5, files=files)]
    for seed, split in ((2, "train"), (3, "test")):
        unavailable = {kind: dict(receipt, archive=f"do_not_read_{split}.tar")
                       for kind, receipt in files.items()}
        records.append(dict(seed=seed, policy_mode="Fixed_Hole_expert", split=split,
                            frames=5, files=unavailable))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(
        format="five_mode_supervised_v1", episodes=3,
        records=records, archives=[dict(file="validation.tar")],
    )))
    index = tmp_path / "condition_index.json"
    index.write_text(json.dumps({"condition_2": {"validation": [dict(
        seed=1, policy_mode="Fixed_Hole_expert", frames=list(range(5)),
    )]}}))
    return manifest, index


def test_end_to_end_reads_only_validation_and_saves_aligned_predictions(tmp_path, monkeypatch):
    manifest, index = write_validation_fixture(tmp_path)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake checkpoint; loader is replaced by a constant model")

    class ConstantModel(nn.Module):
        def forward(self, graph):
            prediction = graph.xyz_world.new_zeros((len(graph.xyz_world), 21))
            prediction[:, 3] = prediction[:, 13] = 1
            return dict(object21=prediction, endpoint_residual=prediction[:, :3])

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(diagnostics, "load_surface_model", lambda *args: ConstantModel().eval())
    monkeypatch.setattr(diagnostics, "ExperimentLogger", partial(ExperimentLogger, root=tmp_path / "experiments"))
    frame_ids = [f"1:Fixed_Hole_expert:{frame}" for frame in range(5)]
    config = dict(
        split="validation", device="cpu", torch_num_threads=1, coplanarity_tolerance_m=0.001,
        batch_size=dict(validation=2),
        manifest=dict(path=str(manifest), sha256=file_sha256(manifest)),
        condition_index=dict(path=str(index), sha256=file_sha256(index)),
        checkpoint=dict(path=str(checkpoint), sha256=file_sha256(checkpoint)),
        cohorts=dict(validation=dict(condition_2=dict(frames=5, frame_ids_sha256=canonical_hash(frame_ids)))),
        output_root=str(tmp_path / "results"),
    )

    output = diagnostics.evaluate(config)

    result = json.loads((output / "results.json").read_text())
    assert result["split"] == "validation"
    assert result["test_samples_read"] is False
    assert result["model_selection_performed"] is False
    assert result["overall"]["frames"] == 5
    assert result["overall"]["metrics"]["endpoint_error_mm"]["mean"] == pytest.approx(2)
    assert [row["frames"] for row in result["subtypes"].values()] == [1] * 5
    with np.load(output / "validation_predictions.npz", allow_pickle=False) as arrays:
        assert arrays["frame_ids"].tolist() == frame_ids
        assert arrays["subtype"].tolist() == list(diagnostics.SUBTYPES)
