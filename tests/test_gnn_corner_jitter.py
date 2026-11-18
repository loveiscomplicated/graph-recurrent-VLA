"""Check train-time corner jitter before graph construction."""

import pytest
import torch

from training.gnn.corner_jitter import (
    corner_prefix_counts, fixed_validation_rows, jitter_std_mm, training_collate,
)
from training.gnn.visibility_data import collate


def sample():
    return (
        torch.tensor([[0.01, 0.02, 0.03], [0.03, 0.02, 0.01], [0.01, 0.01, 0.01]]),
        torch.tensor([0, 1, 0]),
        torch.zeros(3),
        torch.ones(21),
        "frame",
    )


def test_surface_and_labels_are_unchanged_and_edges_use_jittered_xyz():
    row = sample()
    clean = collate([row])
    noisy = collate([row], jitter_std_m=0.001, corner_counts={"frame": 2})
    assert not torch.equal(noisy.graph.xyz_world[0, :2], clean.graph.xyz_world[0, :2])
    torch.testing.assert_close(noisy.graph.xyz_world[0, 2], clean.graph.xyz_world[0, 2], rtol=0, atol=0)
    assert torch.equal(clean.graph.xyz_world[0], row[0])
    assert torch.equal(clean.targets["object21"], noisy.targets["object21"])
    assert torch.equal(clean.graph.tcp_xyz_world, noisy.graph.tcp_xyz_world)
    assert torch.equal(clean.graph.edge_index, noisy.graph.edge_index)
    xyz = noisy.graph.xyz_world.reshape(-1, 3)
    sender, receiver = noisy.graph.edge_index
    offsets = xyz[sender] - xyz[receiver]
    torch.testing.assert_close(noisy.graph.relative_xyz, offsets)
    torch.testing.assert_close(noisy.graph.distance[:, 0], offsets.norm(dim=1))


def test_clean_default_and_reproducible_jitter():
    row = sample()
    state = torch.get_rng_state()
    assert torch.equal(collate([row]).graph.xyz_world[0], row[0])
    assert torch.equal(state, torch.get_rng_state())
    train = training_collate(collate, {"dataset_sha256": "clean"}, 1.0)
    first = train([row]).graph.xyz_world
    assert not torch.equal(train([row]).graph.xyz_world, first)
    torch.set_rng_state(state)
    assert torch.equal(train([row]).graph.xyz_world, first)


def test_padding_and_empty_batch_stay_zero():
    row = sample()
    empty = (row[0][:0], row[1][:0], *row[2:4], "empty")
    noise = collate([empty, row], jitter_std_m=0.001)
    assert not noise.graph.xyz_world[0].any()
    assert not noise.graph.node_valid[0].any()
    all_empty = collate([empty], jitter_std_m=0.001)
    assert not all_empty.graph.xyz_world.any()


def test_prefix_checks_exact_coordinates_and_types():
    fused = sample()
    direct = (fused[0][:2], fused[1][:2], *fused[2:])
    assert corner_prefix_counts([fused], [direct]) == {"frame": 2}
    with pytest.raises(ValueError, match="Detector prefix"):
        corner_prefix_counts([fused], [(direct[0] + 0.001, *direct[1:])])


def test_fixed_validation_matches_across_graph_variants_without_using_training_rng():
    row = sample()
    settings = {"validation_corner_jitter_std_mm": 5.0, "validation_corner_jitter_seed": 23}
    direct = (row[0][:2], row[1][:2], *row[2:])
    fused = (*row[:-1], "original::frame")
    state = torch.get_rng_state()
    direct_identity = {"dataset_sha256": "original"}
    noisy_direct = fixed_validation_rows([direct], direct_identity, settings)[0]
    noisy_fused = fixed_validation_rows([fused], {"dataset_sha256": "fused"}, settings,
                                        corner_counts={"original::frame": 2})[0]
    again = fixed_validation_rows([direct], {"dataset_sha256": "original"}, settings)[0]
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(noisy_direct[0], noisy_fused[0][:2])
    assert torch.equal(noisy_direct[0], again[0])
    assert not torch.equal(noisy_direct[0], direct[0])
    assert torch.equal(row[0][2], noisy_fused[0][2])
    assert direct_identity["resume_from_dataset_sha256"] == "original"
    assert direct_identity["dataset_sha256"] != "original"


@pytest.mark.parametrize("invalid", [True, "1", -1.0, float("nan"), float("inf")])
def test_invalid_standard_deviation(invalid):
    with pytest.raises(ValueError, match="finite nonnegative"):
        jitter_std_mm({"corner_jitter_std_mm": invalid})
