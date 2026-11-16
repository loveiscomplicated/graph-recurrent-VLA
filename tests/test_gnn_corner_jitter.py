"""Check train-time corner jitter before graph construction."""

import pytest
import torch

from training.gnn.corner_jitter import corner_prefix_counts, jitter_std_mm, training_collate
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


@pytest.mark.parametrize("invalid", [True, "1", -1.0, float("nan"), float("inf")])
def test_invalid_standard_deviation(invalid):
    with pytest.raises(ValueError, match="finite nonnegative"):
        jitter_std_mm({"corner_jitter_std_mm": invalid})
