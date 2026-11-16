"""Binding, visibility and causal windows for the temporal-first model."""

from dataclasses import replace

import pytest
import torch

from graph.state_estimation.graph import Source, batch_graphs, build_graph
from graph.state_estimation.models import TemporalFirstEstimator
from tests.test_state21_data import node, observation


def frames(episode="one"):
    peg = node((0.01, 0, 0.5), vertex_id=0)
    hole = node((0.02, 0.01, 0.5), object_id=1, vertex_id=0)
    surface = node((0.01, 0.015, 0.5))
    missing = replace(peg, visible=False, source=Source.MISSING, position=(99, 99, 99))
    return [
        observation([peg, hole, surface], 0, episode),
        observation([hole, missing, surface], 3, episode),
        observation(
            [surface, replace(hole, position=(0.04, 0.02, 0.5)), peg], 8, episode
        ),
        observation(
            [hole, surface, replace(peg, position=(0.01, 0.02, 0.5))], 11, episode
        ),
    ]


def batches(raw):
    return [batch_graphs([build_graph(frame)]) for frame in raw]


def test_binding_order_missing_and_uncertainty():
    from graph.state_estimation.temporal import bind_vertices

    raw = frames()
    binding = bind_vertices(batches(raw[:3]))
    assert binding.keys == [[(0, 0), (1, 0)]]
    assert binding.rows.tolist() == [[[0, -1, 2], [1, 0, 1]]]

    # Unknown identity is excluded from temporal matching even at the same array index.
    uncertain = replace(raw[0].nodes[0], vertex_id=None)
    raw[0] = replace(raw[0], nodes=(uncertain, *raw[0].nodes[1:]))
    assert bind_vertices(batches(raw[:3])).rows.tolist() == [[[-1, -1, 2], [1, 0, 1]]]
    assert bind_vertices(batches(raw[1:2])).keys == [[(1, 0)]]


def check_causality(model):
    raw = frames()
    changed = raw[:2] + [
        replace(
            frame,
            nodes=tuple(
                replace(point, position=(0.2, -0.3, 0.8)) for point in frame.nodes
            ),
        )
        for frame in raw[2:]
    ]
    model.eval()
    first = model.forward_sequence(batches(raw))
    second = model.forward_sequence(batches(changed))
    torch.testing.assert_close(first[:, :2], second[:, :2])
    assert (first[:, 2:] - second[:, 2:]).abs().max() > 1e-4
    next_episode = frames("new")[:2]
    torch.testing.assert_close(
        model.forward_sequence(batches(raw + next_episode))[:, -2:],
        model.forward_sequence(batches(next_episode)),
    )


@pytest.mark.parametrize("window", [1, 3])
def test_temporal_first_causality(window):
    torch.manual_seed(7)
    model = TemporalFirstEstimator(hidden=16, layers=1, history=window)
    check_causality(model)


def test_temporal_first_is_permutation_invariant_and_ignores_hidden_coordinates():
    torch.manual_seed(7)
    model = TemporalFirstEstimator(hidden=16, layers=1, history=3).eval()
    raw = frames()[:3]
    changed = [replace(frame, nodes=frame.nodes[::-1]) for frame in raw]
    changed[1] = replace(
        changed[1],
        nodes=tuple(
            replace(point, position=(-700, 0, 9))
            if point.source == Source.MISSING
            else point
            for point in changed[1].nodes
        ),
    )
    torch.testing.assert_close(
        model(batches(raw))["object21"], model(batches(changed))["object21"]
    )
    model.train()
    model(batches(raw))["object21"].square().sum().backward()
    assert model.temporal.block.attention.in_proj_weight.grad.abs().sum() > 0


def test_binding_rejects_reordered_episode_frames():
    from graph.state_estimation.temporal import bind_vertices

    with pytest.raises(ValueError, match="increase"):
        bind_vertices(batches(frames()[:2][::-1]))
