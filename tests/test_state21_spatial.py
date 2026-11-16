"""Spatial baseline and edge-dependent message tests."""

from dataclasses import replace

import torch

from graph.state_estimation.graph import batch_graphs, build_graph
from graph.state_estimation.models import SpatialEstimator
from tests.test_state21_data import node, observation


def test_edge_features_affect_both_gate_and_message():
    torch.manual_seed(4)
    model = SpatialEstimator(hidden=16, layers=1).eval()
    graph = batch_graphs(
        [
            build_graph(
                observation(
                    [
                        node((0.0, 0.0, 0.5), vertex_id=0),
                        node((0.01, 0.0, 0.5)),
                        node((0.0, 0.01, 0.5)),
                        node((0.0, 0.0, 0.5), 1, 0),
                    ]
                )
            )
        ]
    )
    layer = model.backbone.peg_layers[0]
    seen = {"gate": [], "message": []}
    handles = [
        module.register_forward_hook(
            lambda _, args, output, name=name: seen[name].append(output.detach())
        )
        for name, module in (("gate", layer.gate), ("message", layer.message))
    ]
    first = model([graph])["object21"]
    changed = graph.edge_attr.clone()
    changed[0, :4] *= 2
    second = model([replace(graph, edge_attr=changed)])["object21"]
    for handle in handles:
        handle.remove()
    for values in seen.values():
        assert len(values) == 2
        assert (values[1] - values[0]).abs().max() > 1e-4
    assert (second - first).abs().max() > 1e-6


def test_spatial_shape_empty_missing_and_gradient():
    graph = batch_graphs(
        [
            build_graph(observation()),
            build_graph(
                observation(
                    [
                        node((0.01, 0, 0.5), vertex_id=0),
                    ]
                )
            ),
        ]
    )
    model = SpatialEstimator(hidden=16, layers=1)
    prediction = model([graph])["object21"]
    assert prediction.shape == (2, 21) and torch.isfinite(prediction).all()
    for sl in (slice(3, 7), slice(13, 17)):
        torch.testing.assert_close(prediction[:, sl].norm(dim=-1), torch.ones(2))
    assert (prediction[:, 17:] > 0).all()
    prediction.square().sum().backward()
    assert all(
        torch.isfinite(param.grad).all()
        for param in model.parameters()
        if param.grad is not None
    )
    assert model.backbone.node_encoder[0].weight.grad.abs().sum() > 0
