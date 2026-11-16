"""Check that B2 uses historical surface geometry before temporal attention."""

from dataclasses import replace

import pytest
import torch

from graph.state_estimation.models import SpatialFirstEstimator, TemporalFirstEstimator
from tests.test_state21_temporal_first import batches, check_causality, frames


@pytest.mark.parametrize("window", [1, 3])
def test_spatial_first_causality(window):
    torch.manual_seed(7)
    check_causality(SpatialFirstEstimator(hidden=16, layers=1, history=window))


def test_order_and_history_are_the_intended_differences():
    torch.manual_seed(17)
    temporal_first = TemporalFirstEstimator(hidden=16, layers=1, history=3).eval()
    spatial_first = SpatialFirstEstimator(hidden=16, layers=1, history=3).eval()
    spatial_first.load_state_dict(temporal_first.state_dict())
    raw = frames()[:3]
    surface = replace(raw[0].nodes[-1], position=(0.02, -0.01, 0.49))
    changed = [replace(raw[0], nodes=(*raw[0].nodes[:-1], surface)), *raw[1:]]
    original, modified = batches(raw), batches(changed)
    for model in (temporal_first, spatial_first):
        calls = []
        handles = [
            model.backbone.peg_layers[0].register_forward_hook(
                lambda *args, calls=calls: calls.append("spatial")
            ),
            model.temporal.register_forward_hook(
                lambda *args, calls=calls: calls.append("temporal")
            ),
        ]
        result = model(original)["object21"]
        expected = (
            ["temporal", "spatial"]
            if model is temporal_first
            else ["spatial", "spatial", "spatial", "temporal"]
        )
        assert calls == expected
        for handle in handles:
            handle.remove()
        altered = model(modified)["object21"]
        if model is temporal_first:
            torch.testing.assert_close(result, altered)
        else:
            assert (result - altered).abs().max() > 1e-6
            result.square().sum().backward()
            assert model.temporal.block.attention.in_proj_weight.grad.abs().sum() > 0
