"""Object-level Transformer and pose-independent shape supervision."""

import pytest
import torch

from graph.state_estimation.models import ObjectTransformerEstimator
from tests.test_state21_temporal_first import batches, check_causality, frames
from training.state21_loss import state21_loss


def assert_shape_loss(model, temporal_parameter):
    model.train()
    graphs = batches(frames())
    output = model(graphs)
    assert output["shape"].shape == (1, 4)
    assert (output["shape"] > 0).all()
    target = output["object21"].detach().clone()
    target[:, 17:] = torch.tensor([0.02, 0.007, 0.008, 0.013])
    first = state21_loss(output, target, shape_weight=1.0)
    changed_pose = target.clone()
    changed_pose[:, :3] += 0.05
    changed_pose[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    changed_pose[:, 10:13] -= 0.09
    changed_pose[:, 13:17] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    second = state21_loss(output, changed_pose, shape_weight=1.0)
    torch.testing.assert_close(first["shape"], second["shape"])
    other_size = target.clone()
    other_size[:, 17:] = torch.tensor([0.07, 0.021, 0.034, 0.057])
    assert not torch.isclose(
        first["shape"], state21_loss(output, other_size, shape_weight=1.0)["shape"]
    )
    disabled = state21_loss(output, target, shape_weight=0.0)
    torch.testing.assert_close(first["total"] - disabled["total"], first["shape"])
    negated = target.clone()
    negated[:, 3:7] *= -1
    negated[:, 13:17] *= -1
    torch.testing.assert_close(
        first["total"], state21_loss(output, negated, shape_weight=1.0)["total"]
    )
    first["shape"].backward()
    assert temporal_parameter.grad.abs().sum() > 0
    model.zero_grad()
    state21_loss(model(graphs), target, shape_weight=0.0)["total"].backward()
    assert temporal_parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("window", [1, 3])
def test_object_transformer_causality(window):
    torch.manual_seed(10)
    check_causality(ObjectTransformerEstimator(hidden=16, layers=1, history=window))


def test_object_transformer_shape_gradient():
    torch.manual_seed(10)
    model = ObjectTransformerEstimator(hidden=16, layers=1, history=3)
    assert_shape_loss(model, model.temporal.block.attention.in_proj_weight)
