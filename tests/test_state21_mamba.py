"""Streaming Mamba state, gradients and episode reset contracts."""

import pytest
import torch

from graph.state_estimation.graph import batch_graphs, build_graph
from graph.state_estimation.models import ObjectMambaEstimator
from tests.test_state21_shape import assert_shape_loss
from tests.test_state21_temporal_first import batches, check_causality, frames


def make_model(**options):
    torch.manual_seed(19)
    return ObjectMambaEstimator(hidden=16, layers=1, history=8, **options)


def test_stream_matches_sequence_and_mixer_reference():
    model = make_model().eval()
    sequence = torch.randn(2, 40, 16)
    cache = model.mixer.allocate_inference_cache(2)
    results = []
    for token in sequence.split(1, dim=1):
        output, *cache = model.mixer.step(token, *cache)
        results.append(output)
    torch.testing.assert_close(
        torch.cat(results, dim=1), model.mixer(sequence), atol=2e-6, rtol=1e-5
    )
    incoming = batches(frames())
    outputs, state = [], None
    for frame in incoming:
        output, state = model.step(frame, state)
        outputs.append(output["object21"])
    torch.testing.assert_close(
        torch.stack(outputs, 1), model.forward_sequence(incoming)
    )
    torch.testing.assert_close(outputs[-1], model(incoming)["object21"])
    assert state.steps == (4,)
    with pytest.raises(ValueError, match="increase"):
        model.step(incoming[-1], state)


def test_stream_resets_each_batch_row_independently():
    model = make_model().eval()
    a, b, new = frames("a"), frames("b"), frames("new")
    initial = batch_graphs([build_graph(frame) for frame in (a[0], b[0])])
    following = batch_graphs([build_graph(frame) for frame in (new[0], b[1])])
    _, state = model.step(initial)
    updated, state = model.step(following, state)
    expected = torch.cat(
        [model(batches([new[0]]))["object21"], model(batches(b[:2]))["object21"]]
    )
    torch.testing.assert_close(updated["object21"], expected)
    assert state.steps == (1, 2)
    check_causality(model)


def test_truncated_bptt_detaches_only_prior_graphs():
    model = make_model(tbptt_steps=2)
    incoming = batches(frames()[:3])
    for frame in incoming:
        frame.features.requires_grad_()
    output = model(incoming)["object21"]
    output.square().sum().backward()
    for frame in incoming[:2]:
        assert frame.features.grad is None or not frame.features.grad.any()
    assert incoming[-1].features.grad.abs().sum() > 0


def test_mamba_shape_gradient():
    model = make_model()
    assert_shape_loss(model, model.mixer.in_proj.weight)


def test_stream_rejects_batch_size_changes():
    model = make_model()
    original = batches(frames()[:1])[0]
    _, state = model.step(original)
    different_batch = batch_graphs(
        [build_graph(frames("new")[0]), build_graph(frames("another")[0])]
    )
    with pytest.raises(ValueError, match="batch size changed"):
        model.step(different_batch, state)
    with pytest.raises(ValueError, match="tbptt_steps"):
        make_model(tbptt_steps=-1)
