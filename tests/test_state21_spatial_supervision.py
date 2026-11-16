"""Test the GT association and the two independently gated B2 losses."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from data_collection import state21_vertex_targets
from data_collection.state21 import Episode, Sample
from graph.state_estimation.models import SpatialFirstEstimator
from tests.test_state21_temporal_first import batches, frames
from training.state21_loss import distance_loss, spatial_vertex_errors, state21_loss


def test_vertex_target_keeps_anchor_identity_during_tracking_drift(
    tmp_path, monkeypatch
):
    raw = frames()[:2]
    peg = raw[0].nodes[0]
    raw[1] = replace(
        raw[1], nodes=(replace(peg, position=(0.03, 0, 0.5)), raw[1].nodes[0])
    )
    references = {
        (0, "peg", 0): np.array((0.01, 0, 0.5)),
        (0, "peg", 1): np.array((0.03, 0, 0.5)),
        (1, "box_outer", 0): np.array((0.02, 0.01, 0.5)),
    }
    monkeypatch.setattr(
        state21_vertex_targets, "reference_corners", lambda *args: references
    )
    episode = Episode(
        "one",
        "parent",
        "train",
        tuple(Sample(frame, frame.index, torch.zeros(21)) for frame in raw),
    )
    annotated, _, _ = state21_vertex_targets.annotate_vertices(
        episode, tmp_path, 0, gate_m=0.005, margin_m=0.002
    )
    assert annotated.samples[0].vertex_targets[(0, 0)] == (0.01, 0, 0.5)
    assert (0, 0) not in annotated.samples[1].vertex_targets
    assert (1, 0) in annotated.samples[1].vertex_targets


def test_ambiguous_and_duplicate_matches_are_excluded():
    references = {
        (0, "peg", 0): (0, 0, 0),
        (0, "peg", 1): (0.008, 0, 0),
    }
    tracks = {(0, 0): (0.004, 0, 0), (0, 1): (0.001, 0, 0), (0, 2): (0.002, 0, 0)}
    matches = state21_vertex_targets.match_vertices(
        tracks, references, gate_m=0.005, margin_m=0.002
    )
    assert matches == {(0, 1): (0, "peg", 0)}


def test_spatial_vertex_loss_reaches_gnn_before_temporal_attention():
    torch.manual_seed(13)
    model = SpatialFirstEstimator(hidden=16, layers=1, history=3, vertex_aux=True)
    observations = frames()[:3]
    graphs = batches(observations)
    before = model(graphs)
    torch.testing.assert_close(before["spatial_vertex_xyz"], graphs[-1].position)
    with torch.no_grad():
        model.vertex_head.weight.normal_(std=0.05)
    output = model(graphs)
    torch.testing.assert_close(before["object21"], output["object21"])
    sample = Sample(
        observations[-1], 8, torch.zeros(21), vertex_targets={(0, 0): (0.013, 0, 0.5)}
    )
    errors, raw_errors = spatial_vertex_errors(
        output["spatial_vertex_xyz"], graphs[-1], [SimpleNamespace(sample=sample)]
    )
    torch.testing.assert_close(raw_errors, torch.tensor([0.003]))
    distance_loss(errors, 0.01, 0.003).backward()
    assert model.vertex_head.weight.grad.abs().sum() > 0
    assert next(model.backbone.node_encoder.parameters()).grad.abs().sum() > 0
    assert model.temporal.block.attention.in_proj_weight.grad is None


def test_endpoint_loss_replaces_only_endpoint_objective():
    predicted = torch.zeros(2, 21)
    predicted[:, [3, 13]] = 1
    target = predicted.clone()
    predicted[0, 7] = 0.003
    predicted[1, 7:10] = 0.003 / (3**0.5)
    output = {"object21": predicted, "endpoint_residual": torch.full((2, 3), 0.002)}
    legacy = state21_loss(output, target)
    radial = state21_loss(output, target, endpoint_transition_m=0.003)
    torch.testing.assert_close(radial["endpoint_radial"], torch.tensor(0.15))
    torch.testing.assert_close(radial["state21"], legacy["state21"])
    torch.testing.assert_close(
        radial["total"], legacy["total"] - legacy["weighted/endpoint"] + 0.15
    )
