"""Object21 observations, triangulation, targets and split contracts."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from data_collection.state21 import Episode, Sample, audit_splits, sequence_windows
from graph.state_estimation.codec import (
    State21Scale,
    assemble_state38,
    camera_to_world,
    object21_from_gt,
)
from graph.state_estimation.graph import Frame, Node, Source, batch_graphs, build_graph


def observation(nodes=(), index=0, episode="episode"):
    return Frame(episode, index, (0.0, 0.0, 0.4), tuple(nodes))


def node(xyz, object_id=0, vertex_id=None, source=Source.OBSERVED, confidence=None):
    return Node(
        xyz,
        object_id,
        vertex_id is not None,
        vertex_id,
        source == Source.OBSERVED,
        confidence,
        source,
    )


@pytest.mark.parametrize(
    "points,method",
    [
        ([], "empty"),
        ([(0, 0, 0.5)], "point"),
        ([(0, 0, 0.5), (0, 0, 0.5)], "point"),
        ([(0, 0, 0.5), (0.01, 0, 0.5), (0.02, 0, 0.5)], "line"),
        ([(0, 0, 0.5), (0.01, 0, 0.5), (0, 0.01, 0.5)], "delaunay_2d"),
        ([(0, 0, 0.5), (0.01, 0, 0.5), (0, 0.01, 0.5), (0, 0, 0.51)], "delaunay_3d"),
    ],
)
def test_degenerate_geometry(points, method):
    graph = build_graph(observation(node(point) for point in points))
    assert graph.methods == (method, "empty")
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape == (graph.edge_index.shape[1], 7)
    assert graph.features.shape == (len(points), 13)


def test_edge_geometry_and_batch_isolation():
    nodes = [
        node((0, 0, 0.5), vertex_id=0),
        node((0.01, 0, 0.5)),
        node((0.2, 0, 0.5)),
        node((0.001, 0, 0.5), 1, 0),
        node((0.001, 0.01, 0.5), 1, 1),
    ]
    graph = build_graph(observation(nodes), max_edge_m=0.05)
    assert graph.rejected_edges == 1
    assert set(map(tuple, graph.edge_index.T.tolist())) == {
        (0, 1),
        (1, 0),
        (3, 4),
        (4, 3),
    }
    source, destination = graph.edge_index
    assert torch.equal(graph.object_id[source], graph.object_id[destination])
    torch.testing.assert_close(
        graph.edge_attr[:, :3], graph.position[destination] - graph.position[source]
    )
    torch.testing.assert_close(
        graph.edge_attr[:, 3], graph.edge_attr[:, :3].norm(dim=-1)
    )
    assert graph.edge_attr[:, 4:].argmax(-1).tolist() == [1, 1, 0, 0]
    batch = batch_graphs([graph, build_graph(observation()), graph])
    src, dst = batch.edge_index
    assert torch.equal(src // batch.valid.shape[-1], dst // batch.valid.shape[-1])
    assert not batch.valid[1].any()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"object_id": 2}, "object"),
        ({"position": (0, np.inf, 0.5)}, "finite"),
        ({"confidence": 1.1}, "confidence"),
        ({"visible": 1}, "visibility"),
        ({"source": Source.TRACKER}, "visibility"),
        ({"vertex_id": -1}, "vertex"),
    ],
)
def test_reject_invalid_observation(change, match):
    invalid = replace(node((0, 0, 0.5), vertex_id=0), **change)
    with pytest.raises(ValueError, match=match):
        build_graph(observation([invalid]))


def test_missing_geometry_is_ignored_and_distinct_from_padding():
    hidden = node((np.nan, 0.02, 99), vertex_id=0, source=Source.MISSING)
    frame = observation([hidden])
    graph = build_graph(frame)
    changed = build_graph(replace(frame, nodes=(replace(hidden, position=(9, 8, 7)),)))
    torch.testing.assert_close(graph.features, changed.features)
    batch = batch_graphs([graph, build_graph(observation())])
    assert batch.valid.tolist() == [[True], [False]]
    assert not batch.geometry.any()
    with pytest.raises(ValueError, match="duplicate"):
        build_graph(observation([hidden, hidden]))


def test_codec_world_tcp_camera_and_units():
    gt = {
        "ee_pose": [[0.1, 0.2, 0.3, 1, 0, 0, 0]],
        "peg_pose": [[0.2, 0.2, 0.3, 0.70710678, 0, 0, 0.70710678]],
        "hole_pose": [[0.3, 0.2, 0.3, 1, 0, 0, 0]],
        "peg_half_sizes": [[0.05, 0.01, 0.02]],
        "hole_radius": [0.025],
    }
    target = object21_from_gt(gt)
    torch.testing.assert_close(target[0, :3], torch.tensor([0.1, 0, 0]))
    torch.testing.assert_close(target[0, 7:10], torch.tensor([-0.1, 0.05, 0]))
    torch.testing.assert_close(target[0, 10:13], torch.tensor([0.2, 0, 0]))
    torch.testing.assert_close(target[0, 17:], torch.tensor([0.05, 0.01, 0.02, 0.025]))
    scale = State21Scale()
    torch.testing.assert_close(scale.decode(scale.encode(target)), target)
    state38 = assemble_state38(torch.ones(1, 15), target, torch.zeros(1, 2))
    assert state38.shape == (1, 38)
    torch.testing.assert_close(state38[:, 15:36], target)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    extrinsic = np.column_stack((rotation, [0.1, 0.2, 0.3]))
    np.testing.assert_allclose(
        camera_to_world(np.array([[0.1, 0.4, 0.8]]), extrinsic), [[0.2, 0, 0.5]]
    )
    with pytest.raises(ValueError, match="metres"):
        build_graph(observation([node((100, 200, 300))]))


def test_parent_split_and_source_order():
    target = torch.zeros(21)
    first = Episode(
        "first",
        "root",
        "train",
        (
            Sample(observation(index=2, episode="first"), 2, target),
            Sample(observation(index=7, episode="first"), 7, target),
        ),
    )
    second = Episode(
        "second",
        "another",
        "validation",
        (Sample(observation(episode="second"), 0, target),),
    )
    assert audit_splits([first, second])["episode_counts"] == {
        "train": 1,
        "validation": 1,
    }
    windows = list(sequence_windows([first, second], 3))
    assert [[f.index for f in window] for window, _ in windows] == [[2], [2, 7], [0]]
    with pytest.raises(ValueError, match="parent"):
        audit_splits([first, replace(second, parent_id="root")])
    with pytest.raises(ValueError, match="ordering"):
        audit_splits([replace(first, samples=first.samples[::-1])])
    with pytest.raises(ValueError, match="target frame"):
        audit_splits(
            [replace(second, samples=(replace(second.samples[0], target_index=9),))]
        )


def test_qhull_failure_records_isolated_fallback(monkeypatch):
    from scipy.spatial import QhullError

    from graph.state_estimation import graph as builder

    def fail(_):
        raise QhullError("degenerate fixture")

    monkeypatch.setattr(builder, "Delaunay", fail)
    graph = build_graph(
        observation(
            [
                node((0, 0, 0.5)),
                node((0.02, 0, 0.5)),
                node((0, 0.02, 0.5)),
            ]
        )
    )
    assert graph.methods == ("qhull_isolated", "empty")
    assert graph.edge_index.shape == (2, 0)
    assert graph.geometry.all()


def test_invalid_scale_rotation_and_frame_metadata():
    with pytest.raises(ValueError, match="scaling"):
        State21Scale(sizes_m=(1.0, 1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="floating"):
        State21Scale().encode(torch.ones(20))
    for extrinsic, match in (
        (np.eye(3), "extrinsic"),
        (np.zeros((3, 4)), "proper rotation"),
    ):
        with pytest.raises(ValueError, match=match):
            camera_to_world(np.zeros((1, 3)), extrinsic)
    for frame, match in (
        (observation(index=-1), "Frame index"),
        (observation(episode=""), "episode ID"),
    ):
        with pytest.raises(ValueError, match=match):
            build_graph(frame)
    with pytest.raises(ValueError, match="zero frames"):
        batch_graphs([])
    with pytest.raises(ValueError, match="history"):
        list(sequence_windows([], 0))


def test_explicit_tapnext_manifest_provenance_and_optional_ids(tmp_path):
    import json

    from data_collection.state21 import load_dataset

    path = tmp_path / "observations.json"
    target = [0.0] * 21
    target[3] = target[13] = 1.0
    target[17:] = [0.04, 0.01, 0.01, 0.02]
    manifest = {
        "format": "state21_observations",
        "coordinates": "world_m",
        "provenance": {
            "tracker": "TAPNext++",
            "tracker_sha256": "a" * 64,
            "tracker_training_or_selection_parents": ["upstream_parent"],
        },
        "episodes": [
            {
                "episode_id": "held_out",
                "parent_id": "parent",
                "split": "test",
                "frames": [
                    {
                        "frame_index": 5,
                        "target_frame_index": 5,
                        "tcp_world_m": [0.0, 0.0, 0.4],
                        "object21": target,
                        "source_frame_sha256": "b" * 64,
                        "nodes": [
                            {
                                "position": [0.02, 0.0, 0.5],
                                "object_id": 0,
                                "is_vertex": True,
                                "vertex_id": 3,
                                "identity_verified": True,
                                "visible": True,
                                "confidence": 0.4,
                                "source": "observed",
                            },
                            {
                                "position": [0.03, 0.0, 0.5],
                                "object_id": 1,
                                "is_vertex": True,
                                "visible": False,
                                "source": "tracker",
                            },
                        ],
                    }
                ],
            }
        ],
    }

    def load():
        path.write_text(json.dumps(manifest))
        return load_dataset({"kind": "observations", "manifest": path})

    (episode,), report = load()
    assert episode.samples[0].observation.nodes[1].vertex_id is None
    assert report["verified_visible_vertex_rows"] == {"test": 1}
    manifest["episodes"][0]["parent_id"] = "upstream_parent"
    with pytest.raises(ValueError, match="Tracker"):
        load()
    manifest["provenance"]["tracker_sha256"] = "unknown"
    with pytest.raises(ValueError, match="TAPNext"):
        load()
    manifest["coordinates"] = "world_mm"
    with pytest.raises(ValueError, match="world_m"):
        load()


def test_initial_scene_fingerprint_cannot_cross_splits():
    from data_collection.state21_synthetic import synthetic_episodes

    first, _, validation, *_ = synthetic_episodes(2)
    with pytest.raises(ValueError, match="fingerprint"):
        audit_splits(
            [
                replace(first, reset_fingerprint="same"),
                replace(validation, reset_fingerprint="same"),
            ]
        )


def test_bad_episode_metadata_fails_audit():
    from data_collection.state21_synthetic import synthetic_episodes

    episode = synthetic_episodes(2)[0]
    sample = episode.samples[0]
    for invalid, match in [
        (replace(episode, split="holdout"), "valid split"),
        (replace(episode, samples=()), "Empty episode"),
        (
            replace(episode, samples=(replace(sample, target=torch.zeros(20)),)),
            "finite Object21",
        ),
        (
            replace(
                episode,
                samples=(
                    replace(
                        sample,
                        observation=replace(sample.observation, tcp=(np.nan, 0, 0)),
                    ),
                ),
            ),
            "finite observed TCP",
        ),
    ]:
        with pytest.raises(ValueError, match=match):
            audit_splits([invalid])
    with pytest.raises(ValueError, match="duplicate episode ID"):
        audit_splits([episode, episode])


def test_reject_missing_confidence_and_nonfinite_camera_geometry():
    hidden = node(None, vertex_id=3, source=Source.MISSING, confidence=0.5)
    with pytest.raises(ValueError, match="cannot supply confidence"):
        build_graph(observation([hidden]))
    extrinsic = np.column_stack((np.eye(3), np.zeros(3)))
    with pytest.raises(ValueError, match="finite"):
        camera_to_world([[np.nan, 0, 1]], extrinsic)
