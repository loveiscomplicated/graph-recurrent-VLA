"""Re-evaluate verified local GT geometry on an existing CUDA host.

This does not start a simulator or reconstruct missing detector identities.
It checks all replayed fixture states against saved validation supervision and
runs the same frozen graph interventions as the CPU study.
"""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.gnn.box_hole_diagnostics import evaluate_models, load_geometry, METRICS
from evaluation.gnn.box_hole_geometry import verify_replay
from evaluation.gnn.vertex_metrics import (
    cluster_interval, load_aligned_predictions, statistics,
)
from utils.experiment_logger import ExperimentLogger


def verify(config):
    """Run paired CUDA inference with hashed GT states and untouched checkpoints."""
    torch.set_num_threads(config.get("torch_threads", 4))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA verification was requested but CUDA is unavailable")
    state_path = Path(config["verified_fixture_states"])
    if file_sha256(state_path) != config["verified_fixture_states_sha256"]:
        raise ValueError("Verified fixture states do not match their pinned SHA256")
    states = json.loads(state_path.read_text())
    geometry = load_geometry(config["validation_geometry"])
    predictions, prediction_hashes = load_aligned_predictions(config["predictions"])
    identifiers = geometry["sample_ids"]
    np.testing.assert_array_equal(identifiers, predictions["baseline"]["sample_ids"])
    np.testing.assert_array_equal(geometry["target"], predictions["baseline"]["target"])
    identity = json.loads(Path(config["data_identity"]).read_text())
    manifest = json.loads(Path(config["parent_manifest"]).read_text())
    episodes = np.asarray([str(Path(identifier).parent) for identifier in identifiers])
    if set(episodes) != set(identity["splits"]["validation"]):
        raise ValueError("CUDA replay must use only the complete original validation set")
    if set(episodes) & set(identity["splits"]["test"]):
        raise ValueError("Reserved test episode found in CUDA validation geometry")
    for episode, state in states.items():
        mask = episodes == episode
        verify_replay(state, geometry["target"][mask], geometry["tcp_world"][mask], 2e-6)
    parents = np.asarray([manifest["parent_by_episode"][episode] for episode in episodes])
    count = predictions["baseline"]["counts"]
    groups = dict(all=np.ones(len(identifiers), dtype=bool), target_present=count[:, 1] > 0,
                  high_support_proxy=(count[:, 0] >= 4) & (count[:, 1] >= 4))
    with ExperimentLogger(team="graph", config=config, model="box_hole_cuda_verification",
                          seed=0, job_type="evaluation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        inference, distributions, replay_checks = evaluate_models(config, geometry, states, predictions, output)
        summary = {}
        for group, mask in groups.items():
            summary[group] = {condition: {model: {metric: {
                **statistics(result["errors"][metric][mask]),
                "parent_mean_ci95": cluster_interval(result["errors"][metric][mask], parents[mask], 2000)}
                for metric in METRICS} for model, result in models.items()}
                for condition, models in inference.items()}
        raw_root = Path(config["raw_root"])
        raw_validation_frames = sum((raw_root / identifier).is_file() for identifier in identifiers)
        sources = [Path(__file__), Path("evaluation/gnn/box_hole_diagnostics.py"),
                   Path("evaluation/gnn/box_hole_geometry.py"), Path("graph/gated_mpnn.py"),
                   Path("graph/vertex_aggregation.py")]
        provenance = dict(instance_id=config["instance_id"], device=torch.cuda.get_device_name(0),
                          torch_version=torch.__version__, cuda_version=torch.version.cuda,
                          checkpoint_sha256={key: file_sha256(path) for key, path in config["checkpoints"].items()},
                          prediction_sha256=prediction_hashes, dataset_sha256=identity["dataset_sha256"],
                          geometry_sha256=file_sha256(config["validation_geometry"]),
                          fixture_states_sha256=file_sha256(state_path),
                          source_sha256={str(path): file_sha256(path) for path in sources},
                          validation_frames=len(identifiers), validation_parents=len(set(parents)),
                          raw_validation_frames_available=raw_validation_frames, reserved_test_frames_read=0,
                          simulator_executed=False, replay_checks=replay_checks)
        write_json(output / "summary.json", summary)
        write_json(output / "provenance.json", provenance)
        write_json(output / "effective_config.json", config)
        logger.record(**provenance)
        for path in output.iterdir():
            if path.is_file():
                logger.link_output(path)
        logger.log({"validation/frames": len(identifiers), "validation/parents": len(set(parents))})
        print(f"CUDA verification saved: {output}", flush=True)
