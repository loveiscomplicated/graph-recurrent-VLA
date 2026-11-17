"""Train a fresh GNN on checksummed frozen detector-only rollouts."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import random

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from graph.gated_mpnn import VertexEdgeGatedMPNN
from training.detected_corner_gnn import run_epoch
from utils.experiment_logger import ExperimentLogger, write


def comparable_source_config(config):
    """Keep all data, temporal and GNN settings that must match the cache run."""
    result = deepcopy(config)
    for name in ("trainer", "output", "rollout_source", "fresh_implementation_sha256"):
        result.pop(name, None)
    result["training"].pop("epochs")
    return result


def load_frozen_rollout(config):
    """Verify the prior checkpoint identifies these inputs and the same model."""
    source = config["rollout_source"]
    path = Path(source["checkpoint"]).resolve()
    if file_sha256(path) != source["checkpoint_sha256"]:
        raise ValueError("Source rollout checkpoint hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if comparable_source_config(checkpoint["config"]) != comparable_source_config(config):
        raise ValueError("Frozen rollout source settings do not match the requested fresh run")
    if checkpoint["epoch"] != 20:
        raise ValueError("Frozen rollout source must be the completed 20-epoch run")
    temporal = config["temporal"]
    if (checkpoint["temporal_sha256"] != temporal["sha256"]
            or file_sha256(temporal["checkpoint"]) != temporal["sha256"]):
        raise ValueError("Frozen temporal checkpoint hash mismatch")
    cache_path = path.parent / "rollout_inputs.pt"
    if file_sha256(cache_path) != checkpoint["rollout_cache_sha256"]:
        raise ValueError("Frozen rollout cache hash mismatch")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache["identity"] != checkpoint["identity"]:
        raise ValueError("Frozen rollout cache identity mismatch")
    return path, cache_path, checkpoint, cache


def save_atomic(path, state):
    """Replace a checkpoint only after its new contents are fully written."""
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def checkpoint_state(model, optimizer, config, source, run_id, epoch, shuffle):
    """Save the optimizer and random states as well as GNN weights."""
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                config=config, identity=source["identity"], run_id=run_id,
                gnn_initial_sha256=source["gnn_initial_sha256"],
                rollout_cache_sha256=source["rollout_cache_sha256"],
                temporal_sha256=source["temporal_sha256"], source_run_id=source["run_id"],
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                numpy_rng=shuffle.bit_generator.state, python_rng=random.getstate())


def quiet_epoch(*args, **kwargs):
    """Keep the original objective and metrics while omitting batch prints."""
    with redirect_stdout(io.StringIO()):
        return run_epoch(*args, **kwargs)


def train(config):
    """Initialize GNN and AdamW from scratch; reuse only frozen rollout inputs."""
    source_path, cache_path, source, cache = load_frozen_rollout(config)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This matched training recipe requires CUDA")
    if config["training"]["epochs"] != 40:
        raise ValueError("This recipe is for a fresh 40-epoch run")
    arrays = {name: torch.as_tensor(value, device=device) for name, value in cache["arrays"].items()
              if name != "sample_ids"}
    train_indices = torch.nonzero(~arrays["split"], as_tuple=True)[0]
    validation_indices = torch.nonzero(arrays["split"], as_tuple=True)[0]
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Train and validation splits must be nonempty")
    model = VertexEdgeGatedMPNN(**config["gnn"]).to(device)
    if model.extra_node_dim != 1:
        raise ValueError("GNN must use one detection-source feature")
    initial_hash = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes()
                                           for value in model.state_dict().values())).hexdigest()
    if initial_hash != source["gnn_initial_sha256"]:
        raise ValueError("Fresh GNN initialization differs from the matched original seed")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    shuffle = np.random.default_rng(seed)
    best, best_epoch, history = float("inf"), 0, []
    with ExperimentLogger(team="graph", model="detected_corner_gnn_fresh", config=config,
                          seed=seed) as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        logger.link_output(source_path, checkpoint=True)
        os.link(cache_path, output / "rollout_inputs.pt")
        logger.link_output(output / "rollout_inputs.pt")
        logger.record(**source["identity"], source_run_id=source["run_id"],
                      source_checkpoint_sha256=config["rollout_source"]["checkpoint_sha256"],
                      temporal_frozen=True, gt_ids_used_in_tracker=False, gnn_fresh_initialization=True,
                      optimizer_saved_each_epoch=True, rollout_cache_sha256=source["rollout_cache_sha256"],
                      gnn_initial_sha256=initial_hash)
        print("Fresh GNN and AdamW initialized; frozen detector-only rollout inputs reused", flush=True)
        for epoch in range(1, config["training"]["epochs"] + 1):
            order = torch.as_tensor(shuffle.permutation(len(train_indices)), device=device)
            train_metrics = quiet_epoch(model, arrays, train_indices[order], config, optimizer, epoch, logger, output)
            validation = quiet_epoch(model, arrays, validation_indices, config, None, epoch, logger, output)
            history.append(dict(epoch=epoch, train=train_metrics, validation=validation))
            state = checkpoint_state(model, optimizer, config, source, logger.run_id, epoch, shuffle)
            save_atomic(output / "last.pt", state)
            if validation["loss"] < best:
                best, best_epoch = validation["loss"], epoch
                save_atomic(output / "best.pt", state)
            write(output / "history.json", history)
            print(f"epoch={epoch} train_loss={train_metrics['loss']:.6f} val_loss={validation['loss']:.6f} "
                  f"best_epoch={best_epoch}", flush=True)
        summary = dict(epochs=len(history), best_epoch=best_epoch, best_validation_loss=best,
                       validation=history[best_epoch - 1]["validation"], optimizer_saved_each_epoch=True,
                       source_run_id=source["run_id"])
        write(output / "completed.json", summary)
        for name in ("best.pt", "last.pt", "history.json", "completed.json", "progress.json"):
            logger.link_output(output / name)
        logger.record(**summary)
        print(f"Completed: {output}; {summary}", flush=True)
        return str(output)
