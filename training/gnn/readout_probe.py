"""Controlled readout training on an unchanged, frozen baseline node encoder.

The probe reuses the existing pools and state heads. Its decoding equations are
checked against the original forward pass before training; it defines no new GNN.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.gnn.vertex_metrics import per_frame_errors, statistics
from graph.gated_mpnn import mlp
from graph.readout_probe import COMPONENTS, ReadoutProbe
from training.gnn.cached_data import build_data
from training.gnn.loss import object21_loss
from training.gnn.trainer import build_model, seed_everything
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


INPUT_KEYS = ("h", "confidence", "peg_mask", "target_mask", "peg_anchor", "hole_anchor")


def state_digest(model):
    """Hash tensor names and bytes to detect any frozen-weight mutation."""
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def capture_features(model, loader, device):
    """Capture actual inputs to both pools; targets stay in a separate cache field."""
    captured = {}

    def capture(name):
        def hook(module, arguments):
            captured[name] = tuple(value.detach() for value in arguments)
        return hook

    handles = [getattr(model, name).register_forward_pre_hook(capture(name))
               for name in ("peg_pool", "hole_pool")]
    batches = []
    identifiers = []
    try:
        with torch.inference_mode():
            for batch in loader:
                output = model(batch.graph.to(device))
                peg_h, confidence, peg_mask = captured["peg_pool"]
                hole_h, hole_confidence, target_mask = captured["hole_pool"]
                if not torch.equal(peg_h, hole_h) or not torch.equal(confidence, hole_confidence):
                    raise ValueError("Pool inputs must share exactly identical baseline node features")
                values = dict(h=peg_h, confidence=confidence, peg_mask=peg_mask,
                              target_mask=target_mask, peg_anchor=output["peg_anchor"],
                              hole_anchor=output["hole_anchor"], target=batch.targets["object21"],
                              pretrained_object21=output["object21"],
                              pretrained_endpoint_geom=output["endpoint_geom"],
                              pretrained_endpoint_residual=output["endpoint_residual"])
                values["counts"] = torch.stack([
                    (batch.graph.node_valid & (batch.graph.object_type == kind)).sum(1)
                    for kind in range(3)
                ], dim=1)
                values["xyz_world"] = batch.graph.xyz_world
                values["object_type"] = batch.graph.object_type
                values["tcp_world"] = batch.graph.tcp_xyz_world
                batches.append({key: value.cpu().clone() for key, value in values.items()})
                identifiers.extend(batch.sample_ids)
                if len(batches) % 100 == 0:
                    print(f"Frozen features: {len(identifiers)} frames", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if not batches or len(set(identifiers)) != len(identifiers):
        raise ValueError("Feature extraction requires nonempty unique sample IDs")
    width = max(batch["h"].shape[1] for batch in batches)
    padded_keys = {"h", "confidence", "peg_mask", "target_mask", "xyz_world", "object_type"}
    result = {}
    for key in batches[0]:
        values = []
        for batch in batches:
            value = batch[key]
            if key in padded_keys:
                padding = (0, width - value.shape[1]) if value.ndim == 2 else (0, 0, 0, width - value.shape[1])
                value = F.pad(value, padding, value=-1 if key == "object_type" else 0)
            values.append(value)
        result[key] = torch.cat(values)
    result["sample_ids"] = np.asarray(identifiers)
    return result


def check_parents(identity, manifest):
    """Reject train/validation/test parent leakage before extracting features."""
    parents = manifest["parent_by_episode"]
    by_split = {name: {parents[episode] for episode in episodes}
                for name, episodes in identity["splits"].items()}
    names = list(by_split)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if by_split[left] & by_split[right]:
                raise ValueError(f"Parent leakage between {left} and {right}")
    return {name: {"episodes": len(identity["splits"][name]), "parents": len(values)}
            for name, values in by_split.items()}


def feature_batch(data, indices, device):
    """Select GT-free model inputs and keep labels outside the input mapping."""
    return {key: data[key][indices].to(device) for key in INPUT_KEYS}


def verify_pretrained_replay(source, data, device):
    """Verify that cached-feature decoding reproduces all pretrained outputs."""
    probe = ReadoutProbe(source).to(device).eval()
    maximum = 0.0
    with torch.inference_mode():
        for start in range(0, len(data["target"]), 256):
            indices = slice(start, start + 256)
            output = probe(feature_batch(data, indices, device))
            for key in ("object21", "endpoint_geom", "endpoint_residual"):
                expected = data[f"pretrained_{key}"][indices].to(device)
                torch.testing.assert_close(output[key], expected, atol=1e-6, rtol=1e-5)
                maximum = max(maximum, float((output[key] - expected).abs().max()))
    return maximum


def make_probe(model_config, variant, seed, device):
    """Freshly initialize both heads/pools; shared heads use identical RNG draws."""
    settings = deepcopy(model_config)
    settings["readout"] = "multiseed" if variant.startswith("multiseed") else "attention"
    if variant == "multiseed_one":
        settings["readout_seeds"] = 1
    elif variant == "multiseed_eight":
        settings["readout_seeds"] = 8
    seed_everything(seed, device)
    fresh = build_model(settings, device)
    if variant == "attention_capacity":
        # Existing two-layer attention MLP with a wider intermediate layer.
        # Width 104 gives 6,865 parameters/pool vs multiseed's 6,880 at H=64.
        hidden = settings["hidden"]
        with torch.random.fork_rng(devices=[]):
            for pool in (fresh.peg_pool, fresh.hole_pool):
                pool.attention = mlp(hidden, 104, 1).to(device)
    return ReadoutProbe(fresh).to(device)


def loss_settings(config):
    """Use the source checkpoint's exact original Object21 objective scales."""
    settings = config["task"]["settings"]
    return {key: settings[key] for key in ("pos_unit_m", "consistency_weight", "size_units_m")}


def evaluate_probe(model, data, device, settings, batch_size=256):
    """Evaluate each saved validation frame once, without optimizer updates."""
    outputs = {key: [] for key in ("object21", "endpoint_geom", "endpoint_residual")}
    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data["target"]), batch_size):
            indices = slice(start, start + batch_size)
            output = model(feature_batch(data, indices, device))
            target = data["target"][indices].to(device)
            loss = object21_loss(output, target, **settings)["total"]
            total_loss += float(loss) * len(target)
            for key, value in output.items():
                outputs[key].append(value.cpu())
    output = {key: torch.cat(values) for key, values in outputs.items()}
    return total_loss / len(data["target"]), output


def train_probe(config, source_config, features, variant, seed, root):
    """Train only existing pool/head parameters for the fixed matched budget."""
    device = set_device(config.get("device", "auto"))
    model = make_probe(source_config["model"], variant, seed, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initial_head_hashes = {name: state_digest(getattr(model, name))
                           for name in ("peg_head", "hole_head", "endpoint_residual")}
    settings = loss_settings(source_config)
    epochs = config.get("epochs", source_config["training"]["epochs"])
    batch_size = config.get("batch_size", source_config["training"]["batch_size"])
    optimizer_config = source_config["optimizer"]
    if optimizer_config["name"] != "adamw":
        raise ValueError("This probe currently requires the audited AdamW source protocol")
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config["lr"],
                                 weight_decay=optimizer_config["weight_decay"])
    schedule = source_config["scheduler"]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=schedule["step_size"], gamma=schedule["gamma"])
    generator = torch.Generator().manual_seed(seed)
    effective = deepcopy(config)
    effective.update(variant=variant, seed=seed, source_training_config=source_config,
                     epochs=epochs, batch_size=batch_size, trainable_parameters=parameter_count)
    train = features["train"]
    validation = features["validation"]
    def compute_loss(inputs, target):
        return object21_loss(model(inputs), target, **settings)["total"]

    training_loss = compute_loss
    if config.get("compile", False) and device.type == "cuda":
        training_loss = torch.compile(compute_loss, dynamic=False, fullgraph=True)
    # A single detached feature cache is shared by every variant and training seed.
    train = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in train.items()}
    best_loss = float("inf")
    best_state = None
    history = []
    order_digest = hashlib.sha256()
    with ExperimentLogger(team="graph", config=effective, seed=seed,
                          model=f"frozen_readout_{variant}", job_type="train") as experiment:
        directory = root / variant / f"seed_{seed}" / experiment.run_id
        directory.mkdir(parents=True, exist_ok=False)
        experiment.link_output(config["checkpoint"], checkpoint=True)
        updates = 0
        for epoch in range(1, epochs + 1):
            started = time.monotonic()
            order = torch.randperm(len(train["target"]), generator=generator)
            order_digest.update(order.numpy().tobytes())
            train_loss = torch.zeros((), device=device)
            model.train()
            for indices in order.split(batch_size):
                indices = indices.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = training_loss(feature_batch(train, indices, device), train["target"][indices])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), source_config["training"]["gradient_clip_norm"],
                                               error_if_nonfinite=True)
                optimizer.step()
                train_loss += loss.detach() * len(indices)
                updates += 1
            validation_loss, _ = evaluate_probe(model, validation, device, settings)
            scheduler.step()
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_epoch = epoch
                torch.save(dict(model=best_state, config=effective, epoch=epoch,
                                feature_sha256=config["feature_sha256"]), directory / "best.pt")
            row = dict(epoch=epoch, updates=updates, train_loss=float(train_loss / len(train["target"])),
                       validation_loss=validation_loss, seconds=time.monotonic() - started)
            history.append(row)
            write_json(directory / "history.json", history)
            experiment.log(row, step=epoch)
            print(f"{variant} seed={seed} epoch={epoch}/{epochs} val={validation_loss:.6f} seconds={row['seconds']:.1f}", flush=True)
        final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(best_state)
        _, output = evaluate_probe(model, validation, device, settings)
        errors = per_frame_errors(output, validation["target"])
        np.savez_compressed(directory / "predictions.npz", sample_ids=validation["sample_ids"],
                            target=validation["target"].numpy(), counts=validation["counts"].numpy(),
                            **{key: value.numpy() for key, value in output.items()}, **errors)
        step_counts = sorted({int(state["step"].item()) for state in optimizer.state.values()})
        if step_counts != [updates]:
            raise AssertionError(f"Optimizer step counters differ from update budget: {step_counts}")
        torch.save(dict(model=final_state, optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), config=effective,
                        epoch=epochs, updates=updates, selected_epoch=best_epoch), directory / "last.pt")
        summary = dict(variant=variant, seed=seed, trainable_parameters=parameter_count,
                       best_epoch=best_epoch, best_validation_loss=best_loss, updates=updates,
                       optimizer_step_counts=step_counts, initial_head_sha256=initial_head_hashes,
                       training_order_sha256=order_digest.hexdigest(), feature_sha256=config["feature_sha256"],
                       checkpoint_sha256=file_sha256(directory / "best.pt"),
                       metrics={key: statistics(value) for key, value in errors.items()})
        write_json(directory / "summary.json", summary)
        write_json(directory / "effective_config.json", effective)
        for name in ("best.pt", "last.pt", "predictions.npz", "summary.json", "history.json", "effective_config.json"):
            experiment.link_output(directory / name, kind="checkpoint" if name.endswith('.pt') else "metrics")
        return dict(path=str(directory), **summary)


def train(config):
    """Extract one frozen cache, then run primary pairs before optional controls."""
    device = set_device(config.get("device", "auto"))
    torch.set_num_threads(config.get("cpu_threads", 2))
    payload = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    source_config = deepcopy(payload["config"])
    if source_config["model"].get("global_geometry", False) or source_config["model"].get("readout", "attention") != "attention":
        raise ValueError("Frozen source must be the baseline attention checkpoint")
    source = build_model(source_config["model"], device)
    source.load_state_dict(payload["model"], strict=True)
    source.requires_grad_(False).eval()
    initial_digest = state_digest(source)
    source_config["data"]["cache_dir"] = config["cache_dir"]
    source_config["data"]["num_workers"] = config.get("num_workers", 0)
    source_config["training"]["batch_size"] = config.get("extraction_batch_size", 256)
    bundle = build_data(source_config, torch.Generator().manual_seed(0))
    if bundle.identity["dataset_sha256"] != payload["dataset_sha256"]:
        raise ValueError("Probe cache identity differs from the source checkpoint dataset")
    # Restore the training batch size; extraction is deterministic inference only.
    source_config["training"]["batch_size"] = payload["config"]["training"]["batch_size"]
    manifest = json.loads(Path(config["parent_manifest"]).read_text())
    split_audit = check_parents(bundle.identity, manifest)
    root = Path(config["output"]["root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    feature_path = Path(config.get("feature_path", root / "frozen_features.pt"))
    if config.get("prepared_feature_sha256"):
        if file_sha256(feature_path) != config["prepared_feature_sha256"]:
            raise ValueError("Prepared feature cache checksum mismatch")
        audit = json.loads((feature_path.parent / "feature_audit.json").read_text())
        if audit["source_checkpoint_sha256"] != file_sha256(config["checkpoint"]):
            raise ValueError("Prepared features use a different frozen checkpoint")
        if audit["dataset_sha256"] != bundle.identity["dataset_sha256"]:
            raise ValueError("Prepared features use a different source dataset")
        features = torch.load(feature_path, map_location="cpu", weights_only=False)
        replay_difference = verify_pretrained_replay(source, features["validation"], device)
    else:
        if feature_path.exists():
            raise FileExistsError(f"Refusing to overwrite a prior probe cache: {feature_path}")
        features = {"train": capture_features(source, bundle.train, device),
                    "validation": capture_features(source, bundle.validation, device)}
        replay_difference = verify_pretrained_replay(source, features["validation"], device)
        torch.save(features, feature_path)
        for split, values in features.items():
            raw_keys = ("target", "counts", "xyz_world", "object_type", "tcp_world")
            np.savez_compressed(root / f"{split}_geometry.npz", sample_ids=values["sample_ids"],
                                **{key: values[key].numpy() for key in raw_keys})
    effective = deepcopy(config)
    effective["feature_sha256"] = file_sha256(feature_path)
    effective["source_checkpoint_sha256"] = file_sha256(config["checkpoint"])
    feature_audit = dict(
        source_checkpoint_sha256=effective["source_checkpoint_sha256"],
        feature_sha256=effective["feature_sha256"], dataset_sha256=bundle.identity["dataset_sha256"],
        inventory_sha256=bundle.identity["cache_inventory_sha256"], splits=split_audit,
        pretrained_replay_max_difference=replay_difference, encoder_state_sha256=initial_digest,
        source_all_parameters_frozen=all(not parameter.requires_grad for parameter in source.parameters()),
        source_all_gradients_absent=all(parameter.grad is None for parameter in source.parameters()),
    )
    if not config.get("prepared_feature_sha256"):
        write_json(root / "feature_audit.json", feature_audit)
    if config.get("prepare_only", False):
        return root
    if (root / "probe_runs.json").exists() or (root / "completion.json").exists():
        raise FileExistsError(f"Choose a new output.root for probe results: {root}")
    runs = []
    variants = config.get("variants", ["attention", "multiseed"])
    for variant in variants:
        for seed in config.get("seeds", [0, 1, 2]):
            runs.append(train_probe(effective, source_config, features, variant, seed, root))
            write_json(root / "probe_runs.json", runs)
    if config.get("capacity_control", True):
        for seed in config.get("seeds", [0, 1, 2]):
            runs.append(train_probe(effective, source_config, features, "attention_capacity", seed, root))
            write_json(root / "probe_runs.json", runs)
    if config.get("seed_count_controls", True):
        for variant in ("multiseed_one", "multiseed_eight"):
            runs.append(train_probe(effective, source_config, features, variant, 0, root))
            write_json(root / "probe_runs.json", runs)
    if state_digest(source) != initial_digest or any(parameter.grad is not None for parameter in source.parameters()):
        raise AssertionError("Frozen source mutated or received gradients")
    write_json(root / "completion.json", dict(complete=True, frozen_source_unchanged=True,
                                              encoder_gradients_absent=True, runs=runs))
    return root
