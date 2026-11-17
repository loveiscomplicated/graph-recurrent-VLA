"""Matched Mamba/Transformer motion-before-GNN training and comparison launcher."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.temporal.corner_tracks import confirmed_eligibility, load_dataset, mask_plan
from evaluation.gnn.temporal.corner_tracking import paired_comparison, summarize
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.corner_tracking.model import ConstantVelocityMotion, CornerMotionModel
from graph.temporal.corner_tracking.tracker import AssociationCalibration, TrackingSettings
from training.gnn.tasks import Object21Task
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger, write
from .corner_tracking_calibration import bootstrap_gates, calibrate_uncertainty
from .corner_tracking_rollout import replay_episode
from .corner_tracking_execution import configure_execution


DEFAULTS = dict(
    device="cpu", torch_threads=1, seeds=[0, 1, 2], architectures=["mamba", "transformer"],
    model=dict(backend="torch_reference", d_model=128, layers=1, heads=4,
               d_state=16, d_conv=4, expand=2, coordinate_scale_m=0.05, age_scale_frames=32.0),
    gnn=dict(num_semantic_ids=16, hidden=64, layers=2, dropout=0.0,
             use_confidence=False, use_semantic_ids=False, global_geometry=True,
             readout="multiseed", readout_dim=16, readout_heads=4, extra_node_dim=5),
    tracking=asdict(TrackingSettings()),
    masking=dict(mode_probabilities=[0.25, 0.40, 0.25, 0.10],
                 length_probabilities=[0.20, 0.40, 0.30, 0.10]),
    calibration=dict(minimum_samples=32),
    evaluation=dict(ablations=True),
    execution=dict(compile=False, compile_mode="default", float32_matmul_precision="highest",
                   optimizer_fused=False, graph_batch_size=1, compile_recompile_limit=64),
    training=dict(pretrain_epochs=5, joint_epochs=20, motion_lr=0.001, gnn_lr=0.001,
                  weight_decay=0.0001, node_weight=0.1, gradient_clip_norm=1.0),
    loss=dict(pos_unit_m=0.01, consistency_weight=0.05,
              size_units_m=[0.04, 0.01, 0.01, 0.02]),
    output=dict(root="outputs/corner_tracking"),
)


def merge_settings(base, override):
    """Deep-merge team recipes, retaining one common comparison configuration."""
    result = deepcopy(base)
    for key, value in override.items():
        if (key == "data" and isinstance(value, dict) and "kind" in value
                and value["kind"] != result.get(key, {}).get("kind")):
            result[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_settings(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_config(config):
    """Resolve optional recipe inheritance and reject invalid budgets before I/O."""
    supplied = deepcopy(config)
    if "base_config" in config:
        base_path = Path(config["base_config"]).expanduser().resolve()
        base = yaml.safe_load(base_path.read_text())
        if not isinstance(base, dict) or "base_config" in base:
            raise ValueError("base_config must be a mapping without further inheritance")
        config = merge_settings(base, {key: value for key, value in config.items() if key != "base_config"})
    effective = merge_settings(DEFAULTS, config)
    effective["supplied_config"] = supplied
    if not isinstance(effective.get("data"), dict):
        raise ValueError("data must declare kind=prepared/root or kind=synthetic")
    if effective["data"].get("kind") not in ("prepared", "synthetic"):
        raise ValueError("data.kind must be prepared or synthetic")
    if effective["data"]["kind"] == "prepared" and not effective["data"].get("root"):
        raise ValueError("data.root must point to a prepared complete-episode dataset")
    seeds = effective["seeds"]
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("seeds must be unique integers in [0,2**32)")
    architectures = effective["architectures"]
    if (not isinstance(architectures, list) or not architectures
            or len(set(architectures)) != len(architectures)
            or any(name not in ("mamba", "transformer") for name in architectures)):
        raise ValueError("architectures must be unique mamba/transformer names")
    for key in ("pretrain_epochs", "joint_epochs"):
        value = effective["training"][key]
        if type(value) is not int or value < (1 if key == "joint_epochs" else 0):
            raise ValueError(f"training.{key} has an invalid epoch count")
    for key in ("motion_lr", "gnn_lr", "gradient_clip_norm"):
        if not np.isfinite(effective["training"][key]) or effective["training"][key] <= 0:
            raise ValueError(f"training.{key} must be positive and finite")
    for key in ("weight_decay", "node_weight"):
        if not np.isfinite(effective["training"][key]) or effective["training"][key] < 0:
            raise ValueError(f"training.{key} must be nonnegative and finite")
    if effective["gnn"].get("extra_node_dim") != 5 or effective["gnn"].get("dropout", 0) != 0:
        raise ValueError("Matched workflow requires five GNN provenance channels and dropout=0")
    for name in ("mode_probabilities", "length_probabilities"):
        values = np.asarray(effective["masking"][name])
        if values.shape != (4,) or np.any(values < 0) or not np.isclose(values.sum(), 1):
            raise ValueError(f"masking.{name} must contain four probabilities summing to one")
    for section in ("model", "training", "masking", "calibration", "evaluation", "execution"):
        unexpected = set(effective[section]) - set(DEFAULTS[section])
        if unexpected:
            raise ValueError(f"Unknown {section} settings: {sorted(unexpected)}")
    if type(effective["calibration"]["minimum_samples"]) is not int or effective["calibration"]["minimum_samples"] < 1:
        raise ValueError("calibration.minimum_samples must be a positive integer")
    if type(effective["torch_threads"]) is not int or effective["torch_threads"] < 1:
        raise ValueError("torch_threads must be a positive integer")
    if type(effective["evaluation"]["ablations"]) is not bool:
        raise ValueError("evaluation.ablations must be boolean")
    execution = effective["execution"]
    if any(type(execution[key]) is not bool for key in ("compile", "optimizer_fused")):
        raise ValueError("execution.compile and optimizer_fused must be boolean")
    if execution["compile_mode"] not in ("default", "max-autotune-no-cudagraphs"):
        raise ValueError("compile_mode must be default or max-autotune-no-cudagraphs")
    if execution["float32_matmul_precision"] not in ("highest", "high"):
        raise ValueError("float32_matmul_precision must be highest or high")
    if type(execution["graph_batch_size"]) is not int or execution["graph_batch_size"] < 1:
        raise ValueError("execution.graph_batch_size must be a positive integer")
    if type(execution["compile_recompile_limit"]) is not int or execution["compile_recompile_limit"] < 1:
        raise ValueError("execution.compile_recompile_limit must be a positive integer")
    if effective["model"]["backend"] not in ("torch_reference", "mamba_ssm"):
        raise ValueError("model.backend must be torch_reference or mamba_ssm")
    if effective["model"]["age_scale_frames"] != 32.0 or effective["model"]["coordinate_scale_m"] != 0.05:
        raise ValueError("This initial matched recipe fixes age_scale_frames=32 and coordinate_scale_m=0.05")
    TrackingSettings(**effective["tracking"])
    Object21Task(**effective["loss"])
    return effective


def load_episodes(config):
    """Load only development splits or clearly labelled synthetic fixtures."""
    if config["data"]["kind"] == "prepared":
        return load_dataset(config["data"]["root"])
    from data_collection.temporal.corner_tracking_synthetic import synthetic_dataset
    settings = {key: value for key, value in config["data"].items() if key != "kind"}
    return synthetic_dataset(**settings)


def derived_seed(seed, *parts):
    """Avoid Python's process-randomized hash when pairing masks and shuffle order."""
    return int(canonical_hash([seed, *parts])[:16], 16)


def plans_for(episodes, seed, tag, config, calibration):
    """Build identical masks from the common causal baseline eligibility table."""
    settings = TrackingSettings(**config["tracking"])
    return [mask_plan(episode, derived_seed(seed, tag, episode.name),
                      config["masking"]["mode_probabilities"], config["masking"]["length_probabilities"],
                      confirmed_eligibility(episode, calibration, settings))
            for episode in episodes]


def build_models(config, architecture, seed, device):
    """Identical GNN/encoder/head initialization despite unequal backbone sizes."""
    torch.manual_seed(seed)
    gnn = VertexEdgeGatedMPNN(**config["gnn"]).to(device)
    torch.manual_seed(seed + 1)
    motion = CornerMotionModel(architecture=architecture, **config["model"]).to(device)
    configure_execution(motion, gnn, config["execution"])
    return motion, gnn


def shared_initialization_digest(motion, gnn):
    """Fingerprint common weights for a machine-checkable paired comparison."""
    digest = hashlib.sha256()
    shared = {f"gnn.{key}": value for key, value in gnn.state_dict().items()}
    shared.update({key: value for key, value in motion.state_dict().items()
                   if key.startswith(("record_encoder.", "context_encoder.", "motion_head."))})
    for key, value in sorted(shared.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def implementation_hashes():
    """Record experiment modules and reused model/loss definitions."""
    root = Path(__file__).resolve().parents[3]
    paths = list((root / "graph/temporal/corner_tracking").glob("*.py"))
    paths.extend((root / "training/gnn/temporal").glob("corner_tracking*.py"))
    paths.extend(root / relative for relative in (
        "graph/gated_mpnn.py", "graph/temporal/causal_transformer.py", "graph/temporal/insertion.py",
        "data_collection/temporal/corner_tracks.py", "data_collection/temporal/corner_tracking_synthetic.py",
        "evaluation/gnn/temporal/corner_tracking.py", "training/gnn/loss.py",
        "evaluation/gnn/vertex_metrics.py"))
    return {str(path.relative_to(root)): file_sha256(path) for path in sorted(set(paths))}


def model_state(motion, gnn):
    """Detach a snapshot rather than retaining references to live parameter storage."""
    return dict(motion={key: value.detach().cpu().clone() for key, value in motion.state_dict().items()},
                gnn={key: value.detach().cpu().clone() for key, value in gnn.state_dict().items()})


def save_checkpoint(path, checkpoint):
    """Replace checkpoints atomically so a failed write preserves the previous one."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def restore_training(path, effective, motion, gnn, optimizer):
    """Resume an epoch boundary, allowing only a longer joint-training budget."""
    previous = torch.load(Path(path), map_location="cpu", weights_only=True)
    if previous.get("kind") != "corner_tracking_last":
        raise ValueError("resume_checkpoint must be corner_tracking last.pt")
    original = previous["config"]
    for key in ("dataset_sha256", "implementation_sha256", "architecture", "seed", "model", "gnn", "tracking",
                "masking", "calibration", "loss", "execution"):
        if original[key] != effective[key]:
            raise ValueError(f"Resume checkpoint differs in {key}")
    old_training = {key: value for key, value in original["training"].items() if key != "joint_epochs"}
    new_training = {key: value for key, value in effective["training"].items() if key != "joint_epochs"}
    if old_training != new_training or effective["training"]["joint_epochs"] < original["training"]["joint_epochs"]:
        raise ValueError("Resume must preserve training settings and may only extend joint_epochs")
    motion.load_state_dict(previous["motion"])
    gnn.load_state_dict(previous["gnn"])
    optimizer.load_state_dict(previous["optimizer"])
    torch.set_rng_state(previous["torch_rng"])
    random.setstate(previous["python_rng"])
    numpy_state = previous["numpy_rng"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    if torch.cuda.is_available() and previous["cuda_rng"]:
        torch.cuda.set_rng_state_all(previous["cuda_rng"])
    return previous


@torch.no_grad()
def evaluate_episodes(motion, gnn, episodes, plans, calibration, settings, loss_settings):
    motion.eval()
    gnn.eval()
    return [replay_episode(motion, gnn, episode, plan, calibration, settings, loss_settings)
            for episode, plan in zip(episodes, plans)]


def save_evaluation(directory, episodes, results, report, plans, logger):
    """Save per-frame predictions, causal decisions and the exact input-mask ledger."""
    directory.mkdir(parents=True, exist_ok=False)
    prediction_path = directory / "predictions.npz"
    np.savez_compressed(prediction_path,
                        sample_ids=np.asarray([frame.sample_id for episode in episodes for frame in episode.frames]),
                        prediction=np.concatenate([result["prediction"] for result in results]),
                        target=np.concatenate([result["target"] for result in results]),
                        endpoint_geom=np.concatenate([result["endpoint_geom"] for result in results]),
                        endpoint_residual=np.concatenate([result["endpoint_residual"] for result in results]))
    for filename, value in (("summary.json", report), ("mask_manifest.json", plans),
                            ("tracking.json", [dict(episode=episode.name, frames=result["trace"],
                                                   nodes=result["nodes"], recovery_frames=result["recovery_frames"])
                                               for episode, result in zip(episodes, results)])):
        write(directory / filename, value)
        logger.link_output(directory / filename)
    logger.link_output(prediction_path, kind="trajectory")


def train_one(config, groups, identity, architecture, seed):
    """Pretrain motion, then update motion and GNN jointly with equal epoch budgets."""
    device = set_device(config["device"])
    random.seed(seed)
    np.random.seed(seed)
    motion, gnn = build_models(config, architecture, seed, device)
    initial_digest = shared_initialization_digest(motion, gnn)
    settings = TrackingSettings(**config["tracking"])
    calibration, bootstrap = bootstrap_gates(groups["calibration"], settings.retirement_missing_frames - 1)
    calibration_plans = plans_for(groups["calibration"], seed, "calibration", config, calibration)
    validation_plans = plans_for(groups["validation"], seed, "validation", config, calibration)
    training = config["training"]
    optimizer = torch.optim.AdamW([
        dict(params=motion.parameters(), lr=training["motion_lr"]),
        dict(params=gnn.parameters(), lr=training["gnn_lr"]),
    ], weight_decay=training["weight_decay"],
       fused=config["execution"]["optimizer_fused"] if device.type == "cuda" else False)
    total_epochs = training["pretrain_epochs"] + training["joint_epochs"]
    effective = deepcopy(config)
    effective.update(architecture=architecture, seed=seed, dataset_sha256=identity["dataset_sha256"])
    effective["implementation_sha256"] = canonical_hash(implementation_hashes())
    resumed = None
    if config.get("resume_checkpoint"):
        resumed = restore_training(config["resume_checkpoint"], effective, motion, gnn, optimizer)
        if resumed["epoch"] >= total_epochs:
            raise ValueError("Resume budget has no remaining epochs; extend training.joint_epochs")
    with ExperimentLogger(team="graph", config=effective, seed=seed, model=f"corner_tracking_{architecture}") as logger:
        output = Path(config["output"]["root"]) / architecture / f"seed_{seed}" / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        logger.record(dataset_sha256=identity["dataset_sha256"],
                      shared_initialization_sha256=initial_digest,
                      context_policy="complete episodes; per-track state starts at birth; predict before record",
                      synthetic=identity.get("synthetic", False), bootstrap_calibration=bootstrap,
                      parameter_counts=dict(motion=sum(p.numel() for p in motion.parameters()),
                                            gnn=sum(p.numel() for p in gnn.parameters())),
                      versions=dict(torch=str(torch.__version__), numpy=np.__version__),
                      environment=dict(device=str(device), simulator_executed=False),
                      source_hashes=implementation_hashes())
        write(output / "effective_config.json", effective)
        write(output / "data_identity.json", identity)
        logger.link_output(output / "effective_config.json")
        logger.link_output(output / "data_identity.json")
        history, ledger = [], []
        best_score, best_snapshot, best_calibration = float("inf"), None, None
        best_epoch = None
        start_epoch = 0
        if resumed is not None:
            logger.link_output(config["resume_checkpoint"], checkpoint=True)
            logger.record(resume_checkpoint_sha256=file_sha256(config["resume_checkpoint"]))
            start_epoch = resumed["epoch"]
            history, ledger = resumed["history"], resumed["mask_ledger"]
            best_score, best_snapshot = resumed["best_score"], resumed["best_state"]
            best_epoch, best_calibration = resumed["best_epoch"], resumed["best_calibration"]
            calibration = AssociationCalibration.from_dict(resumed["calibration"])
        for epoch in range(start_epoch, total_epochs):
            phase = "pretrain" if epoch < training["pretrain_epochs"] else "joint"
            plans = plans_for(groups["train"], seed, f"train_epoch_{epoch}", config, calibration)
            ledger.append(dict(epoch=epoch, phase=phase, plans=plans))
            order = np.random.default_rng(derived_seed(seed, "shuffle", epoch)).permutation(len(groups["train"]))
            motion.train()
            gnn.train(phase == "joint")
            gnn.requires_grad_(phase == "joint")
            losses = []
            supervised_nodes = 0
            optimizer_steps = 0
            for index in order:
                optimizer.zero_grad(set_to_none=True)
                result = replay_episode(motion, gnn, groups["train"][index], plans[index],
                                        calibration, settings, config["loss"], collect_diagnostics=False,
                                        compute_object_loss=phase == "joint")
                loss = result["node_loss"]
                if phase == "joint":
                    loss = result["object_loss"] + training["node_weight"] * loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite corner tracking loss")
                if phase == "joint" or result["node_supervision_count"]:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(motion.parameters()) + list(gnn.parameters()),
                                                   training["gradient_clip_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    optimizer_steps += 1
                supervised_nodes += result["node_supervision_count"]
                losses.append(float(loss.detach().cpu()))
            if supervised_nodes == 0:
                raise ValueError("No certified track supervision in this epoch; check corner IDs, source alignment and association gates")
            calibration, calibration_report = calibrate_uncertainty(
                motion, gnn, groups["calibration"], calibration_plans, calibration, settings,
                config["calibration"]["minimum_samples"])
            results = evaluate_episodes(motion, gnn, groups["validation"], validation_plans,
                                        calibration, settings, config["loss"])
            report = summarize(results, calibration)
            row = dict(epoch=epoch + 1, phase=phase, train_loss=float(np.mean(losses)),
                       optimizer_steps=optimizer_steps, supervised_nodes=supervised_nodes,
                       validation=report, calibration=calibration_report)
            history.append(row)
            logger.log(row, step=epoch + 1)
            score = report["object21_loss"]
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite validation Object21 loss")
            if phase == "joint" and score < best_score:
                best_score = score
                best_snapshot = model_state(motion, gnn)
                best_calibration = calibration.as_dict()
                best_epoch = epoch + 1
                checkpoint = dict(kind="corner_tracking", config=effective, **best_snapshot,
                                  calibration=best_calibration, epoch=best_epoch,
                                  validation_score=best_score, shared_initialization_sha256=initial_digest)
                save_checkpoint(output / "best.pt", checkpoint)
                logger.link_output(output / "best.pt")
            last = dict(kind="corner_tracking_last", config=effective, **model_state(motion, gnn),
                        calibration=calibration.as_dict(), optimizer=optimizer.state_dict(),
                        epoch=epoch + 1, history=history, torch_rng=torch.get_rng_state(),
                        python_rng=random.getstate(),
                        numpy_rng=(np.random.get_state()[0], np.random.get_state()[1].tolist(),
                                   *np.random.get_state()[2:]),
                        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        best_score=best_score, best_state=best_snapshot, best_calibration=best_calibration,
                        best_epoch=best_epoch, mask_ledger=ledger,
                        shared_initialization_sha256=initial_digest)
            save_checkpoint(output / "last.pt", last)
            logger.link_output(output / "last.pt")
            write(output / "history.json", history)
            write(output / "training_masks.json", ledger)
            print(f"{architecture} seed={seed} {phase} epoch={epoch + 1}/{total_epochs} "
                  f"train={row['train_loss']:.5f} val_object21={score:.5f}", flush=True)
        logger.link_output(output / "history.json")
        logger.link_output(output / "training_masks.json")
        # A resumed run can retain an earlier best epoch without producing a new
        # improvement. Preserve that selected model in its new output directory.
        if not (output / "best.pt").exists():
            save_checkpoint(output / "best.pt", dict(kind="corner_tracking", config=effective,
                            **best_snapshot, calibration=best_calibration, epoch=best_epoch,
                            validation_score=best_score, shared_initialization_sha256=initial_digest))
            logger.link_output(output / "best.pt")
        motion.load_state_dict(best_snapshot["motion"])
        gnn.load_state_dict(best_snapshot["gnn"])
        calibration = AssociationCalibration.from_dict(best_calibration)
        evaluations = {}
        stress_plans = []
        for episode in groups["validation"]:
            plan = mask_plan(episode, derived_seed(seed, "end_stress", episode.name),
                             [0.0, 0.5, 0.25, 0.25], [0.0, 0.0, 0.0, 1.0],
                             confirmed_eligibility(episode, calibration, settings))
            if plan["stop"] > plan["start"]:
                plan.update(stop=len(episode.frames), actual_length=len(episode.frames) - plan["start"])
            stress_plans.append(plan)
        for label, plans in (("masked", validation_plans),
                             ("natural", [dict(mode="natural", start=0, stop=0) for _ in groups["validation"]]),
                             ("end_stress", stress_plans)):
            results = evaluate_episodes(motion, gnn, groups["validation"], plans, calibration, settings, config["loss"])
            report = summarize(results, calibration)
            save_evaluation(output / label, groups["validation"], results, report, plans, logger)
            evaluations[label] = report
        if config["evaluation"]["ablations"]:
            # These are interventions on this same trained GNN, not separately
            # trained baselines or replacements for the paired backbone study.
            variants = [("current_only", motion)]
            variants.extend((name, ConstantVelocityMotion(name, device))
                            for name in ("hold", "constant_velocity"))
            for name, baseline in variants:
                if name == "current_only":
                    baseline_calibration = AssociationCalibration(calibration.radii.copy())
                else:
                    baseline_calibration, _ = calibrate_uncertainty(
                        baseline, gnn, groups["calibration"], calibration_plans, calibration,
                        settings, config["calibration"]["minimum_samples"])
                results = evaluate_episodes(baseline, gnn, groups["validation"], validation_plans,
                                            baseline_calibration, settings, config["loss"])
                report = summarize(results, baseline_calibration)
                report["role"] = "same trained GNN; masked-input intervention, not independently trained baseline"
                save_evaluation(output / f"ablation_{name}", groups["validation"], results,
                                report, validation_plans, logger)
                evaluations[f"ablation_{name}"] = report
        return dict(architecture=architecture, seed=seed, run_id=logger.run_id, output=str(output),
                    checkpoint=str(output / "best.pt"), best_epoch=best_epoch,
                    shared_initialization_sha256=initial_digest,
                    validation_mask_sha256=canonical_hash(validation_plans),
                    training_mask_sha256=canonical_hash(ledger), evaluation=evaluations)


def compare(config):
    """Run the seed matrix sequentially with identical data, masks and shared weights."""
    effective = resolve_config(config)
    torch.set_num_threads(effective["torch_threads"])
    groups, identity = load_episodes(effective)
    rows = []
    with ExperimentLogger(team="graph", config=effective, model="corner_tracking_comparison",
                          job_type="comparison") as logger:
        output = Path(effective["output"]["root"]) / "comparisons" / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        for seed in effective["seeds"]:
            paired = []
            for architecture in effective["architectures"]:
                row = train_one(effective, groups, identity, architecture, seed)
                paired.append(row)
                rows.append(row)
                write(output / "comparison.json", dict(dataset_sha256=identity["dataset_sha256"], runs=rows))
            for key in ("shared_initialization_sha256", "validation_mask_sha256", "training_mask_sha256"):
                if len({row[key] for row in paired}) != 1:
                    raise RuntimeError(f"Paired comparison differs in {key}")
        aggregate = paired_comparison(rows, groups["validation"])
        write(output / "comparison.json", dict(dataset_sha256=identity["dataset_sha256"],
                                                runs=rows, aggregate=aggregate))
        logger.record(dataset_sha256=identity["dataset_sha256"], synthetic=identity.get("synthetic", False))
        logger.link_output(output / "comparison.json")
        print(f"Matched corner tracking comparison: {output / 'comparison.json'}", flush=True)
        return str(output / "comparison.json")


def train(config):
    """Run one architecture/seed through the same implementation as compare()."""
    effective = resolve_config(config)
    torch.set_num_threads(effective["torch_threads"])
    if len(effective["architectures"]) != 1 or len(effective["seeds"]) != 1:
        raise ValueError("train requires exactly one architecture and seed; use compare for a matrix")
    groups, identity = load_episodes(effective)
    return train_one(effective, groups, identity, effective["architectures"][0], effective["seeds"][0])


def evaluate_checkpoint(config):
    """Replay a saved paired model; reject a different dataset identity."""
    checkpoint_path = Path(config["checkpoint"]).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != "corner_tracking":
        raise ValueError("checkpoint must be a corner_tracking best.pt")
    effective = deepcopy(checkpoint["config"])
    effective["device"] = config.get("device", effective["device"])
    if "data" in config:
        effective["data"] = merge_settings(effective["data"], config["data"])
    groups, identity = load_episodes(effective)
    if identity["dataset_sha256"] != effective["dataset_sha256"]:
        raise ValueError("Evaluation dataset differs from checkpoint")
    torch.set_num_threads(effective["torch_threads"])
    motion, gnn = build_models(effective, effective["architecture"], effective["seed"], set_device(effective["device"]))
    motion.load_state_dict(checkpoint["motion"])
    gnn.load_state_dict(checkpoint["gnn"])
    calibration = AssociationCalibration.from_dict(checkpoint["calibration"])
    settings = TrackingSettings(**effective["tracking"])
    with ExperimentLogger(team="graph", config=config, seed=effective["seed"],
                          model=f"corner_tracking_{effective['architecture']}", job_type="evaluation") as logger:
        logger.link_output(checkpoint_path, checkpoint=True)
        logger.record(dataset_sha256=identity["dataset_sha256"],
                      checkpoint_sha256=file_sha256(checkpoint_path), effective_model_config=effective)
        output = Path(config["output"]["root"]) / logger.run_id
        plans = plans_for(groups["validation"], effective["seed"], "validation", effective, calibration)
        results = evaluate_episodes(motion, gnn, groups["validation"], plans, calibration, settings, effective["loss"])
        save_evaluation(output, groups["validation"], results, summarize(results, calibration), plans, logger)
        return str(output)
