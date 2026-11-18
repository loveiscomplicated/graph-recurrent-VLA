"""Prepare or train controlled C2 continuations with validation-only guarded selection."""

from copy import deepcopy
from itertools import islice
import math
from pathlib import Path

import torch
import yaml

from data_collection.episode_dataset import file_sha256
from data_collection.five_mode_surface_audit import pinned_json
from data_collection.storage import write_json
from evaluation.gnn.five_mode_endpoint_diagnostics import predict_object21
from evaluation.gnn.five_mode_subtype_diagnostics import build_report, physical_errors
from graph.five_mode_hard_routing import load_surface_model
from training.gnn.checkpoint import save_checkpoint
from training.gnn.five_mode_experiment_data import build_data
from training.gnn.loss import object21_loss
from training.gnn.trainer import atomic_torch_save, resolve_config, run_epoch, seed_everything
from training.gnn.tasks import Object21Task
from utils.device_set import set_device
from utils.experiment_logger import ExperimentLogger


VARIANTS = {
    "peg_uniform": ("peg", "uniform"),
    "peg_focus": ("peg", "focus"),
    "peg_box_focus": ("peg_box", "focus"),
}


def validation_report(model, loader, groups, task):
    """Score fixed validation frames with the unchanged Object21 objective and geometry metrics."""
    model.eval()
    prediction, residual = predict_object21(loader, model)
    target = loader.target.cpu()
    report, _, _ = build_report(groups, physical_errors(prediction, target), loader.ids)
    terms = object21_loss(dict(object21=prediction, endpoint_residual=residual), target, **task.loss_settings)
    report["loss"] = float(terms["total"])
    return report


def selection_rejections(candidate, baseline, settings):
    """Gate C2-E improvement and reject overall, subtype or policy regression on validation."""
    reasons = []
    endpoint = "endpoint_error_mm"
    def mean(row, metric):
        return row["metrics"][metric]["mean"]
    if mean(candidate["overall"], endpoint) >= mean(baseline["overall"], endpoint):
        reasons.append("overall_endpoint_not_improved")
    target = "hole_position_error_mm"
    limit = mean(baseline["subtypes"]["C2-E"], target) * (1 - settings["minimum_e_hole_improvement"])
    if mean(candidate["subtypes"]["C2-E"], target) > limit:
        reasons.append("c2_e_hole_improvement_below_minimum")
    if mean(candidate["subtypes"]["C2-E"], endpoint) > mean(baseline["subtypes"]["C2-E"], endpoint):
        reasons.append("c2_e_endpoint_regressed")
    ratio = 1 + settings["maximum_degradation"]
    for metric in settings["guard_metrics"]:
        original = mean(baseline["overall"], metric)
        if mean(candidate["overall"], metric) > original * ratio + settings["absolute_guard_slack"]:
            reasons.append(f"overall/{metric}")
    for category, rows in (("subtype", baseline["subtypes"]), ("policy", baseline["overall"]["by_policy"]),
                           ("motion", baseline["overall"]["by_motion"])):
        candidates = candidate["subtypes"] if category == "subtype" else candidate["overall"][f"by_{category}"]
        for name, original in rows.items():
            if original["frames"] and mean(candidates[name], endpoint) > mean(original, endpoint) * ratio:
                reasons.append(f"{category}/{name}/endpoint")
    return reasons


def verify_prepared_baseline(current, prepared):
    """Allow small CPU/CUDA roundoff while keeping prepared validation thresholds fixed."""
    def compare(left, right, location):
        if isinstance(right, dict):
            if not isinstance(left, dict) or left.keys() != right.keys():
                raise ValueError(f"Prepared baseline structure differs at {location}")
            for key in right:
                compare(left[key], right[key], f"{location}/{key}")
        elif type(right) is float:
            # CUDA FP32 quaternion roundoff is amplified by acos in rotation quantiles.
            # These are diagnostic only; mean-based selection guards retain the tighter check.
            rotation_quantile = (
                "_rotation_error_deg/" in location
                and location.rsplit("/", 1)[-1] in ("median", "p95")
            )
            absolute_tolerance = 0.01 if rotation_quantile else 1e-3
            if not math.isclose(left, right, rel_tol=1e-4, abs_tol=absolute_tolerance):
                raise ValueError(f"Prepared baseline metric differs at {location}; re-prepare on this environment")
        elif left != right:
            raise ValueError(f"Prepared baseline identity differs at {location}")

    compare(current, prepared, "baseline")


def baseline_model(config, device):
    """Load the checksum-pinned epoch-35 model and lock architecture and loss settings."""
    model = load_surface_model(config["baseline"], device, "condition_2")
    checkpoint = torch.load(config["baseline"]["path"], map_location="cpu", weights_only=True)
    if checkpoint["config"]["model"] != config["model"] or checkpoint["config"]["task"] != config["task"]:
        raise ValueError("C2 experiments must preserve the baseline model capacity and Object21 loss")
    return model


def prepare(config, data, reference_model, reference_report, task, device, logger, output):
    """Emit executable arm configs and smoke-test current-data arms without selecting a model."""
    _, audit = pinned_json(config["data"]["surface_audit"])
    if (audit["config"]["manifest"]["sha256"] != config["data"]["manifest"]["sha256"]
            or audit["config"]["condition_index"]["sha256"] != config["data"]["condition_index"]["sha256"]):
        raise ValueError("Preparation surface audit describes a different release or condition index")
    ready_box = all(
        audit["counts"][split]["condition_2"]["box_points"] > 0
        and audit["counts"][split]["condition_2"]["episodes_with_object_ids"]
        == audit["counts"][split]["condition_2"]["episodes"]
        for split in ("train", "validation")
    )
    plan = {}
    for variant, (surface_mode, sampling_mode) in VARIANTS.items():
        arm = deepcopy(config)
        arm.update(phase="train", variant=variant, device=config["preparation"]["training_device"])
        arm["data"]["surface_mode"] = surface_mode
        arm["sampling"]["mode"] = sampling_mode
        arm["output"]["root"] = str(Path(config["training_output_root"]) / variant)
        arm["prepared_baseline"] = dict(path=str(output / "baseline_validation.json"),
                                        sha256=file_sha256(output / "baseline_validation.json"))
        path = output / f"{variant}.yaml"
        path.write_text(yaml.safe_dump(arm, sort_keys=False))
        logger.link_output(path)
        blocked = surface_mode == "peg_box" and not ready_box
        details = dict(config=str(path.resolve()), ready=not blocked)
        if blocked:
            details["reason"] = "No typed Box surface in current C2 train/validation; regenerate and re-audit a candidate release"
        else:
            if surface_mode == "peg_box":
                probe_config = deepcopy(arm)
                probe_config["device"] = str(device)
                # A typed cache alone is insufficient: compare its original inputs with the baseline.
                build_data(probe_config, torch.Generator().manual_seed(config["seed"]))
            data.train.sampling = arm["sampling"]
            details["sampling"] = data.train.sampling_summary()
            if surface_mode == "peg":
                # Probe only current-data arms. Do not present these updates as selected candidates.
                model = deepcopy(reference_model).requires_grad_(True)
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["optimizer"]["lr"],
                                              weight_decay=config["optimizer"]["weight_decay"])
                state = data.train.generator.get_state()
                original_size = data.train.batch_size
                try:
                    data.train.batch_size = config["preparation"]["smoke_batch_size"]
                    details["smoke"] = run_epoch(
                        model, islice(data.train, 1), task, device, optimizer,
                        config["training"]["gradient_clip_norm"],
                    )
                finally:
                    data.train.batch_size = original_size
                    data.train.generator.set_state(state)
        plan[variant] = details
    write_json(output / "experiment_plan.json", dict(
        baseline_checkpoint=config["baseline"], baseline_validation=reference_report,
        variants=plan, selection=config["selection"], test_samples_read=False,
        training_performed="one disposable CPU/GPU minibatch for each current-data arm only",
        baseline_replaced=False,
    ))
    logger.link_output(output / "experiment_plan.json", kind="metrics")
    print(f"Prepared experiments: {output / 'experiment_plan.json'}", flush=True)


def train(config, data, reference_model, baseline, task, device, generator, logger, output):
    """Train one arm; keep the external epoch-35 baseline if no candidate passes every guard."""
    model = deepcopy(reference_model).requires_grad_(True)
    optimizer_settings = {key: value for key, value in config["optimizer"].items() if key != "name"}
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_settings)
    scheduler = None
    if config["scheduler"]["name"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config["scheduler"]["step_size"], gamma=config["scheduler"]["gamma"],
        )
    selected = baseline
    selected_path = Path(config["baseline"]["path"]).resolve()
    selected_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    selected_epoch, stale = 0, 0
    history = []
    early = config["training"]["early_stopping"]
    for epoch in range(1, config["training"]["epochs"] + 1):
        sampling = data.train.sampling_summary()
        metrics = run_epoch(model, data.train, task, device, optimizer, config["training"]["gradient_clip_norm"])
        report = validation_report(model, data.validation, data.validation_groups, task)
        reasons = selection_rejections(report, baseline, config["selection"])
        score = report["overall"]["metrics"]["endpoint_error_mm"]["mean"]
        previous = selected["overall"]["metrics"]["endpoint_error_mm"]["mean"]
        improved = not reasons and score < previous - early["min_delta"]
        stale = 0 if improved else stale + 1
        if improved:
            selected, selected_epoch, selected_path = report, epoch, output / "best.pt"
            selected_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            atomic_torch_save(selected_path, dict(
                kind="best_weights", model=selected_state, config=config,
                dataset_sha256=data.identity["dataset_sha256"], validation_loss=report["loss"],
                validation=report, selected_epoch=epoch, selection_split="validation",
            ))
            logger.link_output(selected_path)
        if scheduler is not None:
            scheduler.step()
        save_checkpoint(
            output / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
            best_loss=selected["loss"], best_model_state=selected_state, config=config,
            dataset_sha256=data.identity["dataset_sha256"], device=device, shuffle_generator=generator,
            early_stopping_state=dict(stale_epochs=stale, selected_epoch=selected_epoch),
        )
        logger.link_output(output / "last.pt")
        history.append(dict(epoch=epoch, train=metrics, validation=report, sampling=sampling,
                            selection_rejections=reasons, selected_epoch=selected_epoch))
        write_json(output / "history.json", history)
        logger.link_output(output / "history.json", kind="metrics")
        logger.log(metrics, step=epoch, prefix="train")
        logger.log(report, step=epoch, prefix="validation")
        print(f"Epoch {epoch}: endpoint={score:.3f} mm, eligible={not reasons}, selected={selected_epoch}", flush=True)
        if epoch >= early["min_epochs"] and stale >= early["patience"]:
            break
    write_json(output / "selection.json", dict(
        split="validation", selected_checkpoint=str(selected_path), selected_epoch=selected_epoch,
        checkpoint_sha256=file_sha256(selected_path), inherited_baseline=selected_epoch == 0,
        validation=selected, test_samples_read=False,
    ))
    logger.link_output(output / "selection.json", kind="metrics")


def run(config):
    """Dispatch explicit preparation or training; never reserve test data for selection."""
    if config.get("phase") not in ("prepare", "train"):
        raise ValueError("phase must be prepare or train")
    if config.get("resume"):
        raise ValueError("This experimental runner does not yet support resume")
    effective = resolve_config(config)
    variant = effective["variant"]
    if variant not in VARIANTS or (effective["data"]["surface_mode"], effective["sampling"]["mode"]) != VARIANTS[variant]:
        raise ValueError("variant must match its explicit surface_mode and sampling.mode")
    if effective["optimizer"]["name"] != "adamw":
        raise ValueError("This controlled comparison requires the AdamW optimizer")
    settings = effective["selection"]
    if (not 0 < settings["minimum_e_hole_improvement"] < 1
            or not 0 <= settings["maximum_degradation"] < 1
            or not math.isfinite(settings["absolute_guard_slack"])
            or settings["absolute_guard_slack"] < 0):
        raise ValueError("Selection improvement/degradation must be valid fractions and guard slack finite >= 0")
    torch.set_num_threads(effective["torch_num_threads"])
    device = set_device(effective["device"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision(effective["matmul_precision"])
    seed_everything(effective["seed"], device)
    generator = torch.Generator().manual_seed(effective["seed"])
    reference_model = baseline_model(effective, device)
    task = Object21Task(**effective["task"]["settings"])
    with ExperimentLogger(team="graph", model=effective["model"]["name"], config=effective,
                          seed=effective["seed"], job_type=effective["phase"]) as logger:
        output = Path(effective["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        print("Loading pinned C2 train/validation; test members are excluded", flush=True)
        data = build_data(effective, generator)
        baseline = validation_report(reference_model, data.reference_validation, data.validation_groups, task)
        write_json(output / "baseline_validation.json", baseline)
        write_json(output / "data_identity.json", data.identity)
        for filename in ("baseline_validation.json", "data_identity.json"):
            logger.link_output(output / filename, kind="metrics")
        logger.link_output(effective["baseline"]["path"], checkpoint=True)
        logger.record(
            dataset_sha256=data.identity["dataset_sha256"], device=str(device), torch_version=str(torch.__version__),
            runner_sha256=file_sha256(Path(__file__)), test_samples_read=False,
            loader_sha256=data.identity["loader_sha256"],
            baseline_sha256=effective["baseline"]["sha256"],
        )
        if effective["phase"] == "prepare":
            prepare(effective, data, reference_model, baseline, task, device, logger, output)
        else:
            _, prepared = pinned_json(effective["prepared_baseline"])
            verify_prepared_baseline(baseline, prepared)
            train(effective, data, reference_model, prepared, task, device, generator, logger, output)
        if file_sha256(Path(effective["baseline"]["path"])) != effective["baseline"]["sha256"]:
            raise ValueError("Baseline checkpoint bytes changed during the experiment")
        return output
