"""Run tracking, validation calibration and frozen Object21 training in order."""

import json
from pathlib import Path

from data_collection.episode_dataset import file_sha256
from main import load_config, load_trainer
from utils.experiment_logger import ExperimentLogger, write


def run_stage(name, config, output, state, logger):
    """Record a concrete effective recipe before invoking one team workflow."""
    recipe_path = output / f"{name}.yaml"
    write(recipe_path, config)
    state.update(phase=name, status="running")
    write(output / "progress.json", state)
    print(f"workflow phase={name} config={recipe_path}", flush=True)
    result = Path(load_trainer(config["trainer"])(config)).resolve()
    state.setdefault("stages", {})[name] = str(result)
    write(output / "progress.json", state)
    logger.link_output(recipe_path)
    return result


def run(config):
    """Complete both stages using validation only; never dispatch held-out test."""
    recipes = {name: load_config(path) for name, path in config["configs"].items()}
    overrides = config.get("execution", {})
    for recipe in recipes.values():
        recipe["device"] = overrides.get("device", recipe.get("device", "cuda:0"))
        if "implementation" in config:
            recipe["implementation"] = config["implementation"]
    tracker_recipe = recipes["train_tracker"]
    for name in ("num_workers", "graph_cache_root"):
        if name in overrides:
            tracker_recipe["data"][name] = overrides[name]
    object_recipe = recipes["train_object21"]
    if "release_manifest" in overrides:
        object_recipe["data"]["manifest"] = overrides["release_manifest"]

    with ExperimentLogger(
        team="graph", model="spatial_temporal_workflow", seed=tracker_recipe["seed"],
        config=config, job_type="workflow",
    ) as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        state = {"stages": {}, "test_episode_files_opened": 0}
        write(output / "effective_config.yaml", config)
        try:
            tracker_run = run_stage("train_tracker", tracker_recipe, output, state, logger)
            tracker_checkpoint = tracker_run / "best.pt"
            tracker_sha256 = file_sha256(tracker_checkpoint)
            for name in ("calibrate_association", "evaluate_tracker"):
                recipes[name].update(
                    checkpoint=str(tracker_checkpoint), checkpoint_sha256=tracker_sha256
                )
            object_recipe["tracker"] = {
                "checkpoint": str(tracker_checkpoint), "sha256": tracker_sha256,
            }
            state["tracker_checkpoint_sha256"] = tracker_sha256

            calibration_run = run_stage(
                "calibrate_association", recipes["calibrate_association"], output, state, logger
            )
            calibration = json.loads((calibration_run / "results.json").read_text())
            association = calibration["selected_association"]
            recipes["evaluate_tracker"]["association"] = association
            object_recipe["association"] = association
            state["association"] = association
            evaluation_run = run_stage(
                "evaluate_tracker", recipes["evaluate_tracker"], output, state, logger
            )
            evaluation = json.loads((evaluation_run / "results.json").read_text())
            track_limit = object_recipe["training"]["max_tracks_per_frame"]
            if evaluation["detector_only_online"]["max_final_tracks"] > track_limit:
                raise ValueError(
                    f"Full validation exceeds {track_limit} lifecycle tracks. "
                    f"Inspect {evaluation_run} before quadratic Stage B rollouts."
                )
            object_run = run_stage("train_object21", object_recipe, output, state, logger)
            object_checkpoint = object_run / "best.pt"
            object_sha256 = file_sha256(object_checkpoint)
            recipes["evaluate_object21"].update(
                checkpoint=str(object_checkpoint), checkpoint_sha256=object_sha256,
                rollout_cache=str(object_run / "frozen_tracker_rollouts.pt"),
            )
            state["object21_checkpoint_sha256"] = object_sha256
            state["rollout_cache_sha256"] = file_sha256(object_run / "frozen_tracker_rollouts.pt")
            run_stage("evaluate_object21", recipes["evaluate_object21"], output, state, logger)
        except Exception as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            write(output / "progress.json", state)
            raise
        state.update(phase="complete", status="completed")
        write(output / "progress.json", state)
        logger.link_output(output / "progress.json")
        logger.record(**state)
    return str(output)
