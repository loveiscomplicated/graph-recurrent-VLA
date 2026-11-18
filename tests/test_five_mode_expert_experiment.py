"""Check typed surface inputs, bounded sampling and validation-only model promotion."""

from collections import Counter
from copy import deepcopy
from functools import partial
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from data_collection.five_mode_surface_audit import surface_counts
from data_collection import five_mode_surface_audit as surface_audit
from data_collection.episode_dataset import canonical_hash, file_sha256
from evaluation.gnn.five_mode_subtype_diagnostics import SUBTYPES, build_report
from tests.test_five_mode_subtype_diagnostics import write_validation_fixture
from training.gnn import five_mode_expert_experiment as workflow
from training.gnn import five_mode_experiment_data as experiment_data
from training.gnn.packed_five_mode_data import pack_episode
from training.gnn.trainer import build_model, resolve_config
from utils.experiment_logger import ExperimentLogger


def test_audit_never_infers_box_identity_from_untyped_coordinates():
    surface = dict(frame_indices=np.arange(2), surface_offsets=np.asarray([0, 2, 3]),
                   surface_world_xyz=np.zeros((3, 3)))
    counts, typed = surface_counts(surface, 2)
    assert not typed
    assert counts["untyped"].sum() == 3
    assert counts["box"].sum() == 0
    surface["surface_object_id"] = np.asarray([0, 1, 1])
    counts, typed = surface_counts(surface, 2)
    assert typed
    np.testing.assert_array_equal(counts["box"], [1, 1])


def test_box_surface_opt_in_preserves_object_types_and_peg_only_default():
    episode = dict(seed=1, frames=1)
    tables = {
        "corners": dict(frame_indices=np.asarray([0]), corner_offsets=np.asarray([0, 2]),
                        corner_world_xyz=np.zeros((2, 3)), corner_object_id=np.asarray([0, 1])),
        "surface": dict(frame_indices=np.asarray([0]), surface_offsets=np.asarray([0, 2]),
                        surface_world_xyz=np.asarray([[1, 2, 3], [4, 5, 6]]),
                        surface_object_id=np.asarray([0, 1])),
        "targets": dict(frame_indices=np.asarray([0]), tcp=np.zeros((1, 3)), object21=np.zeros((1, 21))),
    }
    default = pack_episode(episode, tables, True)
    augmented = pack_episode(episode, tables, True, include_box_surface=True)
    assert default[1].tolist() == [[0, 1, 0]]
    assert augmented[1].tolist() == [[0, 1, 0, 1]]
    np.testing.assert_array_equal(default[0], augmented[0][:, :3])
    del tables["surface"]["surface_object_id"]
    with pytest.raises(ValueError, match="explicit surface_object_id"):
        pack_episode(episode, tables, True, include_box_surface=True)


def sampler(mode):
    episodes = []
    for seed in range(4):
        for policy in ("Fixed_Hole_expert", "Moving_Hole_expert"):
            episode = dict(seed=seed, policy_mode=policy)
            arrays = (np.zeros((5, 3, 3), dtype=np.float32), np.tile([0, 0, 1], (5, 1)),
                      np.zeros((5, 3), dtype=np.float32), np.zeros((5, 21), dtype=np.float32))
            episodes.append((episode, arrays, list(range(5))))
    groups = {name: np.arange(40) % 5 == index for index, name in enumerate(SUBTYPES)}
    return experiment_data.DispersedFrameBatches(
        episodes, "cpu", 8, torch.Generator().manual_seed(7), groups,
        dict(mode=mode, max_episode_frames_per_batch=1, max_seed_frames_per_batch=2),
    )


@pytest.mark.parametrize("mode", ["uniform", "focus"])
def test_sampler_has_equal_draw_budget_caps_and_rng_preserving_summary(mode):
    loader = sampler(mode)
    state = loader.generator.get_state().clone()
    summary = loader.sampling_summary()
    assert torch.equal(state, loader.generator.get_state())
    assert summary["draws"] == 48
    assert summary["maximum_exposures_per_frame"] == 2
    if mode == "focus":
        assert summary["subtype_draws"]["C2-E"] == 16
        assert all(summary["subtype_draws"][name] == 8 for name in SUBTYPES[:-1])
    count = len(loader)
    assert torch.equal(state, loader.generator.get_state())
    batches = list(loader.batch_rows())
    assert len(batches) == count
    exposures = Counter()
    for batch in batches:
        rows = batch.tolist()
        assert len(set(rows)) == len(rows)
        assert max(Counter(loader.episode_ids[row] for row in rows).values()) <= 1
        assert max(Counter(loader.seeds[row] for row in rows).values()) <= 2
        exposures.update(rows)
    assert set(exposures) == set(range(40))
    assert max(exposures.values()) == 2


def reports():
    ids = [f"{row}:{policy}:0" for row, policy in enumerate((
        "Fixed_Hole_expert", "Fixed_Hole_expert_random", "Fixed_Hole_expert_random_expert",
        "Fixed_Hole_random", "Moving_Hole_expert",
    ))]
    groups = {name: np.arange(5) == row for row, name in enumerate(SUBTYPES)}
    errors = dict(endpoint_error_mm=np.asarray([10, 10, 10, 10, 40.0]),
                  hole_position_error_mm=np.asarray([10, 10, 10, 10, 40.0]),
                  peg_position_error_mm=np.full(5, 5.0))
    baseline = build_report(groups, errors, ids)[0]
    errors["endpoint_error_mm"] = np.asarray([9, 9, 9, 9, 38.0])
    errors["hole_position_error_mm"] = np.asarray([10, 10, 10, 10, 38.0])
    return baseline, build_report(groups, errors, ids)[0]


def test_selection_requires_e_hole_gain_and_protects_subtypes_policies_and_motion():
    baseline, candidate = reports()
    settings = dict(minimum_e_hole_improvement=0.05, maximum_degradation=0.10,
                    absolute_guard_slack=0.01, guard_metrics=["peg_position_error_mm"])
    assert workflow.selection_rejections(candidate, baseline, settings) == []
    candidate["subtypes"]["C2-E"]["metrics"]["hole_position_error_mm"]["mean"] = 39
    assert "c2_e_hole_improvement_below_minimum" in workflow.selection_rejections(candidate, baseline, settings)
    candidate = reports()[1]
    candidate["subtypes"]["C2-B"]["metrics"]["endpoint_error_mm"]["mean"] = 20
    candidate["overall"]["by_policy"]["Fixed_Hole_random"]["metrics"]["endpoint_error_mm"]["mean"] = 20
    candidate["overall"]["by_motion"]["Fixed_Hole"]["metrics"]["endpoint_error_mm"]["mean"] = 20
    reasons = workflow.selection_rejections(candidate, baseline, settings)
    assert "subtype/C2-B/endpoint" in reasons
    assert "policy/Fixed_Hole_random/endpoint" in reasons
    assert "motion/Fixed_Hole/endpoint" in reasons


def test_prepared_baseline_tolerates_only_small_numeric_roundoff():
    baseline, _ = reports()
    current = deepcopy(baseline)
    current["overall"]["metrics"]["endpoint_error_mm"]["mean"] += 0.0001
    workflow.verify_prepared_baseline(current, baseline)
    current["overall"]["frames"] += 1
    with pytest.raises(ValueError, match="identity"):
        workflow.verify_prepared_baseline(current, baseline)


def test_box_augmentation_cannot_change_original_observations_or_targets():
    episode = dict(seed=1, policy_mode="Fixed_Hole_expert", split="train",
                   files=dict(corners=dict(sha256="a" * 64)))
    reference = [(episode, (np.zeros((1, 3)),), [0])]
    changed = deepcopy(reference)
    changed[0][1][0][0, 0] = 1
    with pytest.raises(ValueError, match="preserve"):
        experiment_data.verify_same_rows(reference, changed)
    changed = deepcopy(reference)
    changed[0][0]["files"]["corners"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="preserve"):
        experiment_data.verify_same_rows(reference, changed)


def test_surface_audit_opens_no_target_or_test_member(tmp_path, monkeypatch):
    manifest_path, index_path = write_validation_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][1]["files"] = manifest["records"][0]["files"]
    manifest_path.write_text(json.dumps(manifest))
    index = json.loads(index_path.read_text())
    index["condition_2"]["train"] = [dict(seed=2, policy_mode="Fixed_Hole_expert", frames=list(range(5)))]
    index_path.write_text(json.dumps(index))
    read = surface_audit.read_archive_member
    kinds = []

    def surface_only(archive, receipt, kind):
        assert kind == "surface"
        kinds.append(kind)
        return read(archive, receipt, kind)

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(surface_audit, "read_archive_member", surface_only)
    monkeypatch.setattr(surface_audit, "ExperimentLogger", partial(ExperimentLogger, root=tmp_path / "experiments"))
    output = surface_audit.audit(dict(
        splits=["train", "validation"], condition_2_frames=dict(train=5, validation=5),
        manifest=dict(path=str(manifest_path), sha256=file_sha256(manifest_path)),
        condition_index=dict(path=str(index_path), sha256=file_sha256(index_path)),
        output_root=str(tmp_path / "audit"),
    ))
    assert kinds == ["surface", "surface"]
    assert json.loads((output / "results.json").read_text())["identified_box_surface_available"] is False


def test_prepare_and_rejected_training_keep_baseline_and_never_read_test(tmp_path, monkeypatch):
    manifest_path, index_path = write_validation_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][1]["files"] = manifest["records"][0]["files"]
    manifest_path.write_text(json.dumps(manifest))
    index = json.loads(index_path.read_text())
    index["condition_2"]["train"] = [dict(seed=2, policy_mode="Fixed_Hole_expert", frames=list(range(5)))]
    index_path.write_text(json.dumps(index))
    config_path = Path(__file__).parents[1] / "graph/configs/prepare_five_mode_condition_2_experiments.yaml"
    config = yaml.safe_load(config_path.read_text())
    data = config["data"]
    data["manifest"] = dict(path=str(manifest_path), sha256=file_sha256(manifest_path))
    data["reference_manifest"] = deepcopy(data["manifest"])
    data["condition_index"] = dict(path=str(index_path), sha256=file_sha256(index_path))
    for split, seed in (("train", 2), ("validation", 1)):
        ids = [f"{seed}:Fixed_Hole_expert:{frame}" for frame in range(5)]
        data["cohorts"][split] = dict(frames=5, frame_ids_sha256=canonical_hash(ids))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(dict(
        config=dict(manifest=data["manifest"], condition_index=data["condition_index"]),
        counts={split: dict(condition_2=dict(box_points=0, episodes=1, episodes_with_object_ids=1))
                for split in ("train", "validation")},
    )))
    data["surface_audit"] = dict(path=str(audit), sha256=file_sha256(audit))
    config["output"]["root"] = str(tmp_path / "preparation")
    config["training_output_root"] = str(tmp_path / "training")
    config["preparation"].update(training_device="cpu", smoke_batch_size=2)
    effective = resolve_config(config)
    checkpoint_config = deepcopy(effective)
    checkpoint_config["data"]["factory"] = "training.gnn.packed_five_mode_data:build_data"
    checkpoint = tmp_path / "best.pt"
    torch.save(dict(kind="best_weights", model=build_model(effective["model"], torch.device("cpu")).state_dict(),
                    config=checkpoint_config), checkpoint)
    config["baseline"] = dict(path=str(checkpoint), sha256=file_sha256(checkpoint))
    original_hash = file_sha256(checkpoint)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(workflow, "ExperimentLogger", partial(ExperimentLogger, root=tmp_path / "experiments"))

    output = workflow.run(config)

    plan = json.loads((output / "experiment_plan.json").read_text())
    assert plan["variants"]["peg_uniform"]["ready"] is True
    assert plan["variants"]["peg_focus"]["ready"] is True
    assert plan["variants"]["peg_box_focus"]["ready"] is False
    assert not (output / "best.pt").exists()
    arm = yaml.safe_load((output / "peg_focus.yaml").read_text())
    arm["training"]["epochs"] = 1
    arm["training"]["early_stopping"]["min_epochs"] = 1
    monkeypatch.setattr(workflow, "selection_rejections", lambda *args: ["forced rejection"])
    trained = workflow.run(arm)
    selection = json.loads((trained / "selection.json").read_text())
    assert selection["inherited_baseline"] is True
    assert selection["selected_epoch"] == 0
    assert selection["test_samples_read"] is False
    assert file_sha256(checkpoint) == original_hash
    assert not (trained / "best.pt").exists()

    def unexpected_sample_read(*args, **kwargs):
        raise AssertionError("Unavailable Box data must fail before loading training samples")

    monkeypatch.setattr(experiment_data, "read_splits", unexpected_sample_read)
    blocked_arm = yaml.safe_load((output / "peg_box_focus.yaml").read_text())
    with pytest.raises(ValueError, match="Box surface unavailable"):
        workflow.run(blocked_arm)
