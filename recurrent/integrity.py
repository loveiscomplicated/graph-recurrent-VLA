"""Recurrent experiment provenance and strict local result validation."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch

from recurrent.common import RAW_KEYS, environment_record, sha256, versions
from recurrent.execution import atomic_json


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def identity_hash(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True).encode()).hexdigest()


def package_versions():
    return {k: v for k, v in versions().items() if k in (
        "python", "torch", "mani_skill", "sapien", "gymnasium", "numpy")}


def validate_demos(path, *, num_demos=None, start_seed=0, noise=0.02, frac=0.5, horizon=250):
    with np.load(path, allow_pickle=False) as data:
        lengths, seeds = data["lengths"], data["seeds"]
        if lengths.ndim != 1 or lengths.dtype.kind not in "iu" or len(lengths) < 1:
            raise ValueError("Invalid demo lengths")
        if num_demos is not None and len(lengths) != num_demos:
            raise ValueError(f"Expected {num_demos} demo episodes, got {len(lengths)}")
        if np.any(lengths < 1) or np.any(lengths > horizon):
            raise ValueError("Invalid demo episode length")
        if seeds.shape != lengths.shape or seeds.dtype.kind not in "iu" or np.any(seeds < start_seed) or np.any(np.diff(seeds) <= 0):
            raise ValueError("Invalid demo seeds")
        n = int(lengths.sum())
        for key, dim in {"act": 7, **{f"raw_{k}": d for k, d in RAW_KEYS.items()}}.items():
            array = data[key]
            if array.shape != (n, dim) or array.dtype.kind != "f" or not np.isfinite(array).all():
                raise ValueError(f"Invalid demo array {key}: shape/finite values")
        if data["perturbed"].shape != lengths.shape or data["perturbed"].dtype.kind != "b":
            raise ValueError("Invalid demo perturbation flags")
        meta = json.loads(str(data["meta"]))
        args = meta.get("args", {})
        expected = dict(num_demos=len(lengths), start_seed=start_seed, noise=noise,
                        perturb_frac=frac, horizon=horizon)
        if meta.get("source") != "expert" or any(args.get(k) != v for k, v in expected.items()):
            raise ValueError("Demo collection metadata mismatch or missing provenance")
        if meta.get("environment") != environment_record(horizon) or meta.get("versions") is None:
            raise ValueError("Demo environment metadata mismatch or missing versions")
        if any(meta["versions"].get(k) != v for k, v in package_versions().items()):
            raise ValueError("Demo package version mismatch")
        successes = meta.get("expert_success", [])
        if (len(successes) != 2 or successes[0] != len(lengths)
                or successes[1] < len(lengths) or seeds[-1] != start_seed + successes[1] - 1):
            raise ValueError("Demo successful episode/seed provenance mismatch")
        plans = meta.get("shift_plans", {})
        if set(plans) != {str(s) for s in seeds} or any(bool(flag) != (plans[str(seed)] is not None) for seed, flag in zip(seeds, data["perturbed"])):
            raise ValueError("Demo shift plan provenance mismatch")
    return sha256(path)


def training_identity(args, data_hash=None):
    from recurrent.common import resolve
    from recurrent.corruption import level_config
    from recurrent.features import FEATURE_LAYOUT
    from recurrent.models import model_config
    from recurrent.overrides import parse_overrides
    from recurrent.train import CONST_STD, STD_FLOOR

    path = resolve(args.data)
    with np.load(path, allow_pickle=False) as data:
        seeds = data["seeds"]
        order = np.random.default_rng(args.split_seed).permutation(len(seeds))
        n_val = max(1, int(round(len(seeds) * args.val_frac)))
        train = order[n_val:]
        if args.num_demos is not None:
            train = train[:args.num_demos]
        split = {"train": seeds[train].tolist(), "validation": seeds[order[:n_val]].tolist()}
    return canonical({
        "schema": 1, "model": args.model,
        "model_config": model_config(args.model, parse_overrides(args.set).get(args.model)),
        "train_level": args.level, "corruption": level_config(args.level, args.corruption),
        "seed": args.seed, "data_sha256": data_hash or sha256(path),
        "split": split, "split_seed": args.split_seed, "num_demos": args.num_demos,
        "val_frac": args.val_frac, "steps": args.steps, "batch": args.batch,
        "optimizer": {"name": "Adam", "lr": args.lr, "betas": [0.9, 0.999],
                      "eps": 1e-8, "weight_decay": 0, "amsgrad": False},
        "gradient_clip_norm": 1.0, "log_every": args.log_every,
        "normalization": {"std_floor": STD_FLOOR, "const_std": CONST_STD, "features": FEATURE_LAYOUT},
    })


def bind_identity(target, identity, *, existing=()):
    target = Path(target)
    if target.exists():
        if json.loads(target.read_text()) != canonical(identity):
            raise ValueError(f"Identity mismatch: {target}; preserve results and use a new out_dir")
    elif any(Path(p).exists() for p in existing):
        raise ValueError(f"Existing results have no identity: {target}; use a new out_dir")
    else:
        atomic_json(target, identity)


def training_complete(run, identity):
    run = Path(run)
    target = run / "identity.json"
    artifacts = [run / n for n in ("best.pt", "last.pt", "history.json")]
    if not target.exists():
        if run.exists() and any(p.name != ".writer.lock" for p in run.iterdir()):
            raise ValueError(f"Training identity missing: {run}")
        return False
    if json.loads(target.read_text()) != identity:
        raise ValueError(f"Training identity mismatch: {run}")
    # Every existing artifact must carry the same identity, even for incomplete runs.
    history = None
    checkpoints = {}
    for path in artifacts:
        if not path.exists():
            continue
        if path.suffix == ".json":
            history = json.loads(path.read_text())
            embedded = history.get("identity")
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            checkpoints[path.stem] = payload
            embedded = payload.get("identity")
        if embedded != identity:
            raise ValueError(f"Training artifact identity mismatch: {path}")
    if len(checkpoints) != 2 or history is None:
        return False
    config = history["config"]
    if (any(config.get(k) != identity[k] for k in ("model", "train_level", "seed"))
            or any(c.get("config") != config for c in checkpoints.values())):
        raise ValueError(f"Training config/checkpoint inconsistency: {run}")
    rows = history["history"]
    steps = [r["step"] for r in rows]
    expected_steps = list(range(identity["log_every"], identity["steps"] + 1, identity["log_every"]))
    if not expected_steps or expected_steps[-1] != identity["steps"]:
        expected_steps.append(identity["steps"])
    if steps != expected_steps:
        return False
    if any(not math.isfinite(r[k]) for r in rows for k in r if k != "step"):
        raise ValueError(f"Non-finite training history: {run}")
    best_row = min(rows, key=lambda r: r["val_optimization_loss"])
    best, last = checkpoints["best"], checkpoints["last"]
    if last["step"] != identity["steps"] or best["step"] != best_row["step"]:
        return False
    if history["summary"]["best_step"] != best["step"] or history["summary"]["best_val_loss"] != best_row["val_optimization_loss"]:
        return False
    if not {"optimization_loss", "action_mse_z", "command_mse_6d"} <= best["val"].keys():
        raise ValueError(f"Missing checkpoint validation metrics: {run}")
    if any(history["summary"].get("final_" + k) != rows[-1][k]
           for k in ("val_action_mse_z", "val_command_mse_6d")):
        return False
    for key, value in best["val"].items():
        if best_row.get("val_" + key) != value:
            return False
    if last["val"] != rows[-1]:
        return False
    for checkpoint in checkpoints.values():
        if (checkpoint.get("kind") != identity["model"] or checkpoint.get("cfg") != identity["model_config"]
                or checkpoint.get("train_level") != identity["train_level"] or checkpoint.get("seed") != identity["seed"]
                or not checkpoint.get("state_dict")):
            raise ValueError(f"Checkpoint metadata/state missing or inconsistent: {run}")
        if any(not torch.isfinite(t).all() for t in checkpoint["state_dict"].values()):
            raise ValueError(f"Non-finite checkpoint: {run}")
    return True


def archive_incomplete(run):
    paths = [p for p in Path(run).iterdir() if p.name not in {"identity.json", ".writer.lock", "incomplete"}]
    if paths:
        archive = Path(run) / "incomplete" / uuid.uuid4().hex
        archive.mkdir(parents=True)
        for path in paths:
            shutil.move(str(path), archive / path.name)


def expected_keys(identity):
    return {(n, l, s) for n, l, _ in identity["jobs"]
            for s in range(identity["start_seed"], identity["start_seed"] + identity["episodes"])}


def read_eval_rows(path, identity, *, repair=False):
    """Only an unterminated, syntactically incomplete final JSON record is recoverable."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows, keys = [], set()
    expected = expected_keys(identity)
    digest = identity_hash(identity)
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            fragment = line.decode("utf-8", errors="replace").rstrip()
            incomplete = (isinstance(exc, UnicodeDecodeError) and exc.reason == "unexpected end of data"
                          and exc.end == len(line) and line.lstrip().startswith(b"{"))
            if isinstance(exc, json.JSONDecodeError) and fragment.startswith("{"):
                tail = fragment[exc.pos:]
                incomplete = (exc.pos >= len(fragment) or exc.msg.startswith("Unterminated string")
                              or bool(re.fullmatch(r"(?:t|tr|tru|f|fa|fal|fals|n|nu|nul)", tail))
                              or (exc.msg == "Expecting ',' delimiter" and bool(re.fullmatch(r"(?:\.|[eE][+-]?)", tail)))
                              or (exc.msg.startswith("Invalid ") and bool(re.fullmatch(r"u[0-9a-fA-F]{0,3}", tail))))
            if i != len(lines) - 1 or line.endswith(b"\n") or not repair or not incomplete:
                raise ValueError(f"Corrupt evaluation record {path}:{i + 1}") from exc
            # Preserve the complete original before trimming a torn final append.
            backup = path.with_name(path.name + ".partial-" + uuid.uuid4().hex + ".bak")
            shutil.copy2(path, backup)
            raw = b"".join(lines[:i])
            with path.open("wb") as stream:
                stream.write(raw)
            break
        if not isinstance(row, dict):
            raise ValueError(f"Invalid evaluation record: {path}:{i + 1}")
        required = {"condition", "eval_level", "episode_seed", "ckpt", "model", "train_level", "train_seed",
                    "success", "steps", "shift_applied", "mean_action_delta", "mean_infer_ms",
                    "acquisition", "transport", "alignment", "failure_stage", "evaluation_identity",
                    "sim_backend", "initial_state_source"}
        if not required <= row.keys():
            raise ValueError(f"Missing evaluation fields: {path}:{i + 1}")
        if any(not isinstance(row[k], str) for k in ("condition", "eval_level", "ckpt", "model")) or type(row["episode_seed"]) is not int:
            raise ValueError("Invalid evaluation condition/model/seed fields")
        key = (row["condition"], row["eval_level"], row["episode_seed"])
        if key not in expected or key in keys:
            raise ValueError(f"Duplicate or unexpected evaluation episode: {path}:{i + 1}")
        if row["evaluation_identity"] != digest or row["sim_backend"] != identity["backend"] or row["initial_state_source"] != identity["initial_states"]:
            raise ValueError(f"Evaluation row identity mismatch: {path}:{i + 1}")
        if any(type(row[k]) is not bool for k in ("success", "shift_applied", "acquisition", "transport", "alignment")):
            raise ValueError("Invalid evaluation boolean")
        if type(row["steps"]) is not int or not 1 <= row["steps"] <= identity["horizon"] or type(row["episode_seed"]) is not int:
            raise ValueError("Invalid evaluation steps/seed")
        for k in ("mean_action_delta", "mean_infer_ms"):
            if row[k] is None and k == "mean_infer_ms" and identity["backend"] == "physx_cuda":
                continue
            if type(row[k]) not in (int, float) or not math.isfinite(row[k]) or row[k] < 0:
                raise ValueError(f"Invalid evaluation metric {k}={row[k]!r}: {path}:{i + 1} "
                                 f"condition={key[0]} level={key[1]} episode_seed={key[2]}")
        if row["failure_stage"] not in (None, "acquisition", "retention", "transport", "alignment", "insertion"):
            raise ValueError("Invalid failure stage")
        keys.add(key)
        rows.append(row)
    if raw and not raw.endswith(b"\n") and repair:
        with path.open("ab") as stream:
            stream.write(b"\n")
    return rows


def evaluation_complete(path, identity, *, repair=False):
    from recurrent.evaluate import summarize
    path = Path(path)
    target = path.with_suffix(".identity.json")
    summary_path = path.with_suffix(".summary.json")
    complete_path = path.with_suffix(".complete.json")
    if target.exists():
        if json.loads(target.read_text()) != identity:
            raise ValueError(f"Evaluation resume identity mismatch: {path}")
    elif any(p.exists() for p in (path, summary_path, complete_path)):
        raise ValueError(f"Evaluation identity missing: {path}")
    digest = identity_hash(identity)
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    complete = json.loads(complete_path.read_text()) if complete_path.exists() else None
    # Existing metadata must belong to this evaluation even when rows are partial.
    for value, source in ((summary, summary_path), (complete, complete_path)):
        if value is not None and value.get("evaluation_identity") != digest:
            raise ValueError(f"Evaluation metadata identity mismatch: {source}")
    rows = read_eval_rows(path, identity, repair=repair)
    if len(rows) != len(expected_keys(identity)) or summary is None or complete is None:
        return False
    if summary.get("summary") != summarize(rows):
        raise ValueError(f"Summary values mismatch: {summary_path}")
    if complete != {"evaluation_identity": digest, "rows": len(rows), "jsonl_sha256": sha256(path)}:
        raise ValueError(f"Completion metadata mismatch: {complete_path}")
    return True


def finish_evaluation(path, identity, payload):
    rows = read_eval_rows(path, identity)
    if len(rows) != len(expected_keys(identity)):
        raise ValueError(f"Incomplete evaluation: {path}")
    atomic_json(Path(path).with_suffix(".summary.json"), {**payload, "evaluation_identity": identity_hash(identity)})
    atomic_json(Path(path).with_suffix(".complete.json"), {
        "evaluation_identity": identity_hash(identity), "rows": len(rows), "jsonl_sha256": sha256(path)})
