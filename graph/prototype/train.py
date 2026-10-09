"""Reproducible BC; validation selects checkpoints, test is never accessed."""
import argparse
from pathlib import Path
from time import perf_counter

import torch

from .common import config, DEFAULT_CONFIG, seed_all, versions, write_json
from .data import load_dataset, split_indices, sha256
from .graph_builder import GraphBuilder
from .models import make_policy, parameter_count
from .normalization import FeatureNormalizer
from .metrics import action_metrics
from .visualization import plot_curves


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["flat", "center", "surface"], required=True)
    p.add_argument("--dataset", type=Path, default=Path("outputs/graph/dataset.pt"))
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--max-batches", type=int, help="smoke only; incomplete epochs are explicitly recorded")
    args = p.parse_args()
    cfg = config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    if cfg["epochs"] < 1 or (args.max_batches is not None and args.max_batches < 1):
        p.error("epochs/max-batches must be positive")
    seed_all(cfg["seed"], cfg["threads"])
    data, geometry = load_dataset(args.dataset)
    dataset_cfg = data["manifest"]["config"]
    for key in ("peg_points", "hole_points", "sampling_seed", "control_mode"):
        if cfg[key] != dataset_cfg[key]:
            raise ValueError(f"dataset config mismatch: {key}")
    train_idx, val_idx = split_indices(data, "train"), split_indices(data, "validation")
    norm = FeatureNormalizer.fit(geometry.select(train_idx))
    model = make_policy(args.model, cfg)
    builder = GraphBuilder(cfg["surface_k"], cfg["interaction_k"], center=args.model == "center")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    batch_generator = torch.Generator().manual_seed(cfg["seed"])
    # Separate generator prevents parameter counts changing sample order.
    args.output.mkdir(parents=True, exist_ok=True)
    history, best, total_steps = [], float("inf"), 0
    dataset_hash = sha256(args.dataset)
    start = perf_counter()
    for epoch in range(1, cfg["epochs"]+1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=batch_generator)]
        epoch_start = perf_counter()
        train_predictions, train_targets = [], []
        for bi, indices in enumerate(order.split(cfg["batch_size"])):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            inputs = norm.apply(args.model, geometry.select(indices), builder)
            target = data["actions"][indices]
            optimizer.zero_grad(set_to_none=True)
            pred = model(inputs)
            loss = ((pred-target)**2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite BC loss")
            loss.backward()
            optimizer.step()
            train_predictions.append(pred.detach())
            train_targets.append(target)
            total_steps += 1
        model.eval()
        with torch.inference_mode():
            predictions = []
            for indices in val_idx.split(cfg["batch_size"]):
                predictions.append(model(norm.apply(args.model, geometry.select(indices), builder)))
            val_metrics = action_metrics(torch.cat(predictions), data["actions"][val_idx])
        train_metrics = action_metrics(torch.cat(train_predictions), torch.cat(train_targets))
        row = {"epoch": epoch, "train_mse": train_metrics["action_mse"], "val_mse": val_metrics["action_mse"],
               "train_metrics": train_metrics, "validation_metrics": val_metrics,
               "seconds": perf_counter()-epoch_start, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        scheduler.step()
        if row["val_mse"] < best:
            best = row["val_mse"]
            torch.save({"schema_version": 1, "model_name": args.model, "config": cfg,
                        "model_state": model.state_dict(), "normalizer": norm.as_dict(),
                        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch, "validation_mse": best, "dataset_sha256": dataset_hash,
                        "splits": data["manifest"]["splits"], "versions": versions(),
                        "parameters": parameter_count(model)}, args.output/"best.pt")
        write_json(args.output/"history.json", history)
        print(f"{args.model} epoch {epoch}/{cfg['epochs']}: train={row['train_mse']:.6f}, val={row['val_mse']:.6f}, {row['seconds']:.1f}s", flush=True)
    report = {"model": args.model, "config": cfg, "versions": versions(), "parameters": parameter_count(model),
              "training_seconds": perf_counter()-start, "optimizer_steps": total_steps,
              "train_samples": len(train_idx), "validation_samples": len(val_idx),
              "best_validation_mse": best, "dataset_sha256": dataset_hash,
              "smoke_max_batches": args.max_batches, "history": history}
    write_json(args.output/"training.json", report)
    plot_curves(history, args.output/"loss.png")
    print(f"best validation MSE={best:.6g}; parameters={report['parameters']}; total={report['training_seconds']:.1f}s")


if __name__ == "__main__":
    main()
