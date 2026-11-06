from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from recurrent.common import (
    ACTION_DIM,
    DATA_DIR,
    OUTPUT_ROOT,
    RAW_KEYS,
    environment_record,
    experiment_logger,
    resolve,
    sha256,
    versions,
)
from recurrent.corruption import corrupt_sequence, level_config
from recurrent.features import AGE_IDX, FEATURE_LAYOUT, FRESH_IDX, build_features
from recurrent.models import build, model_config, num_params
from recurrent.overrides import parse_overrides

STD_FLOOR = 0.01
CONST_STD = 1e-5


def load_episodes(path):
    data = np.load(path)
    lengths = data["lengths"]
    seeds = data["seeds"] if "seeds" in data else np.arange(len(lengths))
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    eps = []
    for s, n, sd in zip(starts, lengths, seeds):
        raw = {k: data[f"raw_{k}"][s:s + n] for k in RAW_KEYS}
        eps.append((raw, data["act"][s:s + n], int(sd)))
    meta = json.loads(str(data["meta"])) if "meta" in data else {}
    return eps, meta


def split(eps, num_demos, val_frac, split_seed):
    order = np.random.default_rng(split_seed).permutation(len(eps))
    n_val = max(1, int(round(len(eps) * val_frac)))
    val = [eps[i] for i in order[:n_val]]
    train = [eps[i] for i in order[n_val:]]
    if num_demos is not None:
        train = train[:num_demos]
    return train, val


def clean_features(raw):
    n = len(raw["tcp_pose"])
    return build_features(raw, np.ones(n, np.float32), np.zeros(n, np.float32))


def make_stats(train):
    obs = np.concatenate([clean_features(e[0]) for e in train])
    act = np.concatenate([e[1] for e in train])[:, :6]
    raw_std = obs.std(0)
    keep = (raw_std >= CONST_STD).astype(np.float32)
    mean, std = obs.mean(0), np.maximum(raw_std, STD_FLOOR)
    mean[[FRESH_IDX, AGE_IDX]] = 0.0
    std[[FRESH_IDX, AGE_IDX]] = 1.0
    keep[[FRESH_IDX, AGE_IDX]] = 1.0
    return {"obs_mean": mean, "obs_std": std, "obs_keep": keep, "act_mean": act.mean(0), "act_std": act.std(0) + 1e-6}


def normalize_obs(obs, stats):
    keep = stats.get("obs_keep", np.ones_like(stats["obs_mean"]))
    return ((obs - stats["obs_mean"]) / stats["obs_std"] * keep).astype(np.float32)


def dropped_features(stats):
    names = [f"{name}[{i}]" for name, n in FEATURE_LAYOUT for i in range(n)]
    return [n for n, k in zip(names, stats["obs_keep"]) if k == 0]


def episode_tensors(raw, act, stats, level_cfg, rng, cached=None):
    if level_cfg["name"] == "none":
        obs = cached if cached is not None else clean_features(raw)
    else:
        craw, fresh, age = corrupt_sequence(raw, level_cfg, rng)
        obs = build_features(craw, fresh, age)
    a = (act[:, :6] - stats["act_mean"]) / stats["act_std"]
    g = (act[:, 6] > 0).astype(np.float32)
    return normalize_obs(obs, stats), a.astype(np.float32), g


def pad(items, device):
    t = max(len(o) for o, _, _ in items)
    b = len(items)
    obs = np.zeros((b, t, items[0][0].shape[1]), np.float32)
    act = np.zeros((b, t, 6), np.float32)
    grip = np.zeros((b, t), np.float32)
    mask = np.zeros((b, t), np.float32)
    for i, (o, a, g) in enumerate(items):
        n = len(o)
        obs[i, :n], act[i, :n], grip[i, :n], mask[i, :n] = o, a, g, 1
    return [torch.from_numpy(x).to(device) for x in (obs, act, grip, mask)]


def loss_fn(pred, act, grip, mask):
    mse = ((pred[..., :6] - act) ** 2).mean(-1)
    bce = F.binary_cross_entropy_with_logits(pred[..., 6], grip, reduction="none")
    denom = mask.sum()
    return ((mse + bce) * mask).sum() / denom, (mse * mask).sum() / denom, (bce * mask).sum() / denom


def evaluate_loss(model, batch, act_std):
    obs, act, grip, mask = batch
    model.eval()
    with torch.no_grad():
        pred, _ = model(obs)
        total, mse, bce = loss_fn(pred, act, grip, mask)
        scale = torch.as_tensor(act_std, dtype=pred.dtype, device=pred.device)
        cmd = (((pred[..., :6] - act) * scale) ** 2).mean(-1)
        acc = ((pred[..., 6] > 0).float() == grip).float()
        denom = mask.sum()
        out = {
            "optimization_loss": total.item(),
            "action_mse_z": mse.item(),
            "gripper_bce": bce.item(),
            "gripper_accuracy": ((acc * mask).sum() / denom).item(),
            "command_mse_6d": ((cmd * mask).sum() / denom).item(),
        }
    model.train()
    return out


def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    data_path = resolve(args.data)
    level_cfg = level_config(args.level, args.corruption)
    eps, data_meta = load_episodes(data_path)
    train_eps, val_eps = split(eps, args.num_demos, args.val_frac, args.split_seed)
    stats = make_stats(train_eps)
    cached = [clean_features(e[0]) for e in train_eps] if level_cfg["name"] == "none" else [None] * len(train_eps)
    val_rng = np.random.default_rng([args.split_seed, 99])
    val = pad([episode_tensors(e[0], e[1], stats, level_cfg, val_rng) for e in val_eps], device)
    obs_dim = val[0].shape[-1]
    model_cfg = model_config(args.model, parse_overrides(args.set).get(args.model))
    model = build(args.model, obs_dim, ACTION_DIM, model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = num_params(model)
    out_dir = resolve(args.out_dir or OUTPUT_ROOT / "runs" / f"{args.model}_{args.level}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": args.model,
        "model_config": model_cfg,
        "train_level": args.level,
        "corruption": level_cfg,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "data": str(data_path),
        "data_source": data_meta.get("source"),
        "num_demos": args.num_demos,
        "val_frac": args.val_frac,
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "device": str(device),
        "threads": args.threads,
        "obs_dim": obs_dim,
        "std_floor": STD_FLOOR,
        "const_std": CONST_STD,
    }
    print(f"model {args.model} level {args.level} seed {args.seed} device {device} train_eps {len(train_eps)} val_eps {len(val_eps)} params {n_params}")
    with experiment_logger(config=config, seed=args.seed, model=args.model, job_type="train") as logger:
        logger.record(
            dataset=str(data_path),
            dataset_sha256=sha256(data_path),
            environment=environment_record(),
            versions=versions(),
            params=n_params,
            dropped_features=dropped_features(stats),
            val_episode_seeds=[e[2] for e in val_eps],
        )
        t0 = time.time()
        history = []
        best = (float("inf"), 0)
        best_path, last_path = out_dir / "best.pt", out_dir / "last.pt"

        def payload(step, metrics):
            return {
                "kind": args.model,
                "cfg": model_cfg,
                "obs_dim": obs_dim,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "stats": {k: np.asarray(v).tolist() for k, v in stats.items()},
                "train_level": args.level,
                "seed": args.seed,
                "params": n_params,
                "num_train_demos": len(train_eps),
                "data": str(data_path),
                "data_source": data_meta.get("source"),
                "step": step,
                "val": metrics,
                "config": config,
            }

        for step in range(1, args.steps + 1):
            idx = rng.choice(len(train_eps), size=min(args.batch, len(train_eps)), replace=False)
            batch = pad([episode_tensors(train_eps[i][0], train_eps[i][1], stats, level_cfg, rng, cached[i]) for i in idx], device)
            pred, _ = model(batch[0])
            loss, mse, bce = loss_fn(pred, *batch[1:])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % args.log_every == 0 or step == args.steps:
                v = evaluate_loss(model, val, stats["act_std"])
                elapsed = time.time() - t0
                row = {"step": step, "train_loss": loss.item(), "train_action_mse_z": mse.item(), "train_gripper_bce": bce.item(),
                       **{f"val_{k}": x for k, x in v.items()}, "elapsed_s": elapsed}
                history.append(row)
                logger.log({"train": {"optimization_loss": loss.item(), "action_mse_z": mse.item(), "gripper_bce": bce.item()},
                            "val": v, "time": {"elapsed_s": elapsed, "steps_per_s": step / max(elapsed, 1e-9)}}, step=step)
                if v["optimization_loss"] < best[0]:
                    best = (v["optimization_loss"], step)
                    torch.save(payload(step, v), best_path)
                print(f"step {step} loss {loss.item():.4f} val_mse {v['action_mse_z']:.4f} val_bce {v['gripper_bce']:.4f} {elapsed:.0f}s")
        train_time = time.time() - t0
        torch.save(payload(args.steps, history[-1]), last_path)
        summary = {"params": n_params, "train_time_s": train_time, "best_step": best[1], "best_val_loss": best[0],
                   "final_val_action_mse_z": history[-1]["val_action_mse_z"], "final_val_command_mse_6d": history[-1]["val_command_mse_6d"]}
        history_path = out_dir / "history.json"
        history_path.write_text(json.dumps({"config": config, "summary": summary, "history": history}, indent=2))
        logger.log({"final": summary}, step=args.steps)
        for path in (best_path, last_path):
            logger.link_output(path, kind="checkpoint")
        logger.link_output(history_path, kind="metrics")
    print(f"saved {out_dir} best_step {best[1]} time {train_time:.0f}s")
    return out_dir


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--level", default="none")
    p.add_argument("--corruption", default="corruption.yaml")
    p.add_argument("--data", default=str(DATA_DIR / "expert_demos.npz"))
    p.add_argument("--num-demos", type=int, default=None)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--out-dir", default=None)
    return p


if __name__ == "__main__":
    train(get_parser().parse_args())
