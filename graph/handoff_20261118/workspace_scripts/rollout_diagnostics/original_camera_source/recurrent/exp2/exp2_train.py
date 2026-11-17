import argparse
import json
import math
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO = "Graph-Recurrent-VLA/recurrent-bc-state36"
TYPES = ["mlp:step", "stack_mlp:step", "stack_mlp:chunk", "gru:step", "gru:chunk",
         "mamba:step", "mamba:chunk", "gru:step:nohf"]
CFG = {
    "mlp": dict(hidden=256, depth=4),
    "stack_mlp": dict(hidden=256, depth=3, k=8),
    "gru": dict(hidden=144, layers=1, enc_hidden=128),
    "mamba": dict(d_model=96, d_state=16, d_conv=4, expand=2, layers=2),
}
STD_FLOOR = 0.01
CONST_STD = 1e-5


def mlp(sizes, act_last=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or act_last:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, depth=3):
        super().__init__()
        self.net = mlp([obs_dim] + [hidden] * (depth - 1) + [act_dim])

    def forward(self, x, state=None):
        return self.net(x), None


class StackMLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, depth=3, k=8):
        super().__init__()
        self.k = k
        self.net = mlp([obs_dim * k] + [hidden] * (depth - 1) + [act_dim])

    def forward(self, x, state=None):
        b, t, d = x.shape
        buf = state if state is not None else x[:, :1].expand(b, self.k - 1, d)
        seq = torch.cat([buf, x], 1)
        win = seq.unfold(1, self.k, 1).permute(0, 1, 3, 2).reshape(b, t, self.k * d)
        return self.net(win), seq[:, -(self.k - 1):].detach()


class GRUPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=144, layers=1, enc_hidden=128):
        super().__init__()
        self.encoder = mlp([obs_dim, enc_hidden], act_last=True)
        self.rnn = nn.GRU(enc_hidden, hidden, num_layers=layers, batch_first=True)
        self.head = mlp([hidden, enc_hidden, act_dim])

    def forward(self, x, state=None):
        y, state = self.rnn(self.encoder(x), state)
        return self.head(y), state


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner, self.d_state, self.d_conv = d_inner, d_state, d_conv
        self.dt_rank = math.ceil(d_model / 16)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x, state=None):
        b, t, _ = x.shape
        xz = self.in_proj(self.norm(x))
        u, z = xz.chunk(2, -1)
        u = u.transpose(1, 2)
        if state is None:
            conv_buf = u.new_zeros(b, self.d_inner, self.d_conv - 1)
            h = u.new_zeros(b, self.d_inner, self.d_state)
        else:
            conv_buf, h = state
        seq = torch.cat([conv_buf, u], -1)
        u = F.silu(self.conv(seq)).transpose(1, 2)
        dt, bm, cm = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], -1)
        dt = F.softplus(self.dt_proj(dt))
        a = -self.A_log.exp()
        ys = []
        for i in range(t):
            da = torch.exp(dt[:, i, :, None] * a)
            h = da * h + dt[:, i, :, None] * bm[:, i, None, :] * u[:, i, :, None]
            ys.append((h * cm[:, i, None, :]).sum(-1))
        y = (torch.stack(ys, 1) + u * self.D) * F.silu(z)
        return x + self.out_proj(y), (seq[..., -(self.d_conv - 1):], h)


class MambaPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, d_model=96, d_state=16, d_conv=4, expand=2, layers=2):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand) for _ in range(layers)])
        self.head = mlp([d_model, 128, act_dim])

    def forward(self, x, state=None):
        y = self.encoder(x)
        states = []
        for i, blk in enumerate(self.blocks):
            y, s = blk(y, None if state is None else state[i])
            states.append(s)
        return self.head(y), states


REGISTRY = {"mlp": MLPPolicy, "stack_mlp": StackMLPPolicy, "gru": GRUPolicy, "mamba": MambaPolicy}


def build(kind, obs_dim, act_dim):
    return REGISTRY[kind](obs_dim, act_dim, **CFG[kind])


def q_conj(q):
    return q * np.array([1, -1, -1, -1], dtype=q.dtype)


def q_rot(q, v):
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def q_mul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def q_rotvec(q):
    q = np.where(q[..., :1] < 0, -q, q)
    s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., :1])
    return q[..., 1:] / np.maximum(s, 1e-9) * angle


def rot6(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y),
                     2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1)


def features(state36, hole_frame=True, r6=False):
    base = np.concatenate([state36[:, :3], rot6(state36[:, 3:7]), state36[:, 7:]], -1) if r6 else state36
    if not hole_frame:
        return base.astype(np.float32)
    obj = state36[:, 15:36]
    peg_q, tip_rel_hole, hole_q = obj[:, 3:7], obj[:, 7:10], obj[:, 13:17]
    hole_q = hole_q / np.linalg.norm(hole_q, axis=-1, keepdims=True)
    peg_q = peg_q / np.linalg.norm(peg_q, axis=-1, keepdims=True)
    tip_in_hole = q_rot(q_conj(hole_q), tip_rel_hole)
    rel_rot = q_rotvec(q_mul(q_conj(hole_q), peg_q))
    return np.concatenate([base, tip_in_hole, rel_rot], -1).astype(np.float32)


def fetch(data_dir):
    data_dir = Path(data_dir)
    found = sorted(data_dir.rglob("train.npz"))
    if not found:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, repo_type="dataset", local_dir=str(data_dir))
        found = sorted(data_dir.rglob("train.npz"))
    if not found:
        for tar in sorted(data_dir.rglob("*.tar")):
            with tarfile.open(tar) as bundle:
                bundle.extractall(data_dir / "extracted")
        found = sorted(data_dir.rglob("train.npz"))
    if not found:
        raise SystemExit("FAIL train.npz not found")
    return found[0].parent


def load(split_dir, splits, inp, hole_frame, kin=False, r6=False):
    eps = []
    for split in splits:
        with np.load(Path(split_dir) / f"{split}.npz", allow_pickle=False) as d:
            x = d[f"state36_{inp}"]
            a = d["expert_action"]
            off = d["episode_offsets"]
            counts = d["counts"] if "counts" in d.files else None
        if x.shape[1] != 36 or a.shape[1] != 7 or off[-1] != len(x):
            raise SystemExit(f"FAIL unexpected arrays in {split}")
        if kin:
            from kin_filter import filter_episodes
            x = filter_episodes(x, off, counts, kin)
        f = features(x, hole_frame, r6)
        eps += [(f[s:e], a[s:e]) for s, e in zip(off[:-1], off[1:])]
    return eps


def load_mix(split_dir, splits, hole_frame, kin, r6, alphas):
    levels = [[] for _ in alphas]
    for split in splits:
        with np.load(Path(split_dir) / f"{split}.npz", allow_pickle=False) as d:
            gt, g = d["state36_gt"], d["state36_transformer"]
            a, off = d["expert_action"], d["episode_offsets"]
            counts = d["counts"] if "counts" in d.files else None
        for k, al in enumerate(alphas):
            x = (gt + al * (g - gt)).astype(np.float32)
            if kin:
                from kin_filter import filter_episodes
                x = filter_episodes(x, off, counts, kin)
            f = features(x, hole_frame, r6)
            levels[k] += [(f[s:e], a[s:e]) for s, e in zip(off[:-1], off[1:])]
    return levels


def make_stats(eps):
    obs = np.concatenate([e[0] for e in eps])
    act = np.concatenate([e[1] for e in eps])[:, :6]
    raw_std = obs.std(0)
    return {"obs_mean": obs.mean(0), "obs_std": np.maximum(raw_std, STD_FLOOR),
            "obs_keep": (raw_std >= CONST_STD).astype(np.float32),
            "act_mean": act.mean(0), "act_std": act.std(0) + 1e-6}


def prepare(eps, stats):
    out = []
    for o, a in eps:
        on = ((o - stats["obs_mean"]) / stats["obs_std"] * stats["obs_keep"]).astype(np.float32)
        an = ((a[:, :6] - stats["act_mean"]) / stats["act_std"]).astype(np.float32)
        out.append((on, an, (a[:, 6] > 0).astype(np.float32)))
    return out


def pad(items, chunk, device):
    tm = max(len(o) for o, _, _ in items)
    b, d = len(items), items[0][0].shape[1]
    obs = np.zeros((b, tm, d), np.float32)
    act = np.zeros((b, tm, chunk, 6), np.float32)
    grip = np.zeros((b, tm, chunk), np.float32)
    mask = np.zeros((b, tm, chunk), np.float32)
    ks = np.arange(chunk)
    for i, (o, a, g) in enumerate(items):
        t = len(o)
        idx = np.arange(t)[:, None] + ks[None]
        valid = idx < t
        idx = np.minimum(idx, t - 1)
        obs[i, :t] = o
        act[i, :t] = a[idx]
        grip[i, :t] = g[idx]
        mask[i, :t] = valid
    return [torch.from_numpy(x).to(device) for x in (obs, act, grip, mask)]


def loss_fn(pred, act, grip, mask, chunk):
    b, t, _ = pred.shape
    pred = pred.view(b, t, chunk, 7)
    mse = ((pred[..., :6] - act) ** 2).mean(-1)
    bce = F.binary_cross_entropy_with_logits(pred[..., 6], grip, reduction="none")
    denom = mask.sum()
    return ((mse + bce) * mask).sum() / denom


def parse_type(name):
    parts = name.split(":")
    return parts[0], parts[1], not (len(parts) > 2 and parts[2] == "nohf")


def run_name(type_name, seed):
    return type_name.replace(":", "_") + f"_s{seed}"


def train_one(args):
    kind, head, hole_frame = parse_type(args.type)
    chunk = args.chunk if head == "chunk" else 1
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    kin = (json.loads(Path(args.kin_cfg).read_text()) if args.kin_cfg else True) if args.kin else False
    alphas = [float(v) for v in args.mix.split(",")] if args.mix else None
    if alphas:
        lv = load_mix(args.split_dir, args.splits.split(","), hole_frame, kin, args.rot6, alphas)
        stats = make_stats([e for l in lv for e in l])
        levels = [prepare(l, stats) for l in lv]
        eps = levels[0]
    else:
        raw = load(args.split_dir, args.splits.split(","), args.input, hole_frame, kin, args.rot6)
        stats = make_stats(raw)
        eps = prepare(raw, stats)
        levels = [eps]
    obs_dim = eps[0][0].shape[1]
    model = build(kind, obs_dim, 7 * chunk).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    out = Path(args.out) / run_name(args.type, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    params = sum(p.numel() for p in model.parameters())
    print(f"{args.type} seed {args.seed} episodes {len(eps)} obs_dim {obs_dim} params {params} device {device}", flush=True)
    t0 = time.time()
    history = []
    for step in range(1, args.steps + 1):
        idx = rng.choice(len(eps), size=min(args.batch, len(eps)), replace=False)
        lvl = rng.integers(len(levels), size=len(idx)) if len(levels) > 1 else np.zeros(len(idx), int)
        batch = pad([levels[l][i] for l, i in zip(lvl, idx)], chunk, device)
        pred, _ = model(batch[0])
        loss = loss_fn(pred, *batch[1:], chunk)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps:
            history.append({"step": step, "loss": loss.item(), "elapsed_s": time.time() - t0})
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    train_time = time.time() - t0
    torch.save({"kind": kind, "head": head, "chunk": chunk, "hole_frame": hole_frame, "input": args.input, "kin": kin, "rot6": args.rot6, "mix": args.mix,
                "cfg": CFG[kind], "obs_dim": obs_dim, "params": params, "seed": args.seed,
                "steps": args.steps, "batch": args.batch, "lr": args.lr, "splits": args.splits,
                "episodes": len(eps), "train_time_s": train_time,
                "stats": {k: np.asarray(v).tolist() for k, v in stats.items()},
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
               out / "last.pt")
    (out / "history.json").write_text(json.dumps({"train_time_s": train_time, "history": history}))
    print(f"DONE {args.type} seed {args.seed} {train_time:.0f}s", flush=True)


def seed_list(text):
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",")]


def run_all(args):
    split_dir = fetch(args.data_dir)
    types = TYPES if args.types == "all" else args.types.split(",")
    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    jobs = [(t, s) for s in seed_list(args.seeds) for t in types
            if not (out / run_name(t, s) / "last.pt").exists()]
    print(f"split_dir {split_dir} jobs {len(jobs)} parallel {args.jobs}", flush=True)
    running, times, t0 = [], {}, time.time()
    failed = 0
    while jobs or running:
        while jobs and len(running) < args.jobs:
            t, s = jobs.pop(0)
            cmd = [sys.executable, __file__, "one", "--type", t, "--seed", str(s), "--split-dir", str(split_dir),
                   "--out", str(out), "--steps", str(args.steps), "--batch", str(args.batch), "--lr", str(args.lr),
                   "--chunk", str(args.chunk), "--input", args.input, "--splits", args.splits,
                   "--threads", str(args.threads), "--log-every", str(args.log_every)] + (["--kin"] if args.kin else []) + (["--kin-cfg", args.kin_cfg] if args.kin_cfg else []) + (["--rot6"] if args.rot6 else []) + (["--mix", args.mix] if args.mix else [])
            log = open(out / "logs" / f"{run_name(t, s)}.log", "w")
            running.append((t, s, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), time.time(), log))
        time.sleep(2)
        for item in list(running):
            t, s, proc, start, log = item
            code = proc.poll()
            if code is None:
                continue
            log.close()
            running.remove(item)
            sec = time.time() - start
            status = "PASS" if code == 0 else f"FAIL code {code}"
            failed += code != 0
            times.setdefault(t, []).append(sec)
            print(f"{status} {t} seed {s} {sec:.0f}s total {time.time() - t0:.0f}s", flush=True)
    summary = {t: round(float(np.mean(v)), 1) for t, v in times.items()}
    print("mean_seconds_per_run " + json.dumps(summary), flush=True)
    print(f"wall_seconds {time.time() - t0:.0f}", flush=True)
    if failed:
        raise SystemExit(f"FAIL {failed} runs")
    (out / f"summary_{int(t0)}.json").write_text(json.dumps({"mean_seconds": summary, "wall_s": time.time() - t0,
                                                             "steps": args.steps, "jobs": args.jobs}))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("one", "run"):
        q = sub.add_parser(name)
        q.add_argument("--out", default="/work/exp2/runs")
        q.add_argument("--steps", type=int, default=3000)
        q.add_argument("--batch", type=int, default=8)
        q.add_argument("--lr", type=float, default=1e-3)
        q.add_argument("--chunk", type=int, default=16)
        q.add_argument("--input", default="transformer", choices=["transformer", "mamba", "gt"])
        q.add_argument("--splits", default="train,validation,test")
        q.add_argument("--threads", type=int, default=2)
        q.add_argument("--log-every", type=int, default=500)
        q.add_argument("--kin", action="store_true")
        q.add_argument("--rot6", action="store_true")
        q.add_argument("--mix", default="")
        q.add_argument("--kin-cfg", default="")
    one = sub.choices["one"]
    one.add_argument("--type", required=True)
    one.add_argument("--seed", type=int, required=True)
    one.add_argument("--split-dir", required=True)
    one.add_argument("--device", default="auto")
    run = sub.choices["run"]
    run.add_argument("--data-dir", default="/work/exp2/data")
    run.add_argument("--types", default="all")
    run.add_argument("--seeds", default="0-9")
    run.add_argument("--jobs", type=int, default=6)
    args = p.parse_args()
    train_one(args) if args.cmd == "one" else run_all(args)


if __name__ == "__main__":
    main()
