from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from recurrent.common import load_config


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
        if self.k == 1:
            return self.net(x), None
        buf = state if state is not None else x[:, :1].expand(b, self.k - 1, d)
        seq = torch.cat([buf, x], 1)
        win = seq.unfold(1, self.k, 1).permute(0, 1, 3, 2).reshape(b, t, self.k * d)
        return self.net(win), seq[:, -(self.k - 1):].detach()


class RNNPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, cell="gru", hidden=256, layers=1, enc_hidden=128):
        super().__init__()
        self.encoder = mlp([obs_dim, enc_hidden], act_last=True)
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(enc_hidden, hidden, num_layers=layers, batch_first=True)
        self.head = mlp([hidden, enc_hidden, act_dim])

    def forward(self, x, state=None):
        y, state = self.rnn(self.encoder(x), state)
        return self.head(y), state


class CfCCell(nn.Module):
    def __init__(self, in_dim, hidden, backbone=128, backbone_layers=1):
        super().__init__()
        sizes = [in_dim + hidden] + [backbone] * backbone_layers
        self.backbone = nn.Sequential(*[m for i in range(backbone_layers) for m in (nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh())])
        self.ff1 = nn.Linear(backbone, hidden)
        self.ff2 = nn.Linear(backbone, hidden)
        self.time_a = nn.Linear(backbone, hidden)
        self.time_b = nn.Linear(backbone, hidden)

    def forward(self, x, h, ts):
        z = self.backbone(torch.cat([x, h], -1))
        gate = torch.sigmoid(self.time_a(z) * ts + self.time_b(z))
        return torch.tanh(self.ff1(z)) * (1 - gate) + gate * torch.tanh(self.ff2(z))


class CfCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128, backbone=128, backbone_layers=1, enc_hidden=128, ts=1.0):
        super().__init__()
        self.hidden = hidden
        self.ts = ts
        self.encoder = mlp([obs_dim, enc_hidden], act_last=True)
        self.cell = CfCCell(enc_hidden, hidden, backbone, backbone_layers)
        self.head = mlp([hidden, enc_hidden, act_dim])

    def forward(self, x, state=None):
        b, t, _ = x.shape
        u = self.encoder(x)
        h = state if state is not None else x.new_zeros(b, self.hidden)
        outs = []
        for i in range(t):
            h = self.cell(u[:, i], h, self.ts)
            outs.append(h)
        return self.head(torch.stack(outs, 1)), h


class RKNPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, latent_obs=32, num_basis=8, enc_hidden=128, head_hidden=128):
        super().__init__()
        m = latent_obs
        self.m = m
        self.encoder = mlp([obs_dim, enc_hidden, enc_hidden], act_last=True)
        self.w_mean = nn.Linear(enc_hidden, m)
        self.w_var = nn.Linear(enc_hidden, m)
        eye = torch.eye(2 * m).unsqueeze(0).repeat(num_basis, 1, 1)
        self.basis = nn.Parameter(eye + 0.01 * torch.randn(num_basis, 2 * m, 2 * m))
        self.coef = nn.Linear(2 * m, num_basis)
        self.log_q = nn.Parameter(torch.full((2 * m,), -3.0))
        self.log_var0 = nn.Parameter(torch.zeros(2 * m))
        self.head = mlp([4 * m, head_hidden, act_dim])

    def forward(self, x, state=None):
        b, t, _ = x.shape
        m = self.m
        e = self.encoder(x)
        w = self.w_mean(e)
        s = F.softplus(self.w_var(e)) + 1e-4
        if state is None:
            mu = x.new_zeros(b, 2 * m)
            var = self.log_var0.exp().expand(b, 2 * m)
        else:
            mu, var = state
        q = self.log_q.exp()
        outs = []
        for i in range(t):
            c = torch.softmax(self.coef(mu), -1)
            a = torch.einsum("bk,kij->bij", c, self.basis)
            mu = torch.einsum("bij,bj->bi", a, mu)
            var = torch.einsum("bij,bj->bi", a * a, var) + q
            gain = var[:, :m] / (var[:, :m] + s[:, i])
            mu = torch.cat([mu[:, :m] + gain * (w[:, i] - mu[:, :m]), mu[:, m:]], -1)
            var = torch.cat([(1 - gain) * var[:, :m], var[:, m:]], -1)
            outs.append(torch.cat([mu, torch.log(var)], -1))
        return self.head(torch.stack(outs, 1)), (mu, var)


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
    def __init__(self, obs_dim, act_dim, d_model=64, d_state=16, d_conv=4, expand=2, layers=2):
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


REGISTRY = {
    "mlp": MLPPolicy,
    "stack_mlp": StackMLPPolicy,
    "gru": lambda o, a, **kw: RNNPolicy(o, a, cell="gru", **kw),
    "lstm": lambda o, a, **kw: RNNPolicy(o, a, cell="lstm", **kw),
    "cfc": CfCPolicy,
    "rkn": RKNPolicy,
    "mamba": MambaPolicy,
}


def model_config(kind: str, overrides: dict | None = None) -> dict:
    cfg = dict(load_config("models.yaml")[kind])
    cfg.update(overrides or {})
    return cfg


def build(kind: str, obs_dim: int, act_dim: int, cfg: dict | None = None) -> nn.Module:
    if kind not in REGISTRY:
        raise ValueError(f"unknown model {kind}, choose from {list(REGISTRY)}")
    cfg = model_config(kind) if cfg is None else cfg
    return REGISTRY[kind](obs_dim, act_dim, **cfg)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def detach_state(state):
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    return type(state)(detach_state(s) for s in state)
