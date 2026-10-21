"""E2: point cloud 추출의 torch GPU 버전 (pointcloud.py 의 extract 와 같은 단계).

1. mask erosion ERODE_PX (3×3, max_pool 로 구현). erosion 후 유효 점 < MIN_PTS 면 원래 mask
2. depth > 0 픽셀 역투영 ((u+0.5-cx)·z/fx 규약)
3. FPS_PRE(2048) 점 random 선다운샘플
4. SOR (k=16 이웃 평균거리 > mean + 2σ 제거; torch.cdist + topk)
5. FPS ≤ 1024 (triton kernel: 블록 하나가 cloud 하나를 처리, 1024 반복을 kernel 안에서 돈다)
6. cam2world → points_world
CPU 버전과 다른 점: 난수열이 달라 선택되는 점이 다르다(통계적으로 같은 처리).
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

ERODE_PX = 2
MIN_PTS = 50
SOR_K = 16
SOR_STD = 2.0
MAX_PTS = 1024
FPS_PRE = 2048


@triton.jit
def _fps_kernel(xyz_ptr, n_ptr, out_ptr, start_ptr, M: tl.constexpr, NMAX: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    n = tl.load(n_ptr + b)
    offs = tl.arange(0, BLOCK)
    valid = offs < n
    base = xyz_ptr + b * NMAX * 3
    x = tl.load(base + offs * 3 + 0, mask=valid, other=0.0)
    y = tl.load(base + offs * 3 + 1, mask=valid, other=0.0)
    z = tl.load(base + offs * 3 + 2, mask=valid, other=0.0)
    dist = tl.where(valid, float("inf"), -1.0)
    cur = tl.load(start_ptr + b)
    for i in range(M):
        tl.store(out_ptr + b * M + i, cur)
        sel = offs == cur
        cx = tl.sum(tl.where(sel, x, 0.0), 0)
        cy = tl.sum(tl.where(sel, y, 0.0), 0)
        cz = tl.sum(tl.where(sel, z, 0.0), 0)
        d = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz)
        dist = tl.where(valid, tl.minimum(dist, d), -1.0)
        cur = tl.argmax(dist, 0)


def fps_batch(clouds, m=MAX_PTS, gen=None):
    """clouds: list of (n_i,3) cuda float32 → list of index tensors (min(n_i, m),)."""
    out = [None] * len(clouds)
    todo = [i for i, c in enumerate(clouds) if len(c) > m]
    for i, c in enumerate(clouds):
        if len(c) <= m:
            out[i] = torch.arange(len(c), device=c.device)
    if not todo:
        return out
    # NMAX/BLOCK 는 constexpr 이라 값이 바뀌면 재컴파일된다 → FPS_PRE 이하면 항상 같은 크기로 패딩
    nmax = max(FPS_PRE, max(len(clouds[i]) for i in todo))
    BLOCK = triton.next_power_of_2(nmax)
    dev = clouds[todo[0]].device
    xyz = torch.zeros(len(todo), nmax, 3, device=dev, dtype=torch.float32)
    ns = torch.tensor([len(clouds[i]) for i in todo], device=dev, dtype=torch.int32)
    for j, i in enumerate(todo):
        xyz[j, :len(clouds[i])] = clouds[i]
    starts = (torch.rand(len(todo), device=dev, generator=gen) * ns).to(torch.int32)
    idx = torch.empty(len(todo), m, device=dev, dtype=torch.int32)
    _fps_kernel[(len(todo),)](xyz, ns, idx, starts, M=m, NMAX=nmax, BLOCK=BLOCK, num_warps=8)
    for j, i in enumerate(todo):
        out[i] = idx[j].long()
    return out


def erode(mask, px=ERODE_PX):
    """mask (H,W) bool cuda → 3×3 erosion px 회."""
    m = (~mask)[None, None].float()
    for _ in range(px):
        m = F.max_pool2d(m, 3, 1, 1)
    return m[0, 0] < 0.5


def sor(pts, k=SOR_K, std=SOR_STD):
    if len(pts) <= k:
        return pts
    d = torch.cdist(pts, pts)
    md = d.topk(k + 1, largest=False).values[:, 1:].mean(1)
    return pts[md <= md.mean() + std * md.std()]


class GpuPointCloud:
    def __init__(self, K, H, W, device="cuda", seed=0, max_pts=MAX_PTS, fps_pre=FPS_PRE, use_fps=True):
        self.dev = device
        K = torch.as_tensor(K, dtype=torch.float32, device=device)
        v, u = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        self.rx = (u + 0.5 - K[0, 2]) / K[0, 0]
        self.ry = (v + 0.5 - K[1, 2]) / K[1, 1]
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.max_pts, self.fps_pre, self.use_fps = max_pts, fps_pre, use_fps

    @torch.no_grad()
    def __call__(self, masks, depth, c2w):
        """masks: dict name → (H,W) bool cuda tensor, depth (H,W) cuda float (m), c2w (4,4).
        returns dict name → (points_cam (N,3), points_world (N,3)) cuda tensors."""
        depth = torch.as_tensor(depth, device=self.dev, dtype=torch.float32)
        c2w = torch.as_tensor(c2w, device=self.dev, dtype=torch.float32)
        dvalid = depth > 0
        names, clouds = [], []
        for name, mask in masks.items():
            m = mask
            if m.any():
                er = erode(m)
                if (er & dvalid).sum() >= MIN_PTS:
                    m = er
            sel = m & dvalid
            z = depth[sel]
            pts = torch.stack([self.rx[sel] * z, self.ry[sel] * z, z], 1)
            if len(pts) > self.fps_pre:
                pts = pts[torch.randperm(len(pts), device=self.dev, generator=self.gen)[:self.fps_pre]]
            pts = sor(pts)
            names.append(name)
            clouds.append(pts)
        if self.use_fps:
            idx = fps_batch(clouds, self.max_pts, self.gen)
        else:
            idx = [torch.randperm(len(c), device=self.dev, generator=self.gen)[:self.max_pts] for c in clouds]
        out = {}
        for name, c, ix in zip(names, clouds, idx):
            p = c[ix]
            out[name] = (p, p @ c2w[:3, :3].T + c2w[:3, 3])
        return out
