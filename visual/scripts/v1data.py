"""v1 데이터 로더 + 이미지 교란(patch / dropout) + episode bootstrap CI.

데이터: data/v1/<set>/ep{seed:03d}.npz (collect_v1.py, 포맷은 collect.py 와 같음 + policy/split/perturb/meta)
  set = dev | eval | dev_box_move | eval_box_move | eval_lighting ...
이미지 교란은 저장하지 않고 spec(data/v1/perturb_<kind>_<split>.json, seed 로 결정)으로 로드 시 적용한다.
  patch   : 물체(seed 짝/홀 → peg/box) 영역을 20 프레임(1 s) 동안 회색(128) 판으로 가림.
            판 = 해당 구간 GT bbox 합집합 + 10 px. depth 도 판 위치(영역 depth 중앙값 − 5 cm 평면)로 바꾸고,
            평가용 GT seg 는 판 안을 0(=안 보임)으로 만든다. `perturb_frames` (T,) bool 로 구간 표시.
  dropout : 20 프레임 동안 RGB=0, depth=0, GT seg=0 (카메라 끊김).
GT(seg, pose)는 평가/교란 생성에만 쓴다. 모델 입력은 `model_inputs()` 로만 꺼낸다 (rgb, depth, K, cam2world).
"""
import json
import os

import numpy as np

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = f"{ROOT}/data/v1"
CAM = "base_camera"
PERTURB_LEN = 20


def ep_list(set_name):
    d = f"{DATA}/{set_name}"
    return sorted(f[:-4] for f in os.listdir(d) if f.startswith("ep") and f.endswith(".npz"))


def split_of(set_name):
    return set_name.split("_")[0]


def make_spec(kind, set_name):
    """seed 로 결정되는 교란 spec. 처음 호출 시 GT 로 생성해 json 으로 저장, 이후 재사용."""
    path = f"{DATA}/perturb_{kind}_{set_name}.json"
    if os.path.exists(path):
        return json.load(open(path))
    spec = {}
    for ep in ep_list(set_name):
        d = np.load(f"{DATA}/{set_name}/{ep}.npz")
        seed = int(d["seed"])
        rng = np.random.default_rng(seed * 7 + (1 if kind == "patch" else 2))
        T = len(d["frame_idx"])
        seg = d[f"{CAM}/seg"]
        if kind == "patch":
            obj = "peg" if seed % 2 == 0 else "box"
            oid = int(d[f"{obj}_id"])
            vis = (seg == oid).reshape(T, -1).sum(1)
            # 물체가 구간 내내 보이는 시작점 후보 (없으면 가장 많이 보이는 구간)
            starts = [s for s in range(10, T - PERTURB_LEN - 5) if (vis[s:s + PERTURB_LEN] >= 200).all()]
            if not starts:
                starts = [int(np.argmax([vis[s:s + PERTURB_LEN].sum() for s in range(10, T - PERTURB_LEN - 5)])) + 10]
            s0 = int(rng.choice(starts))
            ys, xs = np.nonzero((seg[s0:s0 + PERTURB_LEN] == oid).any(0))
            H, W = seg.shape[1:]
            rect = [int(max(0, xs.min() - 10)), int(max(0, ys.min() - 10)), int(min(W, xs.max() + 11)), int(min(H, ys.max() + 11))]
            spec[ep] = dict(obj=obj, start=s0, end=s0 + PERTURB_LEN, rect=rect)
        else:
            s0 = int(rng.integers(15, T - PERTURB_LEN - 10))
            spec[ep] = dict(start=s0, end=s0 + PERTURB_LEN)
    json.dump(spec, open(path, "w"), indent=1)
    return spec


def load(set_name, ep, perturb=None, cams=(CAM,)):
    """npz → dict (선택한 카메라 key 만). perturb ∈ {None, 'patch', 'dropout'} 은 base_camera 에 적용."""
    z = np.load(f"{DATA}/{set_name}/{ep}.npz")
    keep = [k for k in z.files if "/" not in k or k.split("/")[0] in cams]
    d = {k: z[k] for k in keep}
    T = len(d["frame_idx"])
    d["perturb_frames"] = np.zeros(T, bool)
    if perturb:
        sp = make_spec(perturb, set_name)[ep]
        s, e = sp["start"], sp["end"]
        d["perturb_frames"][s:e] = True
        rgb, dep, seg = d[f"{CAM}/rgb"].copy(), d[f"{CAM}/depth"].copy(), d[f"{CAM}/seg"].copy()
        if perturb == "patch":
            x1, y1, x2, y2 = sp["rect"]
            for t in range(s, e):
                reg = dep[t, y1:y2, x1:x2]
                z0 = float(np.median(reg[reg > 0])) - 0.05 if (reg > 0).any() else 0.5
                rgb[t, y1:y2, x1:x2] = 128
                dep[t, y1:y2, x1:x2] = max(z0, 0.05)
                seg[t, y1:y2, x1:x2] = 0
        else:
            rgb[s:e] = 0
            dep[s:e] = 0
            seg[s:e] = 0
        d[f"{CAM}/rgb"], d[f"{CAM}/depth"], d[f"{CAM}/seg"] = rgb, dep, seg
        d["perturb_spec"] = sp
    return d


def model_inputs(d, cam=CAM):
    """모델 경로가 볼 수 있는 것만 (GT seg/pose 제외). input boundary."""
    return dict(rgb=d[f"{cam}/rgb"], depth=d[f"{cam}/depth"], K=d[f"{cam}/intrinsic"], cam2world=d[f"{cam}/cam2world"],
                timestamp=d["timestamp"])


# ---------------------------------------------------------------- statistics
def boot_ci(per_ep_vals, n=2000, seed=0, stat="pooled"):
    """episode bootstrap 95% CI.
    per_ep_vals: list of 1D arrays (episode 별 프레임 값, NaN 제외) → pooled 평균(프레임 가중)의 CI.
    stat='epmean' 이면 에피소드 평균의 평균."""
    vals = [np.asarray(v, float)[~np.isnan(np.asarray(v, float))] for v in per_ep_vals]
    vals = [v for v in vals if len(v)]
    if not vals:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n_ep=0)
    rng = np.random.default_rng(seed)
    E = len(vals)
    sums = np.array([v.sum() for v in vals])
    cnts = np.array([len(v) for v in vals])
    means = sums / cnts
    idx = rng.integers(0, E, (n, E))
    if stat == "pooled":
        bs = sums[idx].sum(1) / cnts[idx].sum(1)
        m = sums.sum() / cnts.sum()
    else:
        bs = means[idx].mean(1)
        m = means.mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(mean=round(float(m), 4), lo=round(float(lo), 4), hi=round(float(hi), 4), n_ep=E)


def paired_diff_ci(a_eps, b_eps, n=2000, seed=0):
    """같은 에피소드 집합에서 두 설정의 에피소드 평균 차이(a-b) bootstrap CI. 입력은 ep별 평균 dict."""
    keys = sorted(set(a_eps) & set(b_eps))
    diff = np.array([a_eps[k] - b_eps[k] for k in keys if not (np.isnan(a_eps[k]) or np.isnan(b_eps[k]))])
    if not len(diff):
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n_ep=0)
    rng = np.random.default_rng(seed)
    bs = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(mean=round(float(diff.mean()), 4), lo=round(float(lo), 4), hi=round(float(hi), 4), n_ep=len(diff),
                wins=int((diff > 0.01).sum()), losses=int((diff < -0.01).sum()))
