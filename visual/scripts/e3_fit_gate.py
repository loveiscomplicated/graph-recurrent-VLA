"""E3: dev shadow 후보로 물체별 로지스틱 회귀 게이트 학습 → 오답 통과율 ≤ 1% 가 되는 확률 임계값. 결과 results/v1/E3/gate.json"""
import json
import sys

import numpy as np
from scipy.optimize import minimize

FEATS = ("sim", "log_area_r", "log_size_r", "dist", "score")


def feat(c):
    a = c.get("area_r") or 1.0
    s = c.get("size_r") or 1.0
    return [c.get("sim") if c.get("sim") is not None else 0.0, np.log(max(a, 1e-3)), np.log(max(s, 1e-3)),
            c.get("dist") if c.get("dist") is not None else 0.3, c.get("score") or 0.0]


def fit(X, y, l2=1e-2):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    Z1 = np.c_[Z, np.ones(len(Z))]

    def nll(w):
        p = 1 / (1 + np.exp(-Z1 @ w))
        return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)) + l2 * (w[:-1] ** 2).sum()
    w = minimize(nll, np.zeros(Z1.shape[1]), method="L-BFGS-B").x
    return w, mu, sd


def prob(w, mu, sd, X):
    Z = (np.asarray(X) - mu) / sd
    return 1 / (1 + np.exp(-(np.c_[Z, np.ones(len(Z))] @ w)))


if __name__ == "__main__":
    C = json.load(open(sys.argv[1]))["cands"]
    out = {}
    for o in ("peg", "box"):
        cc = [c for c in C if c["obj"] == o]
        X = np.array([feat(c) for c in cc], float)
        y = np.array([c["gt_iou"] >= 0.5 for c in cc], float)
        w, mu, sd = fit(X, y)
        p = prob(w, mu, sd, X)
        neg = p[y == 0]
        thr = float(np.quantile(neg, 0.99))
        tpr = float((p[y == 1] >= thr).mean())
        # 에피소드별로 나눠 과적합 확인 (leave-one-episode-out)
        eps = sorted(set(c["ep"] for c in cc))
        lo_tp = lo_fp = 0
        for ep in eps:
            tr = np.array([c["ep"] != ep for c in cc])
            w2, mu2, sd2 = fit(X[tr], y[tr])
            p2 = prob(w2, mu2, sd2, X[~tr])
            t2 = float(np.quantile(prob(w2, mu2, sd2, X[tr])[y[tr] == 0], 0.99))
            lo_tp += int(((p2 >= t2) & (y[~tr] == 1)).sum())
            lo_fp += int(((p2 >= t2) & (y[~tr] == 0)).sum())
        print(f"{o}: n={len(y)} pos={int(y.sum())} thr={thr:.3f} 학습 TPR={tpr:.2f} FPR=0.01 | LOEO: 정답 통과 {lo_tp}/{int(y.sum())}, 오답 통과 {lo_fp}/{int((1 - y).sum())}")
        print("   w (std 단위):", dict(zip(FEATS + ("bias",), np.round(w, 2))))
        out[o] = dict(w=w.tolist(), mu=mu.tolist(), sd=sd.tolist(), thr=thr, feats=FEATS, tpr=tpr, loeo_tp=lo_tp, loeo_fp=lo_fp)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
