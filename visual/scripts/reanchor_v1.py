"""E3: Re-anchor v1 (streaming SAM2 위에서 동작, 모델 경로는 GT 를 읽지 않는다).

물체별 상태: logit, 안정 면적(최근 confident 10 프레임 면적 중앙값), 마지막 유효 3D 중심/중앙 depth, 기대 3D 크기(첫 프레임 cloud).
트리거 (cfg["triggers"] 로 켜고 끔)
  lost   : object_score_logit < 0 이 2 프레임 연속 (v0 conf)
  spike  : mask 면적 > 1.5 × 안정 면적
  split  : mask 내부 depth 가 3 cm 이상 간격으로 두 덩어리 (작은 쪽 ≥ 10%) → gripper/배경 번짐 의심
  jump   : mask 중앙 depth 가 한 프레임에 3 cm 이상 변함 (occluder 위 FP 의심; 물체는 프레임당 ≤ 1–2 cm 움직임)
  small  : (peg) 보이는 3D 길이 < 0.4 × 기대 길이 이면서 면적 < 0.6 × 안정 면적 (부분 mask 의심)
조치
  split/spike → (cleanup) depth 덩어리 중 마지막 중앙 depth 에 가까운 쪽만 남긴 mask 를 add_new_mask 로 보정 (검출기 없이)
  lost/jump/small/(cleanup 실패) → (detect) GDINO 후보 → SAM2 image decoder mask → 게이트 → 2 프레임 확인 → add_new_points_or_box
게이트 (cfg)
  area   : 후보 면적 ≥ r × 안정 면적
  size3d : 후보 3D 길이 / 기대 길이 ∈ [0.3, 1.6]
  cont   : 후보 3D 중심이 마지막 유효 중심에서 (5 cm + 1 cm × 소실 프레임) 이내
  reid   : DINOv2-S patch feature (mask 평균) cosine ≥ thr  | "sam": v0 SAM2 embedding (thr 0.75) | "none"
  tcp    : (ablation) grasp 후 TCP 기준 peg 예측 위치 3 cm 이내. 예측 = TCP × (마지막으로 잡힌 상태에서 본 TCP→peg 변환)
출력 valid: logit > 0 이고 jump/spike 이상 상태가 아닐 때. 이상 상태 프레임은 valid=False (빈 cloud + age 증가).
"""
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from stream_v1 import StreamTracker

OBJS = {1: "peg", 2: "box"}
PROMPTS = {"peg": "red and white block", "box": "cardboard box"}
MAX_BOX_FRAC = 0.25

DEFAULT = dict(triggers=("lost",), cleanup=False, reid="sam", reid_thr=0.75, gates=(), area_r=0.3, every=1, tcp=False,
               confirm=True, invalid_on_anomaly=False, shadow=False, lr=None)


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def mask_bbox(m):
    ys, xs = torch.nonzero(m, as_tuple=True)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


class Dino:
    def __init__(self, name="facebook/dinov2-small", size=448):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(name).cuda().eval()
        self.size = size
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]
        self.t, self.feat = None, None

    @torch.no_grad()
    def features(self, rgb, t):
        if self.t == t:
            return self.feat
        x = torch.from_numpy(np.ascontiguousarray(rgb)).cuda().permute(2, 0, 1).float()[None] / 255
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.m(pixel_values=x).last_hidden_state[0, 1:]
        g = self.size // 14
        self.feat = h.float().T.reshape(-1, g, g)
        self.t = t
        return self.feat

    def embed(self, feat, mask):
        mm = F.interpolate(mask.float()[None, None], size=feat.shape[-2:], mode="area")[0, 0]
        if mm.sum() < 0.3:
            return None
        return F.normalize((feat * mm).sum((1, 2)) / mm.sum(), dim=0)


class ObjState:
    def __init__(self, name):
        self.name = name
        self.areas = deque(maxlen=10)
        self.lost = 0
        self.age = 0
        self.last_c = None  # 마지막 유효 3D 중심 (world)
        self.last_z = None  # 마지막 중앙 depth
        self.exp_len = None
        self.bank = deque(maxlen=8)
        self.ref0 = None
        self.pending = None
        self.rel_tcp = None
        self.anomaly = False

    @property
    def stable_area(self):
        return float(np.median(self.areas)) if self.areas else None


class ReanchorV1:
    def __init__(self, predictor, imgp=None, gdino=None, dino=None, cfg=None, preprocess="gpu"):
        self.p = predictor
        self.tr = StreamTracker(predictor, preprocess=preprocess, prune=True)
        self.imgp, self.gdino, self.dino = imgp, gdino, dino
        self.cfg = dict(DEFAULT, **(cfg or {}))

    # ------------------------------------------------------------------ geometry helpers
    def _setup_geom(self, K, H, W):
        K = torch.as_tensor(K, dtype=torch.float32, device="cuda")
        v, u = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"), indexing="ij")
        self.rx = (u + 0.5 - K[0, 2]) / K[0, 0]
        self.ry = (v + 0.5 - K[1, 2]) / K[1, 1]

    @torch.autocast("cuda", enabled=False)
    def _pts(self, m, depth, c2w, maxn=1500):
        sel = m & (depth > 0)
        z = depth[sel]
        if len(z) < 30:
            return None, None
        p = torch.stack([self.rx[sel] * z, self.ry[sel] * z, z], 1)
        if len(p) > maxn:
            p = p[torch.randperm(len(p), device="cuda")[:maxn]]
            z = p[:, 2]
        pw = p @ c2w[:3, :3].T + c2w[:3, 3]
        return pw, z

    @staticmethod
    def _extent(pw):
        """(주축 방향 98-2 percentile 길이, 중심) — 작은 SVD 는 CPU 가 빠르다"""
        X = pw.float().cpu().numpy()
        c = X.mean(0)
        _, _, V = np.linalg.svd(X - c, full_matrices=False)
        sp = (X - c) @ V[0]
        lo, hi = np.percentile(sp, [2, 98])
        return float(hi - lo), torch.as_tensor(c, device="cuda")

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def _split(z):
        """depth 두 덩어리: 정렬 후 최대 간격 > 3 cm, 작은 쪽 ≥ 10% → (thr, near_frac)"""
        zs, _ = torch.sort(z)
        if len(zs) < 50:
            return None
        gaps = zs[1:] - zs[:-1]
        i = int(torch.argmax(gaps))
        if float(gaps[i]) < 0.03:
            return None
        frac = (i + 1) / len(zs)
        if min(frac, 1 - frac) < 0.1:
            return None
        return float(0.5 * (zs[i] + zs[i + 1]))

    # ------------------------------------------------------------------ API
    def start(self, rgb, depth, K, c2w, boxes):
        H, W = rgb.shape[:2]
        self._setup_geom(K, H, W)
        self.st = {n: ObjState(n) for n in OBJS.values()}
        self.events = []
        self.t = 0
        out = self.tr.start(rgb, {oid: boxes.get(n) for oid, n in OBJS.items()})
        self.active = {OBJS[o] for o in self.tr.obj_ids}
        return self._post(0, out, rgb, depth, c2w, None)

    def step(self, rgb, depth, c2w, tcp=None):
        t, out = self.tr.step(rgb)
        self.t = t
        return self._post(t, out, rgb, depth, c2w, tcp)

    # ------------------------------------------------------------------ core
    def _post(self, t, out, rgb, depth, c2w, tcp):
        cfg = self.cfg
        depth_g = torch.as_tensor(depth, device="cuda", dtype=torch.float32)
        c2w_g = torch.as_tensor(c2w, device="cuda", dtype=torch.float32)
        H, W = rgb.shape[:2]
        need_img = False
        res = {}
        for oid, name in OBJS.items():
            s = self.st[name]
            if name not in self.active:
                res[name] = dict(mask=torch.zeros(H, W, dtype=torch.bool, device="cuda"), logit=-10.0, valid=False, age=t + 1)
                continue
            m = out[oid]
            lg = self.tr.logit(oid, t)
            area = int(m.sum())
            pw, z = self._pts(m, depth_g, c2w_g) if area >= 30 else (None, None)
            trig = []
            s.lost = s.lost + 1 if lg < 0 else 0
            if "lost" in cfg["triggers"] and s.lost >= 2 and (s.lost - 2) % cfg["every"] == 0:
                trig.append("lost")
            zmed = float(torch.median(z)) if z is not None else None
            sa = s.stable_area
            if lg > 0 and z is not None:
                if "spike" in cfg["triggers"] and sa and area > 1.5 * sa:
                    trig.append("spike")
                if "split" in cfg["triggers"] and self._split(z) is not None:
                    trig.append("split")
                if "jump" in cfg["triggers"] and s.last_z is not None and abs(zmed - s.last_z) > 0.03:
                    trig.append("jump")
                if "small" in cfg["triggers"] and name == "peg" and s.exp_len and sa:
                    ln, _ = self._extent(pw)
                    if ln < 0.4 * s.exp_len and area < 0.6 * sa:
                        trig.append("small")
            if s.pending is not None and "confirm" not in trig:
                trig.append("confirm")
            res[name] = dict(mask=m, logit=lg, area=area, pw=pw, z=z, zmed=zmed, trig=trig)
            if any(x in trig for x in ("lost", "jump", "small", "confirm", "spike", "split")) or (lg > 2 and t % 5 == 0):
                need_img = True
        if need_img and self.imgp is not None:
            from track_v1 import set_image_features
            set_image_features(self.imgp, self.tr.state["cached_features"][t][1], (H, W))
        for oid, name in OBJS.items():
            if name not in self.active:
                continue
            r, s = res[name], self.st[name]
            trig = r["trig"]
            ev = None
            anomaly = any(x in trig for x in ("spike", "split", "jump"))
            # cleanup: depth 덩어리 정리
            if cfg["cleanup"] and not cfg["shadow"] and ("split" in trig or "spike" in trig) and r["z"] is not None:
                thr = self._split(r["z"])
                if thr is not None and s.last_z is not None:
                    dmap = depth_g
                    keep_near = s.last_z < thr  # 마지막 물체 depth 가 있는 쪽 덩어리를 남김
                    clean = r["mask"] & ((dmap < thr) if keep_near else (dmap >= thr)) & (dmap > 0)
                    if s.stable_area and clean.sum() >= 0.3 * s.stable_area:
                        tm = time.perf_counter()
                        newm = self.tr.add_mask(t, oid, clean)
                        r["mask"], r["area"] = newm, int(newm.sum())
                        r["pw"], r["z"] = self._pts(newm, depth_g, c2w_g)
                        r["zmed"] = float(torch.median(r["z"])) if r["z"] is not None else None
                        ev = dict(t=t, obj=name, kind="cleanup", trig=trig, ms=(time.perf_counter() - tm) * 1000,
                                  area_before=int(res[name]["area"]), area_after=r["area"])
                        anomaly = False
                        trig = [x for x in trig if x not in ("spike", "split")]
            if ev is None and any(x in trig for x in ("lost", "jump", "small", "confirm")) or (
                    ev is None and not cfg["cleanup"] and any(x in trig for x in ("spike", "split"))):
                ev = self._detect(t, oid, name, rgb, depth_g, c2w_g, trig, res, tcp)
                if ev.get("accepted"):
                    r["mask"] = ev.pop("mask_t")
                    r["area"] = int(r["mask"].sum())
                    r["pw"], r["z"] = self._pts(r["mask"], depth_g, c2w_g)
                    r["zmed"] = float(torch.median(r["z"])) if r["z"] is not None else None
                    r["logit"] = 1.0
                    s.lost = 0
                    anomaly = False
                else:
                    ev.pop("mask_t", None)
            if ev is not None:
                self.events.append(ev)
            # 상태 갱신
            ok = r["logit"] > 0 and r["pw"] is not None and not anomaly
            s.anomaly = anomaly
            r["valid"] = bool(ok or (r["logit"] > 0 and r["pw"] is not None and not cfg["invalid_on_anomaly"]))
            if ok:
                s.age = 0
                s.last_z = r["zmed"]
                ln, c = self._extent(r["pw"])
                s.last_c = c
                if s.exp_len is None:
                    s.exp_len = ln
                if r["logit"] > 2:
                    s.areas.append(r["area"])
                    if self.cfg["reid"] != "none" and (t == 0 or t % 5 == 0):
                        e = self._embed(rgb, t, r["mask"])
                        if e is not None:
                            if s.ref0 is None:
                                s.ref0 = e
                            else:
                                s.bank.append(e)
                if tcp is not None and name == "peg" and self.cfg["tcp"]:
                    Tt = tcp_mat(tcp)
                    Tp = np.eye(4)
                    Tp[:3, 3] = c.cpu().numpy()
                    s.rel_tcp = np.linalg.inv(Tt) @ Tp
            else:
                s.age += 1
            r["age"] = s.age
        return {n: dict(mask=res[n]["mask"], logit=res[n]["logit"], valid=res[n].get("valid", False), age=res[n].get("age", t + 1))
                for n in OBJS.values()}

    def _embed(self, rgb, t, mask):
        if self.cfg["reid"] == "dino":
            return self.dino.embed(self.dino.features(rgb, t), mask)
        if self.cfg["reid"] == "sam":
            from track_v1 import pool_embed
            return pool_embed(self.imgp, mask.cpu().numpy())
        return None

    def _detect(self, t, oid, name, rgb, depth_g, c2w_g, trig, res, tcp):
        cfg, s = self.cfg, self.st[name]
        H, W = rgb.shape[:2]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", enabled=False):
            boxes, scores = self.gdino(rgb, PROMPTS[name])
        g_ms = (time.perf_counter() - t0) * 1000
        other = "box" if name == "peg" else "peg"
        ob = mask_bbox(res[other]["mask"]) if other in res and res[other].get("valid") else None
        cands = [(b, sc) for b, sc in zip(boxes, scores) if (b[2] - b[0]) * (b[3] - b[1]) < MAX_BOX_FRAC * H * W
                 and (ob is None or box_iou(b, ob) < 0.5)][:6]
        pred_c = None
        if cfg["tcp"] and tcp is not None and name == "peg" and s.rel_tcp is not None:
            pred_c = (tcp_mat(tcp) @ s.rel_tcp)[:3, 3]
        bank = ([s.ref0] if s.ref0 is not None else []) + list(s.bank)
        B = torch.stack(bank) if bank else None
        clist = []
        for b, sc in cands:
            m, _, _ = self.imgp.predict(box=np.array(b, np.float32), multimask_output=False)
            mg = torch.as_tensor(m[0] > 0, device="cuda")
            c = dict(box=[round(float(v), 1) for v in b], score=round(float(sc), 3), area=int(mg.sum()))
            c["area_r"] = c["area"] / s.stable_area if s.stable_area else None
            pw, _ = self._pts(mg, depth_g, c2w_g)
            if pw is not None:
                ln, cc = self._extent(pw)
                c["size_r"] = ln / s.exp_len if s.exp_len else None
                c["dist"] = float(torch.linalg.norm(cc - s.last_c)) if s.last_c is not None else None
                if pred_c is not None:
                    c["tcp_dist"] = float(np.linalg.norm(cc.cpu().numpy() - pred_c))
            if B is not None and cfg["reid"] != "none":
                e = self._embed(rgb, t, mg)
                c["sim"] = float((B @ e).max()) if e is not None else -1.0
            c["mask"] = mg
            clist.append(c)
        if cfg["lr"]:
            from e3_fit_gate import feat, prob
            gl = cfg["lr"][name]
            for c in clist:
                c["p_lr"] = float(prob(np.array(gl["w"]), np.array(gl["mu"]), np.array(gl["sd"]), np.array([feat(c)]))[0])
        ok = []
        for c in clist:
            g = True
            if cfg["lr"]:
                g &= c["p_lr"] >= cfg["lr"][name]["thr_use"]
            if cfg["reid"] != "none" and not cfg["lr"]:
                g &= c.get("sim", -1) >= cfg["reid_thr"]
            if "area" in cfg["gates"] and c["area_r"] is not None:
                g &= c["area_r"] >= cfg["area_r"]
            if "size3d" in cfg["gates"] and c.get("size_r") is not None:
                g &= 0.3 <= c["size_r"] <= 1.6
            if "cont" in cfg["gates"] and c.get("dist") is not None:
                g &= c["dist"] <= 0.05 + 0.01 * s.age
            if cfg["tcp"] and "tcp_dist" in c:
                g &= c["tcp_dist"] <= 0.03
            c["pass"] = bool(g)
            if g:
                ok.append(c)
        best = max(ok, key=lambda c: (c.get("p_lr", 0), c.get("sim", 0), c["score"])) if ok else None
        ev = dict(t=t, obj=name, kind="detect", trig=trig, gdino_ms=round(g_ms, 1), n_cand=len(clist),
                  cands=[{k: v for k, v in c.items() if k != "mask"} for c in clist])
        accepted = False
        if cfg["shadow"]:  # 보정용: 후보 특징만 기록하고 받아들이지 않음
            best = None
        if best is not None:
            if cfg["confirm"]:
                prev = s.pending
                accepted = prev is not None and prev[0] == t - 1 and box_iou(best["box"], prev[1]) >= 0.5
                s.pending = None if accepted else (t, best["box"])
            else:
                accepted = True
        else:
            s.pending = None
        ev["accepted"] = accepted
        if accepted:
            ev["box"] = best["box"]
            ev["chosen"] = {k: v for k, v in best.items() if k != "mask"}
            ev["mask_t"] = self.tr.add_box(t, oid, best["box"])
        torch.cuda.synchronize()
        ev["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return ev

    def reset(self):
        self.tr.reset()


def tcp_mat(p):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
    T[:3, 3] = p[:3]
    return T
