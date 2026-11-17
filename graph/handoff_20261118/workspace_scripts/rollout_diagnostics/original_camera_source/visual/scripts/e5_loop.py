"""E5: vision-in-the-loop 스크립트 정책 — perception 오차가 삽입 성공률에 주는 영향.

collect_v1.V1Policy(expert) 의 waypoint 계획에서 GT 를 쓰던 세 값을 state source 로 바꾼다.
  peg_pose (grasp 계획: frame 0 / refine: 현재 프레임), hole_pose (goal = hole × [-L,0,0]), peg 크기 (L, r: task 파라미터)
state source
  gt          : GT pose
  noise       : GT + 가우시안 노이즈 (위치 σ mm, 회전 σ deg; 질의할 때마다 새로 뽑음)
  vision      : streaming SAM2 (+ re-anchor 설정) → mask → pose_v1 (peg 직육면체 / hole 면) 추정. 매 env step 갱신,
                질의 시 마지막 유효 추정을 쓴다 (peg: 유효 추정 없으면 이전 값, hole: 정지 가정으로 최근 유효값 유지)
GT 는 평가(성공/실패 단계)에만 쓴다. 정책은 vision 경로에서 GT pose 를 읽지 않는다 (box_move 이동도 vision 이 알아채야 함).
실패 단계: grasp (lift 후 peg 가 TCP 에서 3 cm 이상 / 안 들림), align (삽입 직전 peg head 가 hole 축에서 hole 반경 이상 벗어남), insert.
Usage: python scripts/e5_loop.py --cond gt --seeds $(seq 200 239) --name gt
"""
import argparse
import json
import os
import sys
import time

import gymnasium as gym
import numpy as np
import sapien
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if "--tracker" in sys.argv and sys.argv[sys.argv.index("--tracker") + 1] == "sam2long":
    sys.path.insert(0, os.path.join(os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party/sam2long"))  # SAM2Long 의 sam2 패키지가 먼저 import 되게 (이름 충돌)
from collect import CONTROL_MODE, ENV_ID, np0, sp  # noqa: E402
from collect_v1 import V1Policy  # noqa: E402
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAM = "base_camera"


def T2pose(T):
    return sapien.Pose(T)


def pose2T(p):
    return p.to_transformation_matrix()


def icp_translation(src, dst, iters=20, keep=0.7):
    """src(현재 box 점) → dst(기준 box 점) 정렬의 반대 = box 이동량 (dst 기준 src 의 평행이동). trimmed point-to-point."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    if len(src) > 1500:
        src = src[rng.choice(len(src), 1500, replace=False)]
    tree = cKDTree(dst)
    t = np.zeros(3)
    for _ in range(iters):
        dist, idx = tree.query(src - t)
        k = np.argsort(dist)[: int(keep * len(src))]
        t_new = (src[k] - dst[idx[k]]).mean(0)
        t_new[2] = 0.0  # box 는 테이블 위: 수평 이동만
        if np.linalg.norm(t_new - t) < 1e-4:
            t = t_new
            break
        t = t_new
    return t


def canon_roll(T):
    """정사각 단면 peg 는 축(x) 기준 90° roll 이 같은 형상 → z 축이 world +z 에 가장 가까운 roll 로 정규화."""
    best = None
    for k in range(4):
        c, s_ = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        Rx = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
        R = T[:3, :3] @ Rx
        if best is None or R[2, 2] > best[2, 2]:
            best = R
    out = T.copy()
    out[:3, :3] = best
    return out


class GTSource:
    def __init__(self, env):
        self.u = env.unwrapped

    def reset(self, obs):
        pass

    def update(self, obs):
        pass

    def peg(self):
        return pose2T(sp(self.u.peg.pose))

    def hole(self):
        return pose2T(sp(self.u.box_hole_pose))


class NoiseSource(GTSource):
    def __init__(self, env, pos_mm, rot_deg, seed):
        super().__init__(env)
        self.ps, self.rs = pos_mm / 1000, np.deg2rad(rot_deg)
        self.rng = np.random.default_rng(seed)

    def _n(self, T):
        T = T.copy()
        T[:3, 3] += self.rng.normal(0, self.ps, 3)
        T[:3, :3] = Rotation.from_rotvec(self.rng.normal(0, self.rs, 3)).as_matrix() @ T[:3, :3]
        return T

    def peg(self):
        return self._n(super().peg())

    def hole(self):
        return self._n(super().hole())


class SLAdapter:
    """SAM2Long causal streaming 을 ReanchorV1 과 같은 start/step/reset 인터페이스로 (re-anchor 없음)."""

    def __init__(self, pred):
        from stream_sam2long import SAM2LongStream
        self.s = SAM2LongStream(pred)

    def _wrap(self, o, H, W):
        res = {}
        for n in ("peg", "box"):
            if n in o:
                m, lg = o[n]
                res[n] = dict(mask=m, logit=lg, valid=lg > 0, age=0)
            else:
                res[n] = dict(mask=torch.zeros(H, W, dtype=torch.bool, device="cuda"), logit=-10.0, valid=False, age=0)
        return res

    def start(self, rgb, dep, K, c2w, boxes):
        self.H, self.W = rgb.shape[:2]
        return self._wrap(self.s.start(rgb, boxes), self.H, self.W)

    def step(self, rgb, dep, c2w, tcp=None):
        return self._wrap(self.s.step(rgb)[1], self.H, self.W)

    def reset(self):
        self.s.reset()


class VisionSource:
    """streaming tracker + pose 추정. 모델 경로는 관측(rgb, depth, K, cam2world)과 task 파라미터(L, r)만 쓴다."""

    def __init__(self, env, cfg_name, models, box_track=False, peg_tcp=False, inhand="raw", fast_pose=False, reobserve=False,
                 hand_init=False, tracker="sam2"):
        self.tracker = tracker
        self.hand_init = hand_init
        self.box_track, self.peg_tcp, self.inhand = box_track, peg_tcp, inhand
        self.fast_pose, self.reobserve, self.box_moved = fast_pose, reobserve, False
        self.grasped, self.rel_samples = False, []
        self.hist = []  # (t, T) 품질 통과 + 시간 게이트 통과한 peg 추정 (temporal / tcp2)
        self.rel2, self.slip = [], 0  # tcp2: (TCP→peg) 표본, 불일치 연속 횟수
        self.env, self.u = env, env.unwrapped
        self.cfg_name = cfg_name
        self.pred, self.imgp, self.gdino, self.dino = models
        self.log = []

    def _obs(self, obs):
        sd, spr = obs["sensor_data"][CAM], obs["sensor_param"][CAM]
        rgb = np0(sd["rgb"])
        d = np0(sd["depth"])[..., 0].astype(np.float32)
        d[d <= 0] = 0
        ext = np.eye(4, dtype=np.float32)
        ext[:3] = np0(spr["extrinsic_cv"])
        return rgb, d / 1000.0, np0(spr["intrinsic_cv"]), np.linalg.inv(ext)

    def reset(self, obs):
        import pose_v1 as pv
        from e3_run import CFGS, GATE_THR
        from init_v1 import assign, cands
        from reanchor_v1 import ReanchorV1
        cfg = dict(CFGS[self.cfg_name])
        if cfg.get("lr") == "gate":
            g = json.load(open(f"{ROOT}/results/v1/E3/gate.json"))
            for o, t in GATE_THR.items():
                g[o]["thr_use"] = t
            cfg["lr"] = g
        rgb, dep, K, c2w = self._obs(obs)
        p, b = assign(cands(self.gdino, rgb, "peg"), cands(self.gdino, rgb, "box"))
        boxes = {"peg": p[0] if p else None, "box": b[0] if b else None}
        self.ra = ReanchorV1(self.pred, self.imgp, self.gdino, self.dino, cfg) if self.tracker == "sam2" else SLAdapter(self.pred)
        L, r = float(self.u.peg_half_sizes[0, 0]), float(self.u.peg_half_sizes[0, 1])  # task 파라미터 (size="known")
        self.pe = pv.PegEstimator(L, r, fast=self.fast_pose)
        self.he = pv.HoleEstimator(L, r)
        self.K = K
        self.T_peg = self.T_hole = None
        self.hole_hist = []
        self.t = 0
        self.ms = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.ra.start(rgb, dep, K, c2w, boxes)
        self.last_pinfo = {}
        self._estimate(out, rgb, dep, c2w)
        self.base0_ok = bool(self.last_pinfo.get("ok"))  # 시작 프레임 peg 추정 신뢰 (보이는 축 ≥ 0.6 L)
        if self.hand_init:
            self._hand_init(obs, L, r)

    def hand_estimate(self, obs, L, r):
        """hand camera 한 장으로 peg pose 추정 (GDINO → SAM2 image mask → pose_v1). hand camera 자세는 로봇 기구학(카메라 보정)."""
        import pose_v1 as pv
        from e4_pose import erode
        sd, spr = obs["sensor_data"]["hand_camera"], obs["sensor_param"]["hand_camera"]
        rgb = np0(sd["rgb"])
        dep = np0(sd["depth"])[..., 0].astype(np.float32) / 1000.0
        dep[dep <= 0] = 0
        ext = np.eye(4)
        ext[:3] = np0(spr["extrinsic_cv"])
        c2w, K = np.linalg.inv(ext), np0(spr["intrinsic_cv"])
        H, W = rgb.shape[:2]
        with torch.autocast("cuda", enabled=False):
            bxs, _ = self.gdino(rgb, "red and white block")
        bxs = [b for b in bxs if (b[2] - b[0]) * (b[3] - b[1]) < 0.6 * H * W]
        if not bxs:
            return None, dict(reason="no_det")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.imgp.set_image(rgb)
            m, _, _ = self.imgp.predict(box=np.array(bxs[0], np.float32), multimask_output=False)
        m = m[0] > 0
        P, v, u = pv.backproject_px(dep, K, erode(m) if m.sum() > 300 else m)
        if len(P) < 30:
            return None, dict(reason="few_pts")
        pe = pv.PegEstimator(L, r)
        pe.set_view(dep, K, c2w, m)
        Pw = pv.to_world(P, c2w)
        T, hi = pe(Pw, pv.head_flag(rgb[v, u]))
        if T is None:
            return None, hi
        hi["mean"] = Pw.mean(0)
        return canon_roll(T), hi

    def _hand_init(self, obs, L, r):
        """시작 시점: hand camera 추정이 축 방향으로 더 많이 보이면 그것을 쓴다 (dev 에서 효과 없음, 기록용)."""
        Th, hi = self.hand_estimate(obs, L, r)
        base_ext = self.last_pinfo.get("vis_extent", 0.0) if self.T_peg is not None else 0.0
        info = dict(hand_ext=round(hi.get("vis_extent", 0) / L, 2) if Th is not None else None, base_ext=round(base_ext / L, 2))
        if Th is not None and hi.get("vis_extent", 0) > base_ext:
            self.adopt_peg(Th, hi)
            info["used"] = "hand"
        self.hand_info = info

    def adopt_peg(self, T, info):
        """외부(hand camera) 추정을 현재 peg 추정으로 채택: base 추정기 prior, tcp2 초기화용 이력도 갱신."""
        self.T_peg = T
        self.pe.prev, self.pe.prev_mean = T, info["mean"]
        if info.get("ok"):
            self.hist = [(self.t, T)]

    def update(self, obs):
        rgb, dep, K, c2w = self._obs(obs)
        self.t += 1
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.ra.step(rgb, dep, c2w)
        self._estimate(out, rgb, dep, c2w)
        self.ms.append((time.perf_counter() - t0) * 1000)

    def _estimate(self, out, rgb, dep, c2w):
        import pose_v1 as pv
        from e4_pose import erode
        mp = out["peg"]["mask"].cpu().numpy()
        if out["peg"]["valid"] and mp.sum() >= 50:
            P, v, u = pv.backproject_px(dep, self.K, erode(mp) if mp.sum() > 300 else mp)
            if len(P):
                self.peg_rough = pv.to_world(P, c2w).mean(0)  # 일부만 보여도 대략 위치 (시작 탐색용)
            if len(P) >= 30:
                self.pe.set_view(dep, self.K, c2w, mp)
                T, pinfo = self.pe(pv.to_world(P, c2w), pv.head_flag(rgb[v, u]))
                self.last_pinfo = pinfo
                if T is not None and self.inhand != "raw":
                    self._inhand_update(canon_roll(T), pinfo)
                if T is not None:
                    self.T_peg = canon_roll(T)
                    if self.peg_tcp and self.grasped:
                        # TCP→peg 상대 변환 표본 (로봇 상태 사용: ablation)
                        Tt = pose2T(sp(self.u.agent.tcp.pose))
                        self.rel_samples.append(np.linalg.inv(Tt) @ self.T_peg)
        mb = out["box"]["mask"].cpu().numpy()
        self.last_masks = {"peg": out["peg"]["mask"].cpu().numpy(), "box": mb}  # 데모 영상용
        self.last_box_mask = mb
        if out["box"]["valid"] and mb.sum() >= 200:
            P, _, _ = pv.backproject_px(dep, self.K, mb)
            Pw = pv.to_world(P, c2w)
            T, hinfo = self.he(Pw, dep, self.K, c2w, mb)
            if T is not None and not self.box_track:
                # 가장 덜 가려진 관측 유지: hole 면 인라이어 수가 지금까지 최대의 0.9 배 이상일 때만 갱신
                if hinfo["n_plane"] >= 0.9 * getattr(self, "best_n", 0):
                    self.best_n = max(hinfo["n_plane"], getattr(self, "best_n", 0))
                    self.T_hole = T
                    if self.reobserve:
                        self.box_ref = Pw[np.random.default_rng(0).choice(len(Pw), min(1500, len(Pw)), replace=False)]
                        self.move_cnt = 0
                T = None
            if self.reobserve and not self.box_moved and getattr(self, "box_ref", None) is not None:
                # box 이동 감지만 (크기는 다시 보기로 구함): ICP 평행이동 > 1 cm 가 3 프레임 연속
                self.move_cnt = getattr(self, "move_cnt", 0) + 1 if np.linalg.norm(icp_translation(Pw, self.box_ref)) > 0.01 else 0
                if self.move_cnt >= 3:
                    self.box_moved = True
            if T is not None and (self.T_hole is None or np.linalg.norm(T[:3, 3] - self.T_hole[:3, 3]) < 0.004):
                # hole 이 직접 보일 때: 최근 유효값 5개 중앙값 + box 기준 cloud 갱신
                self.hole_hist = (self.hole_hist + [T])[-5:]
                Tm = T.copy()
                Tm[:3, 3] = np.median([h[:3, 3] for h in self.hole_hist], 0)
                self.T_hole = Tm
                self.box_ref = Pw[np.random.default_rng(0).choice(len(Pw), min(1500, len(Pw)), replace=False)]
                self.hole_ref = Tm.copy()
                self.move_cnt = 0
            elif self.box_track and self.T_hole is not None and getattr(self, "box_ref", None) is not None:
                # hole 이 안 보이거나 크게 벗어남: box cloud 를 기준 cloud 에 정합 (평면 이동 가정, 평행이동만)
                d = icp_translation(Pw, self.box_ref)
                if np.linalg.norm(d) > 0.008:
                    self.move_cnt = getattr(self, "move_cnt", 0) + 1
                    if self.move_cnt >= 3:
                        Tm = self.hole_ref.copy()
                        Tm[:3, 3] += d
                        self.T_hole = Tm
                        self.hole_hist = []
                else:
                    self.move_cnt = 0

    def _inhand_update(self, T, pinfo):
        """품질 게이트(보이는 축 길이 ≥ 0.6 L, 표면 잔차 < 2 mm) + 시간 게이트(직전 수용값에서 1 cm + 1 cm/프레임 이내)."""
        if not (pinfo.get("ok") and pinfo.get("rms_mm", 9) < 2.0):
            return
        if self.hist:
            t0, T0 = self.hist[-1]
            if np.linalg.norm(T[:3, 3] - T0[:3, 3]) > 0.01 + 0.01 * (self.t - t0):
                self.jump_cnt = getattr(self, "jump_cnt", 0) + 1
                if self.jump_cnt < 3:  # 3 프레임 연속이면 실제 이동으로 보고 수용
                    return
        self.jump_cnt = 0
        self.hist = (self.hist + [(self.t, T)])[-30:]
        if self.inhand == "tcp2" and self.grasped:
            Tt = pose2T(sp(self.u.agent.tcp.pose))
            if len(self.rel2) >= 5:
                pred = Tt @ self._robust_rel()
                if np.linalg.norm(pred[:3, 3] - T[:3, 3]) > 0.01:
                    self.slip += 1
                    if self.slip >= 5:  # 손 안에서 peg 가 미끄러짐 → 다시 정합
                        self.rel2, self.slip = [], 0
                    return
                self.slip = 0
            self.rel2 = (self.rel2 + [np.linalg.inv(Tt) @ T])[-40:]

    @staticmethod
    def _avg_pose(Ts):
        """위치: 중앙값 근처(5 mm) 평균, 회전: 축(x) 평균 + roll 정규화"""
        P = np.array([T[:3, 3] for T in Ts])
        med = np.median(P, 0)
        keep = np.linalg.norm(P - med, axis=1) < 0.005
        Ts = [T for T, k in zip(Ts, keep) if k] or Ts
        out = Ts[-1].copy()
        out[:3, 3] = np.mean([T[:3, 3] for T in Ts], 0)
        x = np.mean([T[:3, 0] for T in Ts], 0)
        x /= np.linalg.norm(x)
        z0 = out[:3, 2] - (out[:3, 2] @ x) * x
        z0 /= np.linalg.norm(z0)
        out[:3, :3] = np.stack([x, np.cross(z0, x), z0], 1)
        return out

    def _robust_rel(self):
        return self._avg_pose(self.rel2)

    def on_grasp(self):
        """그리퍼를 닫는 순간: 직전(바닥에 놓여 잘 보이던) peg 추정과 TCP 로 손 기준 상대 위치 초기값 (tcp2)."""
        self.grasped, self.rel_samples, self.rel2 = True, [], []
        self.rel_init = None
        if self.inhand == "tcp2" and self.hist:
            Tt = pose2T(sp(self.u.agent.tcp.pose))
            self.rel_init = np.linalg.inv(Tt) @ self.hist[-1][1]

    def peg(self):
        if self.inhand == "tcp2" and self.grasped:
            Tt = pose2T(sp(self.u.agent.tcp.pose))
            if len(self.rel2) >= 5:
                return canon_roll(Tt @ self._robust_rel())
            if getattr(self, "rel_init", None) is not None:
                return canon_roll(Tt @ self.rel_init)
        if self.inhand in ("temporal", "tcp2") and self.hist and self.t - self.hist[-1][0] > 2:
            return self.T_peg  # 품질 통과 추정이 오래됨(손이 움직이는 중일 수 있음) → 최신 원시 추정
        if self.inhand in ("temporal", "tcp2") and self.hist:
            recent = [T for t, T in self.hist if self.t - t <= 5]
            if len(recent) >= 3:
                P = np.array([T[:3, 3] for T in recent])
                if np.linalg.norm(P - np.median(P, 0), axis=1).max() < 0.004:  # 정지 구간: 최근 추정 평균
                    return canon_roll(self._avg_pose(recent))
            return self.hist[-1][1]
        if self.peg_tcp and self.grasped and len(self.rel_samples) >= 3:
            R = np.stack(self.rel_samples[-30:])
            rel = R[-1].copy()
            rel[:3, 3] = np.median(R[:, :3, 3], 0)
            # 회전: 축(x) 평균 후 roll 은 canon 으로
            x = np.median(R[:, :3, 0], 0)
            x /= np.linalg.norm(x)
            z0 = rel[:3, 2] - (rel[:3, 2] @ x) * x
            z0 /= np.linalg.norm(z0)
            rel[:3, :3] = np.stack([x, np.cross(z0, x), z0], 1)
            return canon_roll(pose2T(sp(self.u.agent.tcp.pose)) @ rel)
        return self.T_peg

    def hole(self):
        return self.T_hole


class LoopPolicy(V1Policy):
    """V1Policy(expert) 의 GT 질의를 source 로 바꾼 버전. env.step 마다 source.update."""

    def __init__(self, env, src, rng, box_move=False, look=False, look_rule="v2", reobs_always=False):
        super().__init__(env, "expert", rng, box_move=box_move)
        self.look, self.look_rule, self.reobs_always = look, look_rule, reobs_always
        self.search = False
        self.src = src
        self.stage = "start"
        self.n_steps = 0

    def _act(self, pose):
        super()._act(pose)
        self.n_steps += 1
        self.src.update(self.last_obs())
        if getattr(self, "rec", None) is not None:
            self.rec.frame(self.last_obs(), self.src, self, self.gt_view())

    def gt_view(self):
        """데모 영상 표시용 정답 (정책 계산에는 안 씀)"""
        u = self.u
        return dict(peg=pose2T(sp(u.peg.pose)), hole=pose2T(sp(u.box_hole_pose)), L=float(u.peg_half_sizes[0, 0]),
                    r=float(u.peg_half_sizes[0, 1]), hr=float(u.box_hole_radii[0]))

    def mark(self, msg):
        if getattr(self, "rec", None) is not None:
            self.rec.mark(msg)

    def last_obs(self):
        return self.env.unwrapped._last_obs

    def run(self):
        u = self.u
        L = float(u.peg_half_sizes[0, 0])
        r = float(u.peg_half_sizes[0, 1])
        Tp = self.src.peg()
        if self.search and hasattr(self.src, "hand_estimate") and Tp is None:  # 추정이 아예 없을 때만 (불확실한 경우는 접근 지점 look 이 처리, eval4 500/507/509/524)
            Ts = self.search_peg(L, r)  # 시작 추정이 없거나 불확실: peg 근처로 손을 옮겨 hand camera 로 찾기
            Tp = Ts if Ts is not None else Tp
        if Tp is None:  # hole 은 목표 계산 때 필요 (시작 프레임엔 그리퍼가 hole 을 가릴 수 있음)
            self.stage = "no_estimate"
            return
        self.qlog("grasp")
        obb = trimesh.primitives.Box(extents=2 * np.array([L, r, r]), transform=Tp)
        target_closing = u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        info = compute_grasp_info_by_obb(obb, approaching=np.array([0, 0, -1]), target_closing=target_closing, depth=0.025)
        grasp_pose = u.agent.build_grasp_pose(np.array([0, 0, -1]), info["closing"], info["center"])
        grasp_pose = grasp_pose * sapien.Pose([-max(0.05, L / 2 + 0.01), 0, 0])
        peg_init_pose = T2pose(Tp)
        self.stage = "grasp"
        self.move(grasp_pose * sapien.Pose([0, 0, -0.05]))
        if self.look:
            self.mark("look-then-grasp: re-estimate peg with wrist camera at approach point")
            # 다가가서 보고 잡기: 접근 지점(peg 위 5 cm)에서 hand camera 로 peg 를 다시 추정해 잡을 위치를 다시 계산 (eye-in-hand 능동 지각)
            Th, hi = self.src.hand_estimate(self.last_obs(), L, r)
            self.log["look"] = dict(ok=bool(Th is not None and hi.get("ok")), ext=round(hi.get("vis_extent", 0) / L, 2) if Th is not None else None)
            base_ok = bool(getattr(self.src, "last_pinfo", {}).get("ok"))
            agree = Th is not None and np.linalg.norm(Th[:3, 3] - Tp[:3, 3]) < 0.015
            self.log["look"].update(base_ok=base_ok, agree=bool(agree))
            # 채택: base 추정이 불확실(보이는 축 < 0.6 L)하거나, 두 추정이 1.5 cm 안에서 일치할 때만 (dev 106/109: hand 쪽이 손가락 섞여 틀림)
            if self.look_rule == "v3":  # 시작 프레임 base 추정이 불확실할 때만 (접근 지점에선 그리퍼가 peg 를 가려 판정이 나빠짐)
                adopt = not self.src.base0_ok
            else:  # v2: 접근 지점 base 판정 불확실 또는 두 추정 일치
                adopt = not base_ok or agree
            if Th is not None and hi.get("ok") and adopt:
                G = pose2T(sp(self.u.peg.pose))  # 평가 로그용 GT
                self.log["look"]["err_mm"] = round(float(np.linalg.norm(Th[:3, 3] - G[:3, 3]) * 1000), 1)
                self.src.adopt_peg(Th, hi)
                Tp = Th
                obb = trimesh.primitives.Box(extents=2 * np.array([L, r, r]), transform=Tp)
                info = compute_grasp_info_by_obb(obb, approaching=np.array([0, 0, -1]), target_closing=target_closing, depth=0.025)
                grasp_pose = u.agent.build_grasp_pose(np.array([0, 0, -1]), info["closing"], info["center"])
                grasp_pose = grasp_pose * sapien.Pose([-max(0.05, L / 2 + 0.01), 0, 0])
                peg_init_pose = T2pose(Tp)
                self.move(grasp_pose * sapien.Pose([0, 0, -0.05]))
        self.move(grasp_pose)
        if hasattr(self.src, "on_grasp"):
            self.src.on_grasp()  # 닫기 직전 관측 기준
        self.mark("grasp")
        self.gripper(-1.0)
        self.move(sapien.Pose([0, 0, 0.05]) * sp(u.agent.tcp.pose))
        if np.linalg.norm(np0(u.peg.pose.p) - np0(u.agent.tcp.pose.p)) > L + 0.03:  # 평가용 GT
            self.result_stage = "grasp"
        self.stage = "align"
        if self.src.hole() is None:
            self.stage = "no_estimate"
            return
        goal = T2pose(self.src.hole() @ pose2T(sapien.Pose([-L, 0, 0])))
        insert_pose = goal * peg_init_pose.inv() * grasp_pose
        offset = sapien.Pose([-0.01 - L, 0, 0])
        pre = insert_pose * offset
        self.move(pre)
        pre = self.refine_src(pre, offset, L)
        if self.box_move:
            self.box_move_do()
            pre = self.refine_src(pre, offset, L)
        if self.reobs_always:
            # 삽입 직전 항상 1회 비켜서 다시 보기 (이동 감지 실패·충돌로 인한 손 안 peg 밀림 대응). 손 기준 peg 위치도 새로 학습
            self.reobserve_do(pre, relearn=True)
            pre = self.refine_src(pre, offset, L)
        self.align_err_mm = self.align_error(L)
        self.stage = "insert"
        self.move(pre * sapien.Pose([0.01 + L + 0.05, 0, 0]))

    def qlog(self, tag):
        """평가용 로그: 질의 시점 추정값 vs GT (정책 계산에는 안 씀)"""
        Tp, Th = self.src.peg(), self.src.hole()
        G = pose2T(sp(self.u.peg.pose))
        H = pose2T(sp(self.u.box_hole_pose))
        e = dict(tag=tag, step=self.n_steps)
        if Tp is not None:
            e["peg_mm"] = round(float(np.linalg.norm(Tp[:3, 3] - G[:3, 3]) * 1000), 1)
            e["peg_deg"] = round(float(np.degrees(np.arccos(np.clip(Tp[:3, 0] @ G[:3, 0], -1, 1)))), 1)
        if Th is not None:
            e["hole_mm"] = round(float(np.linalg.norm(Th[:3, 3] - H[:3, 3]) * 1000), 1)
            e["hole_deg"] = round(float(np.degrees(np.arccos(np.clip(Th[:3, 0] @ H[:3, 0], -1, 1)))), 1)
        if hasattr(self.src, "hist"):
            e["n_hist"], e["n_rel2"] = len(self.src.hist), len(self.src.rel2)
            e["hist_age"] = (self.src.t - self.src.hist[-1][0]) if self.src.hist else None
        self.log.setdefault("queries", []).append(e)

    def search_peg(self, L, r):
        """능동 탐색: base 에 일부 보인 peg 의 3D 중심(없으면 작업 영역 중앙) 위 25 cm 로 손을 옮겨 hand camera 로 peg 추정.
        작업 영역 중앙 = task 배치 범위 x∈[-0.1,0.1], y∈[-0.3,0] 의 중심 (환경 사양, GT 아님)."""
        rough = getattr(self.src, "peg_rough", None)
        center = rough if rough is not None else np.array([0.0, -0.15, 0.0])
        q = sp(self.u.agent.tcp.pose).q
        self.log["search"] = dict(from_base=rough is not None)
        for dxy in ((0, 0), (0, 0.08), (0, -0.08)):
            self.move(sapien.Pose([center[0] + dxy[0], center[1] + dxy[1], 0.25], q))
            for _ in range(3):
                self._act(sp(self.u.agent.tcp.pose))
            Th, hi = self.src.hand_estimate(self.last_obs(), L, r)
            if Th is not None and hi.get("ok"):
                G = pose2T(sp(self.u.peg.pose))  # 평가 로그용 GT
                self.log["search"].update(found=True, err_mm=round(float(np.linalg.norm(Th[:3, 3] - G[:3, 3]) * 1000), 1))
                self.src.adopt_peg(Th, hi)  # base0_ok 는 그대로 둔다 → 접근 지점 look 이 더 가까이서 다시 추정
                return Th
        self.log["search"]["found"] = False
        return None

    def reobserve_do(self, pre, relearn=False):
        """box 이동 감지 시: peg 를 hole 면을 따라 옆으로 비켜 hole 을 다시 보고, 원래 pre-insert 자세로 돌아온다 (능동 지각)."""
        Th = self.src.hole()
        side = Th[:3, 1] if Th is not None else np.array([1.0, 0, 0])
        base = np0(self.u.agent.robot.pose.p)
        if side @ (base - pre.p) < 0:  # 로봇 몸 쪽으로 비킨다 (도달 쉬움)
            side = -side
        self.mark("re-observe: side-step 18 cm to see the hole again" + (" + relearn in-hand peg" if relearn else ""))
        self.src.best_n, self.src.box_ref, self.src.T_hole = 0, None, None
        if relearn and hasattr(self.src, "rel2"):
            if len(self.src.rel2) >= 5:  # 새 표본 5개 전까지는 직전 손 기준 위치 유지 (dev 108: 새 표본 0 → 원시 추정 10 mm)
                self.src.rel_init = self.src._robust_rel()
            self.src.rel2, self.src.slip = [], 0  # 비켜 선 동안(box 에 안 가림) 새 관측으로 손 기준 peg 위치 재학습
        # 손(그리퍼 폭 ~20 cm)이 box 면(반폭 ~10 cm)을 가리지 않을 만큼 비킨다
        self.move(sapien.Pose(side * 0.18) * pre)
        for _ in range(8):
            self._act(sp(self.u.agent.tcp.pose))
        if os.environ.get("E5_DEBUG_DIR"):  # 디버그: 비켜 선 상태의 관측 저장
            o = self.last_obs()["sensor_data"][CAM]
            np.savez_compressed(f"{os.environ['E5_DEBUG_DIR']}/reobs_{self.n_steps}.npz", rgb=np0(o["rgb"]),
                                seg=np0(o["segmentation"])[..., 0], box_mask=self.src.last_box_mask)
        self.move(pre)
        self.src.box_moved = False
        self.log["reobserve"] = self.log.get("reobserve", 0) + 1

    def refine_src(self, pre, offset, L, n=3):
        if getattr(self.src, "box_moved", False):
            self.reobserve_do(pre)
        for _ in range(n):
            self.qlog("refine")
            Tp, Th = self.src.peg(), self.src.hole()
            if Tp is None or Th is None:
                continue
            goal = T2pose(Th @ pose2T(sapien.Pose([-L, 0, 0])))
            delta = goal * offset * T2pose(Tp).inv()
            pre = delta * pre
            self.move(pre)
        return pre

    def box_move_do(self):
        u = self.u
        a = self.rng.uniform(0, 2 * np.pi)
        rr = self.rng.uniform(0.02, 0.05)
        bp = sp(u.box.pose)
        u.box.set_pose(sapien.Pose(bp.p + np.array([rr * np.cos(a), rr * np.sin(a), 0.0]), bp.q))
        self.log["box_move_offset_cm"] = round(rr * 100, 2)
        self.mark(f"box teleported by {rr * 100:.1f} cm just before insertion")
        # vision 이 이동을 알아챌 수 있게 몇 step 정지 관측
        for _ in range(5):
            self._act(sp(u.agent.tcp.pose))

    def align_error(self, L):
        """평가용 GT: 삽입 직전 peg head 의 hole 축 수직 거리 (mm)."""
        u = self.u
        H = pose2T(sp(u.box_hole_pose))
        head = np0(u.peg_head_pose.p)
        q = np.linalg.inv(H) @ np.r_[head, 1]
        return float(np.linalg.norm(q[1:3]) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, help="gt | noise | vision")
    ap.add_argument("--pos-mm", type=float, default=0)
    ap.add_argument("--rot-deg", type=float, default=0)
    ap.add_argument("--vcfg", default="none", help="vision 일 때 e3_run CFGS 이름")
    ap.add_argument("--box-move", action="store_true")
    ap.add_argument("--box-track", action="store_true", help="hole 이 안 보일 때 box cloud ICP 로 이동 추적")
    ap.add_argument("--inhand", default="raw", choices=["raw", "temporal", "tcp2"],
                    help="손에 쥔 peg 추정: raw(프레임별), temporal(vision 만: 품질+시간 게이트, 정지 구간 평균), tcp2(+TCP 융합, 미끄러짐 감지)")
    ap.add_argument("--fast-pose", action="store_true", help="pose_v1 빠른 경로 (이전 자세 refine 우선)")
    ap.add_argument("--tracker", default="sam2", choices=["sam2", "sam2long"], help="sam2long: causal SAM2Long streaming (re-anchor 없음)")
    ap.add_argument("--look", action="store_true", help="다가가서 보고 잡기: 접근 지점에서 hand camera 로 peg 재추정")
    ap.add_argument("--look-rule", default="v2", choices=["v2", "v3"])
    ap.add_argument("--video", default=None, help="데모 영상 저장 폴더")
    ap.add_argument("--video-title", default=None)
    ap.add_argument("--search", action="store_true", help="시작 추정이 없거나 불확실하면 peg 근처로 손을 옮겨 hand camera 로 탐색")
    ap.add_argument("--reobs-always", action="store_true", help="삽입 직전 항상 1회 비켜서 hole 다시 보기 + 손 기준 peg 재학습")
    ap.add_argument("--hand-init", action="store_true", help="시작 peg pose 를 hand camera 로도 추정해 더 많이 보이는 쪽 사용")
    ap.add_argument("--reobserve", action="store_true", help="box 이동 감지 시 peg 를 옆으로 비켜 hole 다시 보기")
    ap.add_argument("--peg-tcp", action="store_true", help="(ablation, 로봇 상태 사용) grasp 후 peg = TCP × robust(TCP→peg)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(200, 240)))
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=f"{ROOT}/results/v1/E5")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(8)
    env = gym.make(ENV_ID, obs_mode="rgb+depth+segmentation", control_mode=CONTROL_MODE,
                   sensor_configs=dict(width=512, height=512), max_episode_steps=1000, num_envs=1)
    # 마지막 관측을 꺼낼 수 있게 step/reset 을 감싼다
    uw = env.unwrapped
    orig_step, orig_reset = env.step, env.reset

    def step(a):
        o = orig_step(a)
        uw._last_obs = o[0]
        uw._last_info = o[4]
        return o

    def reset(**kw):
        o = orig_reset(**kw)
        uw._last_obs = o[0]
        return o
    env.step, env.reset = step, reset
    models = None
    if args.cond == "vision":
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from grounding import GDino
        from track_v1 import MODELS
        pred = build_sam2_video_predictor(*MODELS["sam2"]["small"], device="cuda")
        pred.add_all_frames_to_correct_as_cond = True
        if args.tracker == "sam2long":
            import sam2
            assert "sam2long" in sam2.__file__, sam2.__file__
        dino = None
        if args.vcfg.startswith("v1"):
            from reanchor_v1 import Dino
            dino = Dino()
        models = (pred, SAM2ImagePredictor(pred), GDino(), dino)
    res = dict(args=vars(args), episodes={})
    for seed in args.seeds:
        t0 = time.time()
        obs, _ = env.reset(seed=seed)
        if args.cond == "gt":
            src = GTSource(env)
        elif args.cond == "noise":
            src = NoiseSource(env, args.pos_mm, args.rot_deg, seed)
        else:
            src = VisionSource(env, args.vcfg, models, box_track=args.box_track, peg_tcp=args.peg_tcp, inhand=args.inhand,
                               fast_pose=args.fast_pose, reobserve=args.reobserve, hand_init=args.hand_init, tracker=args.tracker)
        src.reset(obs)
        pol = LoopPolicy(env, src, np.random.default_rng(seed * 31 + 1), box_move=args.box_move, look=args.look, look_rule=args.look_rule,
                         reobs_always=args.reobs_always)
        pol.result_stage = None
        pol.search = args.search
        pol.rec = None
        if args.video:
            from demo_video import DemoRecorder
            vt = args.video_title or args.name
            pol.rec = DemoRecorder(f"{args.video}/{args.name}_seed{seed}.mp4", f"{vt} | seed {seed}")
            pol.rec.frame(obs, src, pol, pol.gt_view())
        err = None
        try:
            pol.run()
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:200]
        succ = bool(np.asarray(uw.evaluate()["success"].cpu()).reshape(-1)[0])
        stage = "success" if succ else (pol.result_stage or ("no_estimate" if pol.stage == "no_estimate" else
                                                             ("align" if getattr(pol, "align_err_mm", 99) > (float(uw.box_hole_radii[0]) - float(uw.peg_half_sizes[0, 1])) * 1000 else "insert")))
        e = dict(seed=seed, success=succ, stage=stage, steps=pol.n_steps, align_err_mm=round(getattr(pol, "align_err_mm", -1), 2),
                 seconds=round(time.time() - t0, 1), error=err, **pol.log)
        if args.cond == "vision" and src.ms:
            e["vision_ms_p50"] = round(float(np.median(src.ms)), 1)
            if getattr(src, "hand_info", None) is not None:
                e["hand_info"] = src.hand_info
            e["vision_ms_p95"] = round(float(np.percentile(src.ms, 95)), 1)
        if pol.rec is not None:
            for _ in range(30):  # 결과 화면 1.5 s
                pol.rec.mark(f"RESULT: {'SUCCESS' if succ else 'FAIL'} ({stage}), alignment error {e['align_err_mm']} mm")
                pol.rec.frame(uw._last_obs, src, pol, pol.gt_view())
            pol.rec.close(succ)
        res["episodes"][seed] = e
        print(json.dumps(e), flush=True)
        if args.cond == "vision":
            src.ra.reset()
    E = list(res["episodes"].values())
    res["summary"] = dict(n=len(E), success=round(float(np.mean([x["success"] for x in E])), 3),
                          stages={s: sum(x["stage"] == s for x in E) for s in set(x["stage"] for x in E)},
                          align_err_mm_median=round(float(np.median([x["align_err_mm"] for x in E if x["align_err_mm"] >= 0] or [-1])), 2))
    json.dump(res, open(f"{args.out}/{args.name}.json", "w"), indent=1)
    print("SUMMARY", json.dumps(res["summary"]))


if __name__ == "__main__":
    main()
