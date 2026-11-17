"""E4: 관측 표면 cloud 에 형상 prior 를 맞춰 peg 6D pose 와 hole pose 를 추정한다 (vision 입력만 사용).

형상 (ManiSkill PegInsertionSide 정의)
  peg : 직육면체 half size (L, r, r). +x 절반(head)은 주황 #EC7357, -x 절반(tail)은 흰색 #EDF6F9
  box : 한 변 2L 인 정육면체 (x = hole 방향 depth half L, y/z half L), x 축을 관통하는 정사각 hole (half r+3mm).
        hole pose 원점 = box 중심 평면(x=0) 위의 hole 중심, x 축 = hole 방향. box 는 테이블 위에 똑바로 놓인다 (yaw 만 회전).
크기
  size="known" : task 파라미터 (L, r) 를 쓴다 (로봇이 물체 사양을 안다고 가정)
  size="est"   : 관측에서 추정 (peg: 첫 유효 프레임 cloud 의 주축/단면 extent, box: hole 면 폭/2 = L)

peg 추정 (프레임별)
  1. PCA 주축 → head/tail 방향은 점 색(주황 vs 흰)의 축 방향 평균 위치로 정한다. 한 색만 보이면 이전 프레임 방향과 맞춘다
  2. 직육면체 SDF 최소화 (scipy least_squares, soft_l1) 로 R, t 정밀화. 축 방향 위치는 보이는 끝면이 없으면 불확정 →
     head 색 경계(주황/흰 경계 = peg 중심)를 추가 residual 로 쓴다 (경계가 보이면 축 방향 위치가 정해짐)
  3. (smooth) 등속 KF 대신 단순 지수 평활 + 이상치 게이트 (위치 점프 > 3 cm 는 1 프레임 보류)
hole 추정
  1. box cloud 에서 RANSAC 으로 법선이 수평인 가장 큰 평면 = hole 면 (카메라를 향함)
  2. 면 좌표계(u: 수평, v: 수직)로 점을 2 mm 격자에 찍고, 면 extent 안의 빈 연결 영역(테두리에 닿지 않는 것) 중 최대 = hole
  3. hole 중심 = 면 위 hole 중심을 면 법선 반대(box 안쪽)로 L 만큼 이동. hole x 축 = -법선(카메라에서 멀어지는 방향)
"""
import numpy as np
from scipy.ndimage import binary_closing, label
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

CLEAR = 0.003
CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
HEAD_RGB = np.array([236, 115, 87], float)
TAIL_RGB = np.array([237, 246, 249], float)


def backproject_px(depth, K, mask):
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u]
    x = (u + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (v + 0.5 - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1), v, u


def to_world(p, c2w):
    return p @ c2w[:3, :3].T + c2w[:3, 3]


def head_flag(rgb_px):
    """주황(head) vs 흰(tail): 채도 기준. 조명이 달라도 R-B 차이로 구분."""
    c = rgb_px.astype(float)
    sat = c.max(1) - c.min(1)
    return (c[:, 0] - c[:, 2] > 40) & (sat > 50)


def box_sdf(p, h):
    """원점 중심, half extent h 인 직육면체의 signed distance."""
    q = np.abs(p) - h
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)


def mat_from(rv, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rv).as_matrix()
    T[:3, 3] = t
    return T


def pose7(T):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.r_[T[:3, 3], q[3], q[:3]]


def T_from7(p):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
    T[:3, 3] = p[:3]
    return T


class PegEstimator:
    def __init__(self, L=None, r=None, max_pts=400, seed=0, fast=False):
        self.fast = fast
        self.L, self.r = L, r
        self.prev = None  # 4x4
        self.view = None
        self.rng = np.random.default_rng(seed)
        self.max_pts = max_pts

    def estimate_size(self, P):
        c = P.mean(0)
        _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
        proj = (P - c) @ Vt.T
        ext = np.percentile(proj, 99, 0) - np.percentile(proj, 1, 0)
        return ext[0] / 2, max(ext[1], ext[2]) / 2  # 위에서 보면 단면 한 변이 온전히 보임

    def set_view(self, depth, K, c2w, mask):
        """free-space 검사용 관측 (없으면 SDF 만으로 고름)"""
        self.view = (depth, K, np.linalg.inv(c2w), c2w[:3, 3], mask)

    def free_space_violation(self, T, h, n=8):
        """모델 표면 중 카메라를 향하는 점을 투영했을 때, 관측 depth 가 모델보다 1 cm 이상 멀거나(뒤가 보임) 배경인 비율."""
        depth, K, w2c, cam, mask = self.view
        g = np.linspace(-0.85, 0.85, n)
        pts, nrm = [], []
        for ax in range(3):
            for sgn in (-1, 1):
                a, b = [i for i in range(3) if i != ax]
                uu, vv = np.meshgrid(g, g)
                q = np.zeros((n * n, 3))
                q[:, ax] = sgn * h[ax]
                q[:, a], q[:, b] = uu.ravel() * h[a], vv.ravel() * h[b]
                pts.append(q)
                nn = np.zeros(3)
                nn[ax] = sgn
                nrm.append(np.tile(nn, (n * n, 1)))
        q = np.concatenate(pts)
        nw = np.concatenate(nrm) @ T[:3, :3].T
        pw = q @ T[:3, :3].T + T[:3, 3]
        vd = cam - pw
        front = (vd * nw).sum(1) / np.linalg.norm(vd, axis=1) > 0.3  # 시선과 거의 평행한 면은 제외
        pc = pw[front] @ w2c[:3, :3].T + w2c[:3, 3]
        u = (pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2]).astype(int)
        v = (pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2]).astype(int)
        H, W = depth.shape
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (pc[:, 2] > 0)
        if not ok.any():
            return 0.0
        dz = depth[v[ok], u[ok]]
        viol = (dz <= 0) | (dz > pc[ok, 2] + 0.01)
        return float(viol.mean())

    def __call__(self, P, head):
        """P (N,3) world points, head (N,) bool → 4x4 pose or None, info."""
        if len(P) < 30:
            return None, dict(reason="few_pts")
        n_use = self.max_pts if self.prev is not None else max(self.max_pts, 1500)  # 첫 추정(이전 자세 없음)은 점 표본에 민감 → 더 많이
        if len(P) > n_use:
            ix = self.rng.choice(len(P), n_use, replace=False)
            P, head = P[ix], head[ix]
        if self.L is None:
            self.L, self.r = self.estimate_size(P)
        L, r = self.L, self.r
        c = P.mean(0)
        _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
        ax = Vt[0]
        nh, nt = head.sum(), (~head).sum()
        if nh >= 10 and nt >= 10:
            if ((P[head] - c) @ ax).mean() < ((P[~head] - c) @ ax).mean():
                ax = -ax
        elif self.prev is not None and ax @ self.prev[:3, 0] < 0:
            ax = -ax
        h = np.array([L, r, r])
        w_col = 0.5

        def res(x):
            T = mat_from(x[:3], x[3:])
            q = (P - T[:3, 3]) @ T[:3, :3]
            r1 = box_sdf(q, h)
            # 색 일관성: head 점은 x>0, tail 점은 x<0 (위반 거리만 penalty)
            r2 = np.where(head, np.minimum(q[:, 0], 0), np.maximum(q[:, 0], 0)) * w_col
            # 테이블 prior: 모든 꼭짓점이 테이블 면(z=0) 위 (3 mm 허용). 장면 지식이고 GT 가 아님
            cz = (CORNERS * h) @ T[:3, :3].T[:, 2] + T[2, 3]
            r3 = np.minimum(cz + 0.003, 0) * 5.0
            return np.r_[r1, r2, r3]

        def hyp(axv, flip_anchor=False):
            """축 방향 axv (head 쪽) 가설의 초기 R, t. 중심 = head 끝(보이는 head 점 최대 투영) − L, 또는 tail 끝 + L"""
            yv = Vt[1] if abs(Vt[1] @ axv) < 0.9 else Vt[2]
            yv = yv - (yv @ axv) * axv
            yv /= np.linalg.norm(yv)
            R = np.stack([axv, yv, np.cross(axv, yv)], 1)
            sv = (P - c) @ axv
            if nh >= 10 and nt >= 10:
                ta = 0.5 * (np.percentile(sv[head], 5) + np.percentile(sv[~head], 95))
            elif nh >= 10:
                ta = sv[head].max() - L
            else:
                ta = sv.min() + L
            if flip_anchor and not (nh >= 10 and nt >= 10):
                # 보이는 끝이 실제 끝이 아닐 수 있음 (화면 가장자리에 잘림 / 가림) → 반대쪽 끝 기준
                ta = (sv.min() + L) if nh >= 10 else (sv.max() - L)
            return np.r_[Rotation.from_matrix(R).as_rotvec(), c + ta * axv]

        inits = [hyp(ax)]
        if self.prev is not None:
            Rp = self.prev[:3, :3]
            inits.append(np.r_[Rotation.from_matrix(Rp).as_rotvec(), self.prev[:3, 3]])
            inits.append(np.r_[Rotation.from_matrix(Rp).as_rotvec(), self.prev[:3, 3] + (P.mean(0) - self.prev_mean)])
        if not (nh >= 10 and nt >= 10):
            inits.append(hyp(-ax))
        if self.view is not None:
            for k in (1, 2):
                inits += [hyp(Vt[k]), hyp(-Vt[k])]
            inits += [hyp(ax, True), hyp(-ax, True)]
        sols = []
        if self.fast and self.prev is not None:
            # 빠른 경로 (tracking): 이전 자세에서 refine 1회. 잔차 < 1.5 mm 이고 free-space 위반 < 10% 이면 채택
            x0 = np.r_[Rotation.from_matrix(self.prev[:3, :3]).as_rotvec(), self.prev[:3, 3] + (P.mean(0) - self.prev_mean)]
            sol = least_squares(res, x0, loss="soft_l1", f_scale=0.003, max_nfev=30)
            Tq = mat_from(sol.x[:3], sol.x[3:])
            rq = float(np.sqrt(np.mean(res(sol.x) ** 2)))  # 색·테이블 항 포함: 표면만 보면 앞뒤 뒤집힌 자세도 통과
            fsv = self.free_space_violation(Tq, h) if self.view is not None else 0.0
            if rq < 0.0015 and fsv < 0.1:
                inits = []
                sols = [sol]
        for x0 in inits:
            sol = least_squares(res, x0, loss="soft_l1", f_scale=0.003, max_nfev=40)
            sols.append(sol)
        if not sols:
            return None, dict(reason="no_sol")
        if self.view is not None and len(sols) > 1:
            scores = [sol.cost + 0.02 * len(P) * self.free_space_violation(mat_from(sol.x[:3], sol.x[3:]), h) for sol in sols]
            sol = sols[int(np.argmin(scores))]
        else:
            sol = min(sols, key=lambda z: z.cost)
        T = mat_from(sol.x[:3], sol.x[3:])
        q = (P - T[:3, 3]) @ T[:3, :3]
        rms = float(np.sqrt(np.mean(box_sdf(q, h) ** 2)))
        self.prev, self.prev_mean = T, P.mean(0)
        sx = (P - T[:3, 3]) @ T[:3, 0]
        vis_ext = float(np.percentile(sx, 98) - np.percentile(sx, 2))
        return T, dict(vis_extent=vis_ext, ok=vis_ext >= 0.6 * L, rms_mm=rms * 1000, n=len(P), n_head=int(nh), nfev=sol.nfev)


def ransac_vertical_plane(P, iters=300, thr=0.003, rng=None):
    rng = rng or np.random.default_rng(0)
    best, best_n = None, 0
    for _ in range(iters):
        a, b = P[rng.choice(len(P), 2, replace=False)]
        d = b - a
        d[2] = 0  # 수평 방향 벡터 → 수직 평면 법선 = z × d
        if np.linalg.norm(d) < 1e-3:
            continue
        n = np.cross([0, 0, 1.0], d)
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < thr
        if inl.sum() > best_n:
            best_n, best = inl.sum(), (n, a, inl)
    return best


class HoleEstimator:
    """hole 면(수직 평면, RANSAC) 을 찾은 뒤, 이미지에서 '면 평면보다 뒤가 보이는 픽셀'(= 구멍으로 뒤가 보임) 의 연결 영역 중
    이미지 테두리에 닿지 않고 면 안쪽에 있는 정사각형 크기 영역을 hole 로 고른다. 팔 등 가림(면보다 앞) 과 구분된다."""

    def __init__(self, L=None, r=None, cam_pos=None):
        self.L, self.r = L, r
        self.cam_pos = cam_pos
        self.prev = None

    def __call__(self, P, depth, K, c2w, box_mask):
        import cv2
        if len(P) < 200:
            return None, dict(reason="few_pts")
        fit = ransac_vertical_plane(P)
        if fit is None:
            return None, dict(reason="no_plane")
        n, a, inl = fit
        Q = P[inl]
        c = Q.mean(0)
        cam = c2w[:3, 3]
        if (cam - c) @ n < 0:
            n = -n
        # 인라이어 전체로 수직 평면 법선 최소제곱 정밀화 (수평 성분 PCA: 최소 분산 방향 = 법선), 이후 재인라이어 1회
        for _ in range(2):
            Qh = Q[:, :2] - Q[:, :2].mean(0)
            _, _, Vh = np.linalg.svd(Qh, full_matrices=False)
            n_new = np.array([Vh[1, 0], Vh[1, 1], 0.0])
            n = n_new if n_new @ n > 0 else -n_new
            c = Q.mean(0)
            Q = P[np.abs((P - c) @ n) < 0.003]
        c = Q.mean(0)
        if (cam - c) @ n < 0:
            n = -n
        ez = np.array([0, 0, 1.0])
        eu = np.cross(ez, n)
        uvq = np.stack([(Q - c) @ eu, (Q - c) @ ez], 1)
        lo, hi = np.percentile(uvq, 0.5, 0), np.percentile(uvq, 99.5, 0)
        face_w = hi - lo
        # box mask bbox 주변만 본다
        ys, xs = np.nonzero(box_mask)
        H, W = depth.shape
        pad = 10
        y0, y1, x0, x1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1), max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
        vv, uu = np.mgrid[y0:y1, x0:x1]
        rays_c = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0], (vv + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
        R = c2w[:3, :3]
        rays_w = rays_c @ R.T
        denom = rays_w @ n
        lam = ((c - cam) @ n) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)  # 카메라 z 기준 평면까지 depth (rays_c z=1)
        dz = depth[y0:y1, x0:x1]
        behind = (dz <= 0) | (dz > lam + 0.0025)  # 터널 안쪽 벽(면 바로 뒤)도 hole 로 (원근에 의한 위쪽 bias 방지)
        hitp = cam + lam[..., None] * rays_w
        fu, fv = (hitp - c) @ eu, (hitp - c) @ ez
        inside = (fu > lo[0] - 0.005) & (fu < hi[0] + 0.005) & (fv > lo[1] - 0.005) & (fv < hi[1] + 0.005)
        nl, lab = cv2.connectedComponents(behind.astype(np.uint8), connectivity=4)
        best, best_score = None, 0
        exp_half = (self.r + CLEAR) if self.r is not None else None
        for i in range(1, nl):
            m = lab == i
            yy, xx = np.nonzero(m)
            if yy.min() == 0 or xx.min() == 0 or yy.max() == m.shape[0] - 1 or xx.max() == m.shape[1] - 1:
                continue  # 테두리에 닿음 = 바깥 배경
            if not inside[m].mean() > 0.9:
                continue
            u_, v_ = fu[m], fv[m]
            half = 0.5 * np.array([u_.max() - u_.min(), v_.max() - v_.min()])
            if half.min() < 0.006 or half.max() > 0.06 or half.max() / half.min() > 1.6:
                continue
            if exp_half is not None and not (0.85 * exp_half < half.min() and half.max() < 1.15 * exp_half):  # 일부 가려진 hole 제외
                continue
            score = m.sum()
            if score > best_score:
                best_score, best = score, (0.5 * (u_.max() + u_.min()), 0.5 * (v_.max() + v_.min()), half)
        L = self.L if self.L is not None else float(np.median(face_w)) / 2
        if best is None:
            return None, dict(reason="no_hole", face_w_mm=(face_w * 1000).round(1).tolist())
        hu, hv, half = best
        p_face = c + hu * eu + hv * ez
        x_ax = -n
        Tm = np.eye(4)
        Tm[:3, :3] = np.stack([x_ax, np.cross(ez, x_ax), ez], 1)
        Tm[:3, 3] = p_face + L * x_ax
        self.prev = Tm
        return Tm, dict(L_est=L, hole_half_mm=(half * 1000).round(1).tolist(), face_w_mm=(face_w * 1000).round(1).tolist(),
                        n_plane=int(inl.sum()))


def axis_angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def signed_axis_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
