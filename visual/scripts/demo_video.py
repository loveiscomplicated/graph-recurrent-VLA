"""E5 데모 영상 녹화: 앞 카메라 | 손목 카메라, 추적 mask, 추정 pose(초록 peg / 파랑 hole) vs 정답(흰 선), 단계·오차·이벤트 HUD.
e5_loop.py --video <path> 로 켠다. GT 는 화면 표시(비교)에만 쓴다.
"""
import cv2
import imageio
import numpy as np

CORN = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def proj(Pw, K, w2c):
    pc = Pw @ w2c[:3, :3].T + w2c[:3, 3]
    ok = pc[:, 2] > 0.01
    uv = np.stack([pc[:, 0] / np.maximum(pc[:, 2], 1e-6) * K[0, 0] + K[0, 2], pc[:, 1] / np.maximum(pc[:, 2], 1e-6) * K[1, 1] + K[1, 2]], 1)
    return uv, ok


def draw_box(img, T, half, K, w2c, color, th):
    P = (CORN * half) @ T[:3, :3].T + T[:3, 3]
    uv, ok = proj(P, K, w2c)
    for a, b in EDGES:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[b]), color, th, cv2.LINE_AA)


def draw_hole(img, T, L, hr, K, w2c, color, th):
    # hole 입구 사각형 (hole 좌표계 x = -L 평면) + 축
    sq = np.array([[-L, -hr, -hr], [-L, hr, -hr], [-L, hr, hr], [-L, -hr, hr]])
    P = sq @ T[:3, :3].T + T[:3, 3]
    uv, ok = proj(P, K, w2c)
    if ok.all():
        cv2.polylines(img, [uv.astype(np.int32)], True, color, th, cv2.LINE_AA)
    ax = np.array([[-L - 0.04, 0, 0], [-L, 0, 0]]) @ T[:3, :3].T + T[:3, 3]
    uv, ok = proj(ax, K, w2c)
    if ok.all():
        cv2.arrowedLine(img, tuple(int(v) for v in uv[0]), tuple(int(v) for v in uv[1]), color, th, cv2.LINE_AA, tipLength=0.3)


def overlay(img, m, col, a=0.4):
    if m is not None and m.any():
        img[m] = ((1 - a) * img[m] + a * np.array(col)).astype(np.uint8)


def text(img, lines, y0=18, s=0.5):
    for i, l in enumerate(lines):
        cv2.putText(img, l, (6, y0 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, l, (6, y0 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)


class DemoRecorder:
    def __init__(self, path, title, fps=20):
        self.w = imageio.get_writer(path, fps=fps, macro_block_size=1, quality=7)
        self.title = title
        self.event, self.event_t = "", -100
        self.n = 0

    def mark(self, msg):
        self.event, self.event_t = msg, self.n

    def frame(self, obs, src, pol, gt):
        """obs: env obs, src: state source (vision / gt), pol: LoopPolicy, gt: dict(peg=T, hole=T, L, r, hr)"""
        from collect import np0
        tiles = []
        for cam in ("base_camera", "hand_camera"):
            sd, spr = obs["sensor_data"][cam], obs["sensor_param"][cam]
            img = np0(sd["rgb"]).copy()
            K = np0(spr["intrinsic_cv"])
            w2c = np.eye(4)
            w2c[:3] = np0(spr["extrinsic_cv"])
            if cam == "base_camera" and getattr(src, "last_masks", None) is not None:
                overlay(img, src.last_masks.get("peg"), (255, 60, 60))
                overlay(img, src.last_masks.get("box"), (60, 120, 255))
            L, r, hr = gt["L"], gt["r"], gt["hr"]
            half = np.array([L, r, r])
            draw_box(img, gt["peg"], half, K, w2c, (255, 255, 255), 1)
            draw_hole(img, gt["hole"], L, hr, K, w2c, (255, 255, 255), 1)
            Tp, Th = src.peg(), src.hole()
            if Tp is not None:
                draw_box(img, Tp, half, K, w2c, (40, 255, 40), 2)
            if Th is not None:
                draw_hole(img, Th, L, hr, K, w2c, (40, 160, 255), 2)
            text(img, ["front camera" if cam == "base_camera" else "wrist (hand) camera"], y0=500, s=0.55)
            tiles.append(img)
        img = np.concatenate(tiles, 1)
        Tp, Th = src.peg(), src.hole()
        pe = f"{np.linalg.norm(Tp[:3, 3] - gt['peg'][:3, 3]) * 1000:.1f} mm" if Tp is not None else "-"
        he = f"{np.linalg.norm(Th[:3, 3] - gt['hole'][:3, 3]) * 1000:.1f} mm" if Th is not None else "-"
        lines = [self.title, f"t={pol.n_steps:3d} ({pol.n_steps / 20:.1f}s)  stage: {pol.stage}",
                 f"peg estimate error {pe} | hole estimate error {he}",
                 "green/blue = vision estimate, white = ground truth, red/blue tint = tracked masks"]
        lines.append(">> " + self.event if (self.n - self.event_t < 40 and self.event) else "")  # 줄 수 고정 (프레임 크기 일정)
        bar = np.full((22 * len(lines) + 8, img.shape[1], 3), 30, np.uint8)
        for i, l in enumerate(lines):
            cv2.putText(bar, l, (8, 18 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0) if l.startswith(">>") else (255, 255, 255), 1, cv2.LINE_AA)
        self.w.append_data(np.concatenate([img, bar], 0))
        self.n += 1

    def close(self, result):
        self.w.close()
