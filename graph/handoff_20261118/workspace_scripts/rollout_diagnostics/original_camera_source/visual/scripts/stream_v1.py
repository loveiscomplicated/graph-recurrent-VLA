"""E2: causal streaming SAM2 tracker (미래 프레임 접근 불가, 최근 프레임만 유지, JPEG/전 프레임 로드 없음).

- `CausalFrames`: SAM2 inference_state["images"] 자리에 들어가는 프레임 버퍼. push 한 프레임까지만 접근 가능하고
  미래 index 를 읽으면 FutureFrameAccess 예외. 최근 `keep` 프레임만 GPU 에 유지 (오래된 index 접근도 예외).
- `StreamTracker.start(rgb0, boxes)` → frame 0 mask, `step(rgb_t)` → frame t mask. 내부는 SAM2VideoPredictor 의
  `propagate_in_video(start_frame_idx=t, max_frame_num_to_track=0)` (offline 과 같은 연산).
- 옵션 (E2 ablation)
  preprocess : "pil" (offline 과 같은 PIL bicubic resize, CPU) | "gpu" (torch bicubic on GPU)
  prune      : 오래된 비조건 memory 삭제 (현재 프레임 − keep_mem 이전). SAM2 는 최근 6 프레임 mask memory 와
               최근 15 프레임 object pointer 만 읽으므로 keep_mem=16 이면 결과가 같아야 한다
  model image_size : build 시 hydra override (1024 기본, 768/512)
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class FutureFrameAccess(RuntimeError):
    pass


class CausalFrames:
    def __init__(self, image_size, device="cuda", keep=4, preprocess="gpu", max_frames=100000):
        self.S, self.dev, self.keep, self.pre = image_size, device, keep, preprocess
        self.buf = {}
        self.latest = -1
        self.max_frames = max_frames
        self.mean, self.std = MEAN.to(device), STD.to(device)

    def __len__(self):
        return self.max_frames  # SAM2 는 num_frames 를 상한 체크에만 쓴다

    def push(self, rgb):
        S = self.S
        if self.pre == "pil":
            im = np.array(Image.fromarray(rgb).resize((S, S))) / 255.0
            x = torch.from_numpy(im).permute(2, 0, 1).float().to(self.dev)
        else:
            x = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.dev, non_blocking=True).permute(2, 0, 1).float()[None] / 255.0
            x = F.interpolate(x, size=(S, S), mode="bicubic", align_corners=False)[0].clamp_(0, 1)
        x = (x - self.mean) / self.std
        self.latest += 1
        self.buf[self.latest] = x
        for k in [k for k in self.buf if k <= self.latest - self.keep]:
            del self.buf[k]
        return self.latest

    def __getitem__(self, i):
        if i > self.latest:
            raise FutureFrameAccess(f"frame {i} requested, latest pushed = {self.latest}")
        if i not in self.buf:
            raise FutureFrameAccess(f"frame {i} already dropped (keep={self.keep})")
        return self.buf[i]


class StreamTracker:
    def __init__(self, predictor, preprocess="gpu", prune=True, keep_mem=16, keep_frames=2):
        self.p = predictor
        self.pre, self.prune, self.keep_mem, self.keep_frames = preprocess, prune, keep_mem, keep_frames
        self.state = None

    def _init_state(self, rgb0):
        import importlib
        H, W = rgb0.shape[:2]
        mod = importlib.import_module(type(self.p).__module__)
        frames = CausalFrames(self.p.image_size, keep=self.keep_frames, preprocess=self.pre)
        frames.push(rgb0)  # init_state 가 frame 0 feature 를 미리 계산한다
        orig = mod.load_video_frames
        mod.load_video_frames = lambda **kw: (frames, H, W)
        try:
            st = self.p.init_state(video_path="<stream>")
        finally:
            mod.load_video_frames = orig
        self.frames = frames
        return st

    def start(self, rgb0, boxes):
        """boxes: {obj_id: [x1,y1,x2,y2] or None}"""
        self.state = self._init_state(rgb0)
        self.obj_ids = [o for o, b in boxes.items() if b is not None]
        for oid in self.obj_ids:
            self.p.add_new_points_or_box(self.state, frame_idx=0, obj_id=oid, box=np.asarray(boxes[oid], np.float32))
        return self._propagate(0)

    def _propagate(self, t):
        out = {}
        for fi, oids, res in self.p.propagate_in_video(self.state, start_frame_idx=t, max_frame_num_to_track=0):
            for j, oid in enumerate(oids):
                out[oid] = res[j, 0] > 0
        if self.prune:
            self._prune(t)
        return out

    def step(self, rgb):
        t = self.frames.push(rgb)
        return t, self._propagate(t)

    def logit(self, oid, t):
        od = self.state["output_dict_per_obj"][self.p._obj_id_to_idx(self.state, oid)]
        o = od["cond_frame_outputs"].get(t) or od["non_cond_frame_outputs"].get(t)
        return float(o["object_score_logits"].flatten()[0])

    def _prune(self, t):
        lim = t - self.keep_mem
        st = self.state
        for i, od in st["output_dict_per_obj"].items():
            od["non_cond_frame_outputs"].pop(lim, None)
            st["frames_tracked_per_obj"][i].pop(lim, None)

    def add_box(self, t, oid, box):
        _, oids, res = self.p.add_new_points_or_box(self.state, frame_idx=t, obj_id=oid, box=np.asarray(box, np.float32))
        return res[list(oids).index(oid), 0] > 0

    def add_mask(self, t, oid, mask):
        _, oids, res = self.p.add_new_mask(self.state, frame_idx=t, obj_id=oid, mask=mask)
        return res[list(oids).index(oid), 0] > 0

    def reset(self):
        if self.state is not None:
            self.p.reset_state(self.state)
            self.state = None
