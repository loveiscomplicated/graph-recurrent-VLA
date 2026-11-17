"""B3: SAM2Long 의 causal streaming 래퍼 (물체마다 상태 하나, 프레임 단위).

SAM2Long 공개 코드는 propagate_in_video 에서 전 프레임을 돌고 끝에 pathway 를 고른다(offline). 여기서는 같은 내부 함수
`_run_single_frame_inference(..., mem_pick_indexs, start_frame_idx=0)` 를 프레임마다 부르고, 그 프레임 저장 시점의 1등 pathway
(topk 내림차순 → index 0) mask 를 낸다. eval 40 ep 에서 이 causal 선택은 offline 과 같은 IoU 였다 (REPORT B3).
import 전에 sys.path 맨 앞에 third_party/sam2long 이 있어야 한다 (패키지 이름 sam2 가 vision venv 의 것과 같음).
"""
import numpy as np
import torch

from stream_v1 import CausalFrames

NUM_PATHWAY, IOU_THRE, UNCERTAINTY = 3, 0.1, 2  # 공식 기본값


class SAM2LongStream:
    def __init__(self, predictor, preprocess="gpu"):
        self.p, self.pre = predictor, preprocess
        self.st = {}

    def _state(self, rgb0):
        import importlib
        mod = importlib.import_module(type(self.p).__module__)
        frames = CausalFrames(self.p.image_size, keep=2, preprocess=self.pre)
        frames.push(rgb0)
        orig = mod.load_video_frames
        mod.load_video_frames = lambda **kw: (frames, rgb0.shape[0], rgb0.shape[1])
        try:
            st = self.p.init_state(video_path="<stream>")
        finally:
            mod.load_video_frames = orig
        st["num_pathway"], st["iou_thre"], st["uncertainty"] = NUM_PATHWAY, IOU_THRE, UNCERTAINTY
        return st, frames

    @torch.inference_mode()
    def start(self, rgb0, boxes):
        """boxes: {name: [x1,y1,x2,y2] or None} → {name: (mask bool cuda, logit)}"""
        self.st, self.t = {}, 0
        out = {}
        for name, b in boxes.items():
            if b is None:
                continue
            st, frames = self._state(rgb0)
            _, _, m = self.p.add_new_points_or_box(st, frame_idx=0, obj_id=1, box=np.asarray(b, np.float32))
            self.p.propagate_in_video_preflight(st)
            self.st[name] = dict(st=st, frames=frames, pick=0)
            out[name] = (m[0, 0] > 0, 10.0)
        return out

    @torch.inference_mode()
    def step(self, rgb):
        self.t += 1
        t = self.t
        out = {}
        for name, o in self.st.items():
            o["frames"].push(rgb)
            st = o["st"]
            cur, _, o["pick"] = self.p._run_single_frame_inference(
                inference_state=st, output_dict=st["output_dict"], frame_idx=t, batch_size=1, is_init_cond_frame=False,
                point_inputs=None, mask_inputs=None, reverse=False, run_mem_encoder=True, mem_pick_indexs=o["pick"], start_frame_idx=0)
            st["output_dict"]["non_cond_frame_outputs"][t] = cur
            m = self.p._get_orig_video_res_output(st, cur["pred_masks"][..., 0])[1]
            lg = cur["object_score_logits"]
            lg = float(lg[..., 0].flatten()[0]) if lg.dim() > 2 else float(lg.flatten()[0])
            out[name] = (m[0, 0] > 0, lg)
        return t, out

    def reset(self):
        for o in self.st.values():
            self.p.reset_state(o["st"])
        self.st = {}
