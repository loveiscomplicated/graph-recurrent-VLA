"""B3 추가: SAM2Long (memory tree, training-free, SAM2.1-small) — track_v1.py 와 같은 출력 형식.

주의: 공개 코드는 영상 끝에서 누적 점수가 가장 높은 pathway 를 골라 전 프레임 mask 를 확정한다 (미래 정보 사용, offline).
그래서 이 결과는 "memory tree 가 이 데이터에서 주는 이득의 상한" 이고, 제어 루프용(causal) 성능이 아니다.
설정은 공식 기본값 num_pathway=3, iou_thre=0.1, uncertainty=2. 초기 bbox 는 init_boxes (frame 0), 늦은 초기화 물체는 추적 안 함.
Usage (sam2long venv): $VISION_ENVS/sam2long/bin/python scripts/track_sam2long.py --set eval --tag sam2long_s
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1data  # noqa: E402
from track_v1 import _ARRAY, _array_loader  # noqa: E402  (같은 PIL 전처리)

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OBJS = {1: "peg", 2: "box"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="eval")
    ap.add_argument("--perturb", default=None)
    ap.add_argument("--eps", nargs="*", default=None)
    ap.add_argument("--tag", default="sam2long_s")
    ap.add_argument("--out", default=f"{ROOT}/results/v1/E1/runs")
    a = ap.parse_args()
    torch.set_num_threads(8)
    import sam2.sam2_video_predictor as vp
    from sam2.build_sam import build_sam2_video_predictor
    vp.load_video_frames = _array_loader
    pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", f"{ROOT}/checkpoints/sam2.1_hiera_small.pt", device="cuda")
    sub = a.set + (f"_{a.perturb}" if a.perturb else "")
    out = f"{a.out}/{a.tag}/{sub}"
    os.makedirs(out, exist_ok=True)
    init = json.load(open(f"{ROOT}/results/v1/init/init_boxes_{a.set}.json"))
    eps = a.eps or v1data.ep_list(a.set)
    res = dict(args=vars(a), note="offline: pathway chosen at end of video", episodes={})
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ei, ep in enumerate([eps[0]] + eps):
            rgb = v1data.load(a.set, ep, a.perturb)["base_camera/rgb"]
            T, H, W = rgb.shape[:3]
            _ARRAY["mem"] = rgb
            masks = {n: np.zeros((T, H, W), bool) for n in OBJS.values()}
            cmasks = {n: np.zeros((T, H, W), bool) for n in OBJS.values()}
            torch.cuda.empty_cache()  # offline 방식이라 영상마다 전 프레임 × pathway 를 GPU 에 둔다 → 매번 비움
            torch.cuda.reset_peak_memory_stats()
            objs = [(o, n) for o, n in OBJS.items() if init[ep][n]["box"] is not None and init[ep][n].get("frame", 0) == 0]
            load_ms = ms_total = 0.0
            for oid, n in objs:  # SAM2Long 은 상태 하나에 물체 하나 (iou.item() 가정) → 물체별로 따로
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                st = pred.init_state(video_path="mem")
                st["num_pathway"], st["iou_thre"], st["uncertainty"] = 3, 0.1, 2
                pred.add_new_points_or_box(st, frame_idx=0, obj_id=oid, box=np.array(init[ep][n]["box"], np.float32))
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                _, mlist = pred.propagate_in_video(st)
                torch.cuda.synchronize()
                load_ms += (t1 - t0) * 1000
                ms_total += (time.perf_counter() - t1) * 1000
                for t, m in enumerate(mlist):
                    masks[n][t] = (m[0, 0] > 0).cpu().numpy()
                # causal: 각 프레임 저장 시점의 1등 pathway (topk 내림차순 → index 0). 미래 정보 없이 그 프레임에서 고른 것
                nc = st["output_dict"]["non_cond_frame_outputs"]
                for t in range(1, T):
                    cm = pred._get_orig_video_res_output(st, nc[t]["pred_masks"][..., 0])[1]
                    cmasks[n][t] = (cm[0, 0] > 0).cpu().numpy()
                cmasks[n][0] = masks[n][0]
                pred.reset_state(st)
            ms_per_frame = ms_total / T
            _ARRAY.pop("mem")
            if ei == 0:
                continue
            nan = np.full(T, np.nan)
            pf = np.full(T, ms_per_frame)  # offline 이라 프레임별 시간 없음: 평균만
            cout = out.replace(f"/{a.tag}/", f"/{a.tag}_causal/")
            os.makedirs(cout, exist_ok=True)
            np.savez_compressed(f"{cout}/{ep}_masks.npz", shape=np.array(masks["peg"].shape), peg=np.packbits(cmasks["peg"]),
                                box=np.packbits(cmasks["box"]), peg_logit=nan, box_logit=nan, per_frame_ms=pf)
            np.savez_compressed(f"{out}/{ep}_masks.npz", shape=np.array(masks["peg"].shape), peg=np.packbits(masks["peg"]),
                                box=np.packbits(masks["box"]), peg_logit=nan, box_logit=nan, per_frame_ms=pf)
            res["episodes"][ep] = dict(frames=T, load_ms=round(load_ms, 1), ms_mean=round(ms_per_frame, 2), ms_p50=round(ms_per_frame, 2),
                                       ms_p95=round(ms_per_frame, 2), ms_max=round(ms_per_frame, 1),
                                       gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3), events=[])
            print(a.tag, sub, ep, json.dumps({k: v for k, v in res["episodes"][ep].items() if k != "events"}), flush=True)
    json.dump(res, open(f"{out}/run.json", "w"), indent=1)
    json.dump(dict(res, note="causal: per-frame best pathway at that time"), open(f"{out.replace(f'/{a.tag}/', f'/{a.tag}_causal/')}/run.json", "w"), indent=1)


if __name__ == "__main__":
    main()
