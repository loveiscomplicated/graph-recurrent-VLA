"""B3/E1 추가: DAM4SAM (distractor-aware memory, SAM2.1-small) tracking — track_v1.py 와 같은 입출력 형식 (eval_v1.py 로 평가).

DAM4SAM 공식 코드 (third_party/dam4sam @ aad389b) 를 프레임 단위 API(initialize/track)로 물체마다 따로 돌린다 (물체 1 개용 tracker).
초기 bbox 는 results/v1/init/init_boxes_<set>.json (frame 0, GDINO). 늦은 초기화 물체는 추적 안 함 (SAMURAI 와 같은 처리).
vot-toolkit 대신 scripts/vot_shim (Python 3.12 import 오류 우회).
Usage (dam4sam venv, cwd = third_party/dam4sam):
  PYTHONPATH=scripts/vot_shim:scripts $VISION_ENVS/dam4sam/bin/python \
      scripts/track_dam4sam.py --set eval --tag dam4sam_s
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, f"{ROOT}/third_party/dam4sam")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1data  # noqa: E402
from dam4sam_tracker import DAM4SAMTracker  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="eval")
    ap.add_argument("--perturb", default=None)
    ap.add_argument("--eps", nargs="*", default=None)
    ap.add_argument("--tag", default="dam4sam_s")
    ap.add_argument("--out", default=f"{ROOT}/results/v1/E1/runs")
    a = ap.parse_args()
    torch.set_num_threads(8)
    sub = a.set + (f"_{a.perturb}" if a.perturb else "")
    out = f"{a.out}/{a.tag}/{sub}"
    os.makedirs(out, exist_ok=True)
    init = json.load(open(f"{ROOT}/results/v1/init/init_boxes_{a.set}.json"))
    eps = a.eps or v1data.ep_list(a.set)
    trk = DAM4SAMTracker("sam21pp-S")
    res = dict(args=vars(a), episodes={})
    for ei, ep in enumerate([eps[0]] + eps):  # 첫 ep 은 워밍업 (결과 버림)
        rgb = v1data.load(a.set, ep, a.perturb)["base_camera/rgb"]
        T, H, W = rgb.shape[:3]
        masks = {o: np.zeros((T, H, W), bool) for o in ("peg", "box")}
        pf = np.zeros(T)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for o in ("peg", "box"):
                b = init[ep][o]["box"]
                if b is None or init[ep][o].get("frame", 0) != 0:
                    continue
                for t in range(T):
                    img = Image.fromarray(rgb[t])
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    if t == 0:
                        m = trk.initialize(img, None, bbox=[b[0], b[1], b[2] - b[0], b[3] - b[1]])["pred_mask"]
                    else:
                        m = trk.track(img)["pred_mask"]
                    torch.cuda.synchronize()
                    pf[t] += (time.perf_counter() - t0) * 1000
                    masks[o][t] = m > 0
        if ei == 0:
            continue
        nan = np.full(T, np.nan)  # DAM4SAM 은 object score 를 내지 않음
        np.savez_compressed(f"{out}/{ep}_masks.npz", shape=np.array(masks["peg"].shape), peg=np.packbits(masks["peg"]),
                            box=np.packbits(masks["box"]), peg_logit=nan, box_logit=nan, per_frame_ms=pf)
        res["episodes"][ep] = dict(frames=T, ms_mean=round(float(pf[1:].mean()), 2), ms_p50=round(float(np.median(pf[1:])), 2),
                                   ms_p95=round(float(np.percentile(pf[1:], 95)), 2), ms_max=round(float(pf[1:].max()), 1),
                                   gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3), events=[])
        print(a.tag, sub, ep, json.dumps({k: v for k, v in res["episodes"][ep].items() if k != "events"}), flush=True)
    json.dump(res, open(f"{out}/run.json", "w"), indent=1)


if __name__ == "__main__":
    main()
