"""Step 1 smoke test: import + GPU load for ManiSkill3, Grounding DINO (HF), SAM2.

Usage:
  source env.sh
  python scripts/smoke_test.py --out results/step1
"""
import argparse
import json
import os
import time
import traceback

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = f"{ROOT}/checkpoints"
SAM2_CFGS = {
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", f"{CKPT}/sam2.1_hiera_small.pt"),
}
GDINO_ID = "IDEA-Research/grounding-dino-tiny"


def gb(x):
    return round(x / 1024**3, 3)


def test_env(results, out):
    print("== torch / cuda")
    results["torch"] = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(results["torch"])
    if not results["torch"]["available"]:
        raise RuntimeError("CUDA unavailable; check NVIDIA driver and container GPU passthrough")


def test_maniskill(results, out):
    print("== ManiSkill3")
    import gymnasium as gym
    import mani_skill
    import mani_skill.envs  # noqa: F401  registers envs
    import sapien

    t0 = time.time()
    env = gym.make("PegInsertionSide-v1", obs_mode="rgbd", render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    cams = obs["sensor_data"]
    cam_name = list(cams.keys())[0]
    rgb = cams[cam_name]["rgb"][0].cpu().numpy().astype(np.uint8)
    depth = cams[cam_name]["depth"][0].cpu().numpy()
    Image.fromarray(rgb).save(f"{out}/maniskill_rgb.png")
    render = env.render()
    render = render[0].cpu().numpy() if torch.is_tensor(render) else render
    Image.fromarray(render.astype(np.uint8)).save(f"{out}/maniskill_render.png")
    env.close()
    results["maniskill"] = {
        "ok": True,
        "mani_skill": mani_skill.__version__,
        "sapien": sapien.__version__,
        "camera": cam_name,
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "depth_range": [float(depth.min()), float(depth.max())],
        "sec": round(time.time() - t0, 2),
    }
    print(results["maniskill"])
    return Image.fromarray(rgb)


def test_gdino(results, out, image):
    print("== Grounding DINO (HF transformers)")
    import transformers
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(GDINO_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).cuda().eval()
    load_s = time.time() - t0
    text = "peg. box with hole."
    inputs = processor(images=image, text=text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**inputs)  # warmup
        torch.cuda.synchronize()
        t1 = time.time()
        outputs = model(**inputs)
        torch.cuda.synchronize()
    infer_ms = (time.time() - t1) * 1000
    det = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.25, text_threshold=0.25,
        target_sizes=[image.size[::-1]],
    )[0]
    vis = image.copy()
    d = ImageDraw.Draw(vis)
    dets = []
    labels = det.get("text_labels", det.get("labels"))
    for box, score, lab in zip(det["boxes"].tolist(), det["scores"].tolist(), labels):
        d.rectangle(box, outline=(255, 0, 0), width=2)
        d.text((box[0] + 2, box[1] + 2), f"{lab} {score:.2f}", fill=(255, 255, 0))
        dets.append({"label": str(lab), "score": round(score, 3), "box": [round(v, 1) for v in box]})
    vis.save(f"{out}/gdino_overlay.png")
    results["gdino"] = {
        "ok": True,
        "transformers": transformers.__version__,
        "model": GDINO_ID,
        "device": str(next(model.parameters()).device),
        "load_s": round(load_s, 2),
        "infer_ms": round(infer_ms, 1),
        "peak_mem_gb": gb(torch.cuda.max_memory_allocated()),
        "prompt": text,
        "detections": dets,
    }
    print(results["gdino"])
    del model
    torch.cuda.empty_cache()
    return dets


def test_sam2(results, out, image, dets):
    print("== SAM2")
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    img = np.array(image)
    h, w = img.shape[:2]
    # prompt box: first detection if any, else image center region
    box = np.array(dets[0]["box"]) if dets else np.array([w * 0.3, h * 0.3, w * 0.7, h * 0.7])
    results["sam2"] = {}
    for name, (cfg, ckpt) in SAM2_CFGS.items():
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model = build_sam2(cfg, ckpt, device="cuda")
        pred = SAM2ImagePredictor(model)
        load_s = time.time() - t0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred.set_image(img)
            masks, scores, _ = pred.predict(box=box[None], multimask_output=False)
        m = masks[0].astype(bool)
        vis = img.copy()
        vis[m] = (0.5 * vis[m] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        v = Image.fromarray(vis)
        ImageDraw.Draw(v).rectangle(box.tolist(), outline=(255, 0, 0), width=2)
        v.save(f"{out}/sam2_{name}_mask.png")
        del pred, model
        torch.cuda.empty_cache()
        # video predictor build check
        t1 = time.time()
        vp = build_sam2_video_predictor(cfg, ckpt, device="cuda")
        vp_s = time.time() - t1
        del vp
        torch.cuda.empty_cache()
        results["sam2"][name] = {
            "ok": True,
            "ckpt": os.path.basename(ckpt),
            "image_predictor_load_s": round(load_s, 2),
            "video_predictor_load_s": round(vp_s, 2),
            "mask_pixels": int(m.sum()),
            "score": float(scores[0]),
            "prompt_box": [round(float(v), 1) for v in box],
            "peak_mem_gb": gb(torch.cuda.max_memory_allocated()),
        }
        print(name, results["sam2"][name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/results/step1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = {}
    image, dets = None, []
    for name, fn in [("env", lambda: test_env(results, args.out)),
                     ("maniskill", lambda: test_maniskill(results, args.out)),
                     ("gdino", lambda: test_gdino(results, args.out, image)),
                     ("sam2", lambda: test_sam2(results, args.out, image, dets))]:
        try:
            r = fn()
            if name == "maniskill":
                image = r
            elif name == "gdino":
                dets = r
        except Exception as e:
            traceback.print_exc()
            results[name] = {"ok": False, "error": repr(e)}
            if name == "maniskill":  # fallback image so model tests still run
                image = Image.fromarray(np.full((256, 256, 3), 127, np.uint8))
    with open(f"{args.out}/smoke_test.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("saved", f"{args.out}/smoke_test.json")
    if any(v.get("ok") is False for v in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
