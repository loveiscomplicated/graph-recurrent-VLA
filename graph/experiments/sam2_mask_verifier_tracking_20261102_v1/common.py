"""Independent Task10I experiment; no production interfaces are modified."""
from pathlib import Path
import os, sys, json, hashlib, csv, math, time
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
S = Path('/workspace/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/source')
sys.path.insert(0, str(S))
R = Path('/workspace/task10i_stage4a2_sam2_mask_verifier_tracking_20261102_v1')
P = Path('/workspace/task10i_stage4a2_qwen35_appearance_bf16_independent_20261102_v1')
B = Path('/workspace/task10i_stage4a2_qwen35_verifier_20261102_v1')
CK = Path('/workspace/graph-recurrent-VLA/visual/checkpoints/sam2.1_hiera_small.pt')
CFG = 'configs/sam2.1/sam2.1_hiera_s.yaml'
LABELS = ['PEG', 'BOX', 'ROBOT', 'OTHER', 'UNCERTAIN']
METHODS = ['A', 'B', 'C']
def rd(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''): h.update(block)
    return h.hexdigest()
def wr(name, obj):
    p = R/name; p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False)+'\n'); tmp.replace(p)
def csvwr(name, rows):
    p = R/name; p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w') as f:
        if not rows: return
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        w.writerows({k:json.dumps(x) if isinstance(x,(dict,list)) else x for k,x in r.items()} for r in rows)
def update(**kwargs):
    v = rd(R/'verification.json'); v.update(kwargs); wr('verification.json', v); return v
def selected(candidates, labels, image_id, obj):
    pool = [c for c in candidates if c['image_id']==image_id and c['query_object']==obj and labels.get(c['id'])==obj.upper()]
    return min(pool, key=lambda c:(-c['DINO_score'], c['rank'], c['id'])) if pool else None
def mask_metrics(pred, target, others):
    import numpy as np
    assert pred.dtype == bool and pred.shape == target.shape
    n = int(pred.sum()); g = int(target.sum()); tp = int((pred & target).sum()); union = n+g-tp
    counts = {k:int((pred & m).sum()) for k,m in others.items()}
    return dict(predicted_pixels=n, visible_pixels=g, intersection=tp,
                IoU=tp/union if union else None, pixel_precision=tp/n if n else None,
                pixel_recall=tp/g if g else None, actor_pixels=counts,
                wrong_object=bool(n and any(counts[k]/n>=.5 for k in ['peg','box','robot'] if k in counts)),
                contamination={k:v/n if n else None for k,v in counts.items()})
def rate(rows, key):
    n=sum(bool(r[key]) for r in rows)
    return dict(numerator=n, denominator=len(rows), rate=n/len(rows) if rows else None)
def gpu_mem():
    import torch
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
