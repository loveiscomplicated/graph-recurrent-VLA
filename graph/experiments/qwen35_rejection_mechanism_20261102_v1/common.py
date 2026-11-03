"""Task-local held-out appearance-prompt evaluation. No production imports change."""
from pathlib import Path
import os,sys,json,hashlib,csv,math
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
S=Path('/workspace/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/source');sys.path.insert(0,str(S))
R=Path('/workspace/task10i_stage4a2_qwen35_rejection_mechanism_20261102_v1')
PREV=Path('/workspace/task10i_stage4a2_qwen35_appearance_bf16_independent_20261102_v1')
B=Path('/workspace/task10i_stage4a2_qwen35_verifier_20261102_v1')
N=Path('/workspace/task10i_stage4a2_qwen35_native_resolution_prompt_20261102_v1')
P=Path('/workspace/task10i_stage4a2_qwen35_perception_diagnostic_20261102_v1')
D=Path('/workspace/task10i_stage4a2_dino_native_resolution_20261102_v1')
MODEL=Path('/workspace/full_pipeline_models/dino')
LABELS=['PEG','BOX','ROBOT','OTHER','UNCERTAIN']
def rd(p):return json.loads(Path(p).read_text())
def wr(n,x):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,default=str,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def csvwr(n,rows):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:
  if not rows:return
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(x) if isinstance(x,(dict,list)) else x for k,x in r.items()} for r in rows)
def iou(a,b):
 if a is None or b is None:return 0.
 z=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z
 return z/u if u else 0.
