from pathlib import Path
import os, sys, json, hashlib, csv, math
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
S=Path('/workspace/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/source')
sys.path.insert(0,str(S))
R=Path('/workspace/task10i_stage4a2_qwen35_full_image_grounding_bf16_20261102_v1')
B=Path('/workspace/task10i_stage4a2_qwen35_verifier_20261102_v1')
def rd(p): return json.loads(Path(p).read_text())
def wr(n,x):
 p=R/n; p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(x,indent=2,allow_nan=False,default=str)+'\n'); tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for z in iter(lambda:f.read(4*1024**2),b''): h.update(z)
 return h.hexdigest()
def csvwr(n,rows,fields=None):
 with (R/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0])); w.writeheader(); w.writerows(rows)
def parse(text):
 def pairs(xs):
  d={}
  for k,v in xs:
   if k in d: raise ValueError('duplicate key')
   d[k]=v
  return d
 try:
  d=json.loads(text,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
  if not isinstance(d,dict) or set(d)!={'peg','box'}: raise ValueError('schema')
 except Exception:
  return dict(json_valid=False,objects={k:dict(status='INVALID_JSON',normalized=None,pixel=None) for k in ['peg','box']})
 out={}
 for k,b in d.items():
  if b is None: out[k]=dict(status='NULL',normalized=None,pixel=None); continue
  ok=isinstance(b,list) and len(b)==4 and all(type(x) in (int,float) and math.isfinite(x) for x in b)
  ok=ok and all(0<=x<=1000 for x in b) and b[0]<b[2] and b[1]<b[3]
  out[k]=dict(status='VALID_BOX' if ok else 'INVALID_BOX',normalized=b,pixel=[float(x)*128/1000 for x in b] if ok else None)
 return dict(json_valid=True,objects=out)
