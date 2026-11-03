"""Task-local, frozen 2x2 paired native-resolution/prompt pilot utilities."""
from pathlib import Path
import os,sys,json,hashlib,csv,math,re
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
S=Path('/workspace/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/source');sys.path.insert(0,str(S))
R=Path('/workspace/task10i_stage4a2_qwen35_native_resolution_prompt_20261102_v1')
PREV=Path('/workspace/task10i_stage4a2_qwen35_precision_comparison_20261102_v1')
B=Path('/workspace/task10i_stage4a2_qwen35_verifier_20261102_v1')
VARIANTS=['native128_P0','native512_P0','native128_P1','native512_P1']
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
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def parse(text,width,height):
 s=text.strip();f=re.fullmatch(r'```(?:json)?\s*\n(.*)\n```',s,re.S);wrapper='SINGLE_JSON_FENCE' if f else 'BARE_JSON';payload=f.group(1) if f else s
 def pairs(xs):
  d={}
  for k,v in xs:
   if k in d:raise ValueError('duplicate key')
   d[k]=v
  return d
 try:
  arr=json.loads(payload,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
  assert isinstance(arr,list) and len(arr)==2
  assert all(isinstance(x,dict) and set(x)=={'label','bbox_2d'} for x in arr)
  assert {x['label'] for x in arr}=={'PEG','BOX'}
  d={x['label'].lower():x['bbox_2d'] for x in arr}
 except Exception:
  return dict(json_valid=False,strict_json_valid=False,wrapper=wrapper,objects={k:dict(status='INVALID_JSON',normalized=None,pixel=None) for k in ['peg','box']})
 out={}
 for k,b in d.items():
  if b is None:out[k]=dict(status='NULL',normalized=None,pixel=None);continue
  ok=isinstance(b,list) and len(b)==4 and all(type(x) in (int,float) and math.isfinite(x) for x in b)
  ok=ok and all(0<=x<=1000 for x in b) and b[0]<b[2] and b[1]<b[3]
  out[k]=dict(status='VALID_BOX' if ok else 'INVALID_BOX',normalized=b,pixel=[float(x)*([width,height,width,height][j])/1000 for j,x in enumerate(b)] if ok else None)
 return dict(json_valid=True,strict_json_valid=wrapper=='BARE_JSON',wrapper=wrapper,objects=out)
