"""POST-HOC output-contract diagnosis. Never accepted, never sent to SAM2.

Only complete JSON documents (bare or a whole fence) are examined. Repeated
keys must agree exactly; ambiguous repeated labels/coordinates are not recovered.
Extra fields may be ignored diagnostically. This is not a parser contract change.
"""
from common import *
import collections
import numpy as np
v=rd(R/'verification.json');assert v['stage']=='EVALUATION_COMPLETE'
raw=rd(R/'full_image_grounding_raw_responses.json')['responses'];mf=rd(R/'input_manifest.json');cases={c['id']:c for c in mf['cases']}
with (R/'.execution/object_evaluation.csv').open() as f:original={(r['variant'],r['pair_id'],r['object']):r for r in csv.DictReader(f)}
class Obj(list):pass
def extract(text):
 s=text.strip();f=re.fullmatch(r'```(?:json)?\s*\n(.*)\n```',s,re.S);s=f.group(1) if f else s;issues=[]
 try:a=json.loads(s,object_pairs_hook=lambda x:Obj(x),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
 except Exception:return None,['INCOMPLETE_OR_UNPARSEABLE_JSON']
 if not type(a) is list or len(a)!=2 or not all(isinstance(x,Obj) for x in a):return None,['WRONG_TOP_LEVEL_SCHEMA']
 result={}
 for obj in a:
  vals=collections.defaultdict(list)
  for k,value in obj:vals[k].append(value)
  for k,vs in vals.items():
   if len(vs)>1:
    issues.append('DUPLICATE_KEY_IDENTICAL' if all(x==vs[0] for x in vs) else 'DUPLICATE_KEY_CONFLICTING')
  if any(not all(x==vs[0] for x in vs) for vs in vals.values()):return None,issues
  if set(vals)-{'label','bbox_2d'}:issues.append('EXTRA_KEYS')
  if 'label' not in vals or 'bbox_2d' not in vals:return None,issues+['MISSING_KEYS']
  label=vals['label'][0];b=vals['bbox_2d'][0]
  if label not in ['PEG','BOX'] or label.lower() in result:return None,issues+['AMBIGUOUS_LABEL']
  if b is not None:
   good=type(b) is list and len(b)==4 and all(type(x) in [int,float] and math.isfinite(x) for x in b)
   good=good and all(0<=x<=1000 for x in b) and b[0]<b[2] and b[1]<b[3]
   if not good:return None,issues+['INVALID_BOX']
  result[label.lower()]=b
 if set(result)!= {'peg','box'}:return None,issues+['MISSING_TARGET']
 return result,sorted(set(issues))
def iou(a,b):
 if a is None or b is None:return 0.
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));den=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
 return inter/den if den else 0.
rows=[];responses=[]
for r in raw:
 c=cases[r['id']];d,issues=extract(r['raw_text']);responses.append(dict(variant=r['variant'],pair_id=r['pair_id'],accepted_json=r['parsed']['json_valid'],diagnostic_readable=d is not None,issues=issues))
 for k in ['peg','box']:
  o=original[(r['variant'],r['pair_id'],k)];gb=json.loads(o['GT_bbox']);b=d[k] if d else None;p=[x*c['native_size']/1000 for x in b] if b else None
  rows.append(dict(variant=r['variant'],pair_id=r['pair_id'],parent_id=c['parent_id'],camera=c['camera'],frame_kind=c['frame_kind'],object=k,accepted_json=r['parsed']['json_valid'],diagnostic_readable=d is not None,issues=issues,normalized=b,pixel=p,visibility=o['visibility'],GT_bbox=gb,bbox_IoU=iou(p,gb) if gb else None,correct_IoU05=bool(p and gb and iou(p,gb)>=.5),strict_accepted_correct=o['result']=='CORRECT'))
csvwr('posthoc_coordinate_diagnostic.csv',[{k:json.dumps(x) if isinstance(x,(dict,list)) else x for k,x in r.items()} for r in rows])
metrics={}
for mode in VARIANTS:
 rs=[r for r in rows if r['variant']==mode];rr=[r for r in responses if r['variant']==mode];issues=collections.Counter(i for r in rr for i in r['issues']);objs={}
 for k in ['peg','box']:
  vis=[r for r in rs if r['object']==k and r['visibility']=='VISIBLE'];objs[k]=dict(numerator=sum(r['correct_IoU05'] for r in vis),denominator=len(vis),recall_IoU05=sum(r['correct_IoU05'] for r in vis)/len(vis) if vis else None,mean_bbox_IoU_including_unreadable=float(np.mean([r['bbox_IoU'] for r in vis])) if vis else None,accepted_correct=sum(r['strict_accepted_correct'] for r in vis),diagnostic_only_correct=sum(r['correct_IoU05'] and not r['accepted_json'] for r in vis))
 metrics[mode]=dict(images=len(rr),diagnostic_readable=sum(r['diagnostic_readable'] for r in rr),contract_issue_counts=dict(issues),objects=objs,by_camera={cam:{k:dict(numerator=sum(r['correct_IoU05'] for r in rs if r['camera']==cam and r['object']==k and r['visibility']=='VISIBLE'),denominator=sum(r['camera']==cam and r['object']==k and r['visibility']=='VISIBLE' for r in rs)) for k in ['peg','box']} for cam in ['base_camera','hand_camera']})
for mode in VARIANTS:
 rs=[r for r in rows if r['variant']==mode]
 pp={p:{k:dict(numerator=sum(r['correct_IoU05'] for r in rs if r['parent_id']==p and r['object']==k and r['visibility']=='VISIBLE'),denominator=sum(r['parent_id']==p and r['object']==k and r['visibility']=='VISIBLE' for r in rs)) for k in ['peg','box']} for p in sorted({r['parent_id'] for r in rs})}
 metrics[mode]['per_parent']=pp
 metrics[mode]['parent_macro_recall']={k:float(np.mean([a[k]['numerator']/a[k]['denominator'] for a in pp.values() if a[k]['denominator']])) for k in ['peg','box']}
# Compare localization on the same emission-valid subset to separate parser failures.
paired={}
look={(r['variant'],r['pair_id'],r['object']):r for r in rows}
for prompt in ['P0','P1']:
 a='native128_'+prompt;b='native512_'+prompt;paired[prompt]={}
 for k in ['peg','box']:
  aa=[r for r in rows if r['variant']==a and r['object']==k and r['visibility']=='VISIBLE' and r['diagnostic_readable'] and look[(b,r['pair_id'],k)]['diagnostic_readable'] and look[(b,r['pair_id'],k)]['visibility']=='VISIBLE'];bb=[look[(b,r['pair_id'],k)] for r in aa]
  paired[prompt][k]=dict(common_readable_denominator=len(aa),native128_correct=sum(r['correct_IoU05'] for r in aa),native512_correct=sum(r['correct_IoU05'] for r in bb),native128_mean_IoU=float(np.mean([r['bbox_IoU'] for r in aa])) if aa else None,native512_mean_IoU=float(np.mean([r['bbox_IoU'] for r in bb])) if bb else None)
wr('posthoc_coordinate_diagnostic.json',dict(posthoc=True,accepted=False,SAM2_executions=0,no_clipping=True,rule='Complete JSON only. Duplicate values must match exactly; conflicting values rejected; extra fields ignored only diagnostically. No GT selection/correction. Diagnostic coordinates never promoted to accepted grounding.',conditions=metrics,paired_common_readable=paired,responses=responses,source_sha256=sha(R/'.execution/diagnose.py'),raw_sha256=sha(R/'full_image_grounding_raw_responses.json')))
print('DIAGNOSTIC_ONLY',json.dumps(metrics),flush=True)
