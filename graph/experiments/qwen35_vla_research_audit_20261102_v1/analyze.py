"""Posthoc audit of frozen perception outputs and pinned public VLA sources."""
from pathlib import Path
import csv,json,hashlib,collections
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'outputs/task10i/stage4a2'
R=BASE/'qwen35_vla_research_audit_20261102_v1';P=BASE/'qwen35_perception_diagnostic_20261102_v1';V=BASE/'dino_qwen_candidate_verifier_20261102_v1';D=BASE/'dino_native_resolution_20261102_v1'
def rd(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def local(p):
 q=Path(p);return BASE/q.parts[2].removeprefix('task10i_stage4a2_')/Path(*q.parts[3:])
def wr(name,obj):(R/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def readcsv(p):
 with p.open() as f:return list(csv.DictReader(f))
def dist(xs):return dict(n=len(xs),mean=float(np.mean(xs)),median=float(np.median(xs)),p95=float(np.quantile(xs,.95)))
def bbox(mask):
 y,x=np.where(mask);return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None
def iou(a,b):
 if a is None or b is None:return 0.
 inter=float(np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod());u=float(np.prod(np.subtract(a[2:],a[:2]))+np.prod(np.subtract(b[2:],b[:2]))-inter);return inter/u if u else 0.

index=rd(P/'artifact_index.json')['files'];assert all(sha(P/name)==v['sha256'] for name,v in index.items())
sources=rd(R/'reference_source_index.json');source_checks=0
for repo,e in sources.items():
 for name,item in e['files'].items():assert sha(R/'reference_sources'/repo/name)==item['sha256'];source_checks+=1
for name in ['common.py','prepare.py','infer.py','finalize.py','run.sh','diagnostic_case_selection.json']:
 assert (P/'.execution'/name).read_bytes()==(ROOT/'graph/experiments/qwen35_perception_diagnostic_20261102_v1'/name).read_bytes()
raw=rd(P/'verifier_raw_responses.json')['responses'];requests=rd(P/'request_manifest.json')['requests'];pid=rd(P/'prompt_identity.json')['conditions'];pred={(r['candidate_id'],r['condition']):r for r in raw};old={r['id']:r for r in rd(V/'verifier_raw_responses.json')['responses']}
assert len(raw)==len(pred)==50
truth={r['id']:r for r in readcsv(V/'candidate_evaluation.csv')};groups=rd(P/'input_manifest.json')['groups'];group_for={cid:g for g,ids in groups.items() for cid in ids};assert len(group_for)==10
gmap={r['id']:r for r in rd(D/'.execution/evaluation_manifest.json')['cases']};masks_by_image={};role_checks=0;iou_checks=0;yy,xx=np.indices((512,512))
for cid in group_for:
 t=truth[cid];ident=t['image_id']
 if ident not in masks_by_image:
  g=gmap[ident];assert sha(local(g['GT_path']))==g['GT_sha256']
  with np.load(local(g['GT_path'])) as z:seg=z['actor_segmentation']
  masks={k:np.isin(seg,g['object_ids'][k]) for k in ['peg','box','robot']};masks['background']=~(masks['peg']|masks['box']|masks['robot']);masks_by_image[ident]=masks
 masks=masks_by_image[ident];b=json.loads(t['box']);rect=(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3]);cnt={k:int((rect&m).sum()) for k,m in masks.items()};assert cnt==json.loads(t['actor_pixels'])
 fg=sum(cnt[k] for k in ['peg','box','robot']);dom=max(['peg','box','robot'],key=lambda k:cnt[k]);role=dom.upper() if fg>=16 and cnt[dom]>=16 and cnt[dom]/fg>=.7 else 'OTHER' if fg<16 else 'AMBIGUOUS';assert role==t['GT_role'];role_checks+=1
 for obj in ['peg','box']:assert abs(iou(b,bbox(masks[obj]))-float(t[obj+'_IoU']))<1e-12;iou_checks+=1
image_checks=0
for req in requests:
 p=pid[req['condition']];r=pred[(req['candidate_id'],req['condition'])];assert r['timing']['initial_forward_cache']['cache_length']==0
 for path,h in zip(req['images'],req['image_sha256']):assert sha(local(path))==h;image_checks+=1
 if p['labels']:assert r['strict_label']==(r['raw_text'].strip() if r['raw_text'].strip() in p['labels'] else 'INVALID_OUTPUT')
for cid in group_for:
 a=next(x for x in requests if x['candidate_id']==cid and x['condition']=='original_three');b=next(x for x in requests if x['candidate_id']==cid and x['condition']=='appearance_three')
 assert a['images']==b['images'] and a['image_sha256']==b['image_sha256']
assert pid['appearance_three']['prompt']==pid['appearance_crop']['prompt'] and pid['appearance_three']['preamble']==pid['appearance_crop']['preamble']
clear=[cid for cid in group_for if truth[cid]['GT_role']!='AMBIGUOUS'];excluded=[dict(candidate_id=cid,reason='AMBIGUOUS actor composition; no unique classification target',all_label_outputs={cond:pred[(cid,cond)]['strict_label'] for cond in ['original_three','appearance_three','appearance_crop']}) for cid in group_for if cid not in clear]
summary={};case_rows=[]
for cond,p in pid.items():
 if p['labels'] is None:
  summary[cond]=dict(n=10,scored=False);continue
 expected=lambda cid:('YES' if truth[cid]['GT_role']=='ROBOT' else 'NO') if cond=='robot_binary_crop' else truth[cid]['GT_role']
 good=[cid for cid in clear if pred[(cid,cond)]['strict_label']==expected(cid)]
 summary[cond]=dict(numerator=len(good),denominator=len(clear),rate=len(good)/len(clear),format_valid_numerator=sum(pred[(cid,cond)]['strict_label']!='INVALID_OUTPUT' for cid in group_for),format_denominator=10,by_GT_role={role:dict(numerator=sum(truth[cid]['GT_role']==role and cid in good for cid in clear),denominator=sum(truth[cid]['GT_role']==role for cid in clear)) for role in ['PEG','BOX','ROBOT']})
for cid in group_for:
 t=truth[cid];case_rows.append(dict(candidate_id=cid,group=group_for[cid],GT_proxy_role=t['GT_role'],scored=cid in clear,PEG_IoU=float(t['peg_IoU']),BOX_IoU=float(t['box_IoU']),historical=old[cid]['strict_label'],original=pred[(cid,'original_three')]['strict_label'],appearance_three=pred[(cid,'appearance_three')]['strict_label'],appearance_crop=pred[(cid,'appearance_crop')]['strict_label'],robot_binary=pred[(cid,'robot_binary_crop')]['strict_label'],caption=pred[(cid,'caption_crop')]['raw_text'].strip()))
historical_matches=sum(pred[(cid,'original_three')]['strict_label']==old[cid]['strict_label'] for cid in group_for)
fixed=[r for r in case_rows if r['scored'] and r['original']!=r['GT_proxy_role'] and r['appearance_three']==r['GT_proxy_role']]
regressed=[r for r in case_rows if r['scored'] and r['original']==r['GT_proxy_role'] and r['appearance_three']!=r['GT_proxy_role']]
lat=rd(P/'verifier_latency_vram.json');timing={}
for cond in pid:
 z=[r['timing'] for r in raw if r['condition']==cond and not r['timing']['cold_first_request']];timing[cond]=dict(warm_end_to_end_ms=dist([x['end_to_end_ms'] for x in z]),peak_allocated_GiB=max(x['memory']['peak_allocated'] for x in z)/1024**3)
result=dict(status='COMPLETE_VERIFIED',purpose='Known-case mechanism diagnosis; not held-out accuracy',requests=50,model_loads=1,parents=len({truth[cid]['parent_id'] for cid in group_for}),camera_frames=len(masks_by_image),conditions=summary,excluded=excluded,historical_baseline_reproduction=dict(numerator=historical_matches,denominator=10),appearance_three_fixed=[r['candidate_id'] for r in fixed],appearance_three_regressed=[r['candidate_id'] for r in regressed],latency=timing,scoring_correction='Original diagnostic_metrics.json counted AMBIGUOUS as an incorrect label although no AMBIGUOUS output exists. That raw artifact is preserved; these clear-role metrics exclude it consistently across every condition. Captions remain unscored.',cases=case_rows)
wr('perception_independent_analysis.json',result)
with (R/'perception_cases.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(case_rows[0]));w.writeheader();w.writerows(case_rows)
# Four prespecified historical failures; no GT box is substituted into model input.
ids=groups['robot_wrong']+groups['peg_wrong'];canvas=Image.new('RGB',(840,960),'white');draw=ImageDraw.Draw(canvas)
for i,cid in enumerate(ids):
 row=next(r for r in case_rows if r['candidate_id']==cid);req=next(r for r in requests if r['candidate_id']==cid and r['condition']=='caption_crop');im=Image.open(local(req['images'][0])).convert('RGB');im.thumbnail((400,390));x=(i%2)*420;y=(i//2)*480;canvas.paste(im,(x,y+65));draw.text((x+5,y+5),cid.replace('10i_s1_','').replace('_native512',''),fill='black');draw.text((x+5,y+23),'Original: '+row['original']+' | appearance: '+row['appearance_three'],fill='black');draw.text((x+5,y+41),'GT actor-role: '+row['GT_proxy_role'],fill='black')
canvas.save(R/'perception_prompt_failure_comparison.png')
verification=dict(status='PASS',remote_indexed_artifacts=len(index),public_source_files=source_checks,local_remote_source_files=6,request_image_hashes=image_checks,GT_role_checks=role_checks,independent_bbox_IoUs=iou_checks,strict_classification_or_binary_outputs=40,initial_cache_empty_requests=50,identical_input_prompt_contrast_cases=10,original_raw_SHA_unchanged=sha(P/'verifier_raw_responses.json')==rd(P/'verification.json')['raw_sha256_before_GT'],all_original_artifacts_preserved=all(sha(P/name)==v['sha256'] for name,v in index.items()),analysis_source_sha256=sha(Path(__file__)))
wr('verification.json',verification)
print(json.dumps(dict(summary=summary,excluded=excluded,fixed=len(fixed),regressed=len(regressed),verification=verification),indent=2))
