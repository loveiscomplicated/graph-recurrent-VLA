from common import *
import collections,subprocess
import numpy as np
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json');assert v['stage']=='PREDICTIONS_FROZEN' and v['model_loads']==1 and v['completed_requests']==326
assert sha(R/'verifier_raw_responses.json')==v['raw_sha256_before_GT'] and sha(R/'verifier_predictions.csv')==v['predictions_sha256_before_GT']
assert sha(R/'case_manifest.json')==v['case_manifest_sha256']
cases=rd(R/'case_manifest.json');prompts=rd(R/'prompt_identity.json')['conditions'];raw=rd(R/'verifier_raw_responses.json')['responses'];pred={(r['candidate_id'],r['condition']):r for r in raw};conditions=[k for k in prompts if k!='caption'];summary={};case_rows=[];labelchanges=[]
def rate(rs,fn):
 n=sum(bool(fn(r)) for r in rs);return dict(numerator=n,denominator=len(rs),rate=n/len(rs) if rs else None)
for c in cases['cases']:
 row=dict(**c,outputs={cond:pred[(c['candidate_id'],cond)]['strict_label'] for cond in conditions},caption=pred[(c['candidate_id'],'caption')]['raw_text'] if (c['candidate_id'],'caption') in pred else None);case_rows.append(row)
for condition in conditions:
 bygroup={}
 for group in cases['groups']:
  rs=[r for r in case_rows if r['group']==group];expected=lambda r:'ROBOT' if group=='robot_control' else r['query_object'].upper();bygroup[group]=dict(query_role_agreement=rate(rs,lambda r:r['outputs'][condition]==expected(r)),historical_label_retention=rate(rs,lambda r:r['outputs'][condition]==r['baseline_label']),labels=dict(collections.Counter(r['outputs'][condition] for r in rs)))
 clear=[r for r in case_rows if r['GT_proxy_role']!='AMBIGUOUS'];summary[condition]=dict(by_group=bygroup,clear_proxy_role_accuracy=rate(clear,lambda r:r['outputs'][condition]==r['GT_proxy_role']),strict_valid=rate(case_rows,lambda r:r['outputs'][condition] in LABELS))
 if condition!='baseline':
  for r in case_rows:
   old=r['outputs']['baseline'];new=r['outputs'][condition]
   if old!=new:labelchanges.append(dict(candidate_id=r['candidate_id'],group=r['group'],condition=condition,old_label=old,new_label=new,GT_proxy_role=r['GT_proxy_role'],query_object=r['query_object']))
historical_reproduction=rate(case_rows,lambda r:r['outputs']['baseline']==r['baseline_label'])
audit=rd(R/'all_good_candidate_audit.json')['candidates'];audit_summary={}
for obj in ['peg','box']:
 rs=[r for r in audit if r['object']==obj];rejected=[r for r in rs if not r['accepted']];audit_summary[obj]=dict(good_candidates=len(rs),rejected=len(rejected),rejected_GT_roles=dict(collections.Counter(r['GT_proxy_role'] for r in rejected)),rejected_labels=dict(collections.Counter(r['label'] for r in rejected)),rejected_confusion=dict(collections.Counter(r['GT_proxy_role']+'->'+r['label'] for r in rejected)))
 if obj=='peg':
  audit_summary[obj]['red_like_pixel_strata']={name:dict(candidates=len(z),accepted=sum(r['accepted'] for r in z),rejected=sum(not r['accepted'] for r in z)) for name,z in [('zero',[r for r in rs if r['red_like_pixels']==0]),('positive',[r for r in rs if r['red_like_pixels']>0])]}
  audit_summary[obj]['color_and_shape_by_acceptance']={str(accepted):dict(n=len(z),red_fraction_median=float(np.median([r['red_like_fraction'] for r in z])),aspect_median=float(np.median([r['GT_target_bbox_aspect'] for r in z]))) for accepted,z in [(a,[r for r in rs if r['accepted']==a]) for a in [False,True]]}
wr('mechanism_metrics.json',dict(purpose='Known-error mechanism diagnosis, not performance/generalization evaluation',historical_baseline_reproduction=historical_reproduction,conditions=summary,full_good_candidate_audit=audit_summary,cases=case_rows,label_changes=labelchanges))
csvwr('mechanism_cases.csv',case_rows);csvwr('mechanism_label_changes.csv',labelchanges)
for r in raw:
 assert r['timing']['initial_forward_cache']['cache_length']==0
 if r['condition']!='caption':assert (r['raw_text'].strip() if r['raw_text'].strip() in LABELS else 'INVALID_OUTPUT')==r['strict_label']
requests=rd(R/'request_manifest.json')['requests'];rm={(r['candidate_id'],r['condition']):r for r in requests}
for req in requests:assert all(sha(p)==h for p,h in zip(req['images'],req['image_sha256']))
for case in case_rows:
 cid=case['candidate_id'];base=rm[(cid,'baseline')]
 for condition in conditions:
  req=rm[(cid,condition)]
  assert req['images']==(base['images'][-1:] if condition=='crop_only' else base['images'])
  assert req['image_sha256']==(base['image_sha256'][-1:] if condition=='crop_only' else base['image_sha256'])
protected={p:sha(p) for p in v['protected_before']};assert protected==v['protected_before']
assert json.loads(subprocess.check_output([str(B/'env/bin/python'),'-m','pip','list','--format=json'],text=True))==json.loads(v['packages_before'])
assert {p.name:sha(p) for p in (R/'.execution').glob('*.py')}==v['source_before']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
v.update(stage='COMPLETE_VERIFIED',protected_after=protected,all_protected_SHA_unchanged=True,environments_unchanged=True,raw_sha256_after_analysis=sha(R/'verifier_raw_responses.json'),predictions_sha256_after_analysis=sha(R/'verifier_predictions.csv'),empty_initial_cache_requests=len(raw),identical_image_contrast_cases=len(case_rows),production_changed=False,final_GPU_compute_processes='');wr('verification.json',v)
lat=rd(R/'verifier_latency_vram.json');lat['summary']={}
for condition in prompts:
 ts=[r['timing'] for r in raw if r['condition']==condition and not r['timing']['cold_first_request']];a=np.array([t['end_to_end_ms'] for t in ts]);lat['summary'][condition]=dict(warm_requests=len(ts),mean_ms=float(a.mean()),median_ms=float(np.median(a)),p95_ms=float(np.quantile(a,.95)),peak_allocated_GiB=max(t['memory']['peak_allocated'] for t in ts)/2**30)
wr('verifier_latency_vram.json',lat)
with ExperimentLogger(team='integration',model='Task10I_rejection_mechanism_analysis',seed=42,job_type='evaluate',root=R/'.execution/experiments',config=rd(R/'protocol.json')) as logger:
 logger.log({k:z['clear_proxy_role_accuracy']['rate'] for k,z in summary.items()});logger.link_output(R/'mechanism_metrics.json')
def rr(x):return str(x['numerator'])+'/'+str(x['denominator'])
lines=['# Task 10-I Stage4A-2 — Qwen rejection mechanism diagnosis','',
'Status: COMPLETE_VERIFIED. Known-error/control diagnostic; no new independent accuracy claim.','',
'## Setup','',
'Qwen3.5-9B revision c202236235762e1c871ad0ccb60c8ee5ba337b9a, fullBF16, SDPA, cuda:0, no offload, greedy/thinkingFalse. One model load,326 requests:300 single-label and26 short captions. No new DINO/SAM2/simulator/training. Old root/weights/environment preserved. Stage4A-2 remains PAUSED_SCOPE_REVIEW.','',
'All20 Peg error camera-frames and6 Box error camera-frames were represented by their highest-score rejected IoU>=.5 proposal. Each had a matched same-parent/same-camera initial accepted control. Eight clear Robot negatives were fixed by Hash sampling. Selection uses historical GT/errors and is explicitly diagnostic.','',
'## Full preserved-pool audit','',
'Peg has35 IoU-good rejected proposals:25 PEG-dominant and10 mixed. Box has7 IoU-good rejected proposals:5 ROBOT-dominant,1 PEG-dominant,1 mixed. The latter6 agree with the actor-dominance proxy despite failing the query-label gate. IoU-good does not guarantee that the crop dominant object is the intended target. Do not call all such rejections model recognition errors.','',
'The RGB red-like proxy finds7 Peg proposals with zero matching pixels (all rejected), but28 rejected proposals have positive matching pixels. This is an observational RGB proxy inside GT target pixels, affected by lighting/antialiasing; not a material label or a causal proof.','',
'## Controlled input edits','',
'Baseline: exact prior appearance prompt and3images. Crop-only: identical text, last clean crop only. Remove-color: delete only red-and-white from PEG definition. Remove-shape: delete only elongated rectangular from PEG definition. Explicit binding: replace only first2sentences by image1/2/3 references. All text was frozen before the run; no adaptive tuning. Caption26 is a separate unscored description probe, not access to internal reasoning.','',
'Historical baseline reproduced '+rr(historical_reproduction)+'.','',
'| Condition | Peg error frames -> PEG | Initial Peg control -> PEG | Box bbox-error frames -> BOX | Initial Box control -> BOX | Robot control -> ROBOT | Clear actor-proxy agreement |','|---|---:|---:|---:|---:|---:|---:|']
for condition,z in summary.items():
 b=z['by_group'];lines.append('| '+condition+' | '+' | '.join(rr(b[g]['query_role_agreement']) for g in ['peg_rejected','peg_matched_initial_control','box_bbox_rejected','box_matched_initial_control','robot_control'])+' | '+rr(z['clear_proxy_role_accuracy'])+' |')
lines+=['',
'Box error-frame -> BOX is query-role agreement, NOT semantic accuracy: no rejected Box proposal in the full audit is BOX-dominant. Report changes against the actor proxy separately. Mixed cases remain recorded and are excluded only from clear-proxy accuracy.','',
'## Costs and preservation','',f'Model load {lat["load_ms"]:.2f}ms, loaded GPU{lat["loaded_memory"]["allocated"]/2**30:.3f}GiB. Protected file hashes{len(protected)} unchanged; every request starts with cache0. Raw response SHA unchanged after analysis. CUDA executed only on Vast.ai58576054. Local macOS GPU/Simulator not executed. Shared ExperimentLogger used with W&B disabled. No publishing/commit/production changes.','',
'## Artifacts','',
'[Protocol](protocol.json), [Case manifest](case_manifest.json), [Prompt identity](prompt_identity.json), [Full good-proposal audit](all_good_candidate_audit.json), [Raw outputs](verifier_raw_responses.json), [Mechanism metrics/captions](mechanism_metrics.json), [Label changes](mechanism_label_changes.csv), [Latency/VRAM](verifier_latency_vram.json), [Verification](verification.json), [Artifact index](artifact_index.json).','']
(R/'10i_stage4a2_qwen35_rejection_mechanism.md').write_text('\n'.join(lines))
wr('artifact_index.json',dict(status='COMPLETE_VERIFIED',files={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in R.rglob('*') if p.is_file() and p.name!='artifact_index.json' and p.suffix not in ['.log','.tmp']}))
print('COMPLETE_VERIFIED',json.dumps(dict(reproduced=historical_reproduction,groups={k:{g:z['query_role_agreement'] for g,z in a['by_group'].items()} for k,a in summary.items()})),flush=True)
