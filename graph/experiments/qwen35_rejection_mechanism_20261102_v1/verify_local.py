"""Verify frozen diagnostic responses against original requests, labels and actors."""
from pathlib import Path
import json,csv,hashlib,collections,re
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'outputs/task10i/stage4a2';R=BASE/'qwen35_rejection_mechanism_20261102_v1';P=BASE/'qwen35_appearance_bf16_independent_20261102_v1'
def rd(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def local(p):return P/Path(p).relative_to('/workspace/task10i_stage4a2_qwen35_appearance_bf16_independent_20261102_v1')
v=rd(R/'verification.json');assert v['stage']=='COMPLETE_VERIFIED' and v['model_loads']==1
idx=rd(R/'artifact_index.json')['files'];assert all(sha(R/n)==e['sha256'] for n,e in idx.items())
oldidx=rd(P/'artifact_index.json')['files'];assert all(sha(P/n)==e['sha256'] for n,e in oldidx.items())
assert sha(R/'verifier_raw_responses.json')==v['raw_sha256_before_GT']==v['raw_sha256_after_analysis']
for n,h in v['source_before'].items():assert sha(Path(__file__).parent/n)==h==sha(R/'.execution'/n)
prompts=rd(R/'prompt_identity.json')['conditions'];oldp=rd(P/'prompt_identity.json')['conditions']['appearance_three'];assert prompts['baseline']==oldp
assert prompts['crop_only']['prompt']==oldp['prompt']
assert prompts['remove_peg_color']['prompt']==oldp['prompt'].replace('A red-and-white elongated rectangular piece.','An elongated rectangular piece.')
assert prompts['remove_peg_shape']['prompt']==oldp['prompt'].replace('A red-and-white elongated rectangular piece.','A red-and-white piece.')
raw=rd(R/'verifier_raw_responses.json')['responses'];pred={(r['candidate_id'],r['condition']):r for r in raw};requests=rd(R/'request_manifest.json')['requests'];rm={(r['candidate_id'],r['condition']):r for r in requests};cases=rd(R/'case_manifest.json')['cases'];assert len(raw)==len(pred)==326 and len(cases)==60
checks=0
for req in requests:
 for p,h in zip(req['images'],req['image_sha256']):assert sha(local(p))==h;checks+=1
for r in raw:
 assert r['timing']['initial_forward_cache']['cache_length']==0
 labels=prompts[r['condition']]['labels']
 if labels:assert r['strict_label']==(r['raw_text'].strip() if r['raw_text'].strip() in labels else 'INVALID_OUTPUT')
for c in cases:
 cid=c['candidate_id'];baseline=rm[(cid,'baseline')]
 for cond in prompts:
  if (cid,cond) not in rm:continue
  r=rm[(cid,cond)];assert r['images']==(baseline['images'][-1:] if cond=='crop_only' else baseline['images'])
oldrows={r['id']:r for r in csv.DictReader((P/'candidate_evaluation.csv').open()) if r['condition']=='appearance_three'}
metric=rd(R/'mechanism_metrics.json');rolechecks=0
for row in metric['cases']:
 old=oldrows[row['candidate_id']];assert old['GT_role']==row['GT_proxy_role'] and old['strict_label']==row['baseline_label'];rolechecks+=1
 assert row['outputs']=={cond:pred[(row['candidate_id'],cond)]['strict_label'] for cond in prompts if cond!='caption'}
for cond,z in metric['conditions'].items():
 for group,g in z['by_group'].items():
  rs=[r for r in cases if r['group']==group];expected=lambda r:'ROBOT' if group=='robot_control' else r['query_object'].upper();assert g['query_role_agreement']['numerator']==sum(pred[(r['candidate_id'],cond)]['strict_label']==expected(r) for r in rs) and g['query_role_agreement']['denominator']==len(rs)
images={c['id']:c for c in rd(P/'input_manifest.json')['cases']};cache={};yy,xx=np.indices((512,512));colorchecks=0
for f in rd(R/'all_good_candidate_audit.json')['candidates']:
 r=oldrows[f['candidate_id']];c=images[r['image_id']]
 if c['id'] not in cache:
  with np.load(local(c['GT_path'])) as z:seg=z['actor_segmentation'].copy()
  cache[c['id']]=(np.array(Image.open(local(c['image_path'])).convert('RGB')),seg)
 rgb,seg=cache[c['id']];obj=r['query_object'];b=json.loads(r['box']);mask=np.isin(seg,c['object_ids'][obj])&(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3]);pix=rgb[mask].astype(float);red=(pix[:,0]>100)&(pix[:,0]>1.25*pix[:,1])&(pix[:,0]>1.25*pix[:,2]);assert int(red.sum())==f['red_like_pixels'] and len(pix)==f['target_visible_pixels'] and abs(float(red.mean())-f['red_like_fraction'])<1e-12;colorchecks+=1
verification=dict(status='PASS',indexed_remote_artifacts=len(idx),old_frozen_artifacts_unchanged=len(oldidx),source_files=len(v['source_before']),requests=326,classification_requests=300,caption_requests=26,cases=60,image_hashes=checks,case_proxy_role_checks=rolechecks,independent_color_audits=colorchecks,all_initial_caches_zero=True,all_original_raws_preserved=True,one_model_load=True)
(R/'local_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification,indent=2))
