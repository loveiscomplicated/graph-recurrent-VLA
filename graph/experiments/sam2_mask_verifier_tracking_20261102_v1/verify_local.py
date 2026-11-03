"""Independent artifact readback and metric recomputation; run after download."""
from pathlib import Path
import sys,json,hashlib,csv,math
import numpy as np
from PIL import Image
root=Path(sys.argv[1]).resolve();remote_mode='--remote' in sys.argv
prior=Path('/workspace/task10i_stage4a2_qwen35_appearance_bf16_independent_20261102_v1') if remote_mode else root.parent/'qwen35_appearance_bf16_independent_20261102_v1'
remote='/workspace/task10i_stage4a2_sam2_mask_verifier_tracking_20261102_v1'
old='/workspace/task10i_stage4a2_qwen35_appearance_bf16_independent_20261102_v1'
def rd(p):return json.loads(Path(p).read_text())
def path(p):
    if str(p).startswith(remote+'/'):return root/str(p)[len(remote)+1:]
    if str(p).startswith(old+'/'):return prior/str(p)[len(old)+1:]
    raise ValueError(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
    return h.hexdigest()
index=rd(root/'artifact_index.json')['files']
for n,e in index.items():assert (root/n).stat().st_size==e['bytes'] and sha(root/n)==e['sha256'],n
v=rd(root/'verification.json');assert v['stage']=='COMPLETE_VERIFIED' and v['Qwen_loads']==1 and v['SAM2_loads']==2
assert all(sha(root/'.execution'/n)==h for n,h in v['source_before'].items())
assert sha(root/'verifier_raw_responses.json')==v['raw_sha256_before_GT'] and sha(root/'tracking_predictions.json')==v['tracking_sha256_before_GT']
requests=rd(root/'request_manifest.json')['requests'];raw=rd(root/'verifier_raw_responses.json')['responses'];assert len(requests)==len(raw)<=1972
byid={r['id']:r for r in raw};assert len(byid)==len(raw)
for r in requests:
    assert all(sha(path(p))==h for p,h in zip(r['images'],r['image_sha256']))
    rr=byid[r['id']];assert rr['image_sha256']==r['image_sha256'] and rr['timing']['initial_forward_cache']['cache_length']==0
cm=rd(root/'candidate_manifest.json');mi={r['candidate_id']:r for r in rd(root/'sam2_candidate_manifest.json')['candidates']}
labels=rd(root/'candidate_labels.json');picks=rd(root/'selected_candidates.json')['rows']
with (prior/'selected_boxes.csv').open() as f:
    oldpicks={(r['image_id'],r['object']):r['candidate_id'] or None for r in csv.DictReader(f) if r['method']=='appearance_query_gate'}
assert len(oldpicks)==120
for p in picks:
    pool=sorted([c for c in cm['candidates'] if c['image_id']==p['image_id'] and c['query_object']==p['object'] and labels[p['method']][c['id']]==p['object'].upper()],key=lambda c:(-c['DINO_score'],c['rank'],c['id']))
    assert p['candidate_id']==(pool[0]['id'] if pool else None)
    if p['method']=='A':assert p['candidate_id']==oldpicks[(p['image_id'],p['object'])]
def truth(c):
    assert sha(path(c['GT_path']))==c['GT_sha256']
    with np.load(path(c['GT_path']),allow_pickle=False) as f:s=f['actor_segmentation']
    z={k:np.isin(s,c['object_ids'][k]) for k in ['peg','box','robot']};z['background']=~(z['peg']|z['box']|z['robot']);return z
images={c['id']:c for c in cm['images']};gt={k:truth(c) for k,c in images.items()}
render_checked=0
for c in cm['candidates']:
    info=mi[c['id']]
    if not info['C_images']:continue
    rgb=np.asarray(Image.open(path(images[c['image_id']]['image_path'])).convert('RGB'))
    mask=np.asarray(Image.open(path(info['mask_path'])))>0
    expected=rgb.copy();expected[~mask]=(rgb[~mask].astype(np.float32)*.25).astype(np.uint8)
    x1,y1,x2,y2=c['crop_box'];actual=np.asarray(Image.open(path(info['C_images'][2])).convert('RGB'))
    assert np.array_equal(actual,expected[y1:y2,x1:x2]),c['id']
    render_checked+=1
def checkrow(pred,z,obj,row):
    g=z[obj];tp=int((pred&g).sum());pn=int(pred.sum());gn=int(g.sum());u=pn+gn-tp
    io=tp/u if u else None;expected=None if row['IoU']=='' else float(row['IoU'])
    assert (io is None and expected is None) or (io is not None and expected is not None and abs(io-expected)<1e-12)
    wrong=bool(pn and any(int((pred&m).sum())/pn>=.5 for k,m in z.items() if k not in [obj,'background']))
    assert pn==int(row['predicted_pixels']) and gn==int(row['visible_pixels']) and tp==int(row['intersection']) and wrong==(row['wrong_object']=='True')
    assert (gn>=16 and io is not None and io>=.8)==(row['success08']=='True')
with (root/'sam2_initialization_cases.csv').open() as f:static=list(csv.DictReader(f))
for row in static:
    cid=row['candidate_id'];mask=np.asarray(Image.open(path(mi[cid]['mask_path'])))>0 if cid else np.zeros((512,512),bool)
    checkrow(mask,gt[row['image_id']],row['object'],row)
with (root/'tracking_cases.csv').open() as f:tracking=list(csv.DictReader(f))
lookup={(r['sequence_id'],r['method'],int(r['timestep']),r['object']):r for r in tracking}
seqs={s['id']:s for s in rd(root/'sequence_manifest.json')['sequences']};npred=rd(root/'tracking_predictions.json')['rows']
for seq in seqs.values():
    zz={f['timestep']:truth(dict(**f,object_ids=seq['object_ids'])) for f in seq['frames']}
    for p in [p for p in npred if p['sequence_id']==seq['id']]:
        with np.load(path(p['mask_path']),allow_pickle=False) as f:m=f['masks']
        for t,z in zz.items():
            for i,obj in enumerate(['peg','box']):checkrow(m[t,i],z,obj,lookup[(seq['id'],p['method'],t,obj)])
    print('INDEPENDENT_READBACK',seq['id'],flush=True)
assert len(static)==360 and len(tracking)==7776 and len(seqs)==16 and len(npred)==48
result=dict(status='PASS',artifact_files_verified=len(index),requests_verified=len(requests),input_image_hashes_verified=sum(len(r['images']) for r in requests),mask_input_rendering_verified=render_checked,selections_recomputed=len(picks),static_mask_metrics_recomputed=len(static),tracking_mask_metrics_recomputed=len(tracking),new_parent_generalization=False,CPU_only=True,executed_on='remote CPU' if remote_mode else 'local CPU')
(root/('independent_verification.json' if remote_mode else 'local_verification.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
