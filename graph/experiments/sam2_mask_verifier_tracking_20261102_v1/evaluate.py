"""GT-only evaluator; predictions must already be immutable."""
from common import *
import collections
import numpy as np
from PIL import Image,ImageDraw
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json');assert v['stage']=='ALL_PREDICTIONS_FROZEN'
for name,key in [('verifier_raw_responses.json','raw_sha256_before_GT'),('verifier_predictions.csv','predictions_sha256_before_GT'),('tracking_predictions.json','tracking_sha256_before_GT')]:assert sha(R/name)==v[key]
cm=rd(R/'candidate_manifest.json');images={c['id']:c for c in cm['images']};candidates={c['id']:c for c in cm['candidates']};maskinfo={c['candidate_id']:c for c in rd(R/'sam2_candidate_manifest.json')['candidates']};labels=rd(R/'candidate_labels.json');picks=rd(R/'selected_candidates.json')['rows']
def loadtruth(case):
    assert sha(case['GT_path'])==case['GT_sha256']
    with np.load(case['GT_path'],allow_pickle=False) as z:seg=z['actor_segmentation'].copy()
    masks={k:np.isin(seg,case['object_ids'][k]) for k in ['peg','box','robot']};masks['background']=~(masks['peg']|masks['box']|masks['robot']);return masks
def candidate_mask(cid):
    if cid is None:return np.zeros((512,512),bool)
    c=maskinfo[cid];assert sha(c['mask_path'])==c['mask_sha256'];return np.asarray(Image.open(c['mask_path']))>0
def score(pred,truth,obj):
    m=mask_metrics(pred,truth[obj],{k:z for k,z in truth.items() if k!=obj});n=m['visible_pixels'];m['visibility']='VISIBLE' if n>=16 else 'TINY_VISIBLE' if n else 'ABSENT';m['success08']=bool(n>=16 and m['IoU'] is not None and m['IoU']>=.8);m['success05']=bool(n>=16 and m['IoU'] is not None and m['IoU']>=.5);m['missing_visible']=bool(n>=16 and not m['predicted_pixels']);return m
truth={iid:loadtruth(c) for iid,c in images.items()};static=[];candidate_rows=[]
for r in picks:static.append(dict(**r,**score(candidate_mask(r['candidate_id']),truth[r['image_id']],r['object'])))
for cid,c in candidates.items():
    obj=c['query_object'];met=score(candidate_mask(cid),truth[c['image_id']],obj)
    for method in METHODS:
        lab=labels[method][cid];candidate_rows.append(dict(candidate_id=cid,image_id=c['image_id'],parent_id=c['parent_id'],camera=c['camera'],frame_kind=c['frame_kind'],query_object=obj,method=method,label=lab,accepted=lab==obj.upper(),mask_good08=met['success08'],mask_IoU=met['IoU'],wrong_mask=met['wrong_object'],mask_pixels=met['predicted_pixels']))
csvwr('candidate_evaluation.csv',candidate_rows);csvwr('sam2_initialization_cases.csv',static)
def dist(xs):
    xs=[float(x) for x in xs if x is not None];return dict(n=len(xs),mean=float(np.mean(xs)) if xs else None,median=float(np.median(xs)) if xs else None,p05=float(np.quantile(xs,.05)) if xs else None,p95=float(np.quantile(xs,.95)) if xs else None)
def summary(rows,scene_key):
    out={};scenes=collections.defaultdict(list)
    for r in rows:scenes[scene_key(r)].append(r)
    both=[a for a in scenes.values() if len(a)==2 and all(r['visibility']=='VISIBLE' for r in a)]
    out['scene']=dict(both_mask08=dict(numerator=sum(all(r['success08'] for r in a) for a in both),denominator=len(both),rate=sum(all(r['success08'] for r in a) for a in both)/len(both) if both else None),both_mask05=dict(numerator=sum(all(r['success05'] for r in a) for a in both),denominator=len(both),rate=sum(all(r['success05'] for r in a) for a in both)/len(both) if both else None))
    for obj in ['peg','box']:
        oo=[r for r in rows if r['object']==obj];vv=[r for r in oo if r['visibility']=='VISIBLE'];em=[r for r in oo if r['predicted_pixels'] and r['visibility']!='TINY_VISIBLE'];aa=[r for r in oo if r['visibility']=='ABSENT']
        out[obj]=dict(visible=len(vv),success08=rate(vv,'success08'),success05=rate(vv,'success05'),missing_visible=rate(vv,'missing_visible'),IoU_including_noninitialization=dist([r['IoU'] for r in vv]),pixel_precision=dist([r['pixel_precision'] for r in vv]),pixel_recall=dist([r['pixel_recall'] for r in vv]),wrong_object=rate(em,'wrong_object'),empty_outputs=sum(not r['predicted_pixels'] for r in oo),visibility_counts=dict(collections.Counter(r['visibility'] for r in oo)),unsupported_visible_mask_when_absent=dict(numerator=sum(bool(r['predicted_pixels']) for r in aa),denominator=len(aa),rate=sum(bool(r['predicted_pixels']) for r in aa)/len(aa) if aa else None),contamination={k:dist([r['contamination'].get(k) for r in em]) for k in ['peg','box','robot','background'] if k!=obj})
    return out
def breakdown(rows,scene_key):
    pp={p:summary([r for r in rows if r['parent_id']==p],scene_key) for p in sorted({r['parent_id'] for r in rows})}
    out=dict(image_object_level=summary(rows,scene_key),per_parent=pp,parent_macro={k:dist([v['scene']['both_mask08']['rate'] for v in pp.values()]) if k=='both_mask08' else dist([v[k]['success08']['rate'] for v in pp.values()]) for k in ['peg','box','both_mask08']},by_camera={c:summary([r for r in rows if r['camera']==c],scene_key) for c in ['base_camera','hand_camera']})
    if all('frame_kind' in r for r in rows):out['by_time']={t:summary([r for r in rows if r['frame_kind']==t],scene_key) for t in ['F0','F1']}
    return out
static_metrics={m:breakdown([r for r in static if r['method']==m],lambda r:r['image_id']) for m in METHODS}
candidate_metrics={}
for method in METHODS:
    cr=[r for r in candidate_rows if r['method']==method];candidate_metrics[method]={}
    for obj in ['peg','box']:
        a=[r for r in cr if r['query_object']==obj];good=[r for r in a if r['mask_good08']];bad=[r for r in a if not r['mask_good08']];candidate_metrics[method][obj]=dict(good_mask_retention=rate(good,'accepted'),bad_mask_acceptance=rate(bad,'accepted'),labels=dict(collections.Counter(r['label'] for r in a)))
def paired_ci(rows,scene_key,baseline):
    pp=sorted({r['parent_id'] for r in rows});a=[]
    for pid in pp:
        pair=[]
        for method in [baseline,'C']:
            z=summary([r for r in rows if r['parent_id']==pid and r['method']==method],scene_key)
            pair.append([z['scene']['both_mask08']['rate']]+[z[o]['success08']['rate'] for o in ['peg','box']])
        a.append([b-x if x is not None and b is not None else np.nan for x,b in zip(*pair)])
    a=np.asarray(a);rng=np.random.default_rng(42);draws=rng.integers(0,len(pp),size=(10000,len(pp)));boot=np.nanmean(a[draws],axis=1)
    return {key:dict(parent_n=int(np.isfinite(a[:,i]).sum()),parent_macro_delta=float(np.nanmean(a[:,i])),CI95=[float(x) for x in np.nanquantile(boot[:,i],[.025,.975])]) for i,key in enumerate(['both_mask08','peg_mask08','box_mask08'])}
wr('sam2_initialization_metrics.json',dict(methods=static_metrics,candidate_metrics=candidate_metrics,paired_deltas={m:paired_ci(static,lambda r:r['image_id'],m) for m in ['A','B']}))
seqs={s['id']:s for s in rd(R/'sequence_manifest.json')['sequences']};tracking=[];seqstats=[]
for prediction in rd(R/'tracking_predictions.json')['rows']:
    seq=seqs[prediction['sequence_id']];assert sha(prediction['mask_path'])==prediction['mask_sha256']
    with np.load(prediction['mask_path'],allow_pickle=False) as f:masks=f['masks']
    perobj={obj:[] for obj in ['peg','box']}
    for frame in seq['frames']:
        t=frame['timestep'];gt=loadtruth(dict(**frame,object_ids=seq['object_ids']))
        for i,obj in enumerate(['peg','box']):
            met=score(masks[t,i],gt,obj);counts=dict(met['actor_pixels']);counts[obj]=met['intersection'];foreground=sum(counts[k] for k in ['peg','box','robot']);dom=max(['peg','box','robot'],key=lambda k:counts[k]);dominant=dom if foreground>=16 and counts[dom]/foreground>=.7 else None
            row=dict(method=prediction['method'],sequence_id=seq['id'],parent_id=seq['parent_id'],camera=seq['camera'],timestep=t,object=obj,candidate_id=prediction['initialization'][obj]['candidate_id'],dominant_actor_proxy=dominant,**met);tracking.append(row);perobj[obj].append(row)
    for obj,rr in perobj.items():
        goodstart=rr[0]['success08'];run=maximum=0;switch=0;last=None
        for r in rr:
            bad=r['visibility']=='VISIBLE' and not r['success05'];run=run+1 if bad else 0;maximum=max(maximum,run)
            if r['dominant_actor_proxy'] is not None:
                if last is not None and last!=r['dominant_actor_proxy']:switch+=1
                last=r['dominant_actor_proxy']
            r['initial_video_mask_good08']=goodstart
        seqstats.append(dict(method=prediction['method'],sequence_id=seq['id'],parent_id=seq['parent_id'],camera=seq['camera'],object=obj,initial_video_mask_IoU=rr[0]['IoU'],initial_video_mask_good08=goodstart,longest_visible_failure_steps=maximum,dominant_actor_proxy_changes=switch,initial_candidate_id=prediction['initialization'][obj]['candidate_id']))
csvwr('tracking_cases.csv',tracking);csvwr('tracking_sequences.csv',seqstats)
sk=lambda r:(r['sequence_id'],r['timestep'])
trackmetrics={}
for method in METHODS:
    rr=[r for r in tracking if r['method']==method];trackmetrics[method]=breakdown(rr,sk)
    trackmetrics[method]['conditional_on_actual_video_t0_good08']={obj:summary([r for r in rr if r['object']==obj and r['initial_video_mask_good08'] and r['timestep']>0],sk)[obj] for obj in ['peg','box']}
    trackmetrics[method]['by_window']={name:summary([r for r in rr if lo<=r['timestep']<=hi],sk) for name,lo,hi in [('initial',0,0),('early',1,20),('middle',21,60),('later',61,80)]}
    trackmetrics[method]['sequence_failure_steps']=dist([r['longest_visible_failure_steps'] for r in seqstats if r['method']==method])
    trackmetrics[method]['dominant_actor_proxy_changes']=sum(r['dominant_actor_proxy_changes'] for r in seqstats if r['method']==method)
wr('tracking_metrics.json',dict(methods=trackmetrics,paired_deltas={m:paired_ci(tracking,sk,m) for m in ['A','B']},GT_notes='Actor-dominance changes are a proxy, not a formal MOT identity-switch score; no correction/reacquisition policy is evaluated'))
# Hole position is evaluation-only: projection is not visibility ground truth.
hole=[];parents={p['parent_id']:p for p in rd(R/'input_manifest.json')['parents']}
for iid,case in images.items():
    with np.load(parents[case['parent_id']]['gt_path'],allow_pickle=False) as g:pos=g['physics_hole_pose'][case['timestep'],:3]
    with np.load(case['GT_path'],allow_pickle=False) as g:K=g['intrinsic_cv'];E=g['extrinsic_cv'];depth=g['depth_raw_mm']
    cv=E@np.r_[pos,1];proj=K@cv;uv=proj[:2]/proj[2] if proj[2]>0 else np.array([np.nan,np.nan]);inside=bool(cv[2]>0 and np.isfinite(uv).all() and 0<=uv[0]<512 and 0<=uv[1]<512)
    hole.append(dict(image_id=iid,parent_id=case['parent_id'],camera=case['camera'],frame_kind=case['frame_kind'],timestep=case['timestep'],projection_in_frame=inside,uv=uv.tolist() if np.isfinite(uv).all() else None,camera_z_m=float(cv[2]),visual_status='UNCERTAIN' if inside else 'OUT_OF_VIEW',annotation_reason='Projection alone cannot certify an aperture or visibility; manual RGB review pending' if inside else 'Reference point behind camera or outside image',hole_estimator_executed=False))
wr('hole_readiness_audit.json',dict(cases=hole,notes='Hole is not a separate Actor. No segmentation-as-hole GT, predicted localization, or accuracy claim. In-frame labels require explicit RGB review.'))
csvwr('hole_readiness_cases.csv',hole)
csvwr('failure_cases.csv',[r for r in static if r['visibility']=='VISIBLE' and not r['success08']])
# Small, predetermined comparison: two hash-first F0 and F1 scenes per camera.
panels=[]
for cam in ['base_camera','hand_camera']:
    for fk in ['F0','F1']:
        pool=sorted([c for c in images.values() if c['camera']==cam and c['frame_kind']==fk],key=lambda c:hashlib.sha256(('gallery|'+c['id']).encode()).hexdigest())[:2]
        panels.extend(pool)
canvas=Image.new('RGB',(4*256,8*280),'white');draw=ImageDraw.Draw(canvas)
for row,case in enumerate(panels):
    base=np.asarray(Image.open(case['image_path']).convert('RGB'))
    for col,method in enumerate(['GT']+METHODS):
        rgb=base.copy()
        for obj,color in [('peg',np.array([240,40,60])),('box',np.array([40,180,240]))]:
            mask=truth[case['id']][obj] if method=='GT' else candidate_mask(next(r['candidate_id'] for r in static if r['image_id']==case['id'] and r['method']==method and r['object']==obj))
            rgb[mask]=(.65*rgb[mask]+.35*color).astype(np.uint8)
        canvas.paste(Image.fromarray(rgb).resize((256,256)),(col*256,row*280+24));draw.text((col*256+3,row*280+4),method+' '+case['parent_id']+' '+case['frame_kind']+' '+('Main' if case['camera']=='base_camera' else 'Wrist'),fill='black')
canvas.save(R/'comparison.png')
assert all(sha(R/name)==v[key] for name,key in [('verifier_raw_responses.json','raw_sha256_before_GT'),('verifier_predictions.csv','predictions_sha256_before_GT'),('tracking_predictions.json','tracking_sha256_before_GT')])
with ExperimentLogger(team='integration',model='Task10I_mask_tracking_metrics',seed=42,job_type='evaluate',root=R/'.execution/experiments',config=rd(R/'protocol.json')) as logger:
    logger.log({m+'_static_joint08':static_metrics[m]['image_object_level']['scene']['both_mask08']['rate'] for m in METHODS});logger.link_output(R/'sam2_initialization_metrics.json');logger.link_output(R/'tracking_metrics.json')
update(stage='EVALUATED',GT_opened_after_all_predictions_frozen=True);print('EVALUATED',json.dumps({m:static_metrics[m]['image_object_level']['scene'] for m in METHODS}),flush=True)
