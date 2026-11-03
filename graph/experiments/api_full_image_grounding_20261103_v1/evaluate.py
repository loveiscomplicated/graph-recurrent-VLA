"""GT-only scoring after both API outputs and masks have been frozen."""
from common import *
from collections import Counter
import itertools
import numpy as np
from PIL import Image

def bbox(mask):
    yy, xx = np.where(mask)
    return [int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1] if len(xx) else None

def box_iou(a,b):
    if a is None or b is None: return 0.
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.

def rate(rows,key):
    return dict(numerator=sum(bool(r[key]) for r in rows),denominator=len(rows),
                rate=sum(bool(r[key]) for r in rows)/len(rows) if rows else None)

def dist(values):
    a=[float(x) for x in values if x is not None]
    return dict(n=len(a),mean=float(np.mean(a)) if a else None,median=float(np.median(a)) if a else None,
                p05=float(np.quantile(a,.05)) if a else None,p95=float(np.quantile(a,.95)) if a else None)

def score_mask(pred, truth, obj):
    gt=truth[obj]; n=int(pred.sum()); g=int(gt.sum()); inter=int((pred&gt).sum()); union=n+g-inter
    counts={k:int((pred&m).sum()) for k,m in truth.items()}
    return dict(mask_pixels=n,visible_pixels=g,mask_IoU=inter/union if union else None,
                precision=inter/n if n else None,recall=inter/g if g else None,
                actor_pixels=counts,contamination={k:v/n if n else None for k,v in counts.items() if k!=obj},
                wrong_object=bool(n and any(counts[k]/n>=.5 for k in ['peg','box','robot'] if k!=obj)),
                mask08=bool(g>=16 and union and inter/union>=.8),
                mask05=bool(g>=16 and union and inter/union>=.5))

def summarize(rows, scenes):
    out=dict(scene={'both_mask08':rate([s for s in scenes if s['both_visible']],'both_mask08'),
                    'both_box05':rate([s for s in scenes if s['both_visible']],'both_box05'),
                    'wrong_object_any':rate(scenes,'wrong_object_any')})
    for obj in ['peg','box']:
        rr=[r for r in rows if r['object']==obj]; vv=[r for r in rr if r['visibility']=='VISIBLE']
        absent=[r for r in rr if r['visibility']=='ABSENT']
        nonempty=[r for r in rr if r['mask_pixels'] and r['visibility']!='TINY_VISIBLE']
        attempted=[r for r in rr if r['attempted']]
        ok=[r for r in vv if r['api_status']=='OK' and r['json_valid']]
        out[obj]=dict(mask08=rate(vv,'mask08'),mask05=rate(vv,'mask05'),
            box03=rate(vv,'box03'),box05=rate(vv,'box05'),box07=rate(vv,'box07'),
            visible_detection_recall_definition='valid box with target visible-bbox IoU>=0.5',
            missing=rate(vv,'missing'),abstention=rate(attempted,'null'),
            hallucinated_box_when_absent=rate(absent,'valid_box'),wrong_object=rate(nonempty,'wrong_object'),
            validity_all_assigned=rate(rr,'valid_or_null'),validity_attempted=rate(attempted,'valid_or_null'),
            visibility=dict(Counter(r['visibility'] for r in rr)),
            box_IoU=dist([r['box_IoU'] for r in vv]),
            center_error_pixels=dist([r['center_error_pixels'] for r in vv]),
            mask_IoU_including_misses=dist([r['mask_IoU'] for r in vv]),
            precision=dist([r['precision'] for r in vv]),recall=dist([r['recall'] for r in vv]),
            confusion_bbox_proxy=rate(vv,'confusion_bbox_proxy'),
            conditional_valid_final_response_mask08=rate(ok,'mask08'),
            contamination={k:dist([r['contamination'].get(k) for r in nonempty])
                           for k in ['peg','box','robot','background'] if k!=obj})
    return out

def main():
    v=verify_frozen()
    assert v['stage'] in ['MASKS_FROZEN','COMPLETE']
    for group in ['api_files','raw_files','sam2_files']:
        for name,expected in v[group].items(): assert sha(ROOT/name)==expected,name
    images=read(ROOT/'evaluation_manifest.json')['images']
    predictions=read(ROOT/'full_image_grounding_predictions.json')['rows']
    masks=read(ROOT/'sam2/sam2_predictions.json')['rows']
    lookup={(r['provider'],r['image_id'],r['object']):r for r in masks}
    rows=[]; scenes=[]
    with logger('evaluate') as log:
        for c in images:
            assert sha(c['GT_path'])==c['GT_sha256']
            with np.load(c['GT_path'],allow_pickle=False) as z: seg=z['actor_segmentation'].copy()
            truth={o:np.isin(seg,c['object_ids'][o]) for o in ['peg','box','robot']}
            truth['background']=~(truth['peg']|truth['box']|truth['robot'])
            gtboxes={o:bbox(m) for o,m in truth.items() if o!='background'}
            for provider in MODELS:
                p=next(p for p in predictions if p['provider']==provider and p['image_id']==c['id'])
                scene=[]
                for obj in ['peg','box']:
                    box=p['parsed'][obj]; pb=box['pixels']; gt=gtboxes[obj]
                    m=lookup.get((provider,c['id'],obj)); pred=np.zeros((512,512),bool)
                    if m:
                        path=ROOT/'sam2'/m['mask_file'];assert sha(path)==m['mask_sha256']
                        pred=np.asarray(Image.open(path))>0
                    met=score_mask(pred,truth,obj); g=met['visible_pixels']
                    iou=box_iou(pb,gt)
                    center=math.hypot((pb[0]+pb[2]-gt[0]-gt[2])/2,(pb[1]+pb[3]-gt[1]-gt[3])/2) if pb and gt else None
                    r=dict(provider=provider,image_id=c['id'],parent_id=c['parent_id'],camera=c['camera'],
                        frame_kind=c['frame_kind'],timestep=c['timestep'],object=obj,
                        attempted=p['attempted'],api_status=p['status'],json_valid=p['parsed']['json_valid'],
                        box_status=box['status'],valid_box=box['status']=='VALID',
                        valid_or_null=box['status'] in ['VALID','NULL'],null=box['status']=='NULL',
                        predicted_box=pb,GT_visible_box=gt,
                        visibility='VISIBLE' if g>=16 else 'TINY_VISIBLE' if g else 'ABSENT',
                        box_IoU=iou,box03=bool(g>=16 and iou>=.3),box05=bool(g>=16 and iou>=.5),
                        box07=bool(g>=16 and iou>=.7),center_error_pixels=center,
                        missing=bool(g>=16 and pb is None),
                        confusion_bbox_proxy=bool(pb and iou<.5 and any(box_iou(pb,gtboxes[o])>=.5 for o in ['peg','box','robot'] if o!=obj)),
                        sam2_status='MASK_GENERATED' if m else 'NO_BOX' if not pb else 'SAM2_RUNTIME_FAILURE',**met)
                    rows.append(r);scene.append(r)
                scenes.append(dict(provider=provider,image_id=c['id'],parent_id=c['parent_id'],camera=c['camera'],
                    frame_kind=c['frame_kind'],both_visible=all(r['visibility']=='VISIBLE' for r in scene),
                    both_mask08=all(r['mask08'] for r in scene),both_box05=all(r['box05'] for r in scene),
                    wrong_object_any=any(r['wrong_object'] for r in scene),attempted=p['attempted']))
        metrics={}
        for provider in MODELS:
            rr=[r for r in rows if r['provider']==provider]; ss=[s for s in scenes if s['provider']==provider]
            parents={pid:summarize([r for r in rr if r['parent_id']==pid],[s for s in ss if s['parent_id']==pid])
                     for pid in sorted({r['parent_id'] for r in rr})}
            metrics[provider]=dict(overall=summarize(rr,ss),per_parent=parents,
                parent_macro={o:dist([a[o]['mask08']['rate'] for a in parents.values()]) for o in ['peg','box']},
                parent_macro_joint=dist([a['scene']['both_mask08']['rate'] for a in parents.values()]),
                by_camera={cam:summarize([r for r in rr if r['camera']==cam],[s for s in ss if s['camera']==cam]) for cam in ['base_camera','hand_camera']},
                by_time={t:summarize([r for r in rr if r['frame_kind']==t],[s for s in ss if s['frame_kind']==t]) for t in ['F0','F1']},
                api_status=dict(Counter(p['status'] for p in predictions if p['provider']==provider)))
        ids=sorted({c['parent_id'] for c in images});rng=np.random.default_rng(42)
        draws=rng.integers(0,len(ids),size=(10000,len(ids)));paired={}
        for a,b in itertools.combinations(MODELS,2):
            delta=np.array([metrics[a]['per_parent'][p]['scene']['both_mask08']['rate']-
                            metrics[b]['per_parent'][p]['scene']['both_mask08']['rate'] for p in ids])
            boot=delta[draws].mean(axis=1)
            paired[a+'_minus_'+b]=dict(parent_n=len(ids),macro_delta=float(delta.mean()),CI95=np.quantile(boot,[.025,.975]).tolist())
        attempted_sets=[{p['image_id'] for p in predictions if p['provider']==m and p['attempted']} for m in MODELS]
        common=set.intersection(*attempted_sets)
        matched={m:summarize([r for r in rows if r['provider']==m and r['image_id'] in common],
                             [s for s in scenes if s['provider']==m and s['image_id'] in common]) for m in MODELS}
        write(ROOT/'grounding_evaluation_metrics.json',dict(models=metrics,paired_parent_CI=paired,
            common_attempted_images=len(common),common_attempted=matched,
            CI_interpretation='Includes operational failures; unavailable models have no measured visual accuracy.'))
        csvwrite(ROOT/'grounding_evaluation_cases.csv',rows)
        csvwrite(ROOT/'scene_cases.csv',scenes)
        csvwrite(ROOT/'grounding_failure_cases.csv',[r for r in rows if r['visibility']=='VISIBLE' and not r['mask08']])
        write(ROOT/'evaluation_rows.json',dict(rows=rows,scenes=scenes))
        log.log({m:metrics[m]['overall']['scene']['both_mask08']['rate'] for m in MODELS})
        log.link_output(ROOT/'grounding_evaluation_metrics.json')
    print('EVALUATED',len(rows),'object rows')

if __name__=='__main__':main()
