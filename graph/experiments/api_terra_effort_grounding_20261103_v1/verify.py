"""Independent check using PIL logical operations rather than evaluator numpy scoring."""
from common import *
import numpy as np
from PIL import Image, ImageChops

def count(im):
    return im.convert('L').histogram()[255]

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['strict','fence'],default='strict');args=ap.parse_args()
    prefix='fence_' if args.mode=='fence' else ''
    v=verify_frozen()
    for group in ['api_files','raw_files','sam2_files']:
        for n,h in v[group].items():assert sha(ROOT/n)==h,n
    for p,h in v['protected_before'].items():assert sha(p)==h,p
    data=read(ROOT/(prefix+'evaluation_rows.json'));images={c['id']:c for c in read(ROOT/'evaluation_manifest.json')['images']}
    masks={(r['provider'],r['image_id'],r['object']):r for r in read(ROOT/'sam2/sam2_predictions.json')['rows']}
    ground={}
    for iid,c in images.items():
        with np.load(c['GT_path'],allow_pickle=False) as z:seg=z['actor_segmentation']
        ground[iid]={o:Image.fromarray(np.isin(seg,c['object_ids'][o])) for o in ['peg','box','robot']}
    for row in data['rows']:
        m=masks.get((row['provider'],row['image_id'],row['object'])) if row['predicted_box'] else None
        pred=Image.open(ROOT/'sam2'/m['mask_file']).convert('1') if m else Image.new('1',(512,512))
        gt=ground[row['image_id']][row['object']]
        intersection=count(ImageChops.logical_and(pred,gt));union=count(ImageChops.logical_or(pred,gt))
        expected=intersection/union if union else None
        assert (expected is None and row['mask_IoU'] is None) or abs(expected-row['mask_IoU'])<1e-12
        n=count(pred)
        wrong=bool(n and any(count(ImageChops.logical_and(pred,a))/n>=.5
                           for o,a in ground[row['image_id']].items() if o!=row['object']))
        assert wrong==row['wrong_object']
        assert row['mask08']==bool(count(gt)>=16 and expected is not None and expected>=.8)
        # PIL half-open getbbox gives an independent visible bounding box.
        b=gt.getbbox();assert (list(b) if b else None)==row['GT_visible_box']
        if row['predicted_box'] and b:
            a=row['predicted_box'];left=max(a[0],b[0]);right=min(a[2],b[2]);top=max(a[1],b[1]);bottom=min(a[3],b[3])
            inter=0 if right<=left or bottom<=top else (right-left)*(bottom-top)
            expected_box=inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
        else:expected_box=0
        assert abs(expected_box-row['box_IoU'])<1e-12
    ledger=read(ROOT/'request_ledger.json')
    assert len(ledger['jobs'])<=180 and PRIOR_ACCOUNTED_USD+ledger_total(ledger['jobs'])<=20
    assert len(ledger['jobs'])==len(set(ledger['jobs']))
    for j in ledger['jobs'].values():
        assert j['image_id'] in images
        assert j.get('parsed') is None or all(j['parsed'][o]['pixels'] is None or j['status']=='OK' for o in ['peg','box'])
    result=dict(status='PASS',object_metrics_recomputed=len(data['rows']),scenes=len(data['scenes']),
        protected_files_verified=len(v['protected_before']),source_files_verified=len(v['source_before']),
        artifact_hashes_verified=sum(len(v[g]) for g in ['api_files','raw_files','sam2_files']),
        requests=len(ledger['jobs']),accounted_usd=ledger_total(ledger['jobs']),GT_used_in_inference=False,
        implementation='PIL ImageChops logical operations + getbbox, separate from numpy evaluator',verified_utc=now())
    write(ROOT/(prefix+'independent_verification.json'),result)
    print(json.dumps(result))

if __name__=='__main__':main()
