"""Independent API, PIL mask, source-pixel and pose readback; never runs a model."""
from common import *
import argparse
from functools import lru_cache
import numpy as np


def verify_api():
    freeze=read(ROOT/'api_freeze.json');assert sha(ROOT/'api_predictions.json')==freeze['predictions_sha256']
    assert sha(ROOT/'request_ledger.json')==freeze['ledger_sha256']
    ledger=read(ROOT/'request_ledger.json');assert len(ledger['jobs'])<=840 and accounted(ledger)<=20
    predictions=read(ROOT/'api_predictions.json')['rows'];assert len(predictions)==840
    manifest=read(ROOT/'api_input_manifest.json')['images'];images={e['id']:e for e in manifest}
    assert len(images)==280 and all('object_ids' not in e and 'gt_file' not in e for e in manifest)
    # Reconstruct bytes -> exact payload digest, independently of the HTTP client.
    import base64
    prompt=(ROOT/'grounding_prompt.txt').read_text();requests={}
    for iid,e in images.items():
        path=ROOT/e['image_file'];assert sha(path)==e['image_sha256']
        from PIL import Image
        with Image.open(path) as im:assert im.size==(e['size'],e['size']) and im.mode=='RGB' and not im.info
        request=dict(model='gpt-6.1-terra',reasoning={'effort':'low'},service_tier='default',max_output_tokens=5120,store=False,
            input=[{'role':'user','content':[{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode(),'detail':'original'},
                                           {'type':'input_text','text':prompt}]}])
        requests[iid]=digest(request)
    responses=0;unknown=0
    for j in predictions:
        if j['status']=='NOT_ATTEMPTED':continue
        original=ledger['jobs'][j['id']];assert original['request_sha256']==requests[j['image']]
        if j['status']=='RESERVED':
            unknown+=1;assert all(j['parsed'][o]['pixels'] is None for o in ('peg','box'));continue
        path=ROOT/j['raw_file'];assert sha(path)==j['raw_sha256'];raw=read(path);responses+=1
        if j['status']=='OK':
            assert raw['returned_model'].startswith('gpt-6.1-terra') and raw['served_tier']=='default'
            # Exact JSON decoder and manual coordinate validation; no producer parser.
            try:
                pairs=json.loads(raw['final_text'],object_pairs_hook=lambda p:p)
                assert len(pairs)==2 and {k for k,_ in pairs}=={'peg','box'}
                value=dict(pairs)
                for obj in ('peg','box'):
                    b=value[obj];actual=j['parsed'][obj]
                    if b is None:assert actual['status']=='NULL';continue
                    valid=isinstance(b,list) and len(b)==4 and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1000 for v in b)
                    valid=valid and b[0]<b[2] and b[1]<b[3]
                    if valid:
                        expected=np.asarray(b)*j['size']/1000
                        np.testing.assert_allclose(actual['pixels'],expected,atol=1e-10,rtol=0)
                    else:assert actual['pixels'] is None
            except (ValueError,AssertionError,TypeError):
                assert not j['parsed']['json_valid'],j['id']
        else:assert all(j['parsed'][o]['pixels'] is None for o in ('peg','box'))
        usage=j.get('usage') or {}
        if type(usage.get('input_tokens')) is int and type(usage.get('output_tokens')) is int:
            cost=(usage['input_tokens']+5*usage['output_tokens'])/1e6
            if j.get('served_tier') in ('fast','priority'):cost*=2
            assert abs(cost-j['accounted_usd'])<1e-12
    write(ROOT/'api_verification.json',dict(status='PASS',planned=840,recorded=len(ledger['jobs']),responses=responses,
        unknown_outcomes=unknown,input_images=280,accounted_usd=accounted(ledger),network_calls=0,
        checks='exact RGB/payload hashes; manual JSON/coordinates; raw response hashes; ledger/accounting; no GT metadata'))
    print('API_READBACK_PASS',responses,unknown,flush=True)


@lru_cache(maxsize=16)
def load_masks(path):
    with np.load(path,allow_pickle=False) as z:return z['masks'].copy()


@lru_cache(maxsize=12)
def truth(scene,size,camera):
    e=next(e for e in read(ROOT/'capture_manifest.json')['entries'] if e['scene']==scene and e['size']==size and e['camera']==camera)
    with np.load(ROOT/e['gt_file'],allow_pickle=False) as z:seg=z['actor_segmentation']
    return [np.isin(seg,e['object_ids'][o]) for o in ('peg','box','robot')]


def independent_pose_errors(est,state):
    from scipy.spatial.transform import Rotation
    def mat(p):return Rotation.from_quat([p[4],p[5],p[6],p[3]]).as_matrix()
    out={};peg=est['peg'];hole=est['hole'];pp=state['peg_pose'];hp=state['hole_pose']
    a=mat(pp)[:,0];ha=mat(hp)[:,0];L=state['peg_half'][0]
    if peg['status']=='ESTIMABLE':
        axis=np.asarray(peg['axis']);anchor=np.asarray(peg['anchor']);center=np.asarray(pp[:3]);denom=float(np.sum(axis*a))
        if abs(denom)>1e-10:
            anchor_on_plane=anchor+axis*np.sum((center-anchor)*a)/denom
            out['peg_position_error_mm']=float(np.sqrt(np.sum((anchor_on_plane-center)**2))*1000)
        out['peg_axis_error_deg']=float(np.degrees(np.arccos(np.clip(abs(np.sum(axis*a))/(np.linalg.norm(axis)*np.linalg.norm(a)),-1,1))))
        if peg.get('end_pair_status')=='ESTIMABLE':
            gt=np.asarray(pp[:3])+np.array([-L,L])[:,None]*a;pred=np.asarray(peg['physical_ends'])
            d0=np.sqrt(np.sum((pred-gt)**2,axis=1));d1=np.sqrt(np.sum((pred-gt[::-1])**2,axis=1))
            best=d0 if np.sum(d0*d0)<=np.sum(d1*d1) else d1
            out['peg_max_end_error_mm']=float(best.max()*1000)
    if hole['status']=='ESTIMABLE':
        entrance=np.asarray(hp[:3])-L*ha
        out['hole_center_error_mm']=float(np.sqrt(np.sum((np.asarray(hole['center'])-entrance)**2))*1000)
        axis=np.asarray(hole['normal']);out['hole_axis_error_deg']=float(np.degrees(np.arccos(np.clip(abs(np.sum(axis*ha))/(np.linalg.norm(axis)*np.linalg.norm(ha)),-1,1))))
    return out


def verify_remote():
    from PIL import Image,ImageChops
    from graph_worker import resolve_folder,verify_receipt
    from scipy.spatial.distance import cdist
    # All manifests and successful/failed receipts must retain their frozen bytes.
    protocol=assert_source();protected=0
    for path,h in protocol['protected_before'].items():assert sha(path)==h,path;protected+=1
    manifest=read(ROOT/'capture_manifest.json');assert manifest['status']=='COMPLETE' and len(manifest['entries'])==920
    for e in manifest['entries']:
        for k,h in (('image_file','image_sha256'),('sensor_file','sensor_sha256'),('gt_file','gt_sha256')):assert sha(ROOT/e[k])==e[h]
    maskrows=read(ROOT/'mask_metrics.json')['rows'];mask_checks=0
    def count(im):return im.convert('L').histogram()[255]
    for row in maskrows:
        init=row['parent']+'_track_t000' if row['cohort']=='tracking' else row['scene']
        iid=image_id(init,row['size'],row['camera']);kind='tracking' if row['cohort']=='tracking' else 'static'
        masks=load_masks(str(ROOT/'masks'/row['arm']/kind/job_id(iid,row['repeat'])/'masks.npz'))
        pred=masks[row['timestep'],row['object_id']] if kind=='tracking' else masks[row['object_id']]
        gt=truth(row['scene'],row['size'],row['camera'])[row['object_id']]
        a=Image.fromarray(pred).convert('1');b=Image.fromarray(gt).convert('1')
        inter=count(ImageChops.logical_and(a,b));union=count(ImageChops.logical_or(a,b));value=inter/union if union else None
        assert inter==row['intersection'] and union==row['union']
        assert value==row['mask_iou'] or (value is not None and abs(value-row['mask_iou'])<1e-12)
        assert count(a)==row['mask_pixels'] and count(b)==row['native_visible_pixels']
        mask_checks+=1
    poses=read(ROOT/'graph_pose_metrics.json')['rows'];pointrows=read(ROOT/'graph_point_metrics.json')['rows']
    index={}
    for arm in ('actual','gt_box','gt_mask'):
        freeze=read(ROOT/(arm+'_graph_freeze.json'));assert sha(ROOT/('graph_index_'+arm+'.json'))==freeze['index_sha256']
        for path,h in freeze['receipts'].items():assert sha(ROOT/path)==h
        for r in read(ROOT/('graph_index_'+arm+'.json'))['rows']:index[(arm,r['scene'],r['size'],r['repeat'])]=r
    pose_checks=0;source_checks=0
    for row in poses:
        r=index[row['arm'],row['scene'],row['size'],row['repeat']]
        if row['graph_status']!='COMPLETE':
            assert row['peg_position_error_mm'] is None and not row['peg_estimable'];continue
        folder=resolve_folder(r);verify_receipt(folder);result=read(folder/'result.json')
        state=read(ROOT/'GT'/row['scene']/'state.json');errors=independent_pose_errors(result['estimate'],state)
        for name,value in errors.items():assert row.get(name) is not None and abs(row[name]-value)<2e-5,(row,name,row.get(name),value)
        if not row['peg_estimable']:assert row['peg_position_error_mm'] is None
        if not row['hole_estimable']:assert row['hole_center_error_mm'] is None
        with np.load(folder/'graph.npz',allow_pickle=False) as z:
            assert np.isfinite(z['node_input']).all() and not z['node_input'][:,14:].any()
            for ci,camera in enumerate(CAMERAS):
                take=z['camera']==ci;pixels=z['pixel'][take];actual=z['xyz'][take]
                with np.load(ROOT/'sensors'/row['scene']/str(row['size'])/(camera+'.npz'),allow_pickle=False) as s:
                    v,u=np.divmod(pixels,row['size']);depth=s['depth_raw_mm'][v,u].astype(float)/1000
                    assert (depth>0).all();k=s['intrinsic_cv'];ext=s['extrinsic_cv']
                    cam=np.column_stack(((u+.5-k[0,2])*depth/k[0,0],(v+.5-k[1,2])*depth/k[1,1],depth))
                    independent=(cam-ext[:3,3])@ext[:3,:3]
                    np.testing.assert_allclose(actual,independent,atol=1e-5,rtol=0)
                source_checks+=len(pixels)
            if (folder/'stages.npz').exists():
                with np.load(folder/'stages.npz',allow_pickle=False) as st:
                    np.testing.assert_array_equal(st['final__xyz'],z['xyz'])
        pose_checks+=1
    # Independently reconstruct1024 references and use brute-force cdist on16 fixed real conditions.
    coverage_checks=0
    for spec in protocol['scenes'][:4]:
        reference={0:[],1:[]}
        for c in CAMERAS:
            with np.load(ROOT/'sensors'/spec['id']/'1024'/(c+'.npz'),allow_pickle=False) as sensor:
                raw=sensor['depth_raw_mm'];intr=sensor['intrinsic_cv'];cv=np.vstack([sensor['extrinsic_cv'],[0,0,0,1]])
                inv=np.linalg.inv(cv).astype(np.float32)
                for oid in (0,1):
                    if oid==1 and c!='base_camera':continue
                    support=truth(spec['id'],1024,c)[oid]&(raw>0);v,u=np.where(support);z=raw[v,u].astype(float)/1000
                    cam=np.column_stack([(u+.5-intr[0,2])*z/intr[0,0],(v+.5-intr[1,2])*z/intr[1,1],z])
                    reference[oid].append((cam@inv[:3,:3].T+inv[:3,3]).astype(np.float32))
        reference={oid:np.concatenate(value) for oid,value in reference.items()}
        for size in SIZES:
            rr=index['actual',spec['id'],size,0]
            if rr['status']!='COMPLETE':continue
            folder=resolve_folder(rr)
            with np.load(folder/'graph.npz',allow_pickle=False) as graph:
                for oid in (0,1):
                    take=graph['object_id']==oid;correct=np.zeros(int(take.sum()),bool)
                    camera=graph['camera'][take];pixel=graph['pixel'][take]
                    for ci,c in enumerate(CAMERAS):
                        here=np.flatnonzero(camera==ci);correct[here]=truth(spec['id'],size,c)[oid].ravel()[pixel[here]]
                    points=graph['xyz'][take][correct];ref=reference[oid]
                    count_near=0
                    if len(points):
                        for start in range(0,len(ref),2048):count_near+=int((cdist(ref[start:start+2048],points).min(1)<.004).sum())
                    value=count_near/len(ref) if len(ref) else None
                    scored=next(x for x in pointrows if x['arm']=='actual' and x['scene']==spec['id'] and x['size']==size and x['repeat']==0 and x['object_id']==oid)['surface_coverage_4mm']
                    if value is None:assert scored is None
                    else:assert abs(value-scored)<=1/max(1,len(ref))+1e-12
                    coverage_checks+=1
    timed=read(ROOT/'timing_graph_actual.json');assert timed['status']=='COMPLETE'
    native_checks=0
    for row in timed['rows']:
        if row['status']=='COMPLETE' and row['instrumentation_equal']:
            r=index['actual',row['scene'],row['size'],0]
            assert row['hashes']==read(resolve_folder(r)/'result.json')['hashes'];native_checks+=1
    for arm in ('actual','gt_box'):
        for kind in ('static','tracking'):
            t=read(ROOT/(f'timing_sam_{kind}_{arm}.json'));assert t['status']=='COMPLETE' and t['masks_equal']
            if kind=='static':assert len(t['rows'])==62*4*8
            else:assert len(t['sequences'])==8*4*4 and len(t['frames'])==8*4*4*21
    # Timing statistics are rebuilt by report.py from these raw rows, not copied summaries.
    import importlib.metadata,sam2
    before=read(ROOT/'runtime_before_main.json')
    after=dict(packages=sorted([list((x.metadata.get('Name',''),x.version)) for x in importlib.metadata.distributions()]),
        sam2_source_sha256={str(x):sha(x) for x in Path(sam2.__file__).parent.rglob('*.py')})
    assert before['packages']==after['packages'];assert before['sam2_source_sha256']==after['sam2_source_sha256']
    write(ROOT/'remote_verification.json',dict(status='PASS',PIL_mask_rows=mask_checks,pose_graph_rows=pose_checks,
        source_points_reprojected=source_checks,native_instrumented_Graph_checks=native_checks,
        protected_files=protected,camera_observations=920,bruteforce_common_reference_coverage_checks=coverage_checks,
        package_snapshot_unchanged=True,SAM2_source_unchanged=True,
        GT_used_in_actual_inference=False,models_run_by_verifier=0,
        limitations='Brute-force full reference coverage is independently checked on the first four static states at all sizes; all mask IoUs, final source-pixel reprojection and pose errors are checked.'))
    print('REMOTE_READBACK_PASS',mask_checks,pose_checks,native_checks,flush=True)


def merge():
    a=read(ROOT/'api_verification.json');b=read(ROOT/'remote_verification.json')
    assert a['status']==b['status']=='PASS'
    write(ROOT/'independent_verification.json',dict(status='PASS',api=a,remote=b,
        summary_verification='pending report generation and independent aggregate readback',verified_utc=now()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['api','remote','merge']);a=ap.parse_args()
    {'api':verify_api,'remote':verify_remote,'merge':merge}[a.stage]()
