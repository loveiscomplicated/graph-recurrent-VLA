"""CUDA image/video predictions; all real outputs freeze before any oracle GT read."""
from common import *
import argparse
import time
import numpy as np
import torch
from PIL import Image


def setup(video=False):
    from sam2.build_sam import build_sam2,build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    assert os.environ.get('CONTAINER_ID')=='58576054' and torch.cuda.is_available()
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    torch.set_num_threads(4);torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.cuda.synchronize();t=time.perf_counter()
    if video:model=build_sam2_video_predictor(CONFIG,CHECKPOINT,device='cuda')
    else:model=build_sam2(CONFIG,CHECKPOINT,device='cuda')
    model.eval();torch.cuda.synchronize()
    assert model.image_size==1024 and all(p.dtype==torch.float32 for p in model.parameters())
    return (model if video else SAM2ImagePredictor(model)),dict(load_ms=(time.perf_counter()-t)*1000,
        device=torch.cuda.get_device_name(),torch_version=torch.__version__,checkpoint_sha256=CHECKPOINT_SHA,
        internal_image_size=model.image_size,video=video)


def image_predict(predictor,rgb,boxes,detail=False):
    """Use the existing public predictor; detailed hooks are diagnostic only."""
    from functools import wraps
    spans=[];original=[]
    def hook(obj,name,label):
        fn=getattr(obj,name);original.append((obj,name,fn))
        @wraps(fn)
        def call(*a,**kw):
            torch.cuda.synchronize();t=time.perf_counter();out=fn(*a,**kw);torch.cuda.synchronize()
            spans.append(dict(stage=label,ms=(time.perf_counter()-t)*1000));return out
        setattr(obj,name,call)
    if detail:
        hook(type(predictor._transforms),'__call__','preprocess')
        hook(predictor.model,'forward_image','image_encoder')
        hook(predictor.model.sam_prompt_encoder,'forward','prompt_encoder')
        hook(predictor.model.sam_mask_decoder,'forward','mask_decoder')
        hook(predictor._transforms,'postprocess_masks','restore_original_size')
        hook(predictor,'_prep_prompts','prompt_preparation')
        hook(predictor,'_predict','gpu_mask_prediction')
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    masks=np.zeros((2,*rgb.shape[:2]),bool);scores=[None,None]
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictor.set_image(rgb);torch.cuda.synchronize();encoded=time.perf_counter()
            prompt_ms=[]
            for oid,box in enumerate(boxes):
                if box is None:prompt_ms.append(None);continue
                t=time.perf_counter()
                m,sc,_=predictor.predict(box=np.asarray(box,np.float32),multimask_output=False,return_logits=True)
                masks[oid]=m[0]>0;scores[oid]=float(sc[0]);torch.cuda.synchronize()
                prompt_ms.append((time.perf_counter()-t)*1000)
        end=time.perf_counter()
    finally:
        for obj,name,fn in reversed(original):setattr(obj,name,fn)
    timing=dict(encode_ms=(encoded-start)*1000,prompt_ms=prompt_ms,total_ms=(end-start)*1000,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),prompt_count=sum(b is not None for b in boxes),detail=spans)
    if detail:
        timing['transfer_feature_pack_residual_ms']=max(0.,timing['encode_ms']-sum(s['ms'] for s in spans if s['stage'] in ('image_encoder','preprocess')))
        # Public predict CPU materialization plus thresholding is the residual after GPU prediction and prompt prep.
        timing['cpu_output_residual_ms']=max(0.,sum(x or 0 for x in prompt_ms)-sum(s['ms'] for s in spans if s['stage'] in ('gpu_mask_prediction','prompt_preparation')))
    return masks,timing,scores


def save_masks(folder,masks,timing,**metadata):
    folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'masks.npz',masks=masks)
    write(folder/'result.json',dict(metadata,timing=timing,mask_sha256=array_sha(masks),file_sha256=sha(folder/'masks.npz')))


def smoke():
    assert_source()
    manifest=read(ROOT/'smoke_capture_manifest.json')['entries'];predictor,identity=setup()
    old=read(ROOT/'reference/full_image_grounding_predictions.json')['rows']
    results=[]
    from graph_adapter import geometry,load_inputs,build
    for size in SIZES:
        mm={}
        for camera in CAMERAS:
            e=next(e for e in manifest if e['size']==size and e['camera']==camera)
            prior=next(p for p in old if p['provider']=='low' and e['parent'] in p['image_id'] and '_F0_' in p['image_id'] and camera in p['image_id'])
            boxes=[None if prior['parsed'][o]['normalized'] is None else
                   [v*size/1000 for v in prior['parsed'][o]['normalized']] for o in ('peg','box')]
            rgb=np.asarray(Image.open(ROOT/e['image_file']).convert('RGB'))
            masks,timing,_=image_predict(predictor,rgb,boxes)
            repeated,detail,_=image_predict(predictor,rgb,boxes,detail=True)
            assert np.array_equal(masks,repeated)
            assert masks.shape==(2,size,size)
            mm[camera]=masks
        geo=geometry();inputs=load_inputs(e['scene'],size,geo)
        plain=build(geo,inputs,mm);instrumented=build(geo,inputs,mm,trace=True)
        assert plain['hashes']==instrumented['hashes']
        results.append(dict(size=size,mask_shapes={c:list(m.shape) for c,m in mm.items()},graph_hashes=plain['hashes'],
                            graph_ms=plain['graph_ms'],instrumented_graph_ms=instrumented['graph_ms'],gate=plain['gate']))
        print('SMOKE_PASS',size,'graph_ms',round(plain['graph_ms'],2),flush=True)
    write(ROOT/'smoke_verification.json',dict(status='PASS',results=results,sam2=identity,
        GT_input=False,boxes='previous same-scene Terra low predictions; smoke only; no new API requests'))


def static(arm='actual'):
    assert_source();assert sha(ROOT/'api_predictions.json')==read(ROOT/'api_freeze.json')['predictions_sha256']
    if arm!='actual':assert (ROOT/'actual_graph_freeze.json').exists()
    pred=read(ROOT/'api_predictions.json')['rows'] if arm=='actual' else read(ROOT/'oracle_jobs.json')['rows']
    entries=[e for e in read(ROOT/'capture_manifest.json')['entries'] if e['cohort']!='tracking']
    lookup={e['id']:e for e in entries};predictor,identity=setup();rows=[]
    with logger('sam_static_'+arm) as log:
        for pi,p in enumerate(pred):
            if p['image'] not in lookup:continue
            e=lookup[p['image']];folder=ROOT/'masks'/arm/'static'/p['id']
            if folder.exists():
                r=read(folder/'result.json');assert sha(folder/'masks.npz')==r['file_sha256'];rows.append(r);continue
            rgb=np.asarray(Image.open(ROOT/e['image_file']).convert('RGB'))
            boxes=[p['parsed'][o]['pixels'] for o in ('peg','box')]
            masks,timing,scores=image_predict(predictor,rgb,boxes)
            save_masks(folder,masks,timing,job=p['id'],image=e['id'],scene=e['scene'],parent=e['parent'],
                       size=e['size'],camera=e['camera'],repeat=p['repeat'],arm=arm,scores=scores)
            rows.append(read(folder/'result.json'))
            if len(rows)%30==0:print('SAM_STATIC',arm,len(rows),flush=True)
        assert len(rows)==(744 if arm=='actual' else 248)
        write(ROOT/('sam_static_'+arm+'.json'),dict(rows=rows,identity=identity,frozen_utc=now()))
        log.log(dict(camera_predictions=len(rows)));log.link_output(ROOT/('sam_static_'+arm+'.json'))


def video_predict(model,folder,boxes,callback=None,before_frame=None):
    """Full CPU preload is separately timed; model sees only current temporal horizon."""
    state=None;all_masks=[];times=[];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=model.init_state(video_path=str(folder),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False)
            torch.cuda.synchronize();loaded=time.perf_counter();state['num_frames']=1
            for oid,box in enumerate(boxes,1):
                if box is not None:model.add_new_points_or_box(state,frame_idx=0,obj_id=oid,box=np.asarray(box,np.float32))
            torch.cuda.synchronize();initialized=time.perf_counter()
            h=state['video_height'];w=state['video_width']
            for t in range(21):
                if before_frame:before_frame(t)
                state['num_frames']=t+1;torch.cuda.synchronize();start=time.perf_counter()
                masks=np.zeros((2,h,w),bool)
                if any(b is not None for b in boxes):
                    it=model.propagate_in_video(state,start_frame_idx=t,max_frame_num_to_track=1)
                    try:frame,ids,logits=next(it)
                    finally:it.close()
                    assert frame==t;torch.cuda.synchronize();forwarded=time.perf_counter()
                    cpu=(logits[:,0]>0).cpu().numpy()
                    for j,oid in enumerate(ids):masks[int(oid)-1]=cpu[j]
                else:forwarded=time.perf_counter()
                torch.cuda.synchronize();end=time.perf_counter()
                all_masks.append(masks);times.append(dict(timestep=t,forward_ms=(forwarded-start)*1000,
                    cpu_output_ms=(end-forwarded)*1000,total_ms=(end-start)*1000,executed=any(b is not None for b in boxes)))
                if callback:callback(t,masks)
        return np.stack(all_masks),dict(video_load_first_encode_ms=(loaded-begin)*1000,
            prompt_init_ms=(initialized-loaded)*1000,init_total_ms=(initialized-begin)*1000,frames=times,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
    finally:
        if state is not None:model.reset_state(state)


def tracking(arm='actual'):
    assert_source()
    if arm!='actual':assert (ROOT/'actual_graph_freeze.json').exists()
    predictions=read(ROOT/'api_predictions.json')['rows'] if arm=='actual' else read(ROOT/'oracle_jobs.json')['rows']
    entries=[e for e in read(ROOT/'capture_manifest.json')['entries'] if e['cohort']=='tracking' and e['timestep']==0]
    lookup={e['id']:e for e in entries};model,identity=setup(video=True);rows=[]
    with logger('sam_tracking_'+arm) as log:
        for p in predictions:
            if p['image'] not in lookup:continue
            e=lookup[p['image']];dest=ROOT/'masks'/arm/'tracking'/p['id']
            if dest.exists():
                r=read(dest/'result.json');assert sha(dest/'masks.npz')==r['file_sha256'];rows.append(r);continue
            folder=ROOT/'jpeg'/e['parent']/str(e['size'])/e['camera']
            boxes=[p['parsed'][o]['pixels'] for o in ('peg','box')]
            masks,timing=video_predict(model,folder,boxes)
            assert masks.shape==(21,2,e['size'],e['size'])
            save_masks(dest,masks,timing,job=p['id'],image=e['id'],scene=e['scene'],parent=e['parent'],
                       size=e['size'],camera=e['camera'],repeat=p['repeat'],arm=arm)
            rows.append(read(dest/'result.json'));print('SAM_TRACK',arm,len(rows),flush=True)
        assert len(rows)==(96 if arm=='actual' else 32)
        write(ROOT/('sam_tracking_'+arm+'.json'),dict(rows=rows,identity=identity,frozen_utc=now()))
        log.log(dict(sequences=len(rows)));log.link_output(ROOT/('sam_tracking_'+arm+'.json'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','static','tracking']);ap.add_argument('--arm',default='actual');a=ap.parse_args()
    if a.stage=='smoke':smoke()
    elif a.stage=='static':static(a.arm)
    else:tracking(a.arm)
