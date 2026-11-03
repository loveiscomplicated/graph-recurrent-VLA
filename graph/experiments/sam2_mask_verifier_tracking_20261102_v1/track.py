from common import *
import numpy as np,torch,resource
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json');assert v['stage']=='SEQUENCES_FROZEN' and v['SAM2_loads']==1
assert sha(R/'sequence_manifest.json')==v['sequence_manifest_sha256'] and sha(R/'selected_candidates.json')==v['selected_sha256']
seqs=rd(R/'sequence_manifest.json')['sequences'];picks=rd(R/'selected_candidates.json')['rows'];cands={c['candidate_id']:c for c in rd(R/'sam2_candidate_manifest.json')['candidates']};rows=[];lat=[]
torch.set_num_threads(4);torch.manual_seed(42);update(stage='VIDEO_SAM_LOADING',SAM2_loads=2)
with ExperimentLogger(team='integration',model='Task10I_SAM2_identity_tracking',seed=42,job_type='evaluate',root=R/'.execution/experiments',config=rd(R/'protocol.json')) as logger:
    t=time.perf_counter();model=build_sam2_video_predictor(CFG,str(CK),device='cuda',vos_optimized=False).eval();torch.cuda.synchronize();cost=dict(load_ms=(time.perf_counter()-t)*1000,loaded_memory=gpu_mem(),sequences=lat)
    assert all(p.device.type=='cuda' and p.dtype==torch.float32 for p in model.parameters())
    for seq in seqs:
        assert all(sha(f['JPEG_path'])==f['JPEG_sha256'] for f in seq['frames'])
        for method in METHODS:
            chosen={r['object']:r['candidate_id'] for r in picks if r['method']==method and r['parent_id']==seq['parent_id'] and r['camera']==seq['camera'] and r['frame_kind']=='F0'};assert set(chosen)=={'peg','box'}
            masks=np.zeros((81,2,512,512),dtype=bool);init={};valid={}
            for i,obj in enumerate(['peg','box']):
                cid=chosen[obj];init[obj]=dict(candidate_id=cid,mask_pixels=0)
                if cid:
                    c=cands[cid];assert sha(c['mask_path'])==c['mask_sha256'];m=np.asarray(Image.open(c['mask_path']))>0;init[obj]['mask_pixels']=int(m.sum())
                    if m.any():valid[i+1]=m
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();frames=[];state=None
            if valid:
                try:
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        state=model.init_state(video_path=seq['jpeg_folder'],offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False)
                        for objid,m in valid.items():model.add_new_mask(state,frame_idx=0,obj_id=objid,mask=m)
                        torch.cuda.synchronize();initms=(time.perf_counter()-t)*1000;previous=time.perf_counter()
                        for frame_idx,object_ids,logits in model.propagate_in_video(state,start_frame_idx=0):
                            assert frame_idx==len(frames)
                            for k,objid in enumerate(object_ids):masks[frame_idx,int(objid)-1]=logits[k,0].float().cpu().numpy()>0
                            torch.cuda.synchronize();now=time.perf_counter();frames.append(dict(timestep=int(frame_idx),wall_ms=(now-previous)*1000));previous=now
                    assert len(frames)==81
                finally:
                    if state is not None:model.reset_state(state);del state
            else:initms=0.
            torch.cuda.synchronize();out=R/'tracking_masks'/(seq['id']+'__'+method+'.npz');out.parent.mkdir(exist_ok=True);np.savez_compressed(out,masks=masks)
            timing=dict(sequence_id=seq['id'],method=method,total_ms=(time.perf_counter()-t)*1000,initialization_ms=initms,frames=frames,memory=gpu_mem())
            rows.append(dict(sequence_id=seq['id'],parent_id=seq['parent_id'],camera=seq['camera'],method=method,initialization=init,predicted_frames=81,mask_path=str(out),mask_sha256=sha(out),actual_video_frames=len(frames)));lat.append(timing)
            wr('tracking_predictions.json',dict(GT_opened=False,rows=rows));wr('tracking_latency_vram.json',cost);del masks
            print('TRACKING_PROGRESS',len(rows),'/48',seq['id'],method,flush=True)
    assert len(rows)==48
    cost['CPU_ru_maxrss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;wr('tracking_latency_vram.json',cost)
    update(stage='ALL_PREDICTIONS_FROZEN',tracking_sha256_before_GT=sha(R/'tracking_predictions.json'),completed_tracking_states=48,completed_tracking_camera_frames=3888)
    logger.log(dict(tracking_states=48,camera_frames=3888));logger.link_output(R/'tracking_predictions.json');print('ALL_PREDICTIONS_FROZEN',flush=True)
