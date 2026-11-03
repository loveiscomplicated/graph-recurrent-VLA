from common import *
import numpy as np, torch, resource
from PIL import Image, ImageDraw, ImageFilter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json');assert v['stage']=='PROTOCOL_FROZEN' and v['SAM2_loads']==0
assert sha(R/'candidate_manifest.json')==v['candidate_manifest_sha256']
cm=rd(R/'candidate_manifest.json'); images={c['id']:c for c in cm['images']};candidates=cm['candidates'];raw=[];requests=[];timings=[]
torch.set_num_threads(4);torch.manual_seed(42)
update(stage='STATIC_SAM_LOADING',SAM2_loads=1)
with ExperimentLogger(team='integration',model='Task10I_SAM2_mask_verifier',seed=42,job_type='evaluate',root=R/'.execution/experiments',config=rd(R/'protocol.json')) as logger:
    t=time.perf_counter();model=build_sam2(CFG,str(CK),device='cuda').eval();pred=SAM2ImagePredictor(model);torch.cuda.synchronize()
    assert all(p.device.type=='cuda' and p.dtype==torch.float32 for p in model.parameters())
    cost=dict(load_ms=(time.perf_counter()-t)*1000,loaded_memory=gpu_mem(),images=[],candidates=timings)
    wr('sam2_effective_config.json',dict(dynamic_multimask_via_stability=model.sam_mask_decoder.dynamic_multimask_via_stability,dynamic_multimask_stability_delta=model.sam_mask_decoder.dynamic_multimask_stability_delta,dynamic_multimask_stability_thresh=model.sam_mask_decoder.dynamic_multimask_stability_thresh,mask_threshold=pred.mask_threshold,max_hole_area=pred._transforms.max_hole_area,max_sprinkle_area=pred._transforms.max_sprinkle_area))
    for index,(iid,image) in enumerate(images.items()):
        assert sha(image['image_path'])==image['image_sha256']
        im=Image.open(image['image_path']).convert('RGB');rgb=np.asarray(im);assert rgb.shape==(512,512,3)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred.set_image(rgb)
        torch.cuda.synchronize();cost['images'].append(dict(id=iid,encoding_ms=(time.perf_counter()-t)*1000,memory=gpu_mem()))
        cache={}
        for c in [x for x in candidates if x['image_id']==iid]:
            key=tuple(c['box']);assert all(math.isfinite(x) for x in key) and key[0]<key[2] and key[1]<key[3]
            t=time.perf_counter();reused=key in cache
            if not reused:
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    masks,scores,low=pred.predict(box=np.asarray(key,np.float32),multimask_output=False,return_logits=True)
                torch.cuda.synchronize();assert masks.shape==(1,512,512)
                cache[key]=(masks[0]>0,float(scores[0]))
            mask,score=cache[key];assert mask.dtype==bool
            mp=R/'candidate_masks'/(c['id']+'.png');mp.parent.mkdir(exist_ok=True);Image.fromarray(mask.astype(np.uint8)*255).save(mp)
            bpaths=[image['image_path'],str(P/'candidate_images'/(c['id']+'_marked.png')),str(P/'candidate_images'/(c['id']+'_crop.png'))]
            requests.append(dict(id=c['id']+'__B',candidate_id=c['id'],condition='B',images=bpaths,image_sha256=[sha(p) for p in bpaths]))
            cp=[]
            if mask.any():
                folder=R/'masked_inputs';folder.mkdir(exist_ok=True)
                binary=Image.fromarray(mask.astype(np.uint8)*255);edge=np.asarray(binary.filter(ImageFilter.MaxFilter(3)))!=np.asarray(binary.filter(ImageFilter.MinFilter(3)))
                marked=rgb.copy();marked[edge]=[0,180,220];markedpath=folder/(c['id']+'_marked.png');Image.fromarray(marked).save(markedpath)
                dim=rgb.copy();dim[~mask]=(rgb[~mask].astype(np.float32)*.25).astype(np.uint8)
                crop=c['crop_box'];croppath=folder/(c['id']+'_crop.png');Image.fromarray(dim).crop(crop).save(croppath)
                cp=[image['image_path'],str(markedpath),str(croppath)]
                requests.append(dict(id=c['id']+'__C',candidate_id=c['id'],condition='C',images=cp,image_sha256=[sha(p) for p in cp]))
            crop=c['crop_box'];inside=int(mask[crop[1]:crop[3],crop[0]:crop[2]].sum())
            raw.append(dict(candidate_id=c['id'],image_id=iid,mask_path=str(mp),mask_sha256=sha(mp),mask_pixels=int(mask.sum()),mask_pixels_outside_crop=int(mask.sum())-inside,SAM2_predicted_quality=score,C_status='READY' if cp else 'NO_MASK',B_images=bpaths,C_images=cp))
            timings.append(dict(id=c['id'],mask_ms=(time.perf_counter()-t)*1000,reused=reused,memory=gpu_mem()))
        pred.reset_predictor();wr('sam2_candidate_manifest.json',dict(candidates=raw));wr('sam2_static_latency_vram.json',cost)
        print('STATIC_MASK_PROGRESS',index+1,'/60',len(raw),'candidates',flush=True)
    assert len(raw)==986 and len(requests)<=1972
    requests.sort(key=lambda r:hashlib.sha256(('sam2mask|'+r['id']).encode()).hexdigest())
    wr('request_manifest.json',dict(requests=requests));cost['CPU_ru_maxrss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;wr('sam2_static_latency_vram.json',cost)
    update(stage='INPUTS_FROZEN',sam2_candidate_manifest_sha256=sha(R/'sam2_candidate_manifest.json'),request_manifest_sha256=sha(R/'request_manifest.json'),request_count=len(requests),empty_masks=sum(r['C_status']=='NO_MASK' for r in raw))
    logger.log(dict(candidate_masks=len(raw),requests=len(requests)));logger.link_output(R/'sam2_candidate_manifest.json')
    print('INPUTS_FROZEN',len(requests),flush=True)
