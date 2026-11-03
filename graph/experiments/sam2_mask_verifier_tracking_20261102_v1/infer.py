from common import *
import torch, subprocess, resource
from PIL import Image
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json');assert v['stage']=='INPUTS_FROZEN' and v['Qwen_loads']==0
assert sha(R/'request_manifest.json')==v['request_manifest_sha256'] and sha(R/'prompt_identity.json')==v['prompt_sha256']
requests=rd(R/'request_manifest.json')['requests'];prompt=rd(R/'prompt_identity.json')['prompt'];raw=[];lat=[]
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.init();assert torch.cuda.mem_get_info()[0]>22*1024**3
update(stage='QWEN_LOADING',Qwen_loads=1)
with ExperimentLogger(team='integration',model='Task10I_SAM2_mask_Qwen',seed=42,job_type='evaluate',root=R/'.execution/experiments',config=rd(R/'protocol.json')) as logger:
    t=time.perf_counter();model=Qwen3_5ForConditionalGeneration.from_pretrained(B/'model',local_files_only=True,device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa').eval();torch.cuda.synchronize()
    assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in model.parameters()) and all(b.device.type=='cuda' for b in model.buffers())
    cost=dict(load_ms=(time.perf_counter()-t)*1000,loaded_memory=gpu_mem(),requests=lat)
    proc=AutoProcessor.from_pretrained(B/'model',local_files_only=True,min_pixels=512**2,max_pixels=512**2)
    entries=[];events=[];stack=[];vision=[];vstack=[]
    def pre(m,a,k):
        e=torch.cuda.Event(enable_timing=True);e.record();stack.append(e)
        past=k.get('past_key_values');entries.append(dict(cache_length=past.get_seq_length() if past is not None else 0,tokens=k['input_ids'].shape[1] if k.get('input_ids') is not None else None))
    def post(m,a,k,o):e=torch.cuda.Event(enable_timing=True);e.record();events.append((stack.pop(),e))
    def vp(m,a):e=torch.cuda.Event(enable_timing=True);e.record();vstack.append(e)
    def vo(m,a,o):e=torch.cuda.Event(enable_timing=True);e.record();vision.append((vstack.pop(),e))
    hooks=[model.register_forward_pre_hook(pre,with_kwargs=True),model.register_forward_hook(post,with_kwargs=True),model.model.visual.register_forward_pre_hook(vp),model.model.visual.register_forward_hook(vo)]
    for index,req in enumerate(requests):
        assert all(sha(p)==h for p,h in zip(req['images'],req['image_sha256']))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();ei=len(events);vi=len(vision);ci=len(entries)
        content=[dict(type='image',image=Image.open(p).convert('RGB')) for p in req['images']]+[dict(type='text',text=prompt)]
        x=proc.apply_chat_template([dict(role='user',content=content)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt');prep=(time.perf_counter()-t)*1000;n=x['input_ids'].shape[1];grid=x['image_grid_thw'].tolist();assert len(grid)==3;x=x.to('cuda')
        with torch.inference_mode():out=model.generate(**x,max_new_tokens=16,do_sample=False,use_cache=True)
        torch.cuda.synchronize();ids=out[:,n:].cpu().tolist()[0];text=proc.batch_decode(out[:,n:],skip_special_tokens=True)[0];label=text.strip() if text.strip() in LABELS else 'INVALID_OUTPUT'
        assert entries[ci]['cache_length']==0 and entries[ci]['tokens']==n and len(vision)-vi==1
        timing=dict(id=req['id'],condition=req['condition'],cold_first_request=index==0,preprocessing_ms=prep,end_to_end_ms=(time.perf_counter()-t)*1000,input_tokens=n,output_tokens=len(ids),image_grid_thw=grid,initial_forward_cache=entries[ci],vision_cuda_ms=sum(a.elapsed_time(b) for a,b in vision[vi:]),vision_prefill_cuda_ms=events[ei][0].elapsed_time(events[ei][1]),decode_forward_cuda_ms=sum(a.elapsed_time(b) for a,b in events[ei+1:]),memory=gpu_mem())
        raw.append(dict(id=req['id'],candidate_id=req['candidate_id'],condition=req['condition'],raw_text=text,strict_label=label,output_token_ids=ids,image_sha256=req['image_sha256'],timing=timing));lat.append(timing)
        # Each completed raw answer is atomically durable. No retry or format repair.
        wr('verifier_raw_responses.json',dict(GT_opened=False,responses=raw));wr('verifier_latency_vram.json',cost);del x,out
        if (index+1)%20==0:print('QWEN_PROGRESS',index+1,'/',len(requests),flush=True)
    for h in hooks:h.remove()
    cost['CPU_ru_maxrss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;wr('verifier_latency_vram.json',cost)
    csvwr('verifier_predictions.csv',[dict(candidate_id=r['candidate_id'],condition=r['condition'],strict_label=r['strict_label'],raw_text=r['raw_text']) for r in raw])
    cm=rd(R/'candidate_manifest.json');old=rd(P/'verifier_raw_responses.json')['responses'];labels=dict(A={r['candidate_id']:r['strict_label'] for r in old if r['condition']=='appearance_three'})
    for method in ['B','C']:labels[method]={r['candidate_id']:r['strict_label'] for r in raw if r['condition']==method}
    for c in cm['candidates']:labels['C'].setdefault(c['id'],'NO_MASK')
    picks=[]
    for image in cm['images']:
        for method in METHODS:
            for obj in ['peg','box']:
                pick=selected(cm['candidates'],labels[method],image['id'],obj)
                picks.append(dict(method=method,image_id=image['id'],parent_id=image['parent_id'],camera=image['camera'],frame_kind=image['frame_kind'],timestep=image['timestep'],object=obj,candidate_id=pick['id'] if pick else None,box=pick['box'] if pick else None))
    wr('selected_candidates.json',dict(rows=picks));wr('candidate_labels.json',labels)
    update(stage='STATIC_PREDICTIONS_FROZEN',completed_requests=len(raw),raw_sha256_before_GT=sha(R/'verifier_raw_responses.json'),predictions_sha256_before_GT=sha(R/'verifier_predictions.csv'),selected_sha256=sha(R/'selected_candidates.json'))
    logger.log(dict(requests=len(raw),format_valid=sum(r['strict_label'] in LABELS for r in raw)));logger.link_output(R/'verifier_predictions.csv');print('STATIC_PREDICTIONS_FROZEN',len(raw),flush=True)
