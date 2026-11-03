from common import *
import time,traceback,subprocess,threading,resource
import torch,psutil
from PIL import Image
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json');requests=rd(R/'request_manifest.json')['requests'];prompts=rd(R/'prompt_identity.json')['conditions'];raw=[];lat=dict(requests=[]);rss=[];stop=threading.Event()
def monitor():
 while not stop.is_set():rss.append(psutil.Process().memory_info().rss);stop.wait(.05)
threading.Thread(target=monitor,daemon=True).start()
def mem():return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
try:
 assert os.environ.get('CONTAINER_ID')=='58576054' and v['stage']=='INPUTS_FROZEN' and v['model_loads']==0
 assert sha(R/'request_manifest.json')==v['request_manifest_sha256'] and sha(R/'prompt_identity.json')==v['prompt_identity_sha256'] and sha(R/'protocol.json')==v['protocol_sha256']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.init();torch.cuda.synchronize();assert torch.cuda.mem_get_info()[0]>22*1024**3
 with ExperimentLogger(team='integration',model='Task10I_Qwen_perception_diagnostic',seed=42,job_type='evaluate',root=R/'.execution/experiments',config=rd(R/'protocol.json')) as logger:
  v.update(stage='QWEN_LOADING',model_loads=1,logger_run_id=logger.run_id);wr('verification.json',v)
  with (R/'.execution/MODEL_LOAD_STARTED').open('x') as g:g.write('one BF16 Qwen weight load; no retries\n')
  t=time.perf_counter();model=Qwen3_5ForConditionalGeneration.from_pretrained(B/'model',local_files_only=True,device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa').eval();torch.cuda.synchronize()
  assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in model.parameters()) and all(b.device.type=='cuda' for b in model.buffers())
  lat.update(load_ms=(time.perf_counter()-t)*1000,loaded_memory=mem(),load_CPU_peak_sampled_bytes=max(rss),actual_precision='All parameters BF16 on CUDA, no quantization or offload')
  proc=AutoProcessor.from_pretrained(B/'model',local_files_only=True,min_pixels=512**2,max_pixels=512**2)
  ev=[];ve=[];fs=[];vs=[]
  entries=[]
  def pre(m,a,k):
   e=torch.cuda.Event(enable_timing=True);e.record();fs.append(e)
   past=k.get('past_key_values');entries.append(dict(cache_length=past.get_seq_length() if past is not None else 0,cache_id=id(past) if past is not None else None,sequence_tokens=k['input_ids'].shape[1] if k.get('input_ids') is not None else None))
  def post(m,a,k,o):e=torch.cuda.Event(enable_timing=True);e.record();ev.append((fs.pop(),e))
  def vp(m,a):e=torch.cuda.Event(enable_timing=True);e.record();vs.append(e)
  def vo(m,a,o):e=torch.cuda.Event(enable_timing=True);e.record();ve.append((vs.pop(),e))
  hooks=[model.register_forward_pre_hook(pre,with_kwargs=True),model.register_forward_hook(post,with_kwargs=True),model.model.visual.register_forward_pre_hook(vp),model.model.visual.register_forward_hook(vo)]
  for index,req in enumerate(requests):
   pid=prompts[req['condition']]
   assert all(sha(p)==h for p,h in zip(req['images'],req['image_sha256']))
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();ei=len(ev);vi=len(ve);entry_index=len(entries)
   content=([dict(type='text',text=pid['preamble'])] if pid['preamble'] else [])+[dict(type='image',image=Image.open(p).convert('RGB')) for p in req['images']]+[dict(type='text',text=pid['prompt'])]
   x=proc.apply_chat_template([dict(role='user',content=content)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt');prep=(time.perf_counter()-t)*1000;n=x['input_ids'].shape[1];grid=x['image_grid_thw'].tolist();assert len(grid)==len(req['images']);x=x.to('cuda')
   with torch.inference_mode():out=model.generate(**x,max_new_tokens=pid['max_new_tokens'],do_sample=False,use_cache=True)
   torch.cuda.synchronize();ids=out[:,n:].cpu().tolist()[0];text=proc.batch_decode(out[:,n:],skip_special_tokens=True)[0];wall=(time.perf_counter()-t)*1000;label=(text.strip() if text.strip() in pid['labels'] else 'INVALID_OUTPUT') if pid['labels'] else None;assert len(ve)-vi==1
   timing=dict(id=req['id'],index=index,cold_first_request=index==0,input_tokens=n,output_tokens=len(ids),preprocessing_ms=prep,end_to_end_ms=wall,image_grid_thw=grid,vision_cuda_ms=sum(a.elapsed_time(b) for a,b in ve[vi:]),vision_prefill_cuda_ms=ev[ei][0].elapsed_time(ev[ei][1]),decode_forward_cuda_ms=sum(a.elapsed_time(b) for a,b in ev[ei+1:]),memory=mem())
   first=entries[entry_index];assert first['cache_length']==0 and first['sequence_tokens']==n
   timing.update(condition=req['condition'],initial_forward_cache=first,vision_forward_count=len(ve)-vi)
   raw.append(dict(id=req['id'],candidate_id=req['candidate_id'],condition=req['condition'],raw_text=text,output_token_ids=ids,strict_label=label,image_sha256=req['image_sha256'],timing=timing));lat['requests'].append(timing)
   wr('verifier_raw_responses.json',dict(GT_opened=False,responses=raw));wr('verifier_latency_vram.json',lat);del x,out
   if (index+1)%10==0:print('DIAGNOSTIC_PROGRESS',index+1,'/50',flush=True)
  for h in hooks:h.remove()
  assert len(raw)==50
  csvwr('verifier_predictions.csv',[dict(id=r['id'],candidate_id=r['candidate_id'],condition=r['condition'],strict_label=r['strict_label'],raw_text=r['raw_text']) for r in raw])
  v.update(stage='PREDICTIONS_FROZEN',completed_requests=len(raw),raw_sha256_before_GT=sha(R/'verifier_raw_responses.json'),predictions_sha256_before_GT=sha(R/'verifier_predictions.csv'));wr('verification.json',v)
  logger.log(dict(requests=len(raw),strict_valid=sum(r['strict_label']!='INVALID_OUTPUT' for r in raw)));logger.link_output(R/'verifier_predictions.csv');print('VERIFIER_PREDICTIONS_FROZEN',len(raw),flush=True)
except Exception as ex:
 v.update(stage='RUNTIME_STOPPED',completed_requests=len(raw),error=repr(ex),traceback=traceback.format_exc());wr('verification.json',v);print(traceback.format_exc(),flush=True);raise
finally:
 stop.set();lat.update(CPU_process_peak_sampled_bytes=max(rss) if rss else None,CPU_ru_maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);wr('verifier_latency_vram.json',lat)
