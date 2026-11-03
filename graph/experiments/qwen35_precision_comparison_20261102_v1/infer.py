from common import *
import time, traceback, threading, re, subprocess, resource, tempfile, collections
import numpy as np, torch, bitsandbytes as bnb, psutil
from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json'); mf=rd(R/'input_manifest.json'); ident=rd(R/'model_identity.json')
lat=dict(requests=[],sam2=[]); responses=[]; samrows=[]; stop=threading.Event(); rss=[]
def monitor():
 while not stop.is_set(): rss.append(psutil.Process().memory_info().rss); stop.wait(.05)
threading.Thread(target=monitor,daemon=True).start()
def mem():
 f,t=torch.cuda.mem_get_info()
 return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),device_free=f,total=t)
def gpu(m): return all(p.device.type=='cuda' for p in m.parameters()) and all(p.device.type=='cuda' for p in m.buffers())
def save_raw(): wr('full_image_grounding_raw_responses.json',dict(GT_opened=False,DINO_outputs_opened=False,responses=responses))
try:
 assert v['stage']=='INPUTS_FROZEN' and v['model_loads']==0
 assert sha(R/'input_manifest.json')==v['manifest_sha256'] and sha(R/'grounding_prompt.txt')==v['prompt_sha256'] and sha(R/'model_identity.json')==v['identity_sha256']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.init();torch.cuda.synchronize()
 assert torch.cuda.mem_get_info()[0]>(22 if MODE=='bf16' else 14)*1024**3
 with ExperimentLogger(team='integration',model='Task10I_Stage4A2_QwenPrecision_'+MODE,seed=42,job_type='evaluate',root=R/'.execution/experiments',config=dict(prompt_sha256=v['prompt_sha256'],manifest_sha256=v['manifest_sha256'],protocol=mf['protocol'],generation=ident['generation'])) as logger:
  v.update(logger_run_id=logger.run_id,stage='QWEN_LOADING',model_loads=1);wr('verification.json',v)
  with (R/'.execution/MODEL_LOAD_STARTED').open('x') as guard: guard.write('exactly one original-weight load attempt\n')
  q=BitsAndBytesConfig.from_dict(ident['quantization_config']) if ident['quantization_config'] else None
  kwargs=dict(local_files_only=True,device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa')
  if q is not None: kwargs['quantization_config']=q
  start=time.perf_counter(); model=Qwen3_5ForConditionalGeneration.from_pretrained(B/'model',**kwargs).eval();torch.cuda.synchronize()
  lat.update(load_ms=(time.perf_counter()-start)*1000,after_load=mem(),load_CPU_peak_sampled_bytes=max(rss),load_CPU_RSS_bytes=psutil.Process().memory_info().rss,CPU_peak_sampling_interval_ms=50)
  assert gpu(model)
  q4=[n for n,m in model.named_modules() if isinstance(m,bnb.nn.Linear4bit)];q8=[n for n,m in model.named_modules() if isinstance(m,bnb.nn.Linear8bitLt)]
  expected=ident['expected_precision'];assert q4==expected['linear4bit_names'] and q8==expected['linear8bit_names']
  assert all(p.dtype==torch.bfloat16 for p in model.model.visual.parameters())
  assert all(not isinstance(m,(bnb.nn.Linear4bit,bnb.nn.Linear8bitLt)) for m in model.model.visual.modules())
  assert not isinstance(model.lm_head,(bnb.nn.Linear4bit,bnb.nn.Linear8bitLt)) and model.lm_head.weight.dtype==torch.bfloat16
  if MODE=='bf16': assert all(p.dtype==torch.bfloat16 for p in model.parameters())
  audits=[]
  for n,m in model.named_modules():
   if isinstance(m,(torch.nn.Linear,bnb.nn.Linear4bit,bnb.nn.Linear8bitLt)):
    item=dict(name=n,type=type(m).__name__,weight_dtype=str(m.weight.dtype),device=str(m.weight.device),compute_dtype=str(getattr(m,'compute_dtype',None)),quant_type=getattr(m,'quant_type',None))
    if isinstance(m,bnb.nn.Linear4bit):
     assert m.weight.quant_state is not None and m.weight.quant_state.quant_type=='nf4' and m.compute_dtype==torch.bfloat16
     assert m.weight.quant_state.nested
    if isinstance(m,bnb.nn.Linear8bitLt): assert m.weight.dtype==torch.int8 and m.state.threshold==6.
    audits.append(item)
  lat['actual_precision']=dict(variant=MODE,linear4bit_layers=len(q4),linear8bit_layers=len(q8),vision_quantized_layers=0,vision_parameter_dtypes=sorted({str(p.dtype) for p in model.model.visual.parameters()}),all_parameter_buffers_cuda=True,parameter_dtypes=dict(collections.Counter(str(p.dtype) for p in model.parameters())),parameter_storage_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),hf_device_map=getattr(model,'hf_device_map',None),offload=False,default_lm_head_exclusion_preserved=True)
  wr('.execution/loaded_model_audit.json',dict(summary=lat['actual_precision'],linear_modules=audits));wr('grounding_latency_vram.json',lat)
  print('MODEL_LOADED',MODE,json.dumps(lat['actual_precision']),flush=True)
  processor=AutoProcessor.from_pretrained(B/'model',local_files_only=True,min_pixels=65536,max_pixels=65536)
  ev=[];ve=[];fstack=[];vstack=[]
  def pre(m,a,k): e=torch.cuda.Event(enable_timing=True);e.record();fstack.append(e)
  def post(m,a,k,o): e=torch.cuda.Event(enable_timing=True);e.record();ev.append((fstack.pop(),e))
  def vp(m,a): e=torch.cuda.Event(enable_timing=True);e.record();vstack.append(e)
  def vo(m,a,o): e=torch.cuda.Event(enable_timing=True);e.record();ve.append((vstack.pop(),e))
  hooks=[model.register_forward_pre_hook(pre,with_kwargs=True),model.register_forward_hook(post,with_kwargs=True),model.model.visual.register_forward_pre_hook(vp),model.model.visual.register_forward_hook(vo)]
  oid=id(model);prompt=(R/'grounding_prompt.txt').read_text()
  def request(c):
   assert sha(c['image_path'])==c['image_sha256']
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();ei=len(ev);vi=len(ve)
   image=Image.open(c['image_path']).convert('RGB');assert image.size==(128,128)
   content=[dict(type='image',image=image),dict(type='text',text=prompt)]
   inp=processor.apply_chat_template([dict(role='user',content=content)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt')
   prep=(time.perf_counter()-t)*1000;n=inp['input_ids'].shape[1];grid=inp['image_grid_thw'].tolist();inp=inp.to('cuda')
   with torch.inference_mode(): out=model.generate(**inp,max_new_tokens=128,do_sample=False,use_cache=True)
   torch.cuda.synchronize();ids=out[:,n:].cpu().tolist()[0];text=processor.batch_decode(out[:,n:],skip_special_tokens=True)[0];wall=(time.perf_counter()-t)*1000
   assert len(ve)-vi==1 and gpu(model) and id(model)==oid
   assert n==1292 and grid==[[1,16,16]]
   parsed=parse(text)
   # Frozen wrapper-aware parser; no partial response recovery.
   diagnostic=parsed
   timing=dict(id=c['id'],phase=c['phase'],cold_first_request=len(responses)==0,input_tokens=n,output_tokens=len(ids),preprocessing_ms=prep,end_to_end_ms=wall,image_grid_thw=grid,vision_cuda_ms=sum(a.elapsed_time(b) for a,b in ve[vi:]),first_forward_vision_prefill_cuda_ms=ev[ei][0].elapsed_time(ev[ei][1]),decode_forward_cuda_ms=sum(a.elapsed_time(b) for a,b in ev[ei+1:]),memory=mem())
   row=dict(id=c['id'],raw_text=text,output_token_ids=ids,parsed=parsed,diagnostic_only=diagnostic,input_image_sha256=c['image_sha256'],timing=timing)
   responses.append(row);lat['requests'].append(timing);save_raw();wr('grounding_latency_vram.json',lat)
   del inp,out
   return row
  for c in [c for c in mf['cases'] if c['phase']=='A' and c['input_status']=='AVAILABLE']:request(c)
  wr('.execution/phase_A_raw_responses.json',responses)
  v.update(phase_A_raw_sha256=sha(R/'.execution/phase_A_raw_responses.json'),phase_A_requests=len(responses));wr('verification.json',v)
  machinable=any(o['status']=='VALID_BOX' for r in responses for d in [r['parsed'],r['diagnostic_only']] if d for o in d['objects'].values())
  if machinable:
   for c in [c for c in mf['cases'] if c['phase']=='B' and c['input_status']=='AVAILABLE']:
    request(c)
    if (len(responses)-3)%12==0: print('QWEN_PROGRESS',len(responses)-3,'/96',flush=True)
  else: v['format_status']='GROUNDING_FORMAT_BLOCKED'
  preds=[]
  cmap={c['id']:c for c in mf['cases']}
  for r in responses:
   c=cmap[r['id']]
   for k,p in r['parsed']['objects'].items():
    preds.append(dict(id=r['id'],phase=c['phase'],parent_id=c['parent_id'],frame_kind=c['frame_kind'],timestep=c['timestep'],camera=c['camera'],object=k,json_valid=r['parsed']['json_valid'],strict_json_valid=r['parsed']['strict_json_valid'],wrapper=r['parsed']['wrapper'],status=p['status'],normalized=json.dumps(p['normalized']),pixel=json.dumps(p['pixel']),raw_text=r['raw_text']))
  csvwr('full_image_grounding_predictions.csv',preds)
  v.update(stage='ALL_GROUNDING_FROZEN',grounding_requests=len(responses),raw_response_SHA_before_GT=sha(R/'full_image_grounding_raw_responses.json'),predictions_SHA_before_GT=sha(R/'full_image_grounding_predictions.csv'),phase_B_requests=sum(cmap[r['id']]['phase']=='B' for r in responses));wr('verification.json',v)
  print('GROUNDING_FROZEN',len(responses),v['raw_response_SHA_before_GT'],flush=True)
  for h in hooks:h.remove()
  # Phase C never reads GT or DINO; the same Qwen remains resident, without another request.
  if machinable:
   from visual.runtime.tracking import SAM2Tracker
   import cv2
   ck=Path('/workspace/graph-recurrent-VLA/visual/checkpoints/sam2.1_hiera_small.pt')
   assert sha(ck)=='6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38'
   t=time.perf_counter();sam=SAM2Tracker(checkpoint=ck);torch.cuda.synchronize();assert gpu(sam.predictor) and gpu(model)
   lat['co_residency']=dict(SAM2_load_ms=(time.perf_counter()-t)*1000,loaded_memory=mem(),Qwen_and_SAM2_all_weights_cuda=True,offloading=False,other_models_loaded=False,sam2_checkpoint_sha256=sha(ck))
   predictions={r['id']:r for r in responses}
   for cid in mf['protocol']['sam2_ids']:
    c=cmap[cid];r=predictions.get(cid);masks={}
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
    valid={k:o for k,o in r['parsed']['objects'].items() if o['status']=='VALID_BOX'} if r else {}
    with tempfile.TemporaryDirectory(prefix='qwen_fullimage_sam_') as td:
     if valid:
      rgb=np.asarray(Image.open(c['image_path']).convert('RGB'));assert cv2.imwrite(str(Path(td)/'000000.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
      with torch.inference_mode():
       state=sam.predictor.init_state(video_path=td)
       for k,o in valid.items():
        objid=1 if k=='peg' else 2
        output=sam.predictor.add_new_points_or_box(state,frame_idx=0,obj_id=objid,box=np.asarray(o['pixel'],np.float32))
        mask=(output[2][list(output[1]).index(objid),0]>0).detach().cpu().numpy();assert mask.shape==(128,128)
        masks[k]=mask
       sam.predictor.reset_state(state);del state
    torch.cuda.synchronize();tim=dict(id=cid,wall_ms=(time.perf_counter()-t)*1000,memory=mem(),valid_prompts=len(valid),cold_first_SAM2_frame=not any(z['timing']['valid_prompts'] for z in samrows))
    maskpath=R/'.execution/masks'/f'{cid}.npz';maskpath.parent.mkdir(exist_ok=True);np.savez_compressed(maskpath,**masks)
    samrows.append(dict(id=cid,masks_path=str(maskpath),masks_sha256=sha(maskpath),objects={k:dict(status='INITIALIZED' if k in masks else 'NOT_INITIALIZED',predicted_box=r['parsed']['objects'][k] if r else None) for k in ['peg','box']},timing=tim,production_memory_commit=False,propagation=False))
    lat['sam2'].append(tim);wr('.execution/sam2_raw_initializations.json',samrows);wr('grounding_latency_vram.json',lat)
   v.update(sam2_camera_frames=len(samrows),sam2_box_prompts=sum(x['timing']['valid_prompts'] for x in samrows),sam2_raw_SHA_before_GT=sha(R/'.execution/sam2_raw_initializations.json'))
   print('SAM2_COMPLETE',len(samrows),v['sam2_box_prompts'],flush=True)
  v.update(stage='INFERENCE_COMPLETE',model_object_unchanged=id(model)==oid);wr('verification.json',v)
  logger.log(dict(Qwen_requests=len(responses),sam2_frames=len(samrows)));logger.record(status_detail='inference frozen; evaluation deferred to GT-only process')
except Exception as ex:
 v.update(stage='RUNTIME_STOPPED',completed_requests=len(responses),completed_SAM2_frames=len(samrows),error=repr(ex),traceback=traceback.format_exc(),error_type=type(ex).__name__);wr('verification.json',v);print(traceback.format_exc(),flush=True);raise
finally:
 stop.set();lat.update(CPU_process_peak_sampled_bytes=max(rss) if rss else None,CPU_ru_maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);wr('grounding_latency_vram.json',lat)
