from common import *
import time,traceback,threading,subprocess,resource,tempfile,collections
import numpy as np,torch,psutil
from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration,AutoProcessor
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json');mf=rd(R/'input_manifest.json');protocol=rd(R/'protocol.json')
lat=dict(requests=[],sam2=[]);responses=[];samrows=[];stop=threading.Event();rss=[]
def monitor():
 while not stop.is_set():rss.append(psutil.Process().memory_info().rss);stop.wait(.05)
threading.Thread(target=monitor,daemon=True).start()
def mem():
 free,total=torch.cuda.mem_get_info()
 return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),device_free=free,total=total)
def gpu(m):return all(p.device.type=='cuda' for p in m.parameters()) and all(p.device.type=='cuda' for p in m.buffers())
def save():
 wr('full_image_grounding_raw_responses.json',dict(GT_metrics_opened=False,DINO_outputs_opened=False,responses=responses));wr('grounding_latency_vram.json',lat)
try:
 assert v['stage']=='INPUTS_FROZEN' and v['model_loads']==0
 assert sha(R/'input_manifest.json')==v['manifest_sha256'] and sha(R/'protocol.json')==v['protocol_sha256']
 for p in v['prompt_identities'].values():assert sha(p['path'])==p['sha256']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.init();torch.cuda.synchronize();assert torch.cuda.mem_get_info()[0]>22*1024**3
 with ExperimentLogger(team='integration',model='Task10I_Qwen35_native_resolution_prompt_2x2',seed=42,job_type='evaluate',root=R/'.execution/experiments',config=protocol) as logger:
  v.update(stage='QWEN_LOADING',model_loads=1,logger_run_id=logger.run_id);wr('verification.json',v)
  with (R/'.execution/MODEL_LOAD_STARTED').open('x') as g:g.write('one model load, no retries\n')
  start=time.perf_counter();model=Qwen3_5ForConditionalGeneration.from_pretrained(B/'model',local_files_only=True,device_map={'':0},dtype=torch.bfloat16,attn_implementation='sdpa').eval();torch.cuda.synchronize()
  assert gpu(model) and all(p.dtype==torch.bfloat16 for p in model.parameters())
  lat.update(load_ms=(time.perf_counter()-start)*1000,after_load=mem(),load_CPU_peak_sampled_bytes=max(rss),actual_precision=dict(all_parameters_BF16=True,all_parameter_buffers_cuda=True,quantized_layers=0,offloading=False,hf_device_map=getattr(model,'hf_device_map',None)))
  processor=AutoProcessor.from_pretrained(B/'model',local_files_only=True,min_pixels=512**2,max_pixels=512**2)
  ev=[];ve=[];fs=[];vs=[]
  def pre(m,a,k):e=torch.cuda.Event(enable_timing=True);e.record();fs.append(e)
  def post(m,a,k,o):e=torch.cuda.Event(enable_timing=True);e.record();ev.append((fs.pop(),e))
  def vp(m,a):e=torch.cuda.Event(enable_timing=True);e.record();vs.append(e)
  def vo(m,a,o):e=torch.cuda.Event(enable_timing=True);e.record();ve.append((vs.pop(),e))
  hooks=[model.register_forward_pre_hook(pre,with_kwargs=True),model.register_forward_hook(post,with_kwargs=True),model.model.visual.register_forward_pre_hook(vp),model.model.visual.register_forward_hook(vo)]
  prompts={k:Path(p['path']).read_text() for k,p in v['prompt_identities'].items()};oid=id(model)
  cases={c['id']:c for c in mf['cases']};pairs=sorted({c['pair_id'] for c in cases.values()},key=lambda x:hashlib.sha256(('order|'+x).encode()).hexdigest())
  schedule=[]
  for j,pair in enumerate(pairs):
   # Fixed Latin rotation: each condition appears in each order position four times.
   for mode in VARIANTS[j%4:]+VARIANTS[:j%4]:
    size=int(mode.split('_')[0].replace('native',''));c=cases[pair+f'_native{size}'];schedule.append((mode,c))
  wr('.execution/request_schedule.json',[dict(index=i,variant=m,id=c['id']) for i,(m,c) in enumerate(schedule)])
  for index,(mode,c) in enumerate(schedule):
   assert sha(c['image_path'])==c['image_sha256'];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();ei=len(ev);vi=len(ve)
   image=Image.open(c['image_path']).convert('RGB');assert image.size==(c['width'],c['height'])
   prompt=prompts[mode.split('_')[1]];messages=[dict(role='user',content=[dict(type='image',image=image),dict(type='text',text=prompt)])]
   inp=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt')
   prep=(time.perf_counter()-t)*1000;n=inp['input_ids'].shape[1];grid=inp['image_grid_thw'].tolist();assert grid==[[1,32,32]]
   inp=inp.to('cuda')
   with torch.inference_mode():out=model.generate(**inp,max_new_tokens=128,do_sample=False,use_cache=True)
   torch.cuda.synchronize();ids=out[:,n:].cpu().tolist()[0];text=processor.batch_decode(out[:,n:],skip_special_tokens=True)[0];wall=(time.perf_counter()-t)*1000
   assert len(ve)-vi==1 and gpu(model) and id(model)==oid
   timing=dict(index=index,id=c['id'],variant=mode,cold_first_request=index==0,input_tokens=n,output_tokens=len(ids),preprocessing_ms=prep,end_to_end_ms=wall,image_grid_thw=grid,vision_tokens=256,vision_cuda_ms=sum(a.elapsed_time(b) for a,b in ve[vi:]),first_forward_vision_prefill_cuda_ms=ev[ei][0].elapsed_time(ev[ei][1]),decode_forward_cuda_ms=sum(a.elapsed_time(b) for a,b in ev[ei+1:]),memory=mem())
   responses.append(dict(id=c['id'],pair_id=c['pair_id'],variant=mode,raw_text=text,output_token_ids=ids,parsed=parse(text,c['width'],c['height']),input_image_sha256=c['image_sha256'],prompt_sha256=v['prompt_identities'][mode.split('_')[1]]['sha256'],timing=timing))
   lat['requests'].append(timing);save();del inp,out
   if (index+1)%8==0:print('QWEN_PROGRESS',index+1,'/64',flush=True)
  csvrows=[]
  for r in responses:
   c=cases[r['id']]
   for k,o in r['parsed']['objects'].items():csvrows.append(dict(variant=r['variant'],id=r['id'],pair_id=r['pair_id'],parent_id=c['parent_id'],camera=c['camera'],frame_kind=c['frame_kind'],timestep=c['timestep'],native_size=c['native_size'],object=k,json_valid=r['parsed']['json_valid'],strict_json_valid=r['parsed']['strict_json_valid'],status=o['status'],normalized=json.dumps(o['normalized']),pixel=json.dumps(o['pixel'])))
  csvwr('full_image_grounding_predictions.csv',csvrows)
  v.update(stage='ALL_GROUNDING_FROZEN',grounding_requests=len(responses),raw_SHA_before_GT=sha(R/'full_image_grounding_raw_responses.json'),predictions_SHA_before_GT=sha(R/'full_image_grounding_predictions.csv'));wr('verification.json',v)
  print('ALL_GROUNDING_FROZEN',len(responses),flush=True)
  for h in hooks:h.remove()
  # Same resident Qwen; SAM2 receives only valid Qwen boxes, never GT.
  from visual.runtime.tracking import SAM2Tracker
  import cv2
  checkpoint=Path('/workspace/graph-recurrent-VLA/visual/checkpoints/sam2.1_hiera_small.pt');assert sha(checkpoint)=='6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38'
  t=time.perf_counter();sam=SAM2Tracker(checkpoint=checkpoint);torch.cuda.synchronize();assert gpu(sam.predictor) and gpu(model)
  lat['co_residency']=dict(SAM2_load_ms=(time.perf_counter()-t)*1000,loaded_memory=mem(),Qwen_and_SAM2_all_weights_cuda=True,offloading=False,SAM2_checkpoint_sha256=sha(checkpoint))
  for index,r in enumerate(responses):
   c=cases[r['id']];masks={};torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
   valid={k:o for k,o in r['parsed']['objects'].items() if o['status']=='VALID_BOX'}
   with tempfile.TemporaryDirectory(prefix='qwen_native_sam_') as td:
    if valid:
     rgb=np.asarray(Image.open(c['image_path']).convert('RGB'));assert cv2.imwrite(str(Path(td)/'000000.jpg'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
     with torch.inference_mode():
      state=sam.predictor.init_state(video_path=td)
      for k,o in valid.items():
       objid=1 if k=='peg' else 2
       output=sam.predictor.add_new_points_or_box(state,frame_idx=0,obj_id=objid,box=np.asarray(o['pixel'],np.float32))
       mask=(output[2][list(output[1]).index(objid),0]>0).detach().cpu().numpy();assert mask.shape==(c['height'],c['width']);masks[k]=mask
      sam.predictor.reset_state(state);del state
   torch.cuda.synchronize();tim=dict(id=c['id'],variant=r['variant'],wall_ms=(time.perf_counter()-t)*1000,memory=mem(),valid_prompts=len(valid),cold_first_SAM2_frame=not any(x['timing']['valid_prompts'] for x in samrows))
   path=R/'.execution/masks'/f'{r["variant"]}_{r["pair_id"]}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**masks)
   samrows.append(dict(id=c['id'],pair_id=r['pair_id'],variant=r['variant'],masks_path=str(path),masks_sha256=sha(path),timing=tim,production_memory_commit=False,propagation=False))
   lat['sam2'].append(tim);wr('.execution/sam2_raw_initializations.json',samrows);wr('grounding_latency_vram.json',lat)
   if (index+1)%16==0:print('SAM2_PROGRESS',index+1,'/64',flush=True)
  v.update(stage='INFERENCE_COMPLETE',sam2_camera_frames=len(samrows),sam2_box_prompts=sum(x['timing']['valid_prompts'] for x in samrows),sam2_raw_SHA_before_GT=sha(R/'.execution/sam2_raw_initializations.json'),model_object_unchanged=id(model)==oid);wr('verification.json',v)
  wr('all_predictions_frozen.json',dict(raw_sha256=v['raw_SHA_before_GT'],predictions_sha256=v['predictions_SHA_before_GT'],SAM2_raw_sha256=v['sam2_raw_SHA_before_GT'],requests=len(responses),GT_metrics_opened=False,DINO_opened=False))
  logger.log(dict(Qwen_requests=len(responses),SAM2_frames=len(samrows),SAM2_box_prompts=v['sam2_box_prompts']));logger.link_output(R/'full_image_grounding_predictions.csv')
except Exception as ex:
 v.update(stage='RUNTIME_STOPPED',completed_requests=len(responses),completed_SAM2_frames=len(samrows),error=repr(ex),traceback=traceback.format_exc());wr('verification.json',v);print(traceback.format_exc(),flush=True);raise
finally:
 stop.set();lat.update(CPU_process_peak_sampled_bytes=max(rss) if rss else None,CPU_ru_maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);wr('grounding_latency_vram.json',lat)
