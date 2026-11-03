from common import *
import time,traceback,subprocess,resource,threading
import numpy as np,torch,psutil
from PIL import Image
from visual.runtime.grounding import GDino
from visual.runtime import initialization as init
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json');mf=rd(R/'input_manifest.json');protocol=rd(R/'protocol.json');responses=[];frames=[];lat=dict(calls=[],images=[]);rss=[];stop=threading.Event()
def monitor():
 while not stop.is_set():rss.append(psutil.Process().memory_info().rss);stop.wait(.05)
threading.Thread(target=monitor,daemon=True).start()
def mem():return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
try:
 assert os.environ.get('CONTAINER_ID')=='58576054' and v['stage']=='INPUTS_FROZEN' and v['model_loads']==0
 assert sha(R/'protocol.json')==v['protocol_sha256'] and sha(R/'input_manifest.json')==v['input_manifest_sha256']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.init();torch.cuda.synchronize()
 with ExperimentLogger(team='integration',model='Task10I_DINO_native_resolution',seed=42,job_type='evaluate',root=R/'.execution/experiments',config=protocol) as logger:
  v.update(stage='DINO_LOADING',model_loads=1,logger_run_id=logger.run_id);wr('verification.json',v)
  with (R/'.execution/MODEL_LOAD_STARTED').open('x') as guard:guard.write('single existing DINO weight load; no retry\n')
  t=time.perf_counter();det=GDino(device='cuda:0',model_id=str(MODEL),threshold=.1);torch.cuda.synchronize()
  assert all(p.device.type=='cuda' and p.dtype==torch.float32 for p in det.model.parameters())
  lat.update(load_ms=(time.perf_counter()-t)*1000,loaded_memory=mem(),loaded_CPU_peak_sampled_bytes=max(rss),dtype='FP32',processor_config=det.proc.image_processor.to_dict(),actual_device_map=sorted({str(p.device) for p in det.model.parameters()}),offloading=False)
  wr('model_identity.json',dict(model=protocol['model'],revision=protocol['revision'],checkpoint_hashes=protocol['model_hashes'],runtime=v['runtime'],dtype='FP32',processor=lat['processor_config'],model_attention_implementation=getattr(det.model.config,'_attn_implementation',None),model_class=type(det.model).__name__,offloading=False))
  shapes=[]
  def hook(m,args,kwargs):shapes.append({k:list(x.shape) for k,x in kwargs.items() if hasattr(x,'shape')})
  handle=det.model.register_forward_pre_hook(hook,with_kwargs=True)
  pairs=sorted({c['pair_id'] for c in mf['cases']},key=lambda x:hashlib.sha256(('dino_native_order|'+x).encode()).hexdigest())
  cmap={c['id']:c for c in mf['cases']};schedule=[]
  for j,pair in enumerate(pairs):
   for size in ([128,512] if j%2==0 else [512,128]):schedule.append(cmap[pair+f'_native{size}'])
  wr('.execution/schedule.json',[c['id'] for c in schedule])
  for index,c in enumerate(schedule):
   assert sha(c['image_path'])==c['image_sha256'];torch.cuda.synchronize();start=time.perf_counter();rgb=np.asarray(Image.open(c['image_path']).convert('RGB'));assert rgb.shape==(c['height'],c['width'],3)
   outputs={}
   for obj,prompt in init.PROMPTS.items():
    torch.cuda.reset_peak_memory_stats();t=time.perf_counter();boxes,scores=det(rgb,prompt);torch.cuda.synchronize();wall=(time.perf_counter()-t)*1000
    assert len(shapes)==len(responses)+1 and shapes[-1]['pixel_values']==[1,3,800,800]
    assert len(boxes)==len(scores) and all(len(b)==4 and all(math.isfinite(x) for x in b) and b[0]<b[2] and b[1]<b[3] for b in boxes)
    assert all(scores[j]>=scores[j+1] for j in range(len(scores)-1))
    timing=dict(index=len(responses),id=c['id'],object=obj,native_size=c['native_size'],cold_first_call=not responses,end_to_end_ms=wall,**det.last_ms,memory=mem(),input_shapes=shapes[-1])
    outputs[prompt]=(boxes,scores)
    responses.append(dict(id=c['id'],pair_id=c['pair_id'],parent_id=c['parent_id'],native_size=c['native_size'],object=obj,prompt=prompt,boxes=boxes,scores=scores,timing=timing))
    lat['calls'].append(timing);wr('dino_raw_responses.json',dict(GT_opened=False,responses=responses))
   # Actual existing candidates and assignment, via a GT-free cached detector.
   def cached(image,prompt):
    assert image is rgb;return outputs[prompt]
   candidates={k:init.candidates(cached,rgb,p) for k,p in init.PROMPTS.items()};assigned=init.assign(candidates['peg'],candidates['box'])
   selected={k:dict(box=x[0],score=x[1]) if x is not None else None for k,x in zip(['peg','box'],assigned)}
   fr=dict(id=c['id'],pair_id=c['pair_id'],parent_id=c['parent_id'],native_size=c['native_size'],candidates=candidates,selected=selected,image_sha256=c['image_sha256'])
   frames.append(fr);torch.cuda.synchronize();imtime=dict(id=c['id'],native_size=c['native_size'],cold_first_image=index==0,end_to_end_ms=(time.perf_counter()-start)*1000);lat['images'].append(imtime)
   wr('dino_frame_selections.json',dict(GT_opened=False,frames=frames));wr('dino_latency_vram.json',lat)
   if (index+1)%8==0:print('DINO_PROGRESS',index+1,'/32 images;',len(responses),'/64 forwards',flush=True)
  handle.remove();assert len(responses)==64 and len(frames)==32
  csvwr('dino_predictions.csv',[dict(id=fr['id'],pair_id=fr['pair_id'],native_size=fr['native_size'],object=k,selected_box=x['box'] if x else None,selected_score=x['score'] if x else None) for fr in frames for k,x in fr['selected'].items()])
  v.update(stage='PREDICTIONS_FROZEN',completed_forwards=len(responses),completed_images=len(frames),raw_sha256_before_GT=sha(R/'dino_raw_responses.json'),selections_sha256_before_GT=sha(R/'dino_frame_selections.json'),predictions_sha256_before_GT=sha(R/'dino_predictions.csv'));wr('verification.json',v)
  logger.log(dict(forwards=len(responses),images=len(frames)));logger.link_output(R/'dino_predictions.csv')
  print('DINO_PREDICTIONS_FROZEN',64,flush=True)
except Exception as ex:
 v.update(stage='RUNTIME_STOPPED',completed_forwards=len(responses),completed_images=len(frames),error=repr(ex),traceback=traceback.format_exc());wr('verification.json',v);print(traceback.format_exc(),flush=True);raise
finally:
 stop.set();lat.update(CPU_peak_sampled_bytes=max(rss) if rss else None,CPU_ru_maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);wr('dino_latency_vram.json',lat)
