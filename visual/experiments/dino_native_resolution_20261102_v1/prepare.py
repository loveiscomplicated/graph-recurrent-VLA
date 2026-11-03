from common import *
import subprocess,inspect
import torch,transformers
from PIL import Image
from transformers import AutoProcessor
from visual.runtime import initialization as init
from visual.runtime.grounding import GDino,CAND_THR
assert os.environ.get('CONTAINER_ID')=='58576054'
assert not (R/'protocol.json').exists(),'Do not overwrite/restart'
R.mkdir(parents=True,exist_ok=True)
qv=rd(Q/'verification.json');assert qv['stage']=='COMPLETE_VERIFIED'
old=rd(Q/'input_manifest.json');assert len(old['cases'])==32 and len(old['parents'])==4
# The detector receives a manifest without GT paths, Actor IDs, privileged state or Qwen outputs.
cases=[{k:c[k] for k in ['id','pair_id','parent_id','frame_kind','timestep','camera','native_size','width','height','image_path','image_sha256','RGB_sha256']} for c in old['cases']]
assert all(sha(c['image_path'])==c['image_sha256'] for c in cases)
for pair in {c['pair_id'] for c in cases}:assert {c['native_size'] for c in cases if c['pair_id']==pair}=={128,512}
modelhash={p.name:sha(p) for p in MODEL.iterdir() if p.is_file()};provenance=rd(MODEL/'download_provenance.json')
sourcehash={str(p):sha(p) for p in [S/'visual/runtime/grounding.py',S/'visual/runtime/initialization.py',S/'utils/experiment_logger.py']}
assert init.PROMPTS=={'peg':'red and white block','box':'cardboard box'} and CAND_THR==.1
protocol=dict(task='10I Stage4A-2 DINO paired native-resolution pilot',instance_id='58576054',model='IDEA-Research/grounding-dino-tiny',revision=provenance.get('revision'),model_path=str(MODEL),model_hashes=modelhash,source_hashes=sourcehash,prompts=init.PROMPTS,box_threshold=CAND_THR,text_threshold=CAND_THR,consumer=dict(MAX_BOX_FRAC=init.MAX_BOX_FRAC,MISSING_SCORE=init.MISSING_SCORE,EXCLUSION_IOU=init.EXCLUSION_IOU,top_k=5),model_dtype='GDino default FP32, checked at load',processor_override=None,budget=dict(model_loads=1,detector_forwards=64,images=32,physical_parents=4,camera_frame_pairs=16,extra_warmups=0,retries=0,Qwen_loads=0,SAM2_loads=0,simulator_steps=0,simulator_renders=0,training=0,prompt_search=0),selection='Reuse all 32 native captures from previous frozen four-parent pilot; no addition, filtering or new rendering',source_input_manifest_sha256=sha(Q/'input_manifest.json'),GT_policy='Predictions and consumer selections frozen before GT evaluation. Historical GT/results already known; this is not blind independent generalization.',metrics=dict(visible_min_pixels=16,recall_IoU=.5,additional_IoU=[.3,.7],wrong_object='IoU<.5 and another Actor >=70% of foreground within selected rectangle, >=16 pixels'),primary='Existing per-frame joint selected boxes. Report raw top1 and any-proposal coverage separately; any-proposal success is an oracle diagnostic.',consumer_scope='Apply same candidates/assign independently at each saved frame, including t80. No temporal scans/tracking; not a claim that production initializes every t80.',coordinates='Original pixel target_sizes, no clipping or GT repair. Record finite/order/in-image validity separately; preserve existing DINO behavior on boxes beyond image.',statistics='Physical-parent macro and paired cluster bootstrap (4 parents, exploratory). No AP/rare-error/generalization claim.',precision_comparison=False,production_change=False,original_runner='PAUSED_SCOPE_REVIEW',prior_10J_optimized_prompts='Not used; this experiment changes only native resolution under the current production pair.')
wr('protocol.json',protocol);wr('input_manifest.json',dict(parents=[p['parent_id'] for p in old['parents']],cases=cases))
wr('.execution/evaluation_manifest.json',old)
processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);checks=[]
for size in [128,512]:
 c=next(c for c in cases if c['native_size']==size)
 for obj,prompt in init.PROMPTS.items():
  x=processor(images=Image.open(c['image_path']).convert('RGB'),text=prompt.lower().strip()+'.',return_tensors='pt');assert tuple(x['pixel_values'].shape)==(1,3,800,800)
  checks.append(dict(native_size=size,object=obj,input_shapes={k:list(t.shape) for k,t in x.items()}))
assert not torch.cuda.is_initialized()
protected=dict(qv['protected_after'])
# Protect the full preceding experiment and DINO snapshot, not just its summary.
for n,d in rd(Q/'artifact_index.json')['files'].items():protected[str(Q/n)]=d['sha256']
for n,h in modelhash.items():protected[str(MODEL/n)]=h
protected.update(sourcehash)
assert all(sha(p)==h for p,h in protected.items())
packages=subprocess.check_output([str(PYTHON),'-m','pip','list','--format=json'],text=True)
wr('verification.json',dict(stage='INPUTS_FROZEN',protected_before=protected,packages_before=packages,protocol_sha256=sha(R/'protocol.json'),input_manifest_sha256=sha(R/'input_manifest.json'),evaluation_manifest_sha256=sha(R/'.execution/evaluation_manifest.json'),processor_preflight=checks,model_loads=0,GT_opened=False,Qwen_predictions_opened=False,execution_sources_before={p.name:sha(p) for p in (R/'.execution').glob('*.py')},runtime=dict(torch=torch.__version__,transformers=transformers.__version__,python=sys.version)))
print('INPUTS_FROZEN',len(cases),'images; 64 forwards; prompts',init.PROMPTS,'revision',protocol['revision'],flush=True)
