from common import *
import subprocess, h5py, numpy as np, shutil
from PIL import Image
assert not (R/'input_manifest.json').exists()
(R/'.execution/images').mkdir(parents=True,exist_ok=True)
prompt=(R/'.execution/grounding_prompt.txt').read_text().rstrip('\n')
(R/'grounding_prompt.txt').write_text(prompt)
wr('grounding_prompt_identity.json',dict(sha256=sha(R/'grounding_prompt.txt'),encoding='UTF-8',trailing_newline=False,prompt=prompt,frozen_before_inference=True,additional_instruction_text=False))
run=Path('/workspace/task10i_stage3_20261031_v5_finalization/run')
ds=[rd(run/f'dataset_{x}_train_manifest.json') for x in 'RD']; val=set(rd(run/'shared_validation_manifest.json')['parents'])
eps={}
for d in ds:
 for e in d['episodes']:
  if e['branch']=='full_task':
   p=str(e['parent_id'])
   if p in eps: assert eps[p]['raw_sha256']==e['raw_sha256']
   eps[p]=e
excluded={'10i_s1_0355','10i_s1_0645'}
seed='Task10I-QwenFullImageGrounding-20261102-v1'
eligible=sorted(set(eps)-excluded-val,key=lambda p:hashlib.sha256((seed+'|'+p).encode()).hexdigest())
assert len(eligible)>=24
parents=[dict(eps[p],selection_hash=hashlib.sha256((seed+'|'+p).encode()).hexdigest()) for p in eligible[:24]]
cases=[]
def extract(e,t,camera,phase,kind,role=None):
 cid=role or f"{e['parent_id']}_{kind}_{camera}"
 row=dict(id=cid,parent_id=e['parent_id'],episode=str(e['episode_id']),raw_path=e['raw_path'],raw_sha256=e['raw_sha256'],phase=phase,frame_kind=kind,timestep=t,camera=camera)
 with h5py.File(e['raw_path']) as h:
  if camera+'/rgb' not in h or t>=len(h[camera+'/rgb']): row['input_status']='MISSING_FRAME'
  else:
   rgb=h[camera+'/rgb'][t]; assert rgb.shape==(128,128,3) and rgb.dtype==np.uint8
   path=R/'.execution/images'/f'{cid}.png'; Image.fromarray(rgb).save(path)
   row.update(input_status='AVAILABLE',image_path=str(path),image_sha256=sha(path),RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),width=128,height=128)
 cases.append(row)
for p,t,role in [('10i_s1_0355',0,'P'),('10i_s1_0355',30,'N'),('10i_s1_0645',0,'C1')]:
 e=eps.get(p) or dict(parent_id=p,episode_id=f'10i_stage3_{p}_full_task',raw_path=f'/workspace/task10i_stage3_20261031_v3/run/parents/{p}/full_task/episode.h5')
 e['raw_sha256']=sha(e['raw_path']);extract(e,t,'base_camera','A','F0' if t==0 else 'F1',role)
for e in parents:
 assert sha(e['raw_path'])==e['raw_sha256']
 with h5py.File(e['raw_path']) as h:
  ts=h['timestep'][:].reshape(-1).tolist(); assert ts==list(range(len(ts))), 'noncontiguous stored frame IDs require explicit mapping'
  f1=max((t for t in ts if t<=80),default=None)
 e['F1_actual']=f1; e['F1_reason']='t80 available' if f1==80 else 'last stored timestep <=80' if f1 is not None else 'no available timestep <=80'
 for kind,t in [('F0',0),('F1',f1)]:
  for cam in ['base_camera','hand_camera']:
   if t is not None: extract(e,t,cam,'B',kind)
samids=[]
for cam in ['base_camera','hand_camera']:
 for kind in ['F0','F1']:
  ordered=sorted([x for x in cases if x['phase']=='B' and x['camera']==cam and x['frame_kind']==kind],key=lambda x:hashlib.sha256((seed+'|SAM2|'+x['id']).encode()).hexdigest())
  samids.extend(x['id'] for x in ordered[:4])
protocol=dict(selection_seed=seed,eligible_ids=eligible,excluded_prompt_development_parents=sorted(excluded),exclude_validation=True,selection_uses_GT_or_DINO=False,selection_basis='union of Stage3 R/D train physical parents; full_task branch; no GT/reference availability filter',F1_rule='t80 else last stored timestep <=80, never replaced after inference',sam2_ids=samids,min_visible_pixels=16,spatial_success='valid box and target GT visible bbox IoU>=0.5; threshold sweeps .3/.5/.7 are diagnostics',semantic_wrong='predicted box overlaps a non-target task actor bbox IoU>=0.5 while target IoU<0.5; actor pixel composition also retained',sam2_success='mask IoU>=0.8; null/invalid/unrun fails visible-object end-to-end success',no_new_simulator=True,bootstrap_seed=42,bootstrap_replicates=10000)
wr('input_manifest.json',dict(parents=parents,cases=cases,protocol=protocol,source_manifests={str(run/f'dataset_{x}_train_manifest.json'):sha(run/f'dataset_{x}_train_manifest.json') for x in 'RD'}))
old=rd(B/'model_identity.json'); ck={n:sha(B/'model'/n) for n in old['checkpoint_hashes']}; assert ck==old['checkpoint_hashes']
sources={p:sha(p) for p in old['sources']}; assert sources==old['sources']
old.update(checkpoint_hashes=ck,sources=sources,generation=dict(max_new_tokens=128,do_sample=False,use_cache=True,enable_thinking=False,batch_size=1),processor_resolution=dict(min_pixels=65536,max_pixels=65536),input_contract='one complete original RGB image + exactly frozen prompt only',CPU_disk_offload=False)
# Construct the current identity explicitly: do not copy stale verifier inputs.
old = dict(model=old['model'], revision=old['revision'], checkpoint_hashes=ck,
 sources=sources, runtime=old['runtime'], multimodal_class='Qwen3_5ForConditionalGeneration',
 quantization_config=None, precision='original BF16; no quantization',
 load_arguments=dict(device_map={'':0}, dtype='bfloat16', attn_implementation='sdpa', local_files_only=True),
 generation=dict(max_new_tokens=128, do_sample=False, use_cache=True, enable_thinking=False, batch_size=1),
 processor_resolution=dict(min_pixels=65536, max_pixels=65536),
 input_contract='one original full RGB + frozen task prompt only; actual input snapshot stored during inference',
 CPU_disk_offload=False, comparison='same frozen prompt, parents, images, order and SAM2 subset as previous NF4; matched precision comparison, not a fresh blind test')
wr('model_identity.json',old)
protected=rd(B/'verification.json')['tracked_inputs_before']
# Do not open preserved DINO output before grounding freeze, even for hashing.
deferred={p:h for p,h in protected.items() if any(z in p for z in ['/dino_raw/','/dino_events.json'])}
checked={p:sha(p) for p in protected if p not in deferred}; assert all(checked[p]==protected[p] for p in checked)
for e in parents:
 checked[e['raw_path']]=sha(e['raw_path'])
 if Path(e['gt_path']).is_file(): checked[e['gt_path']]=sha(e['gt_path'])
for p,h in sources.items(): checked[p]=h
packages={x:subprocess.check_output([x,'-m','pip','list','--format=json'],text=True) for x in [str(B/'env/bin/python'),'/workspace/venvs/task10i-stage4a-cu130/bin/python']}
wr('verification.json',dict(stage='INPUTS_FROZEN',stage4a2='PAUSED_SCOPE_REVIEW',protected_before=checked,deferred_DINO_hashes_from_previous_verified_receipt=deferred,model_hashes_before=ck,packages_before=packages,GT_semantic_opened=False,DINO_outputs_opened=False,manifest_sha256=sha(R/'input_manifest.json'),prompt_sha256=sha(R/'grounding_prompt.txt'),identity_sha256=sha(R/'model_identity.json'),model_loads=0,preflight_GPU=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv'],text=True)))
assert sha(R/'grounding_prompt.txt')=='66a8621c89784e51b20be107ad57af726cdf275c34acc04947057cb1ea26c62d'
previous=Path('/workspace/task10i_stage4a2_qwen35_full_image_grounding_20261102_v1')
prior=rd(previous/'input_manifest.json');current=rd(R/'input_manifest.json')
assert current['protocol']==prior['protocol'] and current['parents']==prior['parents']
for a,b in zip(current['cases'],prior['cases']):
 assert {k:v for k,v in a.items() if k!='image_path'}=={k:v for k,v in b.items() if k!='image_path'}
verification=rd(R/'verification.json')
verification.update(original_NF4_artifact_index_SHA_before=sha(previous/'artifact_index.json'),
 source_files_before={str(p):sha(p) for p in (R/'.execution').glob('*.py')},
 precision='BF16',instance_id=58576054,matched_NF4_input_identity_pass=True)
wr('verification.json',verification)
print('FROZEN',len(eligible),'eligible',len(parents),'parents',len(cases),'images',len(samids),'SAM frames',flush=True)
print('PARENTS',[p['parent_id'] for p in parents],flush=True)
