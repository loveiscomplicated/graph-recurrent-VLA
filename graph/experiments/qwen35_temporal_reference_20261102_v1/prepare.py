from common import *
import subprocess
import torch
from PIL import Image
from transformers import AutoProcessor

assert os.environ.get('CONTAINER_ID')=='58576054'
assert not (R/'protocol.json').exists()
R.mkdir(parents=True,exist_ok=True)
prior=rd(V/'verification.json');assert prior['stage']=='COMPLETE_VERIFIED'
cm=rd(V/'.execution/candidate_manifest.json');images={x['id']:x for x in cm['images']}
candidates=[c for c in cm['candidates'] if c['frame_kind']=='F1']
assert len(candidates)==149 and len({c['image_id'] for c in candidates})==8
oldrequests={x['id']:x for x in rd(V/'request_manifest.json')['requests']}
original=rd(V/'prompt_identity.json')
note="If a fourth image is present, it is the initial RGB frame from the same scene and camera, provided only as earlier visual context. Always classify the candidate marked in image 2 of the current scene, not an object in the initial frame."
preamble=original['preamble']+'\n'+note
(R/'semantic_role_prompt.txt').write_text(original['prompt'])
(R/'shared_input_preamble.txt').write_text(preamble)
assert sha(R/'semantic_role_prompt.txt')==original['prompt_sha256']
pid=dict(prompt=original['prompt'],preamble=preamble,semantic_prompt_sha256=sha(R/'semantic_role_prompt.txt'),shared_preamble_sha256=sha(R/'shared_input_preamble.txt'),temporal_note=note,labels=LABELS,text_identical_between_conditions=True,parser='strip then exact uppercase label; no recovery')
wr('prompt_identity.json',pid)
requests=[];anchors={}
for cand in candidates:
 c=images[cand['image_id']]
 anchor=next(x for x in images.values() if x['parent_id']==c['parent_id'] and x['camera']==c['camera'] and x['frame_kind']=='F0')
 assert anchor['timestep']==0 and c['timestep']==80 and anchor['native_size']==c['native_size']==512
 assert sha(anchor['image_path'])==anchor['image_sha256']
 anchors[c['id']]=dict(current_image_id=c['id'],initial_image_id=anchor['id'],parent_id=c['parent_id'],camera=c['camera'],current_timestep=c['timestep'],initial_timestep=anchor['timestep'],initial_RGB_path=anchor['image_path'],initial_SHA256=anchor['image_sha256'])
 old=oldrequests[cand['id']]
 assert all(sha(p)==h for p,h in zip(old['images'],old['image_sha256']))
 for condition in ['current_only','with_initial_RGB']:
  paths=old['images'][:] + ([anchor['image_path']] if condition=='with_initial_RGB' else [])
  hashes=old['image_sha256'][:] + ([anchor['image_sha256']] if condition=='with_initial_RGB' else [])
  requests.append(dict(id=cand['id']+'__'+condition,candidate_id=cand['id'],condition=condition,images=paths,image_sha256=hashes))
requests.sort(key=lambda r:hashlib.sha256(('temporal_reference_v1|'+r['id']).encode()).hexdigest())
wr('request_manifest.json',dict(requests=requests))
wr('input_manifest.json',dict(parents=cm['parents'],camera_frames=list(anchors.values()),candidate_count=len(candidates),conditions=['current_only','with_initial_RGB']))
wr('.execution/candidate_manifest.json',dict(candidates=candidates,images=[x for x in cm['images'] if x['frame_kind']=='F1'],parents=cm['parents']))
protocol=dict(task='Task10I Stage4A-2 paired initial-RGB reference diagnosis',selection='All preserved F1 proposals, no GT or prediction-based filtering',parents=cm['parents'],camera_frames=8,candidates=149,conditions=['current_only','with_initial_RGB'],contrast='Only a fourth unmarked initial RGB image differs; all text, first three current images and candidate selection rules identical',historical_comparison='A shared optional-fourth-image note is added to BOTH arms; current_only vs original frozen verifier is a secondary reproduction check, not the causal contrast',reference='Same parent, same camera, t0, native512; no prior labels, marks, masks, coordinates, privileged states or object correspondence supplied',model='Qwen/Qwen3.5-9B',revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',precision='full BF16',attention='sdpa',device='cuda:0',offloading=False,processor=dict(min_pixels=512**2,max_pixels=512**2),generation=dict(do_sample=False,enable_thinking=False,max_new_tokens=16,use_cache=True,batch_size=1),semantic_prompt_sha256=pid['semantic_prompt_sha256'],shared_preamble_sha256=pid['shared_preamble_sha256'],selection_policy='Per original query pool, strict label must equal role, then highest original DINO score; rank tie break; none => null; no coordinate change, area/top5/null-score/joint filtering',primary_metrics=['Paired PEG good-box retention','Paired BOX good-box retention','PEG and BOX selected IoU>=0.5 recall','Regression candidates 0275/0305 F1 wrist rank0 labels','Wrong-object approvals and nulls'],statistics=dict(unit='parent',clusters=4,bootstrap_replicates=10000,seed=42,exploratory=True),GT_barrier='Freeze all 298 raw predictions before GT; known development scenes, not blind generalization',budget=dict(model_loads=1,Qwen_requests=298,DINO_forwards=0,SAM2_forwards=0,simulator_calls=0,retries=0,extra_warmups=0,prompt_sweeps=0,training=0),original_runner='PAUSED_SCOPE_REVIEW',production=False)
wr('protocol.json',protocol)
ident=rd(V/'model_identity.json');ident.update(experiment=protocol['task'],generation=protocol['generation'],processor=protocol['processor']);wr('model_identity.json',ident)
protected=dict(prior['protected_after'])
for rel,entry in rd(V/'artifact_index.json')['files'].items():protected[str(V/rel)]=entry['sha256']
protected[str(V/'artifact_index.json')]=sha(V/'artifact_index.json')
assert all(sha(p)==h for p,h in protected.items())
assert all(sha(B/'model'/p)==h for p,h in ident['checkpoint_hashes'].items())
processor=AutoProcessor.from_pretrained(B/'model',local_files_only=True,min_pixels=512**2,max_pixels=512**2)
checks=[]
for cand in candidates[:2]:
 tensors=[]
 for condition in protocol['conditions']:
  req=next(x for x in requests if x['candidate_id']==cand['id'] and x['condition']==condition)
  content=[dict(type='text',text=preamble)]+[dict(type='image',image=Image.open(p).convert('RGB')) for p in req['images']]+[dict(type='text',text=pid['prompt'])]
  x=processor.apply_chat_template([dict(role='user',content=content)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt')
  assert len(x['image_grid_thw'])==len(req['images']);tensors.append(x)
  checks.append(dict(id=req['id'],input_tokens=x['input_ids'].shape[1],grid=x['image_grid_thw'].tolist()))
 a,b=tensors;assert torch.equal(a['image_grid_thw'],b['image_grid_thw'][:3]);assert torch.equal(a['pixel_values'],b['pixel_values'][:len(a['pixel_values'])])
assert not torch.cuda.is_initialized()
wr('verification.json',dict(stage='INPUTS_FROZEN',model_loads=0,completed_requests=0,GT_opened=False,protected_before=protected,packages_before=subprocess.check_output([str(B/'env/bin/python'),'-m','pip','list','--format=json'],text=True),request_manifest_sha256=sha(R/'request_manifest.json'),candidate_manifest_sha256=sha(R/'.execution/candidate_manifest.json'),prompt_identity_sha256=sha(R/'prompt_identity.json'),protocol_sha256=sha(R/'protocol.json'),preflight=checks,source_before={p.name:sha(p) for p in (R/'.execution').glob('*.py')}))
print('TEMPORAL_INPUTS_FROZEN 149 candidates x 2 conditions = 298 requests',flush=True)
