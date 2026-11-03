from common import *
import subprocess,copy
import numpy as np,torch
from PIL import Image,ImageDraw
from transformers import AutoProcessor
assert os.environ.get('CONTAINER_ID')=='58576054'
assert not (R/'protocol.json').exists();R.mkdir(parents=True,exist_ok=True)
dv=rd(D/'verification.json');assert dv['stage']=='COMPLETE_VERIFIED'
raw=[r for r in rd(D/'dino_raw_responses.json')['responses'] if r['native_size']==512];mf=rd(D/'input_manifest.json');images={c['id']:c for c in mf['cases'] if c['native_size']==512}
assert len(raw)==32 and len(images)==16 and sum(len(r['boxes']) for r in raw)==251
prior=rd(F/'prompt_identity.json');prompt=prior['prompt'];assert hashlib.sha256(prompt.encode()).hexdigest()=='691ea9e22ec710dad77084db6aa812455d525190f7d6f589821bfaf082889a06'
(R/'verifier_prompt.txt').write_text(prompt)
# Same neutral preamble and three-image order as the successful historical verifier.
preamble='Images in order: full scene, scene with the candidate region marked, candidate crop.'
wr('prompt_identity.json',dict(prompt=prompt,prompt_sha256=sha(R/'verifier_prompt.txt'),preamble=preamble,labels=LABELS,parser='raw.strip() must equal exactly one uppercase allowed label; no recovery',previous_prompt_identity_sha256=sha(F/'prompt_identity.json')))
candidates=[];requests=[]
for r in raw:
 c=images[r['id']];assert sha(c['image_path'])==c['image_sha256'];original=Image.open(c['image_path']).convert('RGB');assert original.size==(512,512)
 for rank,(box,score) in enumerate(zip(r['boxes'],r['scores'])):
  cid=c['id']+'_'+r['object']+f'_rank{rank:03d}';x1,y1,x2,y2=box;padx=(x2-x1)*.1;pady=(y2-y1)*.1
  cropbox=[max(0,int(math.floor(x1-padx))),max(0,int(math.floor(y1-pady))),min(512,int(math.ceil(x2+padx))),min(512,int(math.ceil(y2+pady)))]
  assert cropbox[2]>cropbox[0] and cropbox[3]>cropbox[1],cid
  marked=original.copy();ImageDraw.Draw(marked).rectangle(box,outline=(0,180,220),width=4)
  folder=R/'candidate_images';folder.mkdir(exist_ok=True)
  markedpath=folder/(cid+'_marked.png');croppath=folder/(cid+'_crop.png');marked.save(markedpath);original.crop(cropbox).save(croppath)
  # Request manifest excludes proposal query, score, rank, coordinates and all GT.
  req=dict(id=cid,images=[str(c['image_path']),str(markedpath),str(croppath)],image_sha256=[sha(c['image_path']),sha(markedpath),sha(croppath)])
  requests.append(req);candidates.append(dict(id=cid,image_id=c['id'],pair_id=c['pair_id'],parent_id=c['parent_id'],camera=c['camera'],frame_kind=c['frame_kind'],timestep=c['timestep'],query_object=r['object'],query_prompt=r['prompt'],rank=rank,box=box,DINO_score=score,crop_box=cropbox))
assert len(requests)==251
protocol=dict(task='10I Stage4A-2 DINO→Qwen candidate verifier pilot',images=16,parents=4,native_size=512,model='Qwen/Qwen3.5-9B',revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',precision='full BF16',device='cuda:0',offloading=False,attention='sdpa',processor=dict(min_pixels=512**2,max_pixels=512**2),generation=dict(do_sample=False,enable_thinking=False,max_new_tokens=16,use_cache=True,batch_size=1),prompt_sha256=sha(R/'verifier_prompt.txt'),candidate_pool='All 251 preserved DINO proposals after original box/text threshold0.1; no GT, area/top5/0.3 filtering or candidate sampling; no new DINO forwards',input='Full native RGB + neutral cyan 4px marked copy + 10% margin crop, original order; no DINO query/label/score/rank/coordinates or GT in messages',baseline_legacy='Saved original area25%/top5/null0.3/joint0.3 selection',primary='Within each original DINO query pool, keep candidates whose strict Qwen label equals requested role; choose highest original DINO score, rank tie-break. No candidates remain => null. This changes raw Top1 only by semantic filtering.',secondary=dict(legacy_gate='Keep a saved legacy-selected box only if verifier agrees; no replacement',union_role='Across both query pools select highest original DINO score among strict PEG or BOX labels; diagnostic alternative, no GT relabeling',raw_top1='Highest original DINO score in each query, with no area/top5/null0.3 constraints'),selection_notes='Independent per-object selection, no new overlap or score thresholds. Legacy comparison changes pool/rules as well as verifier; raw Top1 is the matched-pool control. Qwen does not adjust any coordinate. Potential overlap/collision recorded, no automatic repair.',budget=dict(model_loads=1,Qwen_requests=251,DINO_forwards=0,SAM2_forwards=0,extra_warmups=0,retries=0,prompt_variants=0,simulator_resets=0,renders=0,steps=0,training=0),GT_barrier='All 251 raw labels SHA-frozen before GT evaluation; history is known, not blind independent generalization',candidate_GT_role='At least16 foreground pixels and one of peg/box/robot occupies>=70% of foreground => that role; foreground<16 => OTHER; remaining mixed regions AMBIGUOUS. Report clear-role classification separately from Box IoU.',metrics=dict(min_visible_pixels=16,Box_IoU=.5,bootstrap_parent_clusters=4),production=False,original_runner='PAUSED_SCOPE_REVIEW')
wr('protocol.json',protocol);wr('request_manifest.json',dict(requests=requests));wr('.execution/candidate_manifest.json',dict(candidates=candidates,images=list(images.values()),parents=mf['parents']))
# Fixed ordering spreads scene/query ranks; IDs are orchestration only, never in messages.
requests.sort(key=lambda x:hashlib.sha256(('verifier_order|'+x['id']).encode()).hexdigest());wr('request_manifest.json',dict(requests=requests))
processor=AutoProcessor.from_pretrained(B/'model',local_files_only=True,min_pixels=512**2,max_pixels=512**2)
checks=[]
for req in requests[:2]:
 content=[dict(type='text',text=preamble)]+[dict(type='image',image=Image.open(p).convert('RGB')) for p in req['images']]+[dict(type='text',text=prompt)]
 x=processor.apply_chat_template([dict(role='user',content=content)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt');assert x['image_grid_thw'].shape[0]==3;assert x['image_grid_thw'].tolist()[:2]==[[1,32,32],[1,32,32]]
 checks.append(dict(id=req['id'],shapes={k:list(z.shape) for k,z in x.items()},grid=x['image_grid_thw'].tolist()))
assert not torch.cuda.is_initialized()
protected=dict(dv['protected_after'])
for name,entry in rd(D/'artifact_index.json')['files'].items():protected[str(D/name)]=entry['sha256']
for name,entry in rd(Q/'artifact_index.json')['files'].items():protected[str(Q/name)]=entry['sha256']
for name in ['prompt_identity.json','verification.json','vlm_raw_responses.json']:protected[str(F/name)]=sha(F/name)
assert all(sha(p)==h for p,h in protected.items())
ident=rd(Q/'model_identity.json')
for n,h in ident['checkpoint_hashes'].items():assert sha(B/'model'/n)==h
wr('model_identity.json',dict(model=protocol['model'],revision=protocol['revision'],precision='BF16',checkpoint_hashes=ident['checkpoint_hashes'],processor=protocol['processor'],generation=protocol['generation'],prior_successful_verifier='Same role prompt, but historical probe used NF4 and 128 source/256 processor. This is a 512 BF16 pilot, not an isolated one-factor replication.'))
wr('verification.json',dict(stage='INPUTS_FROZEN',model_loads=0,protected_before=protected,packages_before=subprocess.check_output([str(B/'env/bin/python'),'-m','pip','list','--format=json'],text=True),request_manifest_sha256=sha(R/'request_manifest.json'),candidate_manifest_sha256=sha(R/'.execution/candidate_manifest.json'),prompt_identity_sha256=sha(R/'prompt_identity.json'),protocol_sha256=sha(R/'protocol.json'),preflight=checks,GT_opened=False,source_before={p.name:sha(p) for p in (R/'.execution').glob('*.py')}))
print('VERIFIER_INPUTS_FROZEN',251,'requests; 16 frames; original role prompt unchanged',flush=True)
