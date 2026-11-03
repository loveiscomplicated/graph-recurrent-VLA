from common import *
from PIL import Image,ImageDraw
import torch
from transformers import AutoProcessor
v=rd(R/'verification.json');assert v['stage']=='DINO_PREDICTIONS_FROZEN' and v['model_loads']==0
assert sha(R/'dino_raw_responses.json')==v['dino_raw_sha256_before_GT']
mf=rd(R/'input_manifest.json');images={c['id']:c for c in mf['cases']};prompts=rd(R/'prompt_identity.json')['conditions'];candidates=[];requests=[]
for r in rd(R/'dino_raw_responses.json')['responses']:
 c=images[r['id']];assert c['native_size']==512 and sha(c['image_path'])==c['image_sha256'];im=Image.open(c['image_path']).convert('RGB')
 for rank,(box,score) in enumerate(zip(r['boxes'],r['scores'])):
  cid=c['id']+'_'+r['object']+f'_rank{rank:03d}';x1,y1,x2,y2=box;px=(x2-x1)*.1;py=(y2-y1)*.1
  crop=[max(0,math.floor(x1-px)),max(0,math.floor(y1-py)),min(512,math.ceil(x2+px)),min(512,math.ceil(y2+py))];assert crop[0]<crop[2] and crop[1]<crop[3]
  folder=R/'candidate_images';folder.mkdir(exist_ok=True);mp=folder/(cid+'_marked.png');cp=folder/(cid+'_crop.png');marked=im.copy();ImageDraw.Draw(marked).rectangle(box,outline=(0,180,220),width=4);marked.save(mp);im.crop(crop).save(cp)
  paths=[c['image_path'],str(mp),str(cp)];hashes=[sha(p) for p in paths]
  candidates.append(dict(id=cid,image_id=c['id'],pair_id=c['pair_id'],parent_id=c['parent_id'],camera=c['camera'],frame_kind=c['frame_kind'],timestep=c['timestep'],query_object=r['object'],rank=rank,box=box,DINO_score=score,crop_box=crop))
  for condition in prompts:requests.append(dict(id=cid+'__'+condition,candidate_id=cid,condition=condition,images=paths,image_sha256=hashes))
assert 0<len(requests)<=8000 and len(requests)==len(candidates)*2
requests.sort(key=lambda r:hashlib.sha256(('appearance_parent_holdout|'+r['id']).encode()).hexdigest())
wr('request_manifest.json',dict(requests=requests));wr('.execution/candidate_manifest.json',dict(candidates=candidates,images=list(images.values()),parents=[p['parent_id'] for p in mf['parents']]))
proc=AutoProcessor.from_pretrained(B/'model',local_files_only=True,min_pixels=512**2,max_pixels=512**2)
preflight=[]
for cond,p in prompts.items():
 req=next(x for x in requests if x['condition']==cond);content=([dict(type='text',text=p['preamble'])] if p['preamble'] else [])+[dict(type='image',image=Image.open(q).convert('RGB')) for q in req['images']]+[dict(type='text',text=p['prompt'])]
 x=proc.apply_chat_template([dict(role='user',content=content)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt');assert x['image_grid_thw'].tolist()[:2]==[[1,32,32],[1,32,32]];preflight.append(dict(condition=cond,grid=x['image_grid_thw'].tolist(),input_tokens=x['input_ids'].shape[1]))
assert not torch.cuda.is_initialized()
v.update(stage='INPUTS_FROZEN',completed_requests=0,request_count=len(requests),candidate_count=len(candidates),request_manifest_sha256=sha(R/'request_manifest.json'),candidate_manifest_sha256=sha(R/'.execution/candidate_manifest.json'),prompt_identity_sha256=sha(R/'prompt_identity.json'),preflight=preflight);wr('verification.json',v)
print('QWEN_INPUTS_FROZEN',len(candidates),'candidates;',len(requests),'requests',flush=True)
