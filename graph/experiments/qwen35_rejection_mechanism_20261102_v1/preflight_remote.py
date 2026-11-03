from common import *
import subprocess
from PIL import Image
import torch
from transformers import AutoProcessor
assert os.environ.get('CONTAINER_ID')=='58576054' and not (R/'verification.json').exists()
assert all(sha(R/n)==h for n,h in rd(R/'frozen_input_index.json').items())
requests=rd(R/'request_manifest.json')['requests'];prompts=rd(R/'prompt_identity.json')['conditions'];cases=rd(R/'case_manifest.json')['cases'];assert len(requests)==326 and len(cases)==60
old=rd(PREV/'prompt_identity.json')['conditions']['appearance_three'];assert prompts['baseline']==old and prompts['crop_only']['prompt']==old['prompt']
protected=dict(rd(PREV/'verification.json')['protected_after'])
for rel,e in rd(PREV/'artifact_index.json')['files'].items():protected[str(PREV/rel)]=e['sha256']
protected[str(PREV/'artifact_index.json')]=sha(PREV/'artifact_index.json')
assert all(sha(p)==h for p,h in protected.items())
for r in requests:assert all(sha(p)==h for p,h in zip(r['images'],r['image_sha256']))
identity=rd(R/'model_identity.json');assert all(sha(B/'model'/p)==h for p,h in identity['checkpoint_hashes'].items())
proc=AutoProcessor.from_pretrained(B/'model',local_files_only=True,min_pixels=262144,max_pixels=262144);checks=[]
for cond,p in prompts.items():
 r=next(r for r in requests if r['condition']==cond);content=[dict(type='image',image=Image.open(path).convert('RGB')) for path in r['images']]+[dict(type='text',text=p['prompt'])]
 x=proc.apply_chat_template([dict(role='user',content=content)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt');assert len(x['image_grid_thw'])==len(r['images']);checks.append(dict(condition=cond,grid=x['image_grid_thw'].tolist(),input_tokens=x['input_ids'].shape[1]))
assert not torch.cuda.is_initialized()
wr('verification.json',dict(stage='INPUTS_FROZEN',model_loads=0,completed_requests=0,protected_before=protected,packages_before=subprocess.check_output([str(B/'env/bin/python'),'-m','pip','list','--format=json'],text=True),protocol_sha256=sha(R/'protocol.json'),prompt_identity_sha256=sha(R/'prompt_identity.json'),request_manifest_sha256=sha(R/'request_manifest.json'),case_manifest_sha256=sha(R/'case_manifest.json'),frozen_input_index_sha256=sha(R/'frozen_input_index.json'),source_before={p.name:sha(p) for p in (R/'.execution').glob('*.py')},preflight=checks,hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],text=True)))
print('PREFLIGHT_PASS cases60 requests326; no model loads',flush=True)
