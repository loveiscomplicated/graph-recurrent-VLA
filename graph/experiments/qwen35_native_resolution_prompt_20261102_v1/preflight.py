from common import *
import torch
from PIL import Image
from transformers import AutoProcessor
v=rd(R/'verification.json');assert v['stage']=='INPUTS_FROZEN'
mf=rd(R/'input_manifest.json');proc=AutoProcessor.from_pretrained(B/'model',local_files_only=True,min_pixels=512**2,max_pixels=512**2)
good='[{"label":"PEG","bbox_2d":[0,1,1000,999]},{"label":"BOX","bbox_2d":null}]'
for size in [128,512]:
 assert parse(good,size,size)['objects']['peg']['pixel']==[0.,size/1000,size,size*.999]
 assert parse('```json\n'+good+'\n```',size,size)['json_valid']
 for bad in [good.replace('"label":"PEG"','"label":"PEG","label":"PEG"'),good.replace('[0,1,1000,999]','[0,1,1001,999]'),good.replace('[0,1,1000,999]','[false,1,10,20]'),good.replace('[0,1,1000,999]','[0,1,NaN,999]'),'explanation '+good]:
  p=parse(bad,size,size);assert not p['json_valid'] or p['objects']['peg']['status']=='INVALID_BOX'
checks=[]
for size in [128,512]:
 c=next(c for c in mf['cases'] if c['native_size']==size)
 for name,p in v['prompt_identities'].items():
  image=Image.open(c['image_path']).convert('RGB');messages=[dict(role='user',content=[dict(type='image',image=image),dict(type='text',text=Path(p['path']).read_text())])]
  x=proc.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt')
  assert x['image_grid_thw'].tolist()==[[1,32,32]]
  checks.append(dict(native_size=size,prompt=name,total_input_tokens=x['input_ids'].shape[1],grid=x['image_grid_thw'].tolist(),vision_tokens=256,shapes={k:list(z.shape) for k,z in x.items()}))
assert not torch.cuda.is_initialized()
wr('preflight_verification.json',dict(status='PASS',parser_test='PASS invalid JSON/duplicate keys/range/bool/nonfinite rejected; no clipping or salvage',checks=checks,CPU_only=True,model_loads=0,source_at_inference={p.name:sha(p) for p in (R/'.execution').glob('*.py')}))
print('PREFLIGHT_PASS',checks,flush=True)
