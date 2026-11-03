from common import *
import ast, collections, copy, inspect, subprocess
import torch
from PIL import Image
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig
from transformers.quantizers.base import get_keys_to_not_convert
from transformers.integrations.bitsandbytes import replace_with_bnb_linear, should_convert_module
import bitsandbytes as bnb

assert not (P/'protocol.json').exists(), 'Do not overwrite or restart an existing comparison'
P.mkdir(parents=True, exist_ok=True)
prompt=(P/'.execution/grounding_prompt.txt').read_text().rstrip('\n')
assert hashlib.sha256(prompt.encode()).hexdigest()=='40b8c68e18ab9d588f86760ea5023cf0366e85e216fe7d94e18f749348f6f3cb'
mf=rd(DETAIL/'input_manifest.json'); prior=rd(DETAIL/'model_identity.json'); dv=rd(DETAIL/'verification.json')
assert len(mf['cases'])==99 and len(mf['parents'])==24 and len(set(mf['protocol']['sam2_ids']))==16
protected=dict(dv['protected_after']); deferred={p:h for p,h in protected.items() if '/dino_raw/' in p or p.endswith('/dino_events.json') or 'dino_matched_comparison' in p}
checked={p:sha(p) for p in protected if p not in deferred}
assert all(checked[p]==protected[p] for p in checked)
for group in ['required','supporting']:
 for rel,e in rd(DETAIL/'artifact_index.json')[group].items():
  f=DETAIL/rel
  if 'dino_matched_comparison' in rel: deferred[str(f)]=e['sha256'];continue
  checked[str(f)]=sha(f);assert checked[str(f)]==e['sha256'],str(f)
for c in mf['cases']: assert sha(c['image_path'])==c['image_sha256']
ck={n:sha(B/'model'/n) for n in prior['checkpoint_hashes']};assert ck==prior['checkpoint_hashes']
packages={p:subprocess.check_output([p,'-m','pip','list','--format=json'],text=True) for p in dv['packages_before']};assert packages==dv['packages_before']
cfg=AutoConfig.from_pretrained(B/'model',local_files_only=True)
with init_empty_weights(): meta=Qwen3_5ForConditionalGeneration._from_config(cfg,dtype=torch.bfloat16,attn_implementation='sdpa')
default_skips=get_keys_to_not_convert(meta)
skip=sorted(set(default_skips+['model.visual']))
assert not should_convert_module('model.visual.merger.linear_fc1',skip)
assert not should_convert_module('lm_head',skip), 'Preserve default lm_head exclusion when supplying custom skips'
assert should_convert_module('model.language_model.layers.0.mlp.gate_proj',skip)
original_linear={n for n,m in meta.named_modules() if type(m) is torch.nn.Linear}
vision_linear={n for n in original_linear if n.startswith('model.visual.')}
assert len(vision_linear)==110 and any('.merger.' in n for n in vision_linear)
param_count=sum(p.numel() for p in meta.parameters());del meta
configs={
 'bf16':None,
 'vision_bf16_language_nf4':BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True,bnb_4bit_quant_storage=torch.uint8,llm_int8_skip_modules=skip,llm_int8_enable_fp32_cpu_offload=False),
 'vision_bf16_language_int8':BitsAndBytesConfig(load_in_8bit=True,llm_int8_skip_modules=skip,llm_int8_threshold=6.,llm_int8_has_fp16_weight=False,llm_int8_enable_fp32_cpu_offload=False),
}
expected={}
for mode,q in configs.items():
 with init_empty_weights(): m=Qwen3_5ForConditionalGeneration._from_config(cfg,dtype=torch.bfloat16,attn_implementation='sdpa')
 if q is not None: m=replace_with_bnb_linear(m,modules_to_not_convert=skip+list(m._keep_in_fp32_modules or []),quantization_config=q)
 q4=[n for n,z in m.named_modules() if isinstance(z,bnb.nn.Linear4bit)];q8=[n for n,z in m.named_modules() if isinstance(z,bnb.nn.Linear8bitLt)]
 assert not any(n.startswith('model.visual.') for n in q4+q8)
 assert len(q4)==(248 if mode.endswith('nf4') else 0) and len(q8)==(248 if mode.endswith('int8') else 0),(mode,len(q4),len(q8))
 expected[mode]=dict(linear4bit_names=q4,linear8bit_names=q8,vision_linear_names=sorted(vision_linear),parameter_count_before_quantization=param_count,BF16_weight_GiB=param_count*2/2**30,default_exclusions=default_skips,explicit_exclusions=skip,keep_in_fp32_modules=m._keep_in_fp32_modules)
 del m
# Contract checks exercise valid, null, duplicate-key, malformed, nonfinite and range failures.
good='[{"label":"PEG","bbox_2d":[0,1,1000,999]},{"label":"BOX","bbox_2d":null}]'
assert parse(good)['objects']['peg']['pixel']==[0.,.128,128.,127.872]
assert parse('```json\n'+good+'\n```')['json_valid'] and not parse('```json\n'+good+'\n```')['strict_json_valid']
for bad in [good.replace('"label":"PEG"','"label":"PEG","label":"PEG"'),good.replace('[0,1,1000,999]','[0,1,1001,999]'),good.replace('[0,1,1000,999]','[true,1,10,20]'),good.replace('[0,1,1000,999]','[0,1,NaN,999]'),'explanation '+good]:
 assert not parse(bad)['json_valid'] or parse(bad)['objects']['peg']['status']=='INVALID_BOX'
protocol=dict(variants=VARIANTS,order=VARIANTS,model_loads_per_variant=1,max_grounding_requests_per_variant=99,phase_A_requests=3,phase_B_requests=96,SAM2_camera_frames_per_variant=16,max_total_requests=297,max_total_SAM2_frames=48,prompt_variants=0,extra_warmups=0,retries=0,independent=False,evaluation_status='Matched precision RETEST on the previously evaluated 24 train parents; not new independent generalization',baseline='Saved all-module NF4 detailed-prompt retry; no baseline rerun',prompt_source=str(DETAIL/'grounding_prompt.txt'),changed='Only original-weight loading precision/module quantization exclusions; same lm_head default exclusion, prompt, schema, parser, images, resize, generation and SAM2',GT_barrier='All three variants terminate and all available predictions are SHA-frozen before any new GT/DINO evaluation',DINO_forwards=0,production_changes=0,offloading=False,SAM2_co_residency='Attempt with resident Qwen; on resource failure preserve results and stop that variant, no process killing or reload')
wr(P/'protocol.json',protocol);wr(P/'expected_precision.json',expected)
proc=AutoProcessor.from_pretrained(B/'model',local_files_only=True,min_pixels=65536,max_pixels=65536)
messages=[dict(role='user',content=[dict(type='image',image=Image.open(mf['cases'][0]['image_path']).convert('RGB')),dict(type='text',text=prompt)])]
x=proc.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt');rendered=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
assert x['input_ids'].shape[1]==1292 and x['image_grid_thw'].tolist()==[[1,16,16]]
sources={p.name:sha(p) for p in (P/'.execution').glob('*.py')}
commonv=dict(stage4a2='PAUSED_SCOPE_REVIEW',model_loads=0,protected_before=checked,deferred_DINO_hashes_from_previous_verified_receipt=deferred,model_hashes_before=ck,packages_before=packages,GT_semantic_opened=False,DINO_outputs_opened=False,prior_results_known=True,protocol=protocol,execution_sources_before=sources,parser_contract_test='PASS',preflight_GPU=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.free','--format=csv'],text=True))
for mode in VARIANTS:
 root=P/mode;assert not root.exists();root.mkdir()
 (root/'.execution').mkdir();(root/'grounding_prompt.txt').write_text(prompt)
 m=copy.deepcopy(mf);m['protocol']['precision_comparison']=protocol;m['protocol']['independent']=False;wr(root/'input_manifest.json',m)
 wr(root/'grounding_prompt_identity.json',dict(sha256=sha(root/'grounding_prompt.txt'),source=str(DETAIL/'grounding_prompt.txt'),unchanged=True,encoding='UTF-8',trailing_newline=False))
 ident={k:copy.deepcopy(prior[k]) for k in ['model','revision','checkpoint_hashes','runtime','sources','generation','processor_resolution','processor_contract'] if k in prior}
 ident.update(variant=mode,quantization_config=configs[mode].to_dict() if configs[mode] else None,expected_precision=expected[mode],load_arguments=dict(local_files_only=True,device_map={'':0},dtype='torch.bfloat16',attn_implementation='sdpa'),source_identity_sha256=sha(DETAIL/'model_identity.json'))
 ident['processor_contract'].update(actual_rendered_prompt=rendered,input_shapes={k:list(v.shape) for k,v in x.items()},image_grid_thw=x['image_grid_thw'].tolist())
 wr(root/'model_identity.json',ident);(root/'.execution/rendered_template.txt').write_text(rendered)
 vv=copy.deepcopy(commonv);vv.update(stage='INPUTS_FROZEN',variant=mode,manifest_sha256=sha(root/'input_manifest.json'),prompt_sha256=sha(root/'grounding_prompt.txt'),identity_sha256=sha(root/'model_identity.json'));wr(root/'verification.json',vv)
wr(P/'verification.json',dict(stage='ALL_INPUTS_FROZEN',variant_order=VARIANTS,source_sha256=sources,protected_count=len(checked)+len(deferred),parser_contract_test='PASS',meta_only_precision_test='PASS',BF16_weight_GiB=param_count*2/2**30,prompt_sha256=sha(P/'bf16/grounding_prompt.txt')))
print('READY_THREE_PRECISIONS',json.dumps(dict(expected={k:dict(q4=len(v['linear4bit_names']),q8=len(v['linear8bit_names'])) for k,v in expected.items()},skips=skip,BF16_weight_GiB=param_count*2/2**30)),flush=True)
