from common import *
import subprocess
assert not (R/'protocol.json').exists(),'Never overwrite frozen pilot'
R.mkdir(parents=True,exist_ok=True)
mf=rd(PREV/'bf16/input_manifest.json')
salt='Task10I_native_resolution_prompt_pilot_v1|'
rank=sorted(mf['parents'],key=lambda p:hashlib.sha256((salt+p['parent_id']).encode()).hexdigest())
parents=rank[:4];ids={p['parent_id'] for p in parents}
cases=[c for c in mf['cases'] if c['phase']=='B' and c['parent_id'] in ids]
assert len(cases)==16 and len(ids)==4 and all(p['split']=='train' for p in parents)
prompts={}
for name in ['P0','P1']:
 data=(R/'.execution'/f'prompt_{name}.txt').read_text().rstrip('\n');(R/f'prompt_{name}.txt').write_text(data)
 prompts[name]=dict(path=str(R/f'prompt_{name}.txt'),sha256=sha(R/f'prompt_{name}.txt'),words=len(data.split()))
assert prompts['P0']['sha256']=='40b8c68e18ab9d588f86760ea5023cf0366e85e216fe7d94e18f749348f6f3cb'
protocol=dict(task='10I Stage4A-2 native-resolution x prompt pilot',conditions=VARIANTS,model='Qwen/Qwen3.5-9B',revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',precision='BF16 all original weights, no quantization',device='cuda:0',offloading=False,processor=dict(min_pixels=512**2,max_pixels=512**2),generation=dict(max_new_tokens=128,do_sample=False,use_cache=True,enable_thinking=False,batch_size=1),prompt_candidates=1,model_load_budget=1,Qwen_request_budget=64,SAM2_camera_frame_budget=64,physical_parents=4,camera_frames=16,simulator_control_steps=0,training_steps=0,extra_generation_warmups=0,retries=0,selection_salt=salt,selection_rule='SHA256(salt+parent_id) smallest four among prior 24 train parents; no GT/visibility/grounding filters',frame_rule='existing F0 and F1 IDs unchanged',capture_rule='Restore saved observable robot and actor state with seeded geometry; clone base/wrist cameras at 512 preserving pose/FOV/mount; render both sizes in the same scene with zero physics steps',GT_barrier='Capture process may read privileged state for reconstruction and save segmentation sidecars; inference reads RGB and prompt only; segmentation metrics opened after all 64 responses and SAM masks SHA-frozen',historical_results_known=True,independent_generalization=False,holdout_rule='Remaining 20 parents excluded from this pilot. No automatic holdout execution; freeze selected setting first, then separately budget verification.',comparison='Within fresh paired capture only; prior 256-processor scores not baseline; 128 capture uses same 512 processor intentionally',SAM2='All prespecified 16 camera-frames per condition, invalid/null retained; native 128 or 512 masks; Hiera Small, no tracking or production commit',DINO_forwards=0,DINO_outputs_used=False,production_changes=0,original_runner='PAUSED_SCOPE_REVIEW',success_metrics=dict(visible_min_pixels=16,bbox_recall_IoU=.5,SAM2_success_IoU=.8),statistics='Report per-native GT visibility and common-support paired effects. Four-parent cluster bootstrap is exploratory, not independent generalization evidence.')
wr('protocol.json',protocol);wr('grounding_prompt_identity.json',prompts)
wr('selection_manifest.json',dict(parents=parents,cases=cases,remaining_parent_ids=[p['parent_id'] for p in rank[4:]],source_manifest_sha256=sha(PREV/'bf16/input_manifest.json')))
# Do not open saved DINO outputs. Recheck protected source/model/dataset identities.
v=rd(PREV/'bf16/verification.json');protected={}
for path,expected in v['protected_before'].items():
 if '/dino_raw/' in path or path.endswith('/dino_events.json') or 'dino_matched_comparison' in path:continue
 actual=sha(path);assert actual==expected,path;protected[path]=actual
for p in parents:
 for pathkey,hashkey in [('raw_path','raw_sha256'),('gt_path','gt_sha256'),('receipt_path','receipt_sha256')]:
  actual=sha(p[pathkey]);assert actual==p[hashkey];protected[p[pathkey]]=actual
ident=rd(PREV/'bf16/model_identity.json')
for n,h in ident['checkpoint_hashes'].items():assert sha(B/'model'/n)==h
wr('model_identity.json',dict(model=protocol['model'],revision=protocol['revision'],precision='BF16',checkpoint_hashes=ident['checkpoint_hashes'],quantization=None,load_arguments=ident['load_arguments'],processor=protocol['processor'],generation=protocol['generation'],source_identity_sha256=sha(PREV/'bf16/model_identity.json')))
packages=subprocess.check_output([str(B/'env/bin/python'),'-m','pip','list','--format=json'],text=True)
wr('verification.json',dict(stage='PROTOCOL_FROZEN',protected_before=protected,packages_before=packages,model_loads=0,GT_metrics_opened=False,DINO_outputs_opened=False,protocol_sha256=sha(R/'protocol.json'),selection_sha256=sha(R/'selection_manifest.json'),prompt_identities=prompts,source_before={p.name:sha(p) for p in (R/'.execution').iterdir() if p.is_file()}))
print('PROTOCOL_FROZEN', [p['parent_id'] for p in parents],prompts,flush=True)
