from common import *
import subprocess, platform, importlib.util
assert os.environ.get('CONTAINER_ID')=='58576054' and not (R/'protocol.json').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
cm=rd(P/'.execution/candidate_manifest.json'); mf=rd(P/'input_manifest.json')
assert len(cm['candidates'])==986 and len(cm['images'])==60 and len(mf['parents'])==15
parents=sorted(mf['parents'],key=lambda p:hashlib.sha256(('Task10I-SAM2MaskVerifier-20261102-v1|'+p['parent_id']).encode()).hexdigest())[:8]
assert all(p['F1_actual']==80 for p in parents)
old=rd(P/'prompt_identity.json')['conditions']['appearance_three']
prompt='Identify the candidate region marked in image 2.\nImage 1 shows the complete scene. Image 3 is a crop of the same candidate region.\n'+'\n'.join(old['prompt'].splitlines()[2:])
R.mkdir(parents=True,exist_ok=True); (R/'verifier_prompt.txt').write_text(prompt)
wr('prompt_identity.json',dict(prompt=prompt,sha256=sha(R/'verifier_prompt.txt'),source_prompt_sha256=sha(P/'appearance_three_prompt.txt'),labels=LABELS,max_new_tokens=16,change='Only first two sentences; B and C identical text; all appearance definitions retained',mask_render=dict(outside_RGB_multiplier=.25,contour_RGB=[0,180,220],contour_width=2,crop='original box plus10% padding, identical B/C')))
wr('input_manifest.json',mf); wr('candidate_manifest.json',cm);wr('model_identity.json',rd(P/'model_identity.json'))
wr('tracking_manifest.json',dict(parents=parents,cameras=['base_camera','hand_camera'],timesteps=list(range(81)),camera_sequences=16,native_frames=1296,selection='SHA256 Task10I-SAM2MaskVerifier-20261102-v1|parent, first8; no result filtering'))
protocol=dict(task='Task10I SAM2 mask verifier and identity maintenance',methods=dict(A='Frozen appearance labels',B='Shared binding prompt, original box marked/full/crop',C='Same prompt, mask contour and outside-dimmed crop'),primary_comparison='C minus B',historical_comparison='C minus A',candidate_selection='Same-query matching role then original DINO score desc, rank asc, ID asc',known_data=True,generalization_claim=False,GT_barrier='RGB/masks/prompt only in VLM; no new semantic GT reading until static and tracking predictions frozen',model=rd(P/'model_identity.json'),sam2=dict(checkpoint=str(CK),sha256=sha(CK),config=CFG,weights='FP32',compute='BF16 autocast',multimask_output=False,threshold=0.,postprocessing='installed build defaults frozen, no threshold edits'),tracking=dict(initialization='Predicted selected image mask only, add_new_mask; independent states',re_grounding=False,offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False,jpeg_quality=100,jpeg_subsampling=0,future_frames_in_model=False),metrics=dict(visible_min_pixels=16,mask_primary_IoU=.8,secondary_IoU=.5,wrong_actor_fraction=.5,bootstrap_seed=42,bootstrap_replicates=10000,bootstrap_unit='physical parent',exploratory_goals=dict(both_mask_success=.8,wrong_object_max=.05)),budget=dict(Qwen_loads=1,SAM2_loads=2,SAM2_candidate_prompts=986,Qwen_requests_max=1972,tracking_states_max=48,tracking_camera_frames_max=3888,native_frames=1296,DINO_forwards=0,physics_steps=0,training=0,retries=0),production=False,stage4a2='PAUSED_SCOPE_REVIEW')
wr('protocol.json',protocol)
protected=dict(rd(P/'verification.json')['protected_after'])
for n,e in rd(P/'artifact_index.json')['files'].items(): protected[str(P/n)]=e['sha256']
protected[str(P/'artifact_index.json')]=sha(P/'artifact_index.json');protected[str(CK)]=sha(CK)
samroot=Path(importlib.util.find_spec('sam2').origin).parent
samfiles={str(p):sha(p) for p in samroot.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml']}
protected.update(samfiles)
assert all(sha(p)==h for p,h in protected.items())
wr('sam2_identity.json',dict(checkpoint=str(CK),sha256=sha(CK),config=CFG,installed_source=samfiles))
wr('verification.json',dict(stage='PROTOCOL_FROZEN',protected_before=protected,protocol_sha256=sha(R/'protocol.json'),prompt_sha256=sha(R/'prompt_identity.json'),candidate_manifest_sha256=sha(R/'candidate_manifest.json'),input_manifest_sha256=sha(R/'input_manifest.json'),tracking_manifest_sha256=sha(R/'tracking_manifest.json'),packages_before=subprocess.check_output([str(B/'env/bin/python'),'-m','pip','list','--format=json'],text=True),source_before={p.name:sha(p) for p in (R/'.execution').glob('*') if p.is_file() and p.suffix in ['.py','.sh']},python=platform.python_version(),Qwen_loads=0,SAM2_loads=0))
print('PROTOCOL_FROZEN 60 images,986 candidates,16 sequences,1972 maximum Qwen',flush=True)
