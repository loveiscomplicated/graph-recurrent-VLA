from common import *
import subprocess,platform
import numpy as np
assert os.environ.get('CONTAINER_ID')=='58576054' and not (R/'protocol.json').exists()
R.mkdir(parents=True,exist_ok=True)
oldsel=rd(N/'selection_manifest.json');previous=rd('/workspace/task10i_stage4a2_qwen35_precision_comparison_20261102_v1/bf16/input_manifest.json')
excluded={p['parent_id'] for p in oldsel['parents']};remaining=oldsel['remaining_parent_ids'];assert len(excluded)==4 and len(remaining)==20 and not excluded&set(remaining)
parents={p['parent_id']:p for p in previous['parents']};eligible=[];unavailable=[]
required=['physics_qpos','physics_qvel','physics_ee_pose','physics_peg_pose','physics_box_pose','physics_hole_pose']
for pid in remaining:
 p=parents[pid];assert p['split']=='train'
 # Inspect restoration-field availability only. No GT visibility/IoU or past predictions.
 with np.load(p['gt_path'],allow_pickle=False) as z:missing=[k for k in required if k not in z.files]
 if missing:unavailable.append(dict(parent_id=pid,reason='MISSING_RECORDED_PHYSICS_FOR_NATIVE512_RESTORATION',missing_fields=missing));continue
 assert (Path(p['raw_path']).parent.parent/'attempt_identity.json').exists()
 eligible.append(p)
assert len(eligible)==15 and len(unavailable)==5
ids={p['parent_id'] for p in eligible};cases=[c for c in previous['cases'] if c['phase']=='B' and c['parent_id'] in ids];assert len(cases)==60
oldpid=rd(P/'prompt_identity.json')['conditions'];prompts={k:oldpid[k] for k in ['original_three','appearance_three']}
for k,p in prompts.items():(R/(k+'_prompt.txt')).write_text(p['prompt'])
wr('prompt_identity.json',dict(conditions=prompts,source_sha256=sha(P/'prompt_identity.json'),prompt_sha256={k:sha(R/(k+'_prompt.txt')) for k in prompts},changed=False))
di=rd(D/'model_identity.json');qi=rd(P/'model_identity.json')
protocol=dict(task='10I Stage4A-2 frozen appearance prompt BF16 parent-held-out candidate verifier',model=qi['model'],revision=qi['revision'],precision='BF16',device='cuda:0',attention='sdpa',offloading=False,processor=qi['processor'],generation=qi['generation'],conditions=list(prompts),planned_parents=20,eligible_parents=15,unavailable_parents=5,camera_frames=60,
 selection='All 20 parents held out from native512 appearance-prompt development; 5 excluded before capture/inference solely for absent recorded physics fields. No replacement or visibility/result filtering. Prior other-prompt evaluations exist; not globally unseen research data.',
 frame_rule='Existing F0=t0, F1=t80 or original <=80 fallback; Main and Wrist separately. No new frame choice.',
 capture_rule='Restore recorded qpos/qvel and actor poses with historical seed, same camera pose/FOV/mount; native512 and128 calibration checks; no physics steps, actions, rollout, or optimization. Constructor and seeded environment initialization are counted, not claimed as zero resets.',
 candidate_pool='All proposals after existing DINO box/text thresholds0.1 from frozen two prompts; no area/top5/score/GT subsampling before Qwen; same candidates and RGB inputs in both Qwen conditions.',
 dino=dict(model=di['model'],revision=di['revision'],model_hashes=di['checkpoint_hashes'],dtype='FP32',device='cuda:0',prompts=dict(peg='red and white block',box='cardboard box'),threshold=.1),
 policies=dict(legacy='Original area25%/Top5/null0.3/joint0.3 selector',raw_top1='Highest score within each original query',original_query_gate='Keep matching role from original prompt, highest original DINO score',appearance_query_gate='Primary: matching role from frozen appearance prompt, highest original DINO score; else null'),
 metrics=dict(visible_min_pixels=16,bbox_success_IoU=.5,GT_role='>=16 foreground pixels and >=70% one peg/box/robot actor; mixed excluded only from role accuracy, retained in selection metrics',bootstrap_unit='physical parent',bootstrap_replicates=10000,bootstrap_seed=42),
 budget=dict(Qwen_loads=1,DINO_loads=1,DINO_forwards=120,Qwen_requests='2 x every preserved DINO proposal; hard cap 8000. Stop if exceeded; no post-result sampling.',SAM2_forwards=0,training=0,simulation_steps=0,retries=0,prompt_edits=0,extra_warmups=0),
 GT_barrier='Reconstruction reads physics state and writes segmentation; inference request path has RGB+frozen text only. All Qwen responses freeze before GT semantic evaluation.',production=False,original_runner='PAUSED_SCOPE_REVIEW')
wr('protocol.json',protocol);wr('selection_manifest.json',dict(planned_parent_ids=remaining,development_excluded_parent_ids=sorted(excluded),parents=eligible,cases=cases,unavailable=unavailable,source_manifest_sha256=sha(N/'selection_manifest.json')));wr('model_identity.json',qi)
protected=dict(rd(P/'verification.json')['protected_after'])
for rel,e in rd(P/'artifact_index.json')['files'].items():protected[str(P/rel)]=e['sha256']
protected[str(P/'artifact_index.json')]=sha(P/'artifact_index.json')
for p in parents.values():
 for pathkey,hashkey in [('raw_path','raw_sha256'),('gt_path','gt_sha256')]:protected[p[pathkey]]=p[hashkey]
for root,identity in [(B/'model',qi),(MODEL,di)]:
 for rel,h in identity['checkpoint_hashes'].items():assert sha(root/rel)==h;protected[str(root/rel)]=h
assert all(sha(p)==h for p,h in protected.items())
wr('verification.json',dict(stage='PROTOCOL_FROZEN',model_loads=0,dino_loads=0,protected_before=protected,packages_before=subprocess.check_output([str(B/'env/bin/python'),'-m','pip','list','--format=json'],text=True),protocol_sha256=sha(R/'protocol.json'),selection_sha256=sha(R/'selection_manifest.json'),runtime=dict(torch='2.14.1+cu130',transformers='5.19.0',python=platform.python_version()),source_before={p.name:sha(p) for p in (R/'.execution').glob('*.py')}))
print('PROTOCOL_FROZEN eligible15 planned20 frames60; appearance and original text unchanged',flush=True)
