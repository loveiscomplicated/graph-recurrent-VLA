from common import *
import subprocess
assert os.environ.get('CONTAINER_ID')=='58576054' and not (R/'protocol.json').exists()
R.mkdir(parents=True,exist_ok=True)
selection=rd(R/'.execution/diagnostic_case_selection.json');oldrequests={r['id']:r for r in rd(V/'request_manifest.json')['requests']};oldpid=rd(V/'prompt_identity.json')
appearance='''Identify the object shown in the last image, which is a crop of the candidate.
If other images are provided, the first shows the full scene and the second shows the candidate outlined.
Classify the candidate itself using its visible appearance:
ROBOT: Gray or white robot arm links, or dark or black gripper fingers. A separate object held by the gripper is not a robot part.
PEG: A red-and-white elongated rectangular piece. It can be held by the gripper or partly outside the image.
BOX: A tan or brown block-shaped fixture. An opening may be visible, depending on the view.
Use color, shape, and visible parts together. Do not classify a different object elsewhere in the scene.
Choose OTHER for a clearly different object, or UNCERTAIN if the candidate cannot be identified.
Return exactly one label: PEG, BOX, ROBOT, OTHER, UNCERTAIN.
Do not add explanations or punctuation.'''
caption='Describe the main object visible in this close-up image. State its visible colors, shape, and recognizable parts. Name the object if you can identify it. If it is unclear, say so. Use one or two short sentences.'
binary='Does the object in this close-up image belong to the robot arm, gripper, or fingers? A separate workpiece held by a gripper is not itself a robot part. Answer YES if it is a robot part, NO if it is a separate object, or UNCERTAIN if you cannot tell. Return only YES, NO, or UNCERTAIN.'
prompts=dict(original_three=dict(preamble=oldpid['preamble'],prompt=oldpid['prompt'],images='three',max_new_tokens=16,labels=LABELS),appearance_three=dict(preamble='',prompt=appearance,images='three',max_new_tokens=16,labels=LABELS),appearance_crop=dict(preamble='',prompt=appearance,images='crop',max_new_tokens=16,labels=LABELS),caption_crop=dict(preamble='',prompt=caption,images='crop',max_new_tokens=96,labels=None),robot_binary_crop=dict(preamble='',prompt=binary,images='crop',max_new_tokens=16,labels=['YES','NO','UNCERTAIN']))
wr('prompt_identity.json',dict(conditions=prompts,original_semantic_prompt_sha256=oldpid['prompt_sha256'],notes='Prompts frozen once; no output-driven editing. Appearance descriptions use known simulator assets, not generic zero-shot object-role generalization.'))
for name,p in prompts.items():(R/(name+'_prompt.txt')).write_text((p['preamble']+'\n' if p['preamble'] else '')+p['prompt'])
requests=[]
for ids in selection['groups'].values():
 for cid in ids:
  old=oldrequests[cid];assert all(sha(p)==h for p,h in zip(old['images'],old['image_sha256']))
  for condition,p in prompts.items():
   paths=old['images'] if p['images']=='three' else old['images'][-1:];hashes=old['image_sha256'] if p['images']=='three' else old['image_sha256'][-1:]
   requests.append(dict(id=cid+'__'+condition,candidate_id=cid,condition=condition,images=paths,image_sha256=hashes))
requests.sort(key=lambda r:hashlib.sha256(('perception_diagnostic_v1|'+r['id']).encode()).hexdigest());assert len(requests)==50
wr('request_manifest.json',dict(requests=requests));wr('input_manifest.json',selection)
protocol=dict(task='Task10I Qwen perception-vs-role diagnostic',model='Qwen/Qwen3.5-9B',revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',precision='BF16',device='cuda:0',attention='sdpa',offloading=False,processor=dict(min_pixels=512**2,max_pixels=512**2),generation=dict(batch_size=1,do_sample=False,enable_thinking=False,use_cache=True,max_new_tokens='16 labels/binary, 96 brief description'),cases=10,conditions=list(prompts),selection='Historical GT/proxy roles and errors used to choose 2 robot errors, 2 robot correct, 2 peg errors, 2 peg correct, 2 box correct. Diagnostic only, no generalization denominator.',contrasts='original_three vs appearance_three isolates text; appearance_three vs appearance_crop isolates removal of scene/marker images at identical appearance text. Caption and binary are separate perception probes.',budget=dict(model_loads=1,Qwen_requests=50,retries=0,warmups=0,prompt_sweeps=0,DINO_forwards=0,SAM2_forwards=0,simulator=0,training=0),GT_policy='Known cases selected with historical GT; inference receives RGB and frozen text only. All 50 outputs freeze before posthoc scoring.',production=False,original_runner='PAUSED_SCOPE_REVIEW')
wr('protocol.json',protocol);ident=rd(V/'model_identity.json');wr('model_identity.json',ident)
prior=rd(V/'verification.json');protected=dict(prior['protected_after'])
for rel,entry in rd(V/'artifact_index.json')['files'].items():protected[str(V/rel)]=entry['sha256']
protected[str(V/'artifact_index.json')]=sha(V/'artifact_index.json')
assert all(sha(p)==h for p,h in protected.items())
assert all(sha(B/'model'/p)==h for p,h in ident['checkpoint_hashes'].items())
wr('verification.json',dict(stage='INPUTS_FROZEN',model_loads=0,completed_requests=0,protected_before=protected,packages_before=subprocess.check_output([str(B/'env/bin/python'),'-m','pip','list','--format=json'],text=True),request_manifest_sha256=sha(R/'request_manifest.json'),prompt_identity_sha256=sha(R/'prompt_identity.json'),protocol_sha256=sha(R/'protocol.json'),source_before={p.name:sha(p) for p in (R/'.execution').glob('*.py')}))
print('DIAGNOSTIC_INPUTS_FROZEN 50 requests',flush=True)
