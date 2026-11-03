from common import *
import subprocess, numpy as np
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json');assert v['stage']=='EVALUATION_COMPLETE'
mf=rd(R/'input_manifest.json');raw=rd(R/'full_image_grounding_raw_responses.json')['responses'];rows=list(csv.DictReader((R/'full_image_grounding_predictions.csv').open()))
assert len({p['parent_id'] for p in mf['parents']})==24
assert len(mf['protocol']['sam2_ids'])==16 and len(set(mf['protocol']['sam2_ids']))==16
assert len(raw)==99 and len(rows)==198 and v['model_loads']==1
assert sum(c['phase']=='A' for c in mf['cases'])==3 and sum(c['phase']=='B' for c in mf['cases'])==96
assert sha(R/'full_image_grounding_raw_responses.json')==v['raw_response_SHA_before_GT']==v['raw_response_SHA_after_GT']
assert sha(R/'full_image_grounding_predictions.csv')==v['predictions_SHA_before_GT']==v['predictions_SHA_after_GT']
assert all(sha(c['image_path'])==c['image_sha256'] for c in mf['cases'] if c['input_status']=='AVAILABLE')
assert sha(R/'input_manifest.json')==v['manifest_sha256'] and sha(R/'grounding_prompt.txt')==v['prompt_sha256'] and sha(R/'model_identity.json')==v['identity_sha256']
for r in raw:
 assert parse(r['raw_text'])==r['parsed']
 for k,o in r['parsed']['objects'].items():
  cr=next(x for x in rows if x['id']==r['id'] and x['object']==k);assert cr['status']==o['status'] and json.loads(cr['pixel'])==o['pixel']
protected=dict(v['protected_before']);protected.update(v['deferred_DINO_hashes_from_previous_verified_receipt']);protected.update(v['GT_reference_files']);protected.update(v.get('DINO_matched_source_hashes',{}))
after={p:sha(p) for p in protected};assert after==protected
ck={n:sha(B/'model'/n) for n in v['model_hashes_before']};assert ck==v['model_hashes_before']
packages={p:subprocess.check_output([p,'-m','pip','list','--format=json'],text=True) for p in v['packages_before']};assert packages==v['packages_before']
sam=rd(R/'sam2_initialization_metrics.json');sr=rd(R/'.execution/sam2_raw_initializations.json')
assert len(sr)==16
for s in sr:
 assert sha(s['masks_path'])==s['masks_sha256']
 with np.load(s['masks_path']) as z:
  assert all(z[k].shape==(128,128) for k in z.files)
  assert set(z.files)=={k for k,o in s['objects'].items() if o['status']=='INITIALIZED'}
assert sha(R/'.execution/sam2_raw_initializations.json')==v['sam2_raw_SHA_before_GT']
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
assert not gpu.strip(),gpu
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
active=[x for x in processes if 'python' in x and any(k in x for k in [' -m graph.task10i_stage4a2.','/run_pipeline.py','/registered_runner.py'])]
assert not active,active
metrics=rd(R/'grounding_evaluation_metrics.json')
with ExperimentLogger(team='integration',model='Task10I_Stage4A2_QwenFullImageGroundingBF16Metrics',seed=42,job_type='evaluate',root=R/'.execution/experiments',config=dict(manifest_sha256=v['manifest_sha256'],prompt_sha256=v['prompt_sha256'],inference_run_id=v['logger_run_id'],production_changes=False)) as logger:
 logger.log(dict(requests=len(raw),independent_requests=96,parents=24,strict_json_rate=metrics['primary']['image_object_level']['JSON_validity']['rate'],coordinate_validity=metrics['primary']['image_object_level']['coordinate_contract_validity']['rate']))
 logger.record(final_status=metrics['final_status'])
 for n in ['grounding_evaluation_metrics.json','sam2_initialization_metrics.json','grounding_latency_vram.json']:logger.link_output(R/n,kind='metrics')
 v['evaluation_logger_run_id']=logger.run_id
previous=Path('/workspace/task10i_stage4a2_qwen35_full_image_grounding_20261102_v1')
assert sha(previous/'artifact_index.json')==v['original_NF4_artifact_index_SHA_before']
prior_index=rd(previous/'artifact_index.json');nf4_checked=0
for section in ['required','supporting']:
 for name,item in prior_index[section].items():
  assert sha(previous/name)==item['sha256'], name
  nf4_checked+=1
assert all(sha(path)==digest for path,digest in v['source_files_before'].items())
v.update(original_NF4_artifacts_unchanged=True,original_NF4_files_verified=nf4_checked)
v.update(stage='COMPLETE_VERIFIED',completion='VERIFIED',protected_file_count=len(protected),protected_after=after,all_protected_SHA_unchanged=True,model_processor_unchanged=True,environments_unchanged=True,GT_barrier_pass=True,raw_CSV_readback_pass=True,coordinate_contract_checks_pass=True,final_GPU_processes=gpu,active_original_stage4a2_runners=active,production_changes=0,new_DINO_forwards=0,new_models=0,prompt_sweeps=0,new_simulator_rollouts=0,training=0,publishing=0,git_commits=0,execution_sources={p.name:sha(p) for p in (R/'.execution').glob('*.py')})
wr('verification.json',v)
required=['grounding_prompt.txt','grounding_prompt_identity.json','model_identity.json','input_manifest.json','full_image_grounding_raw_responses.json','full_image_grounding_predictions.csv','grounding_evaluation_metrics.json','grounding_failure_cases.csv','sam2_initialization_metrics.json','sam2_initialization_cases.csv','grounding_latency_vram.json','verification.json','full_image_grounding_comparison.png']
wr('artifact_index.json',dict(required={n:dict(path=n,sha256=sha(R/n),bytes=(R/n).stat().st_size) for n in required},supporting={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in (R/'.execution').rglob('*') if p.is_file() and p.suffix in ['.py','.json','.csv','.png','.npz']},report='docs/visual-graph-integration/10i_stage4a2_qwen35_full_image_grounding_bf16.md',publication=False))
print('FINAL_VERIFIED',metrics['final_status'],len(protected),'protected',flush=True)
