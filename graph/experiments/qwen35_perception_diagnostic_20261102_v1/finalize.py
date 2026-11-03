from common import *
import subprocess,collections
from utils.experiment_logger import ExperimentLogger
v=rd(R/'verification.json');assert v['stage']=='PREDICTIONS_FROZEN' and v['completed_requests']==50 and v['model_loads']==1
assert sha(R/'verifier_raw_responses.json')==v['raw_sha256_before_GT'] and sha(R/'verifier_predictions.csv')==v['predictions_sha256_before_GT']
with (V/'candidate_evaluation.csv').open() as f:truth={r['id']:r for r in csv.DictReader(f)}
raw=rd(R/'verifier_raw_responses.json')['responses'];prompts=rd(R/'prompt_identity.json')['conditions'];rows=[];groups=rd(R/'input_manifest.json')['groups'];group_for={cid:group for group,ids in groups.items() for cid in ids}
for r in raw:
 t=truth[r['candidate_id']];p=prompts[r['condition']];expected=('YES' if t['GT_role']=='ROBOT' else 'NO') if r['condition']=='robot_binary_crop' else t['GT_role'];actual=r['raw_text'].strip();valid=actual in p['labels'] if p['labels'] else None
 assert r['timing']['initial_forward_cache']['cache_length']==0
 if p['labels']:assert r['strict_label']==(actual if valid else 'INVALID_OUTPUT')
 rows.append(dict(id=r['id'],candidate_id=r['candidate_id'],group=group_for[r['candidate_id']],condition=r['condition'],GT_proxy_role=t['GT_role'],historical_label=t['strict_label'],expected=expected if p['labels'] else None,response=r['raw_text'],format_valid=valid,correct=actual==expected if p['labels'] else None))
csvwr('diagnostic_cases.csv',rows)
metrics={}
for condition,p in prompts.items():
 z=[r for r in rows if r['condition']==condition];metrics[condition]=dict(n=len(z),scored=p['labels'] is not None,correct=sum(r['correct'] is True for r in z) if p['labels'] else None,valid=sum(r['format_valid'] is True for r in z) if p['labels'] else None,by_group={g:dict(n=sum(r['group']==g for r in z),correct=sum(r['group']==g and r['correct'] is True for r in z) if p['labels'] else None,responses=[r['response'] for r in z if r['group']==g]) for g in groups})
wr('diagnostic_metrics.json',dict(conditions=metrics,interpretation='GT-selected known-case diagnosis; not independent accuracy. Captions unscored, must inspect verbatim. Robot proxy excludes background and may not describe the intended object in mixed crops.'))
protected={p:sha(p) for p in v['protected_before']};assert protected==v['protected_before']
assert json.loads(subprocess.check_output([str(B/'env/bin/python'),'-m','pip','list','--format=json'],text=True))==json.loads(v['packages_before'])
assert all(sha(B/'model'/p)==h for p,h in rd(R/'model_identity.json')['checkpoint_hashes'].items())
assert v['source_before']=={p.name:sha(p) for p in (R/'.execution').glob('*.py')}
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
v.update(stage='COMPLETE_VERIFIED',protected_after=protected,all_protected_SHA_unchanged=True,environments_unchanged=True,model_processor_unchanged=True,initial_cache_empty_verified=50,raw_sha256_after_GT=sha(R/'verifier_raw_responses.json'),predictions_sha256_after_GT=sha(R/'verifier_predictions.csv'),production_changed=False);wr('verification.json',v)
with ExperimentLogger(team='integration',model='Task10I_Qwen_perception_diagnostic_scoring',seed=42,job_type='evaluate',root=R/'.execution/experiments',config=rd(R/'protocol.json')) as logger:
 logger.log({k:dict(correct=z['correct'],n=z['n']) for k,z in metrics.items() if z['scored']});logger.link_output(R/'diagnostic_metrics.json')
files={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in R.rglob('*') if p.is_file() and p.name!='artifact_index.json' and not p.name.endswith('.log') and p.suffix!='.tmp'};wr('artifact_index.json',dict(status='COMPLETE_VERIFIED',files=files))
print('PERCEPTION_COMPLETE',json.dumps({k:dict(n=z['n'],correct=z['correct'],valid=z['valid']) for k,z in metrics.items()}),flush=True)
