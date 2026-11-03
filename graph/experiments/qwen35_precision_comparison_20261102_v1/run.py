from common import *
import subprocess,time,traceback
PY=str(B/'env/bin/python')
assert rd(P/'verification.json')['stage']=='ALL_INPUTS_FROZEN'
# One invocation owns the complete sequence, and never restarts a child.
with (P/'.execution/COMPARISON_STARTED').open('x') as f:f.write('single sequential execution\n')
results={}
for mode in VARIANTS:
 env=dict(os.environ,QWEN_PRECISION_VARIANT=mode)
 print('BEGIN_VARIANT',mode,flush=True)
 with (P/mode/'.execution/infer.log').open('w') as log:
  result=subprocess.run([PY,'-B','-u',str(P/'.execution/infer.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
 v=rd(P/mode/'verification.json')
 results[mode]=dict(returncode=result.returncode,stage=v['stage'],model_loads=v['model_loads'],grounding_requests=v.get('grounding_requests',v.get('completed_requests',0)),raw_sha256=sha(P/mode/'full_image_grounding_raw_responses.json') if (P/mode/'full_image_grounding_raw_responses.json').exists() else None,predictions_sha256=sha(P/mode/'full_image_grounding_predictions.csv') if (P/mode/'full_image_grounding_predictions.csv').exists() else None,error=v.get('error'))
 wr(P/'execution_progress.json',results);print('END_VARIANT',mode,json.dumps(results[mode]),flush=True)
# No semantic GT or saved DINO output has been opened by any preceding program.
wr(P/'all_predictions_frozen.json',dict(variants=results,GT_opened=False,DINO_opened=False,all_inference_processes_terminated=True))
for mode in VARIANTS:
 if results[mode]['grounding_requests']!=99 or results[mode]['predictions_sha256'] is None:continue
 env=dict(os.environ,QWEN_PRECISION_VARIANT=mode)
 with (P/mode/'.execution/evaluate.log').open('w') as log:
  result=subprocess.run([PY,'-B','-u',str(P/'.execution/evaluate.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
 results[mode]['evaluation_returncode']=result.returncode
 print('EVALUATION',mode,result.returncode,flush=True)
 wr(P/'execution_progress.json',results)
v=rd(P/'verification.json');v.update(stage='INFERENCE_AND_EVALUATION_FINISHED',results=results);wr(P/'verification.json',v)
print('THREE_PRECISION_EXECUTION_FINISHED',flush=True)
