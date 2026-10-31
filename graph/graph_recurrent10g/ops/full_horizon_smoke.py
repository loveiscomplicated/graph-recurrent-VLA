"""Independent 250-action engineering rollout, never a Blind Final seed."""
import sys,time,threading,subprocess
from pathlib import Path
root=Path(sys.argv[1]);source=Path(sys.argv[2]);sys.path.insert(0,str(source))
from graph.plan_a_bc.common import read,write,sha
from graph.graph_recurrent10g.orchestrator import verify_all_stages
from graph.graph_recurrent10g.eval_runner import load_model,vision,episode
verify_all_stages(root)
import graph.graph_recurrent10g.eval_runner as runtime
actual_backend={}
original_environment=runtime.environment
def audited_environment(path):
 env=original_environment(path);task=env.unwrapped
 actual_backend.update(backend=str(task.backend),device=str(task.device),control_freq=task.control_freq,physics_freq=task.sim_freq)
 return env
runtime.environment=audited_environment
choice=read(root/'model_selection_frozen.json')['conditions']['A']
cfg,m,norm,codec,ck=load_model(root,'A',root/choice['path']);detector,tracker=vision(root)
samples=[];stop=threading.Event()
def monitor():
 while not stop.is_set():
  line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip()
  samples.append(line);stop.wait(.5)
t=threading.Thread(target=monitor,daemon=True);t.start()
try:
 result=episode(root,root/'full_horizon_engineering_smoke',m,norm,codec,detector,tracker,
   cfg['evaluation']['dev_seeds'][0],choice['step'],250,smoke=True)
finally:stop.set();t.join(timeout=3)
assert result['actions']==250 or result['termination'] in ('success','termination','time_limit')
write(root/'full_horizon_smoke_verification.json',dict(status='PASS',result=result,memory_samples=samples,
  actual_backend=actual_backend,scope='engineering only, Development parent; same vision/policy/controller; no Blind Final access',
  harness_sha256=sha(Path(__file__))))
print('FULL-HORIZON ENGINEERING SMOKE PASS',result['actions'],flush=True)
