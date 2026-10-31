"""tmux-owned engineering gates followed by the entire authorized Full pipeline."""
import os,subprocess,sys
from pathlib import Path
base=Path('/workspace/task10g_f1_abc_20261031');source=base/'source';sys.path.insert(0,str(source))
from graph.plan_a_bc.common import read,write,sha,now
from graph.graph_recurrent10g.config import create
from graph.graph_recurrent10g.orchestrator import verify_all_stages
assert os.getenv('TMUX'),'driver must be owned by tmux'
mini=base/'mini_v5';full=base/'full';f1=Path('/workspace/task10f1_gt_pretrain_20261030')
def command(args,log):
 print('RUN',args,flush=True)
 with log.open('a') as out:
  subprocess.run([sys.executable,'-u','-B',*args],cwd=source,stdout=out,stderr=subprocess.STDOUT,check=True)
if not (mini/'config.json').exists():create(mini,f1,True,reuse_verified_data=base/'mini_v2')
status=read(mini/'pipeline_state.json')['status'] if (mini/'pipeline_state.json').exists() else 'PENDING'
if status not in ('ENGINEERING_INTERRUPTED','COMPLETED'):
 command(['-m','graph.graph_recurrent10g.orchestrator','--root',str(mini),'--stop-after','PARALLEL_TRAIN_A_B_C'],mini/'pipeline.log')
if not (mini/'miniature_restart_verification.json').exists():
 command([str(base/'miniature_restart_test.py'),str(mini),str(source)],mini/'restart_harness.log')
verify_all_stages(mini)
assert read(mini/'miniature_restart_verification.json')['status']=='PASS'
if not (mini/'full_horizon_smoke_verification.json').exists():
 command([str(base/'full_horizon_smoke.py'),str(mini),str(source)],mini/'horizon_smoke.log')
smoke=read(mini/'full_horizon_smoke_verification.json');assert smoke['status']=='PASS'
folder=mini/'full_horizon_engineering_smoke'
assert sha(folder/'trace.json')==smoke['result']['trace_sha256']
assert sha(folder/'rollout.mp4')==smoke['result']['video_sha256']
identity=read(mini/'source_identity.json')
for name,h in identity['hashes'].items():
 assert sha(source/name)==h,'source changed after miniature: '+name
if not (full/'config.json').exists():create(full,f1,False,miniature_root=mini)
write(full/'launch_gate_evidence.json',dict(status='PASS',at=now(),miniature_root=str(mini),
 miniature_completion_sha256=sha(mini/'completion_audit.json'),restart_sha256=sha(mini/'miniature_restart_verification.json'),
 full_horizon_sha256=sha(mini/'full_horizon_smoke_verification.json'),driver_sha256=sha(Path(__file__)),
 no_callback=True,no_agent_dependency=True))
command(['-m','graph.graph_recurrent10g.orchestrator','--root',str(full)],full/'pipeline.log')
verify_all_stages(full)
print('FULL AUTHORIZED TASK10G PIPELINE COMPLETED',flush=True)
