"""Independent supervised continuation through data freeze, A/B and final test."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback
from graph.plan_a_bc.common import read,write,sha,now


def stage(root,name,args):
    write(root/'pipeline_state.json',dict(status='RUNNING',phase=name,updated_at=now()))
    print('STAGE',name,flush=True)
    with (root/(name+'.log')).open('a') as f:
        subprocess.run([sys.executable,'-u','-B',*args],stdout=f,stderr=subprocess.STDOUT,check=True)


def run(root):
    if (root/'pipeline_state.json').exists() and read(root/'pipeline_state.json').get('status')=='COMPLETE':return
    write(root/'pipeline_state.json',dict(status='RUNNING',phase='data',updated_at=now()))
    while not (root/'data_pipeline_complete.json').exists():
        if (root/'data_pipeline_failure.json').exists():raise RuntimeError('data producer failed; preserve partial data')
        time.sleep(5)
    stage(root,'freeze',['-m','graph.geometry_reach10d.freeze','--root',str(root)])
    stage(root,'contracts',['-m','graph.geometry_reach10d.contracts','--root',str(root)])
    if not (root/'training_smoke_verification.json').exists():
        stage(root,'smoke',['-m','graph.geometry_reach10d.train','--root',str(root),'--smoke'])
    for condition in ['A','B']:
        if not (root/(condition+'_complete.json')).exists():
            stage(root,'train_'+condition,['-m','graph.geometry_reach10d.train','--root',str(root),'--condition',condition])
    a,b=[read(root/(c+'_complete.json')) for c in ['A','B']]
    assert a['updates']==b['updates']==11487
    assert a['initial_weight_sha256']==b['initial_weight_sha256']
    assert a['total_timesteps']==b['total_timesteps']
    selected={c:read(root/(c+'_selected.json')) for c in ['A','B']}
    for cp in selected.values():assert sha(cp['path'])==cp['sha256']
    if not (root/'model_selection_frozen.json').exists():
        write(root/'model_selection_frozen.json',dict(created_at=now(),selected=selected,
            criterion='first minimum validation Method B loss; final data not accessed',
            dataset_ab_sha256=sha(root/'dataset_ab_manifest.json')),immutable=True)
    for condition in ['A','B']:
        stage(root,'final_'+condition,['-m','graph.geometry_reach10d.evaluate','--root',str(root),
            '--checkpoint',selected[condition]['path'],'--condition',condition,'--mode','final'])
    stage(root,'report',['-m','graph.geometry_reach10d.report','--root',str(root)])
    assert read(root/'verification.json')['full_experiment_complete']
    write(root/'pipeline_state.json',dict(status='COMPLETE',phase='complete',completed_at=now(),
        report=str(root/'10d_geometry_conditioned_reach_bc.md')))
    print('TASK10D COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:run(a.root)
    except BaseException as e:
        current=read(a.root/'pipeline_state.json') if (a.root/'pipeline_state.json').exists() else {}
        write(a.root/'pipeline_state.json',dict(current,status='FAILED',exception=type(e).__name__,message=str(e),updated_at=now()))
        write(a.root/'pipeline_failure.json',dict(exception=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
        try:subprocess.run([sys.executable,'-B','-m','graph.geometry_reach10d.report','--root',str(a.root)],timeout=120)
        except BaseException:pass
        raise
