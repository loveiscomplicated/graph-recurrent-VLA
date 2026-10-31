"""Guarded 10-F data/cache -> fixed A/B -> frozen final live evaluation."""
import argparse,os,subprocess,sys,time
from pathlib import Path
import h5py,numpy as np
from graph.plan_a_bc.common import read,write,sha
from graph.corrective_reach10f.recovery_contracts import verify
from utils.experiment_logger import ExperimentLogger


def data_gate(root):
    assert read(root/'full_data_complete.json')['status']=='PASS'
    assert read(root/'full_cache/corrective_cache_manifest.json')['status']=='PASS'
    assert read(root/'matched_hold_v4_completion_gate.json')['status']=='PASS'
    protected=read(root/'original_raw_protection.json');assert protected['status']=='PASS'
    for row in protected['branches']:
        original=Path('/workspace/task10f_corrective_reach/attempt04/raw')/row['branch']/'episode.h5'
        assert sha(original)==row['sha256'],'original failed pilot raw changed'
    checks={};all_hold=[]
    for name,count in [('additional10mm',40),('development',32)]:
        folder=root/name;receipts=read(folder/'recovery_pilot_manifest.json')
        assert len(receipts)==count and all(r['approved'] for r in receipts) and not (folder/'failure.json').exists()
        output=root/(name+'_raw_contracts.json')
        if not output.exists():verify(folder,output)
        assert read(output)['pairs']==count and read(output)['status']=='PASS'
        for r in receipts:
            with h5py.File(folder/'raw'/r['branch']/'N/episode.h5') as f:
                segment=f['segment'][:];a=int((segment==0).sum());d=int((segment==1).sum());x=f['tcp_pose'][:]
                error=float(np.linalg.norm(x[a+d,:3]-x[a,:3]));assert error<.001
                all_hold.append(dict(group=name,branch=r['branch'],matched_duration_steps=d,anchor_error_m=error))
        checks[name]=dict(pairs=count,raw_contracts_sha256=sha(output),matched_hold_max_m=max(x['anchor_error_m'] for x in all_hold if x['group']==name))
    assert len(all_hold)==72
    payload=dict(status='PASS',checks=checks,matched_hold=all_hold,
         full_cache_manifest_sha256=sha(root/'full_cache/corrective_cache_manifest.json'),plan_sha256=sha(root/'full_experiment_plan.json'))
    destination=root/'full_data_contracts.json'
    if destination.exists():assert read(destination)==payload
    else:write(destination,payload,immutable=True)


def run(root,dino,sam):
    assert os.getenv('CONTAINER_ID')=='57640710' and os.getenv('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    root=root.resolve();started=time.monotonic()
    with ExperimentLogger(team='graph',model='Task10F_guarded_full_continuation',seed=42,config=dict(plan_sha256=sha(root/'full_experiment_plan.json')),
                          root=root/'experiments',job_type='train') as log:
        identity=root/'continuation_identity.json';attempt=2
        while identity.exists():
            # Keep every failed attempt and its ExperimentLogger record.
            identity=root/f'continuation_attempt{attempt}_identity.json';attempt+=1
        write(identity,dict(run_id=log.run_id,plan_sha256=sha(root/'full_experiment_plan.json')),immutable=True)
        while not ((root/'full_data_complete.json').exists() and (root/'full_cache/corrective_cache_manifest.json').exists()):
            if time.monotonic()-started>12*3600:raise TimeoutError('data/cache gate did not complete within 12h')
            for n in ['additional10mm/failure.json','development/failure.json','nominal/failure.json','full_cache/failure.json']:
                if (root/n).exists():raise RuntimeError('required source/cache gate failed: '+n)
            for n,receipt in [('full_data','full_data_complete.json'),('full_cache','full_cache/corrective_cache_manifest.json')]:
                if (root/receipt).exists():continue
                status=subprocess.check_output(['supervisorctl','-c',str(root/'supervisor.conf'),'status',n],text=True)
                if 'EXITED' in status or 'FATAL' in status:raise RuntimeError(n+' stopped before completion: '+status)
            time.sleep(30)
        data_gate(root);write(root/'continuation_state.json',dict(stage='DATA_CACHE_PASS',timestamp=time.time()))
        log.record(data_cache_contract='PASS',data_contract_sha256=sha(root/'full_data_contracts.json'))
        subprocess.run([sys.executable,'-u','-B','-m','graph.corrective_reach10f.finetune','--root',str(root)],check=True)
        assert (root/'model_selection_frozen.json').exists()
        write(root/'continuation_state.json',dict(stage='MODEL_SELECTION_FROZEN',timestamp=time.time()))
        subprocess.run([sys.executable,'-u','-B','-m','graph.corrective_reach10f.offline_diagnostics','--root',str(root)],check=True)
        subprocess.run([sys.executable,'-u','-B','-m','graph.corrective_reach10f.recovery_eval','--root',str(root),'--dino',str(dino),'--sam',str(sam)],check=True)
        assert read(root/'final_evaluation/evaluation_complete.json')['status']=='PASS'
        write(root/'continuation_state.json',dict(stage='FINAL_EVALUATION_COMPLETE',timestamp=time.time(),
              model_selection_sha256=sha(root/'model_selection_frozen.json'),evaluation_sha256=sha(root/'final_evaluation/evaluation_complete.json')))
        log.link_output(root/'continuation_state.json',kind='metrics')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dino',type=Path,required=True);p.add_argument('--sam',type=Path,required=True);a=p.parse_args();run(a.root,a.dino,a.sam)
