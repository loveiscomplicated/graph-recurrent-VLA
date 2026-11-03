"""Finite remote experiment DAG; no API credentials, training or remote publishing."""
from common import *
import subprocess
import time


def run(name,script,*args):
    status=read(ROOT/'pipeline_status.json') if (ROOT/'pipeline_status.json').exists() else dict(completed=[])
    if name in status['completed']:return
    status.update(active=name,updated_utc=now());write(ROOT/'pipeline_status.json',status)
    with (ROOT/(name+'.log')).open('a') as log:
        r=subprocess.run([sys.executable,str(SOURCE/script),*map(str,args)],stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:
        status.update(active=None,failed=name,exit_code=r.returncode,updated_utc=now());write(ROOT/'pipeline_status.json',status)
        raise RuntimeError(f'{name} failed with exit {r.returncode}; preserved log/partial outputs')
    status['completed'].append(name);status.update(active=None,updated_utc=now());write(ROOT/'pipeline_status.json',status)


def freeze_masks(arm):
    path=ROOT/(arm+'_masks_freeze.json')
    if path.exists():
        for name,h in read(path)['files'].items():assert sha(ROOT/name)==h
        return
    files={}
    for kind in ('static','tracking'):
        summary=ROOT/(f'sam_{kind}_{arm}.json');assert summary.exists();files[summary.name]=sha(summary)
        for result in (ROOT/'masks'/arm/kind).glob('*/result.json'):
            r=read(result);assert sha(result.parent/'masks.npz')==r['file_sha256']
            files[str(result.relative_to(ROOT))]=sha(result)
            files[str((result.parent/'masks.npz').relative_to(ROOT))]=r['file_sha256']
    write(path,dict(files=files,frozen_utc=now()))


def graphs(arm):
    name='graphs_'+arm
    status=read(ROOT/'pipeline_status.json')
    if name in status['completed']:return
    status.update(active=name,updated_utc=now());write(ROOT/'pipeline_status.json',status)
    processes=[];handles=[]
    for shard in range(4):
        log=(ROOT/(f'graph_{arm}_{shard}.log')).open('a');handles.append(log)
        processes.append(subprocess.Popen([sys.executable,str(SOURCE/'graph_worker.py'),'quality','--arm',arm,'--shard',str(shard),'--shards','4'],
                                         stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes]
    for h in handles:h.close()
    if codes!=[0]*4:
        status.update(active=None,failed=name,exit_codes=codes,updated_utc=now());write(ROOT/'pipeline_status.json',status)
        raise RuntimeError(f'{name} worker failure {codes}')
    subprocess.run([sys.executable,str(SOURCE/'graph_worker.py'),'merge','--arm',arm,'--shards','4'],check=True)
    status['completed'].append(name);status.update(active=None,updated_utc=now());write(ROOT/'pipeline_status.json',status)


def evaluate():
    name='evaluate_GT'
    status=read(ROOT/'pipeline_status.json')
    if name in status['completed']:return
    status.update(active=name,updated_utc=now());write(ROOT/'pipeline_status.json',status)
    processes=[];handles=[]
    for shard in range(4):
        log=(ROOT/(f'evaluate_{shard}.log')).open('a');handles.append(log)
        processes.append(subprocess.Popen([sys.executable,str(SOURCE/'evaluate.py'),'--shard',str(shard),'--shards','4'],stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes]
    for h in handles:h.close()
    assert codes==[0]*4,codes
    subprocess.run([sys.executable,str(SOURCE/'evaluate.py'),'--merge','--shards','4'],check=True)
    status['completed'].append(name);status.update(active=None,updated_utc=now());write(ROOT/'pipeline_status.json',status)


def main():
    os.environ.pop('OPENAI_API_KEY',None)
    assert_source();assert read(ROOT/'capture_manifest.json')['status']=='COMPLETE'
    assert sha(ROOT/'api_predictions.json')==read(ROOT/'api_freeze.json')['predictions_sha256']
    if not (ROOT/'pipeline_status.json').exists():write(ROOT/'pipeline_status.json',dict(completed=[],started_utc=now()))
    run('sam_static_actual','sam_worker.py','static','--arm','actual')
    run('sam_tracking_actual','sam_worker.py','tracking','--arm','actual');freeze_masks('actual')
    graphs('actual')
    run('oracle_jobs','graph_worker.py','oracle_jobs')
    run('sam_static_gt_box','sam_worker.py','static','--arm','gt_box')
    run('sam_tracking_gt_box','sam_worker.py','tracking','--arm','gt_box');freeze_masks('gt_box')
    graphs('gt_box');graphs('gt_mask')
    # All quality workers have exited before any official latency measurement.
    for arm in ('actual','gt_box'):
        run('timing_static_'+arm,'timing.py','static','--arm',arm)
        run('timing_tracking_'+arm,'timing.py','tracking','--arm',arm)
    run('timing_graph','timing.py','graph')
    evaluate()
    run('verify_remote','verify.py','remote')
    status=read(ROOT/'pipeline_status.json');status.update(active=None,status='COMPLETE',finished_utc=now());write(ROOT/'pipeline_status.json',status)
    print('REMOTE_EXPERIMENT_COMPLETE',flush=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-masks');a=ap.parse_args()
    if a.freeze_masks:freeze_masks(a.freeze_masks)
    else:main()
