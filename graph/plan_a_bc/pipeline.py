"""Fail-closed sequential cache -> E0 -> E1 -> frozen Test12 experiment."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import os
import traceback

import torch

from utils.experiment_logger import ExperimentLogger
from .common import read,write,sha,digest,tree_identity,now,checkpoint_load


def run(root):
    root=Path(root).resolve();paths=read(root/'runtime_paths.json');cfg=read(root/'config.json')
    state=dict(status='RUNNING',phase='await_original_dataset',started_at=now(),stages={})
    write(root/'pipeline_state.json',state)
    with ExperimentLogger(team='integration',model='plan_a_cached_graph_bc_periodic_rollout',seed=42,
        config=dict(comparison_uuid=cfg.get('comparison_uuid',root.name),protocol=cfg,no_publish=True),
        root=root/'pipeline_runs',job_type='comparison') as log:
        write(root/'pipeline_run.json',dict(run_id=log.run_id,directory=str(log.directory)))
        def stage(name,args):
            if (root/'execution_source_manifest.json').exists():
                frozen=read(root/'execution_source_manifest.json')
                for relative,expected in frozen['files'].items():
                    if sha(Path(__file__).resolve().parents[2]/relative)!=expected:
                        raise ValueError('execution source changed after freeze: '+relative)
            state['phase']=name;state['stages'][name]=dict(status='RUNNING',started_at=now());write(root/'pipeline_state.json',state)
            print('STAGE_BEGIN',name,flush=True);started=time.perf_counter()
            with (root/(name+'.log')).open('x') as output:
                subprocess.run([sys.executable,'-u','-B','-m',*args],stdout=output,stderr=subprocess.STDOUT,check=True)
            state['stages'][name]=dict(status='PASS',wall_seconds=time.perf_counter()-started,ended_at=now())
            write(root/'pipeline_state.json',state);log.log(dict(completed_stages=len(state['stages'])));print('STAGE_PASS',name,flush=True)
        try:
            # Download is independently supervised; a dropped SSH session does
            # not interrupt it. It reads only an already-published original.
            deadline=time.monotonic()+3600
            while not (root/'dataset_transfer_verification.json').exists():
                if (root/'dataset_transfer_failure.json').exists():raise RuntimeError('original dataset transfer failed')
                if time.monotonic()>deadline:raise TimeoutError('dataset resource preparation exceeded one hour')
                time.sleep(10)
            if read(root/'dataset_transfer_verification.json')['status']!='PASS':raise ValueError('dataset hash gate')
            if read(root/'runtime_verification.json')['status']!='PASS':raise ValueError('simulator runtime gate')
            if read(root/'frozen09d_source_verification.json')['status']!='PASS':raise ValueError('frozen09D source gate')
            # All hyperparameters and seeds are already registered; only the
            # final implementation identity is sealed before any policy output.
            source=Path(__file__).resolve().parents[2];files=tree_identity(source)
            cfg.update(comparison_uuid=cfg.get('comparison_uuid',root.name[-32:]),
                source_sha256=digest(files),execution_source_frozen_at=now())
            write(root/'config.json',cfg)
            write(root/'execution_source_manifest.json',dict(files=files,sha256=digest(files)),immutable=True)
            stage('contracts',['unittest','graph.plan_a_bc.tests','graph.plan_a_geometry.tests','graph.plan_a_geometry.test_provenance','-v'])
            stage('geometry_regression',['graph.plan_a_bc.regression','--root',str(root)])
            common=['--root',str(root),'--dataset',paths['dataset'],'--dino',paths['dino'],'--sam',paths['sam'],'--device','cuda']
            stage('cache_pilot',['graph.plan_a_bc.cache',*common,'--pilot'])
            # Train96/Validation12 ONLY; no Test12 graphs enter fitting or selection.
            stage('cache_full',['graph.plan_a_bc.cache',*common])
            stage('training_smoke',['graph.plan_a_bc.train','--root',str(root),'--device','cuda','--smoke'])
            for variant in ['E0','E1']:
                stage('train_'+variant,['graph.plan_a_bc.train','--root',str(root),'--variant',variant,'--device','cuda'])
            selection={}
            for variant in ['E0','E1']:
                run=Path(read(root/(variant+'_run.json'))['directory']);status=read(run/'status.json')
                if status['status']!='COMPLETE' or status['updates']!=11487:raise ValueError('budget incomplete')
                selection[variant]={kind:read(run/(kind+'.json')) for kind in ['last','validation_best']}
                for receipt in selection[variant].values():
                    if sha(receipt['path'])!=receipt['sha256']:raise ValueError('selected checkpoint hash')
            write(root/'model_selection_frozen.json',dict(frozen_at=now(),models=selection,
                rule=cfg['training']['validation_best'],no_test_outputs_seen=True),immutable=True)
            stage('cache_final_test',['graph.plan_a_bc.cache',*common,'--final-test'])
            stage('offline_final_test',['graph.plan_a_bc.final','--root',str(root),'--offline-only'])
            for variant in ['E0','E1']:
                run=Path(read(root/(variant+'_run.json'))['directory'])
                for kind,receipt in selection[variant].items():
                    stage('test_'+variant+'_'+kind,['graph.plan_a_bc.rollout','--root',str(root),'--run',str(run),
                        '--checkpoint',receipt['path'],'--device','cuda','--final-kind',kind])
            state.update(status='EXPERIMENT_EXECUTED',phase='report',ended_at=now());write(root/'pipeline_state.json',state)
            stage('report',['graph.plan_a_bc.final','--root',str(root),'--report-only'])
            state.update(status='COMPLETE',phase='complete',ended_at=now());write(root/'pipeline_state.json',state)
        except BaseException as exc:
            state.update(status='FAILED',exception=type(exc).__name__,message=str(exc),ended_at=now())
            write(root/'pipeline_state.json',state)
            write(root/'pipeline_error.json',dict(error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            try:
                from .final import report
                report(root)
            except BaseException as report_exc:
                write(root/'report_error.json',dict(error_type=type(report_exc).__name__))
            raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)


if __name__=='__main__':main()
