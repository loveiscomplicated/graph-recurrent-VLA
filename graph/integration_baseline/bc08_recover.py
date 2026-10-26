"""Audited successor plan after an orchestration-only Task08 failure.

Completed Small5 runs remain immutable. No updates, seeds, codecs, initial weights
or datasets change. Full training receives the successor source identity.
"""
import argparse
import copy
from pathlib import Path

from graph.integration_baseline.bc08_codec import digest
from graph.integration_baseline.bc08_train import load_checkpoint,make_identity,normalizer_hash,tensor_hash,restricted_load
from graph.integration_baseline.direct_runtime import atomic,read,sha
from utils.experiment_logger import ExperimentLogger


ALLOWED_CHANGED={
    'graph/integration_baseline/bc08_pipeline.py',
    'graph/integration_baseline/bc08_eval.py',
    'graph/integration_baseline/bc08_report.py',
    'graph/integration_baseline/tests/test_bc08.py',
}


def audit_sources(old,root):
    root=Path(root).resolve();changed=[]
    for name,expected in old['source_hashes'].items():
        path=Path(name).resolve();rel=path.relative_to(root).as_posix()
        if not path.is_file():raise ValueError('source removed during recovery: '+rel)
        if sha(path)!=expected:
            if rel not in ALLOWED_CHANGED:raise ValueError('learning/data/architecture source changed: '+rel)
            changed.append(dict(path=rel,before=expected,after=sha(path)))
    return changed


def recover(old_path,*,maximum_parallel_training_jobs=2):
    old_path=Path(old_path).resolve();old_root=old_path.parent;old=read(old_path)
    source_root=Path(__file__).resolve().parents[2]
    for name,expected in old['input_hashes'].items():
        if sha(name)!=expected:raise ValueError('original frozen input changed: '+name)
    changes=audit_sources(old,source_root)
    state=read(old_root/'status.json')
    if state['status']!='ERROR' or set(state['stages'])!={'overfit_A','overfit_C'}:
        raise ValueError('recovery requires only completed overfit runs and no later executed stages')
    adopted={};extra_inputs={str(old_path):sha(old_path)}
    for method in ['A','C']:
        stage=state['stages']['overfit_'+method]
        directory=Path(stage['directory']);completion=read(directory/'completion.json')
        if stage['status']!='COMPLETE' or completion['updates']!=2000 or old['overfit_updates']!=2000:
            raise ValueError('complete fixed-budget Small5 result required')
        receipt_path=directory/'last_valid.json';receipt=read(receipt_path)
        cp,codec=load_checkpoint(receipt,make_identity(old,old_path,'overfit',method))
        if completion['selected']!=receipt or receipt['global_step']!=2000:
            raise ValueError('Small5 selected checkpoint/receipt mismatch')
        if normalizer_hash(cp['normalizer'])!=old['robot_normalizer_identity'] or codec.identity!=digest(old['codecs']['overfit'][method]):
            raise ValueError('Small5 normalizer/codec mismatch')
        adopted[method]=dict(receipt=receipt,identity=cp['identity'],directory=str(directory.resolve()),
            original_run_id=receipt['run_id'],completion_sha256=sha(directory/'completion.json'))
        for p in [receipt_path,Path(receipt['path']),directory/'completion.json',directory/'batch_identity.jsonl']:
            extra_inputs[str(p.resolve())]=sha(p)
    if sha(Path(adopted['A']['directory'])/'batch_identity.jsonl')!=sha(Path(adopted['C']['directory'])/'batch_identity.jsonl'):
        raise ValueError('Small5 A/C batch identity mismatch')
    init=restricted_load(old['initial_path'])
    if tensor_hash(init['model'])!=old['initial_weight_identity']:raise ValueError('initial weights changed')
    seed_path=old_root/'seeds_manifest.json';extra_inputs[str(seed_path)]=sha(seed_path)
    new=copy.deepcopy(old)
    if maximum_parallel_training_jobs not in (2,3):raise ValueError('supported maximum is two or explicitly authorized three')
    new.update(adopted_overfit=adopted,maximum_parallel_training_jobs=maximum_parallel_training_jobs,
        recovery=dict(parent_plan=str(old_path),parent_plan_sha256=sha(old_path),
        parent_comparison_run_id=old_root.name,reason='Notify tmux after worker process exit, including CUDA teardown; preserve completed Small5',
        changed_sources=changes,retrained_overfit=False,changed_learning_settings=False,
        execution_policy=dict(previous_maximum=old.get('maximum_parallel_training_jobs',2),requested_maximum=maximum_parallel_training_jobs,
            authorization='User explicitly requested concurrent Full A/B/C' if maximum_parallel_training_jobs==3 else 'original two-job limit')))
    new['input_hashes'].update(extra_inputs)
    new['source_hashes']={str(p.resolve()):sha(p) for folder in ['graph','recurrent','envs','visual/runtime','utils','tools']
        for p in sorted((source_root/folder).rglob('*.py'))}
    with ExperimentLogger(team='integration',model='task08_comparison_recovery',seed=42,config=new['config'],
        root=old['experiments_root'],job_type='comparison') as logger:
        root=logger.directory
        protected=read(old['protection_manifest']) if old.get('protection_manifest') else {}
        for directory in [old_root,*[Path(v['directory']) for v in adopted.values()]]:
            for p in directory.rglob('*'):
                if p.is_file():protected[str(p.resolve())]=sha(p)
        protection=root/'recovery_protected_before.json'
        atomic(protection,protected,immutable=True)
        new['protection_manifest']=str(protection.resolve());new['input_hashes'][str(protection.resolve())]=sha(protection)
        atomic(root/'plan.json',new,immutable=True)
        atomic(root/'recovery_audit.json',new['recovery'],immutable=True)
        atomic(root/'seeds_manifest.json',read(seed_path),immutable=True)
        gate=copy.deepcopy(state['resource_gate']);gate['reused_from_comparison']=old_root.name
        atomic(root/'resource_gate.json',gate,immutable=True)
        atomic(root/'status.json',dict(status='PREPARED',stage='overfit_closed_loop',stages=state['stages'],resource_gate=gate))
        for method in ['A','C']:
            marker=read(old_root/f'overfit-{method}-train.json');marker['status']='COMPLETE'
            atomic(root/f'overfit-{method}-train.json',marker,immutable=True)
        from graph.integration_baseline.bc08_report import snapshot
        snapshot(root,new)
        logger.record(source_run_id=old_root.name,recovery_audit=new['recovery'])
        logger.link_output(root/'plan.json',kind='metrics');logger.link_output(root/'recovery_audit.json',kind='metrics')
        print('PLAN='+str((root/'plan.json').resolve()),flush=True)
        return root/'plan.json'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--from-plan',type=Path,required=True)
    p.add_argument('--maximum-parallel-training-jobs',type=int,choices=[2,3],default=2)
    a=p.parse_args();recover(a.from_plan,maximum_parallel_training_jobs=a.maximum_parallel_training_jobs)


if __name__=='__main__':main()
