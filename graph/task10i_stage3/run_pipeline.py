"""tmux-owned Stage3 coordinator. Invariants fail closed; no model/publisher calls."""
import argparse
from collections import Counter
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .core import STAGE2,POOL,RUNTIME_ROOT,read,write,freeze,sha,lock,verify_receipt,allocations,verified
from utils.experiment_logger import ExperimentLogger

def stage(root,name,status='RUNNING',**extra):
    write(root/'pipeline_state.json',dict(status=status,stage=name,updated=time.time(),pid=os.getpid(),**extra))
    print(name,status,flush=True)

def source_identity():
    source=Path(__file__).resolve().parents[2]
    return {str(p.relative_to(source)):sha(p) for base in ('graph','recurrent','envs','utils')
            for p in (source/base).rglob('*.py')}

def inputs(root):
    receipt=verify_receipt();freeze(root/'stage3_input_identity.json',dict(receipt=receipt,stage2_root=str(STAGE2),pool=str(POOL),instance=57987772))
    freeze(root/'stage2_selection_identity_verified.json',dict(status='PASS',files=receipt['files'],receipt_sha256=sha(STAGE2/'stage3_input_receipt.json')))
    # Every historical raw and GT source remains read-only, including excluded episodes.
    inventory=read(STAGE2/'existing_source_file_inventory.json')
    files=inventory.get('files',inventory)
    for path,entry in files.items():
        if not isinstance(entry,dict) or 'sha256' not in entry:continue
        p=Path(path);assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],path
    freeze(root/'existing_source_identity_verified.json',dict(status='PASS',inventory_sha256=sha(STAGE2/'existing_source_file_inventory.json'),files=files))
    plan=allocations();freeze(root/'parent_branch_collection_plan.json',plan)
    # Reuse Stage1.5's complete canonical tree fingerprint; do not regenerate candidates.
    original=read(STAGE2/'input_pool_inventory.json')
    for rel,entry in original.items():
        assert (POOL/rel).stat().st_size==entry['bytes'] and sha(POOL/rel)==entry['sha256'],rel
    freeze(root/'canonical_pool_identity_verified.json',dict(status='PASS',files=len(original),inventory_sha256=sha(STAGE2/'input_pool_inventory.json')))
    checkpoints={}
    for base in (Path('/workspace/task10f1_gt_pretrain_20261030'),RUNTIME_ROOT):
        for path in base.rglob('*.pt'):
            checkpoints[str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
    freeze(root/'existing_checkpoint_identity.json',checkpoints)
    # Exercise the complete contract reader on all105 preserved logical episodes
    # before any research collection, rather than trusting counts or SHA alone.
    from .assembly import audit_episode
    preserved=[dict(e,source='existing_'+e['source']) for e in read(STAGE2/'existing_episode_preservation_manifest.json')['episodes'] if e['disposition']=='PRESERVE']
    nominal={e['parent_id']:e for e in preserved if e['branch']=='nominal'}
    schema=read(RUNTIME_ROOT/'normalizer.json')['schema']
    rows=[audit_episode(root/'engineering_existing_readback',e,schema,nominal) for e in preserved]
    freeze(root/'existing_contract_readback.json',dict(status='PASS',episodes=len(rows),rows=rows))
    return plan

def request(root,parents,workers,label,crash=None):
    """Independent processes, disjoint ownership, bounded completion and log files."""
    root.mkdir(parents=True,exist_ok=True);procs=[];started=time.perf_counter()
    for i in range(workers):
        pp=root/f'{label}_request_{i}.json';freeze(pp,dict(parents=parents[i::workers]))
        log=(root/f'{label}_worker_{i}.log').open('a')
        env=os.environ.copy()
        if crash is not None:env['TASK10I_ENGINEERING_CRASH_AT']=str(crash)
        p=subprocess.Popen([sys.executable,'-B','-u','-m','graph.task10i_stage3.worker','--root',str(root),'--request',str(pp)],
            stdout=log,stderr=subprocess.STDOUT,env=env)
        procs.append((p,log))
    write(root/f'{label}_live_handles.json',dict(started=time.time(),pids=[p.pid for p,_ in procs],workers=workers))
    write(root/'candidate_attempt_cursor.json',dict(label=label,pids=[p.pid for p,_ in procs],
        issued_parents=[p['candidate']['id'] for p in parents],status='RUNNING',retries=0))
    # Waiting belongs to this independent coordinator, not a Codex background terminal.
    codes=[]
    try:
        for p,log in procs:codes.append(p.wait(timeout=7200));log.close()
    except BaseException:
        # Leave live worker handles for an explicit resume audit. Do not duplicate them.
        write(root/f'{label}_observation_error.json',dict(pids=[p.pid for p,_ in procs],traceback=traceback.format_exc()))
        raise
    elapsed=time.perf_counter()-started
    receipts=[verified(p.parent) for p in root.glob('parents/*/*/receipt.json')]
    result=dict(exit_codes=codes,wall_seconds=elapsed,workers=workers,parents=len(parents),
        committed_actions=sum(r['actions'] for r in receipts),
        committed_bytes=sum(p.stat().st_size for p in root.glob('parents/*/*/*') if p.is_file()),
        isolated_processes=True,disjoint_parent_ownership=True)
    write(root/f'{label}_execution.json',result)
    write(root/'candidate_attempt_cursor.json',dict(label=label,status='WORKERS_TERMINAL',exit_codes=codes,
        issued_parents=[p['candidate']['id'] for p in parents],retries=0))
    if crash is None:assert all(c==0 for c in codes),result
    else:assert codes==[91],result
    return result

def results(root,parents):
    return [read(root/'parents'/p['candidate']['id']/'parent_result.json') for p in parents]

def passed(root,parents):
    rr=results(root,parents)
    assert all(r['status']=='PASS' for r in rr),rr
    return rr

def compare_raw(a,b,parents):
    import h5py
    import numpy as np
    count=0
    for p in parents:
        cid=p['candidate']['id']
        for branch in p['branches']:
            pa=a/'parents'/cid/branch;pb=b/'parents'/cid/branch
            verified(pa);verified(pb)
            with h5py.File(pa/'episode.h5') as x,h5py.File(pb/'episode.h5') as y:
                def check(k,v):
                    if isinstance(v,h5py.Dataset):np.testing.assert_array_equal(v[:],y[k][:],err_msg=cid+'/'+branch+'/'+k)
                x.visititems(check)
            with np.load(pa/'gt_supervision.npz') as x,np.load(pb/'gt_supervision.npz') as y:
                assert x.files==y.files
                for k in x.files:np.testing.assert_array_equal(x[k],y[k],err_msg=k)
            count+=1
    return dict(status='PASS',episodes=count,all_sensor_action_GT_arrays_exact=True)

def engineering(root,plan):
    done=root/'engineering_gates.json'
    if done.exists():
        gate=read(done);assert gate['status']=='PASS'
        for rel,h in gate['evidence_sha256'].items():assert sha(root/rel)==h,rel
        for out in root.glob('engineering_*'):
            if out.is_dir():
                for p in out.glob('parents/*/*/receipt.json'):verified(p.parent)
        return gate
    with (root/'contract_tests.log').open('a') as log:
        subprocess.run([sys.executable,'-B','-m','unittest','graph.task10i_stage3.test_contracts','-v'],stdout=log,stderr=subprocess.STDOUT,check=True)
    # Fixed role representatives, including the largest real corrective template.
    values=list(plan.values())
    common=next(p for p in values if {a['dataset'] for a in p['allocations']}=={'R','D'} and len(p['branches'])==18)
    r=next(p for p in values if {a['dataset'] for a in p['allocations']}=={'R'})
    d=next(p for p in values if {a['dataset'] for a in p['allocations']}=={'D'})
    v=next(p for p in values if {a['dataset'] for a in p['allocations']}=={'validation'})
    smoke=[dict(candidate=p['candidate'],branches=list(dict.fromkeys(['full_task']+p['branches']))) for p in (common,r,d,v)]
    stage(root,'INITIAL_RESET_FULL_TASK_CORRECTIVE_SMOKE')
    sr=root/'engineering_smoke';request(sr,smoke,1,'smoke');passed(sr,smoke)
    # A real interrupted simulator episode, followed by exact reset reconstruction.
    stage(root,'SAVE_RESUME_MINIATURE_END_TO_END')
    small=[dict(candidate=common['candidate'],branches=['full_task','nominal'])]
    cr=root/'engineering_crash_resume'
    if not (cr/'crash_execution.json').exists():request(cr,small,1,'crash',crash=5)
    cid=small[0]['candidate']['id']
    if not (cr/'resume_execution.json').exists():
        assert not (cr/'parents'/cid/'full_task'/'receipt.json').exists()
        request(cr,small,1,'resume')
    passed(cr,small)
    saved=sha(cr/'parents'/cid/'full_task'/'receipt.json')
    request(cr,small,1,'verified_skip')
    assert sha(cr/'parents'/cid/'full_task'/'receipt.json')==saved
    assert list((cr/'parents'/cid/'interrupted').glob('full_task.*'))
    compare_raw(sr,cr,small)
    # Real three-worker benchmark first, then one and two. Identical candidates,
    # fresh runtimes and complete HDF5/GT readback under every concurrency setting.
    stage(root,'PARALLEL_COLLECTION_BENCHMARK')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.free,utilization.gpu','--format=csv'],capture_output=True,text=True,check=True).stdout
    other=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command=p.read_bytes()
            if b'handoff10h' in command or b'task10h_runner' in command:other.append(int(p.parent.name))
        except OSError:pass
    freeze(root/'benchmark_resource_context.json',dict(gpu_before=gpu,other_Task10H_pids=other,
        other_processes_interrupted=False,cpu_affinity_count=len(os.sched_getaffinity(0)),free_disk_bytes=shutil.disk_usage(root).free))
    bench=[dict(candidate=p['candidate'],branches=['full_task','nominal']) for p in (common,r,d)]
    measurements=[]
    for n in (3,1,2):
        out=root/f'engineering_benchmark_{n}'
        if (out/'benchmark_execution.json').exists():m=read(out/'benchmark_execution.json')
        else:m=request(out,bench,n,'benchmark')
        passed(out,bench);measurements.append(m)
    for n in (2,3):compare_raw(root/'engineering_benchmark_1',root/f'engineering_benchmark_{n}',bench)
    workers=min(measurements,key=lambda r:r['wall_seconds'])['workers']
    from .assembly import miniature
    miniature(root,sr,smoke)
    result=dict(status='PASS',full_task_states=4,roles=['common','R-exclusive','D-exclusive','validation'],
        largest_template_branches=18,actual_simulator_crash=True,verified_episode_skip=True,
        benchmark=measurements,workers=workers,parallel_exact_sensor_action_GT=True,
        engineering_excluded_from_research=True,miniature_end_to_end=True,
        evidence_sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and
            (p.name=='receipt.json' and any(s.startswith('engineering_') for s in p.relative_to(root).parts))})
    freeze(done,result);return result

def candidate_catalog():
    result={}
    for e in read(POOL/'eligible_candidate_pool.json'):
        result[e['candidate_id']]=dict(id=e['candidate_id'],seed=e['reset_seed'],kind='new',source=e)
    assert len(result)==300
    return result

def collect(root,plan,workers):
    stage(root,'EXPERT_VALIDATION_AND_CORRECTIVE_COLLECTION')
    primary=[dict(candidate=p['candidate'],branches=list(dict.fromkeys(['full_task']+p['branches']))) for p in plan.values()]
    request(root,primary,workers,'primary')
    initial_results=results(root,primary)
    # Runtime/alignment faults are infrastructure gates, never biological/task attrition.
    infra=[r for r in initial_results if r['status']=='FAIL' and r['reason'] in ('RUNTIME_ERROR','CONTROLLER_ERROR','DATA_ALIGNMENT_FAILURE','INITIAL_RESET_MISMATCH','VISIBILITY_REGRESSION')]
    assert not infra,('infrastructure or identity failure; investigate original evidence',infra)
    frozen=read(STAGE2/'stage3_demonstration_plan.json')['per_parent_allocations']
    mapping={dataset:{a['new_parent']:a['new_parent'] for a in frozen if a['dataset']==dataset} for dataset in ('R','D','validation')}
    failed={r['candidate_id'] for r in initial_results if r['status']=='FAIL'}
    queues=read(STAGE2/'stage3_collection_reserve_protocol.json')['queues'];events=[];catalog=candidate_catalog()
    allocations0=[a for p in plan.values() for a in p['allocations']]
    # Original slot order is the frozen per-parent allocation order, never a new selection.
    common=set(read(STAGE2/'sampling_trace.json')['common'][:44])
    def active(dataset):return set(mapping[dataset].values())
    def needed(slot,datasets):
        return list(dict.fromkeys(['full_task']+[b['branch'] for a in frozen if a['new_parent']==slot and a['dataset'] in datasets for b in a['branch_templates']]))
    stage(root,'RESERVE_REPLACEMENT_IF_NEEDED')
    # Durable event replay supports an interrupted coordinator without changing queues.
    if (root/'reserve_events.json').exists():
        events=read(root/'reserve_events.json')
        for e in events:
            if e['event']=='ASSIGNED':
                for dataset in e['datasets']:mapping[dataset][e['slot']]=e['candidate_id']
            if e['event']=='FAILED':failed.add(e['candidate_id'])
    roles=[('validation','validation'),('common','R'),('R','R'),('D','D')]
    while failed & (active('R')|active('D')|active('validation')):
      for role,dataset in roles:
        ordered=[]
        for a in frozen:
            slot=a['new_parent']
            if a['dataset']!=dataset or slot in ordered:continue
            if (role=='common')!=(slot in common) and role!='validation':continue
            ordered.append(slot)
        for slot in ordered:
            if mapping[dataset][slot] not in failed:continue
            # Shared physical parent failures are global; common slots replace both.
            assigned=False
            for rank,cid in enumerate(queues[role],1):
                used=active('R')|active('D') if role=='common' else active(dataset)
                reason='GLOBALLY_FAILED' if cid in failed else 'CURRENT_DATASET_OR_UNION' if cid in used else None
                datasets=['R','D'] if role=='common' else [dataset]
                event=dict(role=role,slot=slot,failed_parent=mapping[dataset][slot],reserve_rank=rank,candidate_id=cid,datasets=datasets)
                if reason:
                    if not any(e.get('role')==role and e.get('slot')==slot and e.get('candidate_id')==cid and e['event']=='SKIP' for e in events):
                        events.append(dict(**event,event='SKIP',reason=reason));write(root/'reserve_events.json',events)
                    continue
                required=needed(slot,datasets)
                # A shared tail reserve may already have an approved different template.
                request(root,[dict(candidate=catalog[cid],branches=required)],1,f'reserve_{role}_{slot}_{rank}')
                result=read(root/'parents'/cid/'parent_result.json')
                if result['status']=='FAIL':
                    if result['reason'] in ('RUNTIME_ERROR','CONTROLLER_ERROR','DATA_ALIGNMENT_FAILURE','INITIAL_RESET_MISMATCH','VISIBILITY_REGRESSION'):
                        raise RuntimeError('reserve infrastructure failure: '+str(result))
                    failed.add(cid);events.append(dict(**event,event='FAILED',reason=result['reason']))
                    write(root/'reserve_events.json',events);continue
                for ds in datasets:mapping[ds][slot]=cid
                events.append(dict(**event,event='ASSIGNED',branches=required))
                write(root/'reserve_events.json',events);assigned=True;break
            assert assigned,('RESERVE_EXHAUSTED',role,slot)
    freeze(root/'stage3_final_parent_selection.json',dict(mapping=mapping,failed=sorted(failed),events=events,
        original_selection_preserved=True,selection_identity_sha256=sha(STAGE2/'selection_identity.json')))
    return mapping

def execute(root,logger):
    stage(root,'INPUT_AND_SOURCE_VERIFICATION')
    freeze(root/'collector_source_identity.json',source_identity())
    if not (root/'run_identity.json').exists():freeze(root/'run_identity.json',dict(run_id=logger.run_id,instance=57987772))
    else:logger.record(source_run_id=read(root/'run_identity.json')['run_id'],resume=True)
    plan=inputs(root)
    gate=engineering(root,plan)
    from graph.corrective_reach10f.recovery_collect import SAFETY,FEEDBACK
    freeze(root/'expert_protocol_frozen.json',dict(source_sha256=sha(Path(__file__).resolve().parents[2]/'recurrent/expert.py'),
        horizon=250,control_hz=20,physics_hz=100,controller='pd_ee_delta_pose',physics='physx_cpu',renderer='sapien_cuda',
        success='installed official evaluate success; grasp and all expert phases recorded',retries=0,
        full_task_physics='finite states, joint limits, all5 physics substep contact separations; finger/peg compliance recorded separately with no artificial force cap'))
    freeze(root/'corrective_protocol_frozen.json',dict(SAFETY=SAFETY,FEEDBACK=FEEDBACK,anchors=[10,25],
        masks='exact approved nominal prefix BC on; disturbance off; fresh expert recovery on',
        solver_cache='not serialized; disk snapshot and 2x5-step observed replay required'))
    freeze(root/'collection_execution_manifest.json',dict(workers=gate['workers'],gates_sha256=sha(root/'engineering_gates.json'),
        original_parent_count=90,source_sha256=sha(root/'collector_source_identity.json'),automatic_infra_retries=0,
        engineering_excluded=True,all_later_stages_automatic=True))
    mapping=collect(root,plan,gate['workers'])
    from .assembly import assemble,finalize
    stage(root,'R_D_VALIDATION_DATASET_ASSEMBLY');assemble(root,plan,mapping)
    stage(root,'FULL_ALIGNMENT_AND_PHYSICS_AUDIT');finalize(root)
    for p in root.glob('*.json'):
        logger.link_output(p,kind='reports',checkpoint=p.with_name(p.name+'.tracking.json').exists())
    stage(root,'COMPLETED',status='COMPLETED',run_id=read(root/'run_identity.json')['run_id'])

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    assert os.environ.get('CONTAINER_ID')=='57987772'
    with lock(root/'run.lock'):
        if (root/'pipeline_state.json').exists() and read(root/'pipeline_state.json')['status']=='COMPLETED':
            from .assembly import audit_completion
            audit_completion(root);return
        # Refuse to duplicate surviving workers after loss of the coordinator.
        for handle in root.glob('*_live_handles.json'):
            for pid in read(handle)['pids']:
                proc=Path('/proc')/str(pid)/'cmdline'
                if proc.exists() and b'graph.task10i_stage3.worker' in proc.read_bytes():
                    raise RuntimeError(f'live worker {pid}: observe/rejoin original job before resume')
        with ExperimentLogger(team='graph',model='Task10I_Stage3_expert_raw_dataset',seed=42,job_type='prepare',root=root/'experiments',
            config=dict(stage2_root=str(STAGE2),pool_readonly=True,instance=57987772,training=False,publishing=False)) as logger:
            try:execute(root,logger)
            except BaseException as e:
                write(root/'pipeline_error.json',dict(error=str(e),traceback=traceback.format_exc()))
                stage(root,'INCOMPLETE',status='INCOMPLETE',error=str(e),run_id=logger.run_id)
                from .assembly import incomplete_report
                incomplete_report(root,str(e))
                raise

if __name__=='__main__':main()
