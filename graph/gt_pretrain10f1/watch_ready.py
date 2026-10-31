"""Local lightweight DATA_READY bridge using existing project event primitives.

Normal progress makes zero model calls. The source instance is strictly read-only.
A sealed dataset is validated, copied locally and to57987772, then one verified
CLI callback can own final dataset freeze/pretraining/evaluation. Not a scheduler
for source collection, and never a source process controller.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

from . import ready_contract


def inherited_ops(path):
    spec=importlib.util.spec_from_file_location('task10f1_existing_event_ops',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def ssh(cfg,host,code,timeout=45):
    endpoint=cfg[host]
    cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15',
         '-p',str(endpoint['port']),endpoint['host'],code]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError('SSH_READ_FAILED_'+str(p.returncode))
    return p.stdout


def remote_probe(cfg):
    code='from pathlib import Path; import json,hashlib; p=Path('+repr(cfg['source']['root'])+')/"gnn_dataset_ready.json"; '
    code+='print(json.dumps(dict(status="WAITING_DATA",marker_exists=False) if not p.exists() else dict(marker_exists=True,marker_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),marker=json.loads(p.read_text()))))'
    out=ssh(cfg,'source','python3 -c '+shlex.quote(code))
    return json.loads(next(line for line in reversed(out.splitlines()) if line.startswith('{')))


def remote_validate(cfg):
    code=Path(ready_contract.__file__).read_text()
    code+='\nprint(json.dumps(dict(validation=validate('+repr(cfg['source']['root'])+','+repr(cfg['official_encoder_sha256'])+'), marker=read_json(Path('+repr(cfg['source']['root'])+')/"gnn_dataset_ready.json"))))\n'
    command=cfg['source']['python_setup']+' -c '+shlex.quote(code)
    out=ssh(cfg,'source',command,timeout=180)
    return json.loads(next(line for line in reversed(out.splitlines()) if line.startswith('{')))


def rsync_cmd(endpoint):
    return 'ssh -o BatchMode=yes -o ConnectTimeout=15 -p '+str(endpoint['port'])


def transfer(directory,cfg,event_id,validated):
    dest=directory/'imports'/event_id;partial=dest.with_name(dest.name+'.partial')
    if dest.exists():
        check=ready_contract.validate(dest,cfg['official_encoder_sha256'])
        if check['marker_sha256']!=validated['validation']['marker_sha256']:raise ValueError('local transfer identity conflict')
    else:
        partial.mkdir(parents=True,exist_ok=True)
        names=[row['path'] for row in validated['marker']['files']]+['gnn_dataset_ready.json']
        if any('\n' in n or '\r' in n for n in names):raise ValueError('unsafe files-from name')
        listing=directory/'transfers'/f'{event_id}.files';listing.parent.mkdir(exist_ok=True)
        listing.write_text('\n'.join(names)+'\n')
        ep=cfg['source']
        subprocess.run(['rsync','-az','--files-from',str(listing),'-e',rsync_cmd(ep),
            ep['host']+':'+ep['root']+'/',str(partial)+'/'],check=True,timeout=1200)
        check=ready_contract.validate(partial,cfg['official_encoder_sha256'])
        if check['marker_sha256']!=validated['validation']['marker_sha256']:raise ValueError('source marker changed in transfer')
        partial.rename(dest)
    # Only sealed files copied; no source directory traversal and no source writes.
    target=cfg['target']['root']+'/imports/'+event_id
    ssh(cfg,'target','mkdir -p '+shlex.quote(target))
    ep=cfg['target']
    subprocess.run(['rsync','-az','-e',rsync_cmd(ep),str(dest)+'/',ep['host']+':'+target+'/'],check=True,timeout=1200)
    code=Path(ready_contract.__file__).read_text()
    code+='\nprint(json.dumps(validate('+repr(target)+','+repr(cfg['official_encoder_sha256'])+')))\n'
    out=ssh(cfg,'target',cfg['target']['python_setup']+' -c '+shlex.quote(code),timeout=180)
    remote=json.loads(next(line for line in reversed(out.splitlines()) if line.startswith('{')))
    if remote['marker_sha256']!=check['marker_sha256']:raise ValueError('target transfer identity mismatch')
    return dict(status='PASS',local_path=str(dest),target_path=target,
        source_marker_sha256=validated['validation']['marker_sha256'],local_marker_sha256=check['marker_sha256'],
        target_marker_sha256=remote['marker_sha256'],files=remote['files_verified'],
        bytes=remote['bytes_verified'],local_readback='PASS',target_readback='PASS',source_modified=False)


def dispatch(directory,cfg,event,ops,launcher=None,busy=None):
    lock=ops.claim_lock(directory/'dispatch.lock')
    if lock is None:return 'LOCKED'
    try:
        journal=ops.read(directory/'invocation_journal.json',dict(events={},total_calls=0))
        eid=event['event_id']
        if eid in journal['events']:return journal['events'][eid]['status']
        verification=ops.read(directory/'callback_verification.json',{})
        if verification.get('status')!='PASS':return 'HANDOFF_ONLY_CALLBACK_UNVERIFIED'
        if busy is None:busy=ops.main_busy(ops.read(directory/'conversation_context.json'))
        if busy:return 'PENDING_MAIN_IDLE'
        if os.getenv('TASK10F1_EVENT_CALLBACK')=='1':return 'RECURSION_BLOCKED'
        if journal['total_calls']>=cfg['max_ready_callbacks']:return 'CALL_LIMIT_BLOCKED'
        # Durable claim before even launching; a crash or failed invocation cannot
        # silently create a second handler for the same dataset content.
        dest=directory/'callbacks'/eid;dest.mkdir(parents=True,exist_ok=True)
        journal['events'][eid]=dict(status='CLAIMED',at=ops.stamp(),directory=str(dest))
        journal['total_calls']+=1;ops.write(directory/'invocation_journal.json',journal)
        handoff=dict(event=event,original_requests=str(directory/'original_user_requests.txt'),
            preparation_handoff=cfg['preparation_handoff'],source_instance=57640710,target_instance=57987772,
            target_root=cfg['target']['root'],transfer=ops.read(directory/'transfers'/f'{eid}.json'),
            guard='Existing Task10F collection/training processes are read-only. Do not duplicate any existing F1 stage; no GRU/action training, commit, publishing, messages, recursive Codex, or new watchers.')
        ops.write(dest/'handoff.json',handoff)
        prompt=f'''Authorized Task10-F1 DATA_READY continuation for event {eid}. Own this event exactly once.
Read VERBATIM {directory/'original_user_requests.txt'} (initial instruction AND latest steering), repository AGENTS.md and required docs, {cfg['preparation_handoff']}, then {dest/'handoff.json'}.
Dataset READY is not pretraining completion. First check latest main lifecycle and invocation ownership. Inspect live F1 status on57987772; preserve completed Full-task caches/engineering and wait for ongoing own data preparation without duplicating it. Task10-F57640710 source and all its processes are read-only.
Reverify imported files/GT/splits, runtime/camera/geometry compatibility and source identity. Different current torch runtimes were observed (F1 2.14.1+cu130 vs F 2.8.0+cu128): do not silently merge. Resolve with actual compatibility evidence or a separate coherent dataset/runtime, reporting limitations. No guessed GT or centroid labels. Exclude failed legacy Pilot.
Only after actual support is established, freeze FINAL dataset composition/parent splits/normalization/budget in new release files, preserving initial plans. Use existing E0 L2/d64 Node21/Edge4 and5x64 Readout320, Spatial Head, absolute and valid same-state delta loss, CUDA Packed Graphs and ExperimentLogger. Implement any remaining validated dataset adapter before training. Do not perform architecture/loss search or train GRU/action modules.
Run remaining real pretraining, held-out absolute/displacement/baselines, oracle/predicted38 and same-checkpoint A/B extraction, frozen hashes/costs, full artifacts/readback and report in docs/visual-graph-integration/10f1_gt_supervised_gnn_pretraining.md. Read the entire original completion criteria. A callback, smoke, successful exit or watcher is not completion.
If a required contract is unavailable, preserve evidence and leave a precise BLOCKED handoff instead of fabricating data or starting arbitrary training. Write findings.json in {dest}; do not invoke Codex or a new watcher, spawn agents, send messages, commit or publish. Never stop/restart source Task10-F jobs.'''
        (dest/'prompt.txt').write_text(prompt)
        if launcher:
            launcher(event,dest);journal['events'][eid]['status']='FIXTURE_DISPATCHED'
        else:
            env=dict(os.environ,TASK10F1_EVENT_CALLBACK='1')
            for key in list(env):
                if key in ('OPENAI_API_KEY','CODEX_API_KEY','DISCORD_WEBHOOK_URL') or (key.startswith('CODEX_') and key!='CODEX_HOME'):env.pop(key,None)
            env['PATH']=str(Path(cfg['codex_executable']).parent)+os.pathsep+env.get('PATH','')
            proc=subprocess.Popen(ops.callback_command(cfg,dest),stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,env=env,start_new_session=True)
            from tools.discord_notifier import Sanitizer
            clean=Sanitizer({k:v for k,v in os.environ.items() if len(v)>=8})
            def capture():
                with (dest/'stdout_sanitized.jsonl').open('w') as f:
                    for line in proc.stdout:f.write(clean(line));f.flush()
            threading.Thread(target=capture,daemon=True).start();ops.RUNNING_CALLBACKS[proc.pid]=proc
            proc.stdin.write(prompt);proc.stdin.close()
            journal['events'][eid].update(status='STARTED',pid=proc.pid)
            ops.write(directory/'active_callback.json',dict(pid=proc.pid,event_id=eid,directory=str(dest)))
        ops.write(directory/'invocation_journal.json',journal)
        return journal['events'][eid]['status']
    except Exception as exc:
        if 'journal' in locals() and event['event_id'] in journal['events']:
            journal['events'][event['event_id']].update(status='LAUNCH_FAILED',error_type=type(exc).__name__)
            ops.write(directory/'invocation_journal.json',journal)
        return 'LAUNCH_FAILED'
    finally:lock.close()


def reap(directory,ops):
    active=ops.read(directory/'active_callback.json')
    if not active:return
    proc=ops.RUNNING_CALLBACKS.get(active['pid'])
    if proc is not None:
        rc=proc.poll()
        if rc is None:return
    else:
        try:os.kill(active['pid'],0);return
        except ProcessLookupError:rc=None
    journal=ops.read(directory/'invocation_journal.json')
    journal['events'][active['event_id']].update(status='EXITED' if rc==0 else 'CALLBACK_FAILED' if rc is not None else 'EXIT_STATUS_UNKNOWN',exit_code=rc,ended_at=ops.stamp())
    ops.write(directory/'invocation_journal.json',journal);ops.write(directory/'last_callback.json',active)
    (directory/'active_callback.json').unlink()


def tick(directory,cfg,ops):
    snapshot=remote_probe(cfg);state=ops.read(directory/'watch_state.json',{})
    status='WAITING_DATA';marker=snapshot.get('marker',{})
    if snapshot.get('marker_exists') and marker.get('status')=='DATA_READY':
        sig=snapshot['marker_sha256']
        if state.get('candidate_sha256')!=sig:
            state.update(candidate_sha256=sig,candidate_since=time.time());status='WAITING_STABLE_MARKER'
        elif time.time()-state['candidate_since']<cfg['stability_seconds']:status='WAITING_STABLE_MARKER'
        else:
            checked=ops.read(directory/'last_validation.json',{})
            if checked.get('marker_sha256')!=sig:
                try:
                    payload=remote_validate(cfg)
                    checked=dict(status='PASS',marker_sha256=sig,payload=payload)
                except Exception as exc:checked=dict(status='NOT_READY',marker_sha256=sig,error_type=type(exc).__name__)
                ops.write(directory/'last_validation.json',checked)
            if checked['status']!='PASS':status='WAITING_VALID_CONTRACT'
            else:
                payload=checked['payload']
                content=[(row['path'],row['sha256'],row['bytes']) for row in payload['marker']['files']]
                eid=ops.fingerprint(['Task10F1_DATA_READY',57640710,marker['dataset_id'],sorted(content)])
                ep=directory/'events'/f'{eid}.json'
                if not ep.exists():
                    event=dict(event_id=eid,kind='DATA_READY',created_at=ops.stamp(),dataset_id=marker['dataset_id'],
                        marker_sha256=sig,validation=payload['validation'],source_instance=57640710)
                    ops.write(ep,event)
                    with (directory/'event_journal.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                event=ops.read(ep);proof=directory/'transfers'/f'{eid}.json'
                if not proof.exists():ops.write(proof,transfer(directory,cfg,eid,payload))
                status=dispatch(directory,cfg,event,ops)
    state.update(status=status,at=ops.stamp(),source=snapshot,pid=os.getpid(),normal_progress_model_calls=0,
        actual_READY_invocations=ops.read(directory/'invocation_journal.json',{}).get('total_calls',0),
        source_process_control=False)
    ops.write(directory/'watch_state.json',state);reap(directory,ops);return status


def run(directory,once=False):
    directory=Path(directory).resolve();cfg=json.loads((directory/'watch_config.json').read_text());ops=inherited_ops(cfg['event_ops'])
    lock=ops.claim_lock(directory/'watcher.lock')
    if lock is None:raise SystemExit('Task10F1 watcher already owns lock')
    if not (directory/'invocation_journal.json').exists():
        ops.write(directory/'invocation_journal.json',dict(events={},total_calls=0))
    ops.write(directory/'watcher_pid.json',dict(pid=os.getpid(),started_at=ops.stamp()))
    while True:
        try:tick(directory,cfg,ops)
        except Exception as exc:ops.write(directory/'watch_error.json',dict(error_type=type(exc).__name__,at=ops.stamp(),source_modified=False))
        if once:return
        time.sleep(cfg['poll_seconds'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--once',action='store_true')
    a=p.parse_args();run(a.directory,a.once)
