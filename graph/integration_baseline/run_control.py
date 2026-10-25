"""Task06-C independent tmux launcher and cooperative status/stop/resume CLI."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import traceback

from graph.integration_baseline.direct_runtime import (
    REPO, atomic, append, read, sha, now, lock, Events, StopRequested, stop_gate, check_space,
)


STAGES=['collection','vision','validation','training','evaluation','report']


def path_for(run):
    if not run or len(run)!=32 or any(c not in '0123456789abcdef' for c in run):
        raise ValueError('expected ExperimentLogger UUID hex')
    path=REPO/'experiments'/run
    if not path.is_dir():raise FileNotFoundError(path)
    return path


def settings():
    os.environ.update(WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')


def prepare(config_path,preflight,*,smoke=False):
    from utils.experiment_logger import ExperimentLogger
    from graph.integration_baseline.direct_errors import source_inventory
    settings();config=read(config_path)
    config['discord_enabled']=False # user explicitly waived Discord for this run
    config['discord_required']=False
    if smoke:
        config['fixture_only']=True
        config['collection'].update(target_successes=2,max_attempts=4,start_seed=29000,gate_episodes=2)
        config['splits']={'train':2,'validation':0,'test':0}
    logger=ExperimentLogger(team='integration',model='graph_gnn_gru_direct_expert_bc',seed=config['seed'],config=config,job_type='prepare')
    run=logger.directory
    for directory in ['collection','vision','training','errors','plots','logs','stages','preflight']:
        (run/directory).mkdir()
    preflight=Path(preflight)
    for name in ['protected.json','git_status.txt','callback_actual.json','callback_cli.log','source_protection.json','automation_verification.json']:
        if (preflight/name).exists():shutil.copy2(preflight/name,run/'preflight'/name)
    actual=read(preflight/'callback_actual.json') if (preflight/'callback_actual.json').exists() else {}
    automation=read(preflight/'automation_verification.json') if (preflight/'automation_verification.json').exists() else {}
    direct_pass=actual.get('returncode')==0 and actual.get('proof')=={'callback_fixture':'PASS'}
    callback_pass=direct_pass and (smoke or automation.get('actual_tmux_dispatch') is True)
    atomic(run/'callback_verification.json',dict(actual_cli_success=callback_pass,
        fixture_result=actual,cli_version=subprocess.check_output(['codex','--version'],text=True).strip(),
        sandbox='workspace-write',approval_policy='never',automation=automation),immutable=True)
    atomic(run/'discord_verification.json',dict(status='USER_DISABLED',actual_delivery=False),immutable=True)
    source=source_inventory();atomic(run/'source_hashes.json',source,immutable=True)
    for name in source:
        target=run/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    pilot=read('experiments/9fec4f02fc9d4172a48e97d5eff90abd/data_pipeline_costs.json')
    # Values are pilot measurements, not a promise of full-episode speed/memory.
    estimated_obs=120*(669/5+1);maximum_obs=160*251
    budget=dict(free_bytes=shutil.disk_usage(run).free,pilot_actions=669,pilot_observations=674,
        pilot_hdf5_bytes=26009353,pilot_collect_seconds=121.296,pilot_mask_frames=16,pilot_vision_seconds=39.594996083993465,
        estimated_success_observations=estimated_obs,estimated_hdf5_bytes=estimated_obs*26009353/674,
        estimated_graph_packet_bytes=estimated_obs*635559/16,
        upper_attempt_observations=maximum_obs,upper_hdf5_bytes_at_pilot_ratio=maximum_obs*26009353/674,
        estimated_collect_seconds=120*121.296/5,estimated_vision_seconds=estimated_obs*39.594996083993465/16,
        limitations='Prefix Vision point counts, SAM2 cache and late occlusion may change cost. Training duration unknown.',
        reserve_bytes=config['minimum_free_bytes'],pilot_costs_sha256=sha('experiments/9fec4f02fc9d4172a48e97d5eff90abd/data_pipeline_costs.json'))
    atomic(run/'preflight/cost_estimate.json',budget,immutable=True)
    # Pilot and official evaluation sources occupy 2000..2029; current normal pilot uses 2000..2004.
    from recurrent.common import load_config
    evaluation=load_config('experiment.yaml')['eval']
    heldout=set(range(evaluation['start_seed'],evaluation['start_seed']+evaluation['episodes']))
    pilot_seeds={e['seed'] for e in read('experiments/9fec4f02fc9d4172a48e97d5eff90abd/pilot_manifest.json')['episodes']}
    selected=set(range(config['collection']['start_seed'],config['collection']['start_seed']+config['collection']['max_attempts']))
    if selected&(heldout|pilot_seeds):raise ValueError('collection seeds overlap prior evaluation/pilot')
    atomic(run/'preflight/seed_audit.json',dict(pilot=sorted(pilot_seeds),evaluation=sorted(heldout),
        collection_start=min(selected),collection_end=max(selected),overlap=[],normal=True,noise=0,hole_shift=None),immutable=True)
    atomic(run/'status.json',dict(status='PREPARED',stage='preflight',run_id=run.name,time=now()))
    logger.record(pipeline_run_id=run.name,launch_state='PREPARED',discord='USER_DISABLED',source_hashes_path=str(run/'source_hashes.json'))
    logger.finish()
    print(json.dumps(dict(run_id=run.name,path=str(run),fixture_only=smoke)),flush=True)
    return run


def source_gate(run,*,allow_repair=False):
    from graph.integration_baseline.direct_errors import source_inventory,ALLOWED_REPAIR,changed
    before=read(run/'source_hashes.json');after=source_inventory();difference=changed(before,after)
    if difference:
        verified=list((run/'errors').glob('*/repair_verified.json'))
        if not allow_repair or not verified or not difference.issubset(ALLOWED_REPAIR):
            raise ValueError('source hashes changed since frozen run: '+str(sorted(difference)))
        approved=read(verified[-1])['source_hashes']
        if after!=approved:raise ValueError('source differs from verified repair revision')


def launch(run,*,resume=False,after_repair=False):
    settings();config=read(run/'config.yaml');check_space(run,config)
    source_gate(run,allow_repair=after_repair or bool(list((run/'errors').glob('*/repair_verified.json'))))
    if config['auto_callback_required'] and not read(run/'callback_verification.json')['actual_cli_success']:
        atomic(run/'status.json',dict(status='BLOCKED_AUTO_CODEX_CALLBACK',run_id=run.name,stage='preflight'))
        raise RuntimeError('BLOCKED_AUTO_CODEX_CALLBACK')
    if not config.get('fixture_only'):
        gate=read(run/'preflight/verification.json') if (run/'preflight/verification.json').exists() else {}
        if gate.get('status')!='PASS':raise RuntimeError('real collection/vision/training smoke and automation tests must pass before launch')
    session=('grvla-06c-smoke-' if config.get('fixture_only') else 'grvla-06c-')+run.name[:8]
    exists=subprocess.run(['tmux','has-session','-t','='+session],capture_output=True).returncode==0
    if exists:raise RuntimeError('session already exists: '+session)
    if resume:
        resume_gate(run)
        if (run/'STOP').exists():
            archive=run/'stages'/('stop-cleared-'+str(os.stat(run/'STOP').st_mtime_ns)+'.json')
            atomic(archive,dict(time=now(),reason='explicit resume command'),immutable=True)
            (run/'STOP').unlink()
    elif (run/'launches.jsonl').exists():raise RuntimeError('previous launch exists; use resume')
    command=[sys.executable,'-m','graph.integration_baseline.run_control','supervise','--run',run.name]
    if after_repair:command+=['--after-repair']
    script=run/'launch.sh'
    text='#!/bin/sh\nset -eu\ncd '+shlex.quote(str(REPO))+'\n'
    text+='export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1\n'
    text+='export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg\n'
    text+='exec '+shlex.join(command)+' >> '+shlex.quote(str(run/'logs/supervisor.log'))+' 2>&1\n'
    # Launch scripts are revisioned as well; never rewrite a previous command.
    if script.exists():script=run/f'launch-{len(list(run.glob("launch*.sh")))}.sh'
    script.write_text(text);script.chmod(0o700)
    subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(REPO),shlex.join(['/bin/sh',str(script)])],check=True)
    append(run/'launches.jsonl',dict(time=now(),session=session,resume=resume,script=str(script.relative_to(run))))
    atomic(run/'tmux.json',dict(session=session,launch_script=str(script.relative_to(run)),time=now()))
    print(json.dumps(dict(run_id=run.name,tmux_session=session,status='LAUNCHED')),flush=True)


def stage(run,name):
    from graph.integration_baseline.direct_dataset import collect,freeze_split,process_vision,DirectDataset
    from graph.integration_baseline.direct_train import gate,train,offline_test
    from utils.experiment_logger import ExperimentLogger
    settings();config=read(run/'config.yaml')
    os.environ.pop('DISCORD_WEBHOOK_URL',None) if not config.get('discord_enabled',False) else None
    events=Events(run,required=config['discord_required'])
    logger=ExperimentLogger(team='integration',model='graph_gnn_gru_direct_expert_bc',seed=config['seed'],
        config=dict(pipeline_run_id=run.name,stage=name,parameters=config),job_type='train' if name=='training' else 'evaluate')
    logger.record(source_run_id=run.name,stage=name,pipeline_run_id=run.name)
    append(run/'execution_runs.jsonl',dict(stage=name,run_id=logger.run_id,time=now(),pid=os.getpid()))
    try:
        torch_config(config)
        stop_gate(run)
        if name=='collection':collect(run,config,events);freeze_split(run,config)
        elif name=='vision':
            process_vision(run,config,events,frame_limit=32 if config.get('fixture_only') else None)
        elif name=='validation':
            dataset=DirectDataset(run,smoke=config.get('fixture_only',False))
            dataset.validate(partial=config.get('fixture_only',False))
            report=gate(run,config,dataset,smoke=config.get('fixture_only',False))
            events.emit('dataset_validation_complete',training_gate=report)
        elif name=='training':train(run,config,DirectDataset(run),events,logger)
        elif name=='evaluation':offline_test(run,config,DirectDataset(run),events,logger)
        elif name=='report':
            from graph.integration_baseline.direct_report import generate
            generate(run,config);events.emit('final_artifacts_complete',path=str(run/'graph_gru_training.html'))
        atomic(run/'stages'/f'{name}.json',dict(status='COMPLETE',exit_code=0,stage=name,time=now(),execution_run_id=logger.run_id),immutable=True)
    except StopRequested:
        logger.record(normal_stop=True);return 75
    except BaseException as exc:
        from tools.discord_notifier import Sanitizer
        clean=Sanitizer()
        atomic(run/'stage_failure.json',dict(error_type=type(exc).__name__,traceback=clean(traceback.format_exc()),stage=name,time=now()))
        logger.finish(failed=True);raise
    finally:events.close();logger.finish()
    return 0


def torch_config(config):
    import torch
    torch.set_num_threads(config['threads'])
    if config['device']!='cpu' or config['vision_device']!='cpu':
        raise ValueError('CPU is the frozen executed backend; no automatic fallback')


def supervise(run,after_repair=False):
    """Blocking child wait, not a periodic Codex/status watcher."""
    from graph.integration_baseline.direct_errors import bundle,dispatch
    settings();config=read(run/'config.yaml')
    if not config.get('discord_enabled'):os.environ.pop('DISCORD_WEBHOOK_URL',None)
    events=Events(run,required=config['discord_required']);active_stage='preflight'
    with lock(run/'process.lock'):
        try:
            source_gate(run,allow_repair=after_repair or bool(list((run/'errors').glob('*/repair_verified.json'))))
            stages=STAGES[:3] if config.get('fixture_only') else STAGES
            for name in stages:
                active_stage=name
                if (run/'stages'/f'{name}.json').exists():continue
                source_gate(run,allow_repair=after_repair or bool(list((run/'errors').glob('*/repair_verified.json'))))
                stop_gate(run)
                atomic(run/'status.json',dict(status='RUNNING',stage=name,run_id=run.name,supervisor_pid=os.getpid(),time=now()))
                command=[sys.executable,'-m','graph.integration_baseline.run_control','stage','--run',run.name,'--stage',name]
                with (run/'logs'/f'{name}.log').open('a') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=REPO)
                    atomic(run/'status.json',dict(status='RUNNING',stage=name,run_id=run.name,supervisor_pid=os.getpid(),stage_pid=child.pid,time=now()))
                    def request_stop(*_):atomic(run/'STOP',dict(time=now(),reason='supervisor signal'))
                    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
                    code=child.wait() # OS exit event, no polling or timed Codex dispatch
                append(run/'stages/exits.jsonl',dict(stage=name,exit_code=code,time=now()))
                if code==75:raise StopRequested()
                if code!=0:
                    raise ChildProcessError(f'stage {name} exited with code {code}')
                if not (run/'stages'/f'{name}.json').exists():raise RuntimeError('missing required stage artifact')
            atomic(run/'status.json',dict(status='SMOKE_COMPLETE' if config.get('fixture_only') else 'COMPLETE',
                stage='done',run_id=run.name,time=now(),exit_code=0))
            events.emit('pipeline_complete',fixture_only=config.get('fixture_only',False))
            return 0
        except StopRequested:
            atomic(run/'status.json',dict(status='STOPPED',stage=active_stage,run_id=run.name,time=now(),exit_code=75))
            events.emit('normal_stop',stage=active_stage);return 0
        except BaseException as exc:
            atomic(run/'status.json',dict(status='FAILED',stage=active_stage,run_id=run.name,time=now(),exit_code=1,error_type=type(exc).__name__))
            failure_code=code if 'code' in locals() and code!=0 else 1
            error_bundle=bundle(run,active_stage,failure_code)
            events.emit('fatal_error',stage=active_stage,error_type=type(exc).__name__,bundle=str(error_bundle))
            # Fixtures validate failure handling separately and never launch repair on their own errors.
            if not config.get('fixture_only'):
                try:dispatch(run,error_bundle,events)
                except BaseException as dispatch_error:
                    atomic(run/'status.json',dict(status='BLOCKED_AUTO_CODEX_CALLBACK',stage=active_stage,
                        run_id=run.name,error_type=type(dispatch_error).__name__,bundle=str(error_bundle),time=now()))
                    events.emit('callback_dispatch_failed',error_type=type(dispatch_error).__name__,bundle=str(error_bundle))
            return 1
        finally:events.close()


def resume_gate(run):
    from graph.integration_baseline.direct_dataset import receipts,validate_receipt,DirectDataset
    from graph.integration_baseline.direct_train import optimizer,identity,restore_checkpoint
    from graph.integration_baseline.model import IntegrationPolicy
    config=read(run/'config.yaml');torch_config(config)
    if read(run/'status.json')['status'] in ('COMPLETE','SMOKE_COMPLETE'):
        raise RuntimeError('completed runs cannot be resumed')
    if (run/'robot_state_schema.json').exists():
        schema=read(run/'robot_state_schema.json')
        for e in receipts(run):validate_receipt(run,e,schema)
    if (run/'vision_manifest.json').exists():
        dataset=DirectDataset(run)
        dataset.validate()
        if (run/'training/last_valid.json').exists():
            model=IntegrationPolicy(dataset.schema,**config['model']);opt,scheduler=optimizer(model,config)
            restore_checkpoint(run,'last_valid',model,opt,scheduler,identity(run,config,dataset))
        elif ((run/'stages/training.json').exists() or (run/'training/progress.json').exists() or
              ((run/'stage_failure.json').exists() and read(run/'stage_failure.json').get('stage')=='training')):
            raise RuntimeError('previous training without checkpoint; fresh reinitialization is forbidden')
    return True


def main():
    os.chdir(REPO);settings()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','launch','status','stop','resume','supervise','stage','repair','resume-gate','export'])
    p.add_argument('--run');p.add_argument('--stage',choices=STAGES);p.add_argument('--bundle',type=Path)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('direct_config.json'))
    p.add_argument('--preflight',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--after-repair',action='store_true')
    p.add_argument('--callback-fixture',action='store_true')
    p.add_argument('--episode',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.command=='prepare':prepare(args.config,args.preflight,smoke=args.smoke);return 0
    if args.command=='export':
        from graph.integration_baseline.direct_dataset import export_npz
        export_npz(args.episode,args.output);return 0
    run=path_for(args.run)
    if args.command=='status':
        print(json.dumps(read(run/'status.json'),indent=2))
        for filename in ['collection/progress.json','training/progress.json','training/best.json']:
            if (run/filename).exists():print(filename,json.dumps(read(run/filename)))
    elif args.command=='stop':atomic(run/'STOP',dict(time=now(),reason='user CLI stop'));print('Cooperative stop requested; current episode/chunk will finalize.')
    elif args.command in ('launch','resume'):launch(run,resume=args.command=='resume',after_repair=args.after_repair)
    elif args.command=='supervise':return supervise(run,args.after_repair)
    elif args.command=='stage':return stage(run,args.stage)
    elif args.command=='resume-gate':resume_gate(run);print('RESUME_GATE_PASS')
    elif args.command=='repair':
        from graph.integration_baseline.direct_errors import repair,callback_fixture
        if args.callback_fixture:
            if not read(run/'config.yaml').get('fixture_only'):raise RuntimeError('callback fixture requires a fixture run')
            callback_fixture(args.bundle);return 0
        events=Events(run)
        try:repair(run,args.bundle,events)
        except BaseException as exc:
            atomic(run/'status.json',dict(status='NEEDS_USER_INTERVENTION',stage='repair',error_type=type(exc).__name__,time=now()))
            events.emit('recovery_failed',error_type=type(exc).__name__,bundle=str(args.bundle));raise
        finally:events.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
