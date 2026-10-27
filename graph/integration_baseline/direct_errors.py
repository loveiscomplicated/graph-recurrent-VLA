"""One fatal event -> one sandboxed Codex repair session. No timer callbacks."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from uuid import uuid4

from graph.integration_baseline.direct_runtime import REPO, atomic, read, sha, now, lock
from tools.discord_notifier import Sanitizer


ALLOWED_REPAIR = {'graph/integration_baseline/direct_dataset.py',
    'graph/integration_baseline/direct_train.py','graph/integration_baseline/direct_runtime.py',
    'graph/integration_baseline/direct_errors.py','graph/integration_baseline/run_control.py',
    'graph/integration_baseline/direct_report.py','graph/integration_baseline/tests/test_direct_training.py'}


def source_inventory():
    paths=[*Path('graph').rglob('*.py'),*Path('recurrent').rglob('*.py'),*Path('envs').rglob('*.py'),
           *Path('utils').rglob('*.py'),*Path('tools').rglob('*.py'),*Path('visual/runtime').rglob('*.py')]
    for root in ['graph','recurrent','envs','utils','tools']:
        for pattern in ['*.toml','*.yaml','*.json','*.md']:
            paths.extend(Path(root).rglob(pattern))
    paths += [Path('AGENTS.md'),Path('README.md'),Path('visual/scripts/stream_v1.py')]
    return {str(p):sha(p) for p in sorted(paths) if p.is_file()}


def artifact_inventory(run):
    run=Path(run)
    files=[]
    for name in ['collection','vision','training']:
        files.extend(p for p in (run/name).rglob('*') if p.is_file())
    files.extend(run/name for name in ['config.yaml','dataset_manifest.json','split_manifest.json',
        'vision_manifest.json','robot_state_schema.json','expert_action_schema.json','dataset_validation.json'] if (run/name).exists())
    return {str(p.relative_to(run)):sha(p) for p in files}


def bundle(run,stage,code):
    run=Path(run);clean=Sanitizer()
    directory=run/'errors'/uuid4().hex;directory.mkdir(parents=True)
    failure=read(run/'stage_failure.json') if (run/'stage_failure.json').exists() else dict(error_type='NonzeroExit',traceback='stage exited without an exception record')
    status=read(run/'status.json')
    data=dict(failure,run_id=run.name,stage=stage,exit_code=code,time=now(),status=status,
              config_sha256=sha(run/'config.yaml'),reproduction=f'{sys.executable} -m graph.integration_baseline.run_control stage --run {run.name} --stage {stage}',
              source_hashes=source_inventory())
    for name,key in [('dataset_manifest.json','dataset_manifest'),('split_manifest.json','split_manifest'),
                     ('training/last_valid.json','last_valid_checkpoint'),('training/progress.json','training_progress')]:
        if (run/name).exists():data[key]=read(run/name)
    log=run/'logs'/f'{stage}.log'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-20000));data['recent_log']=f.read().decode(errors='replace')
    # Error bundles have no RGB-D payload. Sanitize every serialized string.
    atomic(directory/'bundle.json',json.loads(clean(json.dumps(data))),immutable=True)
    atomic(directory/'config.json',read(run/'config.yaml'),immutable=True)
    return directory/'bundle.json'


def dispatch(run,error_bundle,events,*,launcher=None,fixture=False):
    run=Path(run)
    with lock(run/'errors/dispatch.lock'):
        marker=run/'errors/callback_dispatch.json'
        if marker.exists() or os.environ.get('GRVLA_REPAIR_SESSION')=='1':
            events.emit('manual_intervention_required',reason='automatic repair budget consumed or recursive invocation blocked')
            return dict(status='BLOCKED_REPAIR_LIMIT')
        verification=read(run/'callback_verification.json') if (run/'callback_verification.json').exists() else {}
        if not verification.get('actual_cli_success'):
            atomic(run/'status.json',dict(status='BLOCKED_AUTO_CODEX_CALLBACK',stage='error',bundle=str(error_bundle)))
            events.emit('callback_blocked',bundle=str(error_bundle),reason='actual CLI preflight did not pass')
            return dict(status='BLOCKED_AUTO_CODEX_CALLBACK')
        session='grvla-06c-repair-'+run.name[:8]
        cmd=[sys.executable,'-m','graph.integration_baseline.run_control','repair','--run',run.name,'--bundle',str(error_bundle)]
        if fixture:cmd+=['--callback-fixture']
        command=shlex.join(cmd)+' >> '+shlex.quote(str(run/'errors/repair.log'))+' 2>&1'
        config=read(run/'config.yaml') if (run/'config.yaml').exists() else {}
        if config.get('runtime_bin_directory'):
            command='export PATH='+shlex.quote(config['runtime_bin_directory'])+':"$PATH"; '+command
        record=dict(status='DISPATCHED',session=session,bundle=str(error_bundle),attempt=1,time=now())
        # Record before launch; a launch failure consumes the attempt and cannot loop.
        atomic(marker,record,immutable=True)
        if launcher is None:
            subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(REPO),command],check=True)
        else:launcher(session,cmd)
        events.emit('recovery_dispatch',session=session,bundle=str(error_bundle),attempt=1)
        return record


def changed(before,after):
    return {p for p in before.keys()|after.keys() if before.get(p)!=after.get(p)}


def codex_command(cwd,result,schema):
    return ['codex','exec','--ignore-user-config','--sandbox','workspace-write','-c','approval_policy="never"',
            '--ephemeral','--color','never','-C',str(cwd),'--output-schema',str(schema),'-o',str(result),'-']


def callback_fixture(error_bundle):
    """Explicit initial automation fixture; no real training failure or repair."""
    directory=Path(error_bundle).parent;workspace=directory/'fixture_workspace';workspace.mkdir()
    atomic(workspace/'bundle.json',dict(error_type='INDEPENDENT_AUTOMATION_FIXTURE',exit_code=7),immutable=True)
    schema=directory/'fixture_schema.json';result=directory/'fixture_result.json'
    atomic(schema,dict(type='object',additionalProperties=False,properties=dict(callback_fixture=dict(type='string')),
                       required=['callback_fixture']),immutable=True)
    prompt='Read bundle.json only. This is an explicitly authorized isolated error callback fixture, not a real training failure. Write proof.json containing exactly {"callback_fixture":"PASS"}. Do not spawn agents, change any other file, inspect secrets or call network tools. Return {"callback_fixture":"PASS"} matching the output schema.'
    env=os.environ.copy();env.pop('DISCORD_WEBHOOK_URL',None)
    from tools.discord_notifier import Sanitizer
    proc=subprocess.run(codex_command(workspace,result,schema),input=prompt,text=True,capture_output=True,env=env,timeout=90)
    (directory/'fixture_cli.log').write_text(Sanitizer()(proc.stdout+'\n'+proc.stderr))
    passed=(proc.returncode==0 and result.exists() and read(result)=={'callback_fixture':'PASS'}
            and (workspace/'proof.json').exists() and read(workspace/'proof.json')=={'callback_fixture':'PASS'})
    atomic(directory/'fixture_verification.json',dict(status='PASS' if passed else 'FAILED',actual_cli_success=passed,
        exit_code=proc.returncode,sandbox='workspace-write',structured_output=True),immutable=True)
    if not passed:raise RuntimeError('actual sandboxed Codex callback fixture failed')


def repair(run,error_bundle,events):
    """Require a structured diagnosis, allowed-file audit and independent tests."""
    run=Path(run);directory=Path(error_bundle).parent
    with lock(run/'errors/repair.lock'):
        os.environ['GRVLA_REPAIR_SESSION']='1'
        events.emit('recovery_start',bundle=str(error_bundle))
        before=source_inventory();protected=artifact_inventory(run)
        atomic(directory/'protected_artifacts.json',protected,immutable=True)
        result=directory/'repair_result.json';schema=directory/'response_schema.json'
        atomic(schema,dict(type='object',additionalProperties=False,
            properties={k:dict(type=v) for k,v in {'confirmed_implementation_bug':'boolean','safe_to_resume':'boolean',
                'diagnosis':'string','reproduction_test':'string','fix_summary':'string'}.items()},
            required=['confirmed_implementation_bug','safe_to_resume','diagnosis','reproduction_test','fix_summary']),immutable=True)
        prompt=f'''Task06-C ERROR EVENT, explicitly authorized maximum ONE repair attempt.
Read AGENTS.md and required repository docs, then {error_bundle}.
Diagnose and reproduce the failure. Only confirmed implementation bugs in the direct
collector/loader/trainer/runner may be fixed. Allowed source paths: {sorted(ALLOWED_REPAIR)}.
Never modify expert, Vision algorithms, graph/GNN/GRU architecture, loss, action or
input schema, config/hyperparameters, split, dataset, checkpoints, thresholds, or existing artifacts.
Never install dependencies, commit, publish, send messages, spawn agents, or call Codex.
Do not start/resume/monitor training yourself. Performance and speed are not bugs.
Add a minimal regression to graph/integration_baseline/tests/test_direct_training.py
that reproduces this exact confirmed bug; run it plus the existing integration contracts.
If uncertain, unsupported backend, OOM requiring config change, corrupt data, or incomplete
checkpoint recovery, set safe_to_resume=false. Return JSON matching the supplied schema.
Your output does not itself authorize resume: the independent runner rechecks contracts,
source changes, dataset/checkpoint integrity. Do not inspect secrets or RGB-D unless
strictly necessary locally; never put their contents in output.'''
        env=os.environ.copy();env.pop('DISCORD_WEBHOOK_URL',None)
        cmd=codex_command(REPO,result,schema)
        log=directory/'codex.log'
        try:
            proc=subprocess.run(cmd,input=prompt,text=True,capture_output=True,env=env,timeout=1800)
            log.write_text(Sanitizer()(proc.stdout+'\n'+proc.stderr))
        except subprocess.TimeoutExpired as exc:
            text=(exc.stdout or b'')+(exc.stderr or b'')
            if isinstance(text,bytes):text=text.decode(errors='replace')
            log.write_text(Sanitizer()(text));raise
        after=source_inventory();modifications=changed(before,after)
        atomic(directory/'source_changes.json',dict(before=before,after=after,changed=sorted(modifications)),immutable=True)
        outcome=read(result) if result.exists() else {}
        if proc.returncode or not outcome.get('confirmed_implementation_bug') or not outcome.get('safe_to_resume'):
            raise RuntimeError('repair could not establish a confirmed safe implementation fix')
        if not modifications or not modifications.issubset(ALLOWED_REPAIR):
            raise RuntimeError('repair changed forbidden sources or made no source fix')
        for name,digest in protected.items():
            if not (run/name).is_file() or sha(run/name)!=digest:
                raise RuntimeError('repair modified a protected dataset/checkpoint/config artifact')
        if not outcome.get('reproduction_test'):raise RuntimeError('missing reproduction test')
        with (directory/'tests.log').open('x') as output:
            tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','graph/integration_baseline/tests','-t','.', '-v'],
                                  stdout=output,stderr=subprocess.STDOUT,env=env,timeout=600)
        if tests.returncode:raise RuntimeError('repair contract tests failed')
        regression=Path('graph/integration_baseline/tests/test_direct_training.py')
        if str(regression) not in modifications:
            raise RuntimeError('repair did not add the required reproducing regression test')
        # This gate runs in a fresh process and verifies dataset, checkpoint and exact config.
        with (directory/'resume_gate.log').open('x') as output:
            gate=subprocess.run([sys.executable,'-m','graph.integration_baseline.run_control','resume-gate','--run',run.name],
                                 stdout=output,stderr=subprocess.STDOUT,env=env,timeout=600)
        if gate.returncode:raise RuntimeError('post-repair integrity/resume gate failed')
        atomic(directory/'repair_verified.json',dict(status='PASS',tests=True,resume_gate=True,source_hashes=after),immutable=True)
        events.emit('recovery_success',changed_files=sorted(modifications))
        # Separate tmux execution; one-use dispatch marker remains for the run forever.
        subprocess.run([sys.executable,'-m','graph.integration_baseline.run_control','resume','--run',run.name,'--after-repair'],check=True,env=env)
