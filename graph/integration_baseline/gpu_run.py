"""Opt-in CUDA execution of a frozen, already-collected direct dataset.

Raw dataset UUID/config/split remain immutable. A new ExperimentLogger UUID
owns CUDA graphs, training, tests and reports; physics is not rerun.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

from graph.integration_baseline.direct_runtime import REPO,atomic,read,sha,Events,now
from graph.integration_baseline.run_control import settings,torch_config,launch


def prepare(data,preflight,*,smoke=False,callback_required=True,vision_backend='legacy_fp32'):
    if vision_backend not in ('legacy_fp32','visual_stream'):
        raise ValueError('unknown vision backend: '+str(vision_backend))
    from utils.experiment_logger import ExperimentLogger
    from graph.integration_baseline.direct_errors import source_inventory
    settings();data=Path(data).resolve();preflight=Path(preflight).resolve()
    manifest=read(data/'dataset_manifest.json');split=read(data/'split_manifest.json')
    if manifest['dataset_id']!=data.name or split['dataset_sha256']!=sha(data/'dataset_manifest.json'):
        raise ValueError('source dataset identity/split checksum mismatch')
    if manifest['successes']!=120 or {k:len(v) for k,v in split['splits'].items()}!={'train':96,'validation':12,'test':12}:
        raise ValueError('the approved 120 episode frozen split is required')
    config=read(Path(__file__).with_name('direct_config.json'))
    original=read(data/'config.yaml')
    for key in ['model','graph','training','seed','vision','splits']:
        if config[key]!=original[key]:raise ValueError('frozen baseline settings changed: '+key)
    if vision_backend=='visual_stream':config['vision']['backend']=vision_backend
    config.update(device='cuda',vision_device='cuda',execution_mode='existing_data_cuda',discord_enabled=False,
        report_document='docs/visual-graph-integration/06c_direct_expert_bc_training_cuda_{run_id_short}.md',fixture_only=smoke,
        auto_callback_required=callback_required)
    codex=shutil.which('codex')
    if not codex:raise RuntimeError('Codex executable required for the one-shot error callback')
    config['runtime_bin_directory']=str(Path(codex).parent)
    from huggingface_hub.constants import HF_HOME
    config['model_cache_directory']=str(Path(HF_HOME).resolve())
    config['numerics']=dict(scope='BC_model',dtype='float32',amp=False,tf32=False,cublas_workspace_config=':4096:8')
    torch_config(config)
    logger=ExperimentLogger(team='integration',model='graph_gnn_gru_direct_expert_bc_cuda',seed=42,
        config=config,job_type='prepare')
    run=logger.directory
    for name in ['preflight','vision','training','errors','plots','logs','stages']:(run/name).mkdir()
    atomic(run/'data_source.json',dict(directory=str(data),dataset_id=data.name,
        dataset_sha256=sha(data/'dataset_manifest.json'),split_sha256=sha(data/'split_manifest.json'),
        provenance=read(data/'download_provenance.json') if (data/'download_provenance.json').exists() else {}),immutable=True)
    for name in ['dataset_manifest.json','split_manifest.json','robot_state_schema.json','expert_action_schema.json','executed_conditions.json']:
        if (data/name).exists():shutil.copy2(data/name,run/name)
    atomic(run/'stages/collection.json',dict(status='COMPLETE',exit_code=0,stage='collection',
        source_dataset_id=data.name,reused=True,physics_executed_on_this_host=False,time=now()),immutable=True)
    actual=read(preflight/'callback_verification.json') if (preflight/'callback_verification.json').exists() else dict(actual_cli_success=False)
    atomic(run/'callback_verification.json',actual,immutable=True)
    atomic(run/'discord_verification.json',dict(status='USER_DISABLED',actual_delivery=False),immutable=True)
    for name in ['environment.json','dependency_tests.log','callback_verification.json','download.json']:
        if (preflight/name).exists():shutil.copy2(preflight/name,run/'preflight'/name)
    sources=source_inventory();atomic(run/'source_hashes.json',sources,immutable=True)
    for name in sources:
        target=run/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    atomic(run/'status.json',dict(status='PREPARED',stage='preflight',run_id=run.name,time=now()))
    logger.record(source_run_id=data.name,pipeline_run_id=run.name,dataset_sha256=sha(data/'dataset_manifest.json'),
        physics_reused=True,device='cuda',discord='USER_DISABLED');logger.finish()
    print('RUN_ID='+run.name,flush=True)
    return run


def smoke(run):
    from graph.integration_baseline.direct_dataset import process_vision,DirectDataset,validate_receipt
    from graph.integration_baseline.direct_train import gate,train,offline_test,seed_all
    from utils.experiment_logger import ExperimentLogger
    settings();config=read(run/'config.yaml');torch_config(config)
    source=Path(read(run/'data_source.json')['directory'])
    # Validate every frozen success episode independently before doing GPU work.
    manifest=read(source/'dataset_manifest.json');schema=read(source/'robot_state_schema.json')
    for e in manifest['episodes']:validate_receipt(source,e,schema)
    ids=read(source/'split_manifest.json')['splits']['train'][:2]
    events=Events(run);tick=perf_counter()
    try:
        graphs=process_vision(run,config,events,episode_ids=ids)
        dataset=DirectDataset(run,smoke=True);validation=dataset.validate(partial=True)
        learning=gate(run,config,dataset,smoke=True)
        report=dict(status='PASS',device='cuda',dataset_id=source.name,source_episodes_validated=120,
            split_unchanged=True,graph_episode_ids=ids,graph_frames=sum(len(e['packets']) for e in graphs['episodes']),
            elapsed_seconds=perf_counter()-tick,validation=validation,learning_gate=learning)
        atomic(run/'preflight/cuda_smoke.json',report,immutable=True)
        atomic(run/'status.json',dict(status='SMOKE_COMPLETE',stage='done',run_id=run.name,time=now()))
        events.emit('cuda_smoke_complete',frames=report['graph_frames'],elapsed_seconds=report['elapsed_seconds'])
    finally:events.close()


def finalize_preflight(run,smoke_run):
    smoke_result=read(smoke_run/'preflight/cuda_smoke.json')
    if smoke_result['status']!='PASS':raise ValueError('CUDA smoke failed')
    if read(run/'data_source.json')!=read(smoke_run/'data_source.json'):raise ValueError('smoke used a different dataset')
    full_config=read(run/'config.yaml');smoke_config=read(smoke_run/'config.yaml')
    full_config.get('numerics',{}).setdefault('scope','BC_model')
    smoke_config.get('numerics',{}).setdefault('scope','BC_model')
    for key in ['fixture_only','auto_callback_required','report_document']:
        full_config.pop(key,None);smoke_config.pop(key,None)
    if full_config!=smoke_config:
        raise ValueError('smoke/full numerical, model, data or runtime configuration differs')
    # Graphs are immutable and use source dataset identity, so they can be reused.
    shutil.copytree(smoke_run/'vision',run/'vision',dirs_exist_ok=True,copy_function=os.link)
    shutil.copy2(smoke_run/'preflight/cuda_smoke.json',run/'preflight/cuda_smoke.json')
    atomic(run/'preflight/verification.json',dict(status='PASS',cuda_smoke_run_id=smoke_run.name,
        real_cuda_forward_backward_optimizer_checkpoint_rng=True,source_dataset_validated=120,
        smoke_sha256=sha(smoke_run/'preflight/cuda_smoke.json')),immutable=True)
    from graph.integration_baseline.direct_report import status_document,verify_html
    status_document(run)
    atomic(run/'preflight/initial_html_verification.json',verify_html(run/'graph_gru_training.html'),immutable=True)


def main():
    os.chdir(REPO);settings()
    os.environ.pop('DISCORD_WEBHOOK_URL',None)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','smoke','launch'])
    p.add_argument('--data',type=Path);p.add_argument('--preflight',type=Path);p.add_argument('--run');p.add_argument('--smoke-run')
    p.add_argument('--fixture',action='store_true')
    p.add_argument('--vision-backend',choices=['legacy_fp32','visual_stream'],default='legacy_fp32',
                   help='prepare only; opt-in backend for a new run, never an existing cache')
    p.add_argument('--without-callback',action='store_true',help='only after explicit user waiver of the unavailable callback')
    args=p.parse_args()
    if args.command=='prepare':prepare(args.data,args.preflight,smoke=args.fixture,
        callback_required=not args.without_callback,vision_backend=args.vision_backend)
    else:
        if args.vision_backend!='legacy_fp32':p.error('--vision-backend is prepare-only; existing run config is immutable')
        from graph.integration_baseline.run_control import path_for
        run=path_for(args.run)
        if args.command=='smoke':smoke(run)
        else:finalize_preflight(run,path_for(args.smoke_run));launch(run)


if __name__=='__main__':main()
