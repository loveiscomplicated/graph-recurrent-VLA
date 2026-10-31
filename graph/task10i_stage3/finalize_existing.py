"""Separate finalization repair. Original collector, raw store and failure run stay immutable."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import traceback

SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'vendor'))
from .core import read,write,freeze,sha,lock
from .run_pipeline import stage
from .assembly import finalize,audit_completion,incomplete_report,MANIFESTS
from utils.experiment_logger import ExperimentLogger

def main(previous,root):
    assert read(previous/'pipeline_state.json')['status']=='INCOMPLETE'
    root.mkdir(parents=True,exist_ok=True)
    with lock(root/'run.lock'):
        original_source=previous.parent/'source'
        collector=read(previous/'collector_source_identity.json')
        for name,h in collector.items():assert sha(original_source/name)==h,name
        changed=[name for name,h in collector.items() if sha(SOURCE/name)!=h]
        assert changed==['graph/task10i_stage3/assembly.py'],changed
        # Identity inputs and already assembled references are byte copies.
        names=['stage3_input_identity.json','stage2_selection_identity_verified.json','collector_source_identity.json',
            'runtime_identity.json','expert_protocol_frozen.json','corrective_protocol_frozen.json','parent_branch_collection_plan.json',
            'collection_execution_manifest.json','run_identity.json','stage3_final_parent_selection.json','existing_source_identity_verified.json',
            'existing_checkpoint_identity.json','canonical_pool_identity_verified.json','engineering_gates.json','benchmark_resource_context.json',
            'existing_contract_readback.json','contract_tests.log',*MANIFESTS.values()]
        for name in names:
            if (root/name).exists():assert sha(root/name)==sha(previous/name),name
            else:shutil.copy2(previous/name,root/name)
        for folder in previous.iterdir():
            if folder.is_dir() and (folder.name=='parents' or folder.name.startswith('engineering_')):
                target=root/folder.name
                if not target.exists():target.symlink_to(folder.resolve(),target_is_directory=True)
        if not (root/'masks').exists():shutil.copytree(previous/'masks',root/'masks')
        for name in ('primary_execution.json','reserve_events.json'):
            if (previous/name).exists() and not (root/name).exists():shutil.copy2(previous/name,root/name)
        source_files={str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob('*') if p.is_file() and p.suffix in ('.py','.md')}
        freeze(root/'finalizer_source_identity.json',source_files)
        if (root/'pipeline_state.json').exists() and read(root/'pipeline_state.json')['status']=='COMPLETED':audit_completion(root);return
        with ExperimentLogger(team='graph',model='Task10I_Stage3_existing_raw_finalization',seed=42,job_type='prepare',root=root/'experiments',
            config=dict(collection_root=str(previous),raw_recollection=False,training=False,publishing=False)) as logger:
            logger.record(source_run_id=read(root/'run_identity.json')['run_id'])
            identity=dict(collection_root=str(previous),collection_source_root=str(original_source),finalization_source_root=str(SOURCE),
                finalization_run_id=logger.run_id,original_error=read(previous/'pipeline_state.json')['error'],
                correction='disambiguate disturbance/recovery orientation CSV keys; isolated vendored Markdown renderer',
                scientific_protocol_changed=False,raw_recollected=False,changed_original_source_files=changed,
                original_state_sha256=sha(previous/'pipeline_state.json'))
            if not (root/'finalization_identity.json').exists():freeze(root/'finalization_identity.json',identity)
            stage(root,'FINALIZATION_OF_ALREADY_COLLECTED_DATA')
            try:
                import markdown
                assert '<table>' in markdown.markdown('| a | b |\n|---|---|\n|1|2|',extensions=['markdown.extensions.tables'])
                finalize(root)
                for p in root.glob('*.json'):logger.link_output(p,kind='reports',checkpoint=p.with_name(p.name+'.tracking.json').exists())
                stage(root,'COMPLETED',status='COMPLETED',run_id=read(root/'run_identity.json')['run_id'])
            except BaseException as e:
                write(root/'pipeline_error.json',dict(error=str(e),traceback=traceback.format_exc()))
                stage(root,'INCOMPLETE',status='INCOMPLETE',error=str(e),run_id=logger.run_id)
                incomplete_report(root,str(e));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('previous',type=Path);p.add_argument('root',type=Path);a=p.parse_args();main(a.previous,a.root)
