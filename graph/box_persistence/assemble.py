"""Assemble a fresh portable evidence tree; original files remain untouched."""
import argparse,json,shutil,difflib,subprocess
from pathlib import Path
from datetime import datetime,timezone
from graph.integration_baseline.io import write_json,sha256
from utils.experiment_artifacts import register_existing_output

BASE=Path('/workspace/task09e_plan_b')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    ids=dict(geometry_id='bcbbd28541d847c0b8664506b815f7d6',capture_id='c488c5a8d9904ba2af541ba5d6bc7dca',
        motion_id='f087c72159334c528e00c235408fb2d7',motion_warm_id='44f909a6fb194eafb45f05d6fe2d65e8',
        motion_oracle_id='79b4d27c5d1e46cc9649f31771716ada')
    paths={'geometry_id':Path('/dev/shm/task09e_plan_b/geometry')/ids['geometry_id'],
        'capture_id':BASE/'control_capture'/ids['capture_id'],
        'motion_id':Path('/dev/shm/task09e_plan_b/motion')/ids['motion_id'],
        'motion_warm_id':Path('/dev/shm/task09e_plan_b/motion_warm')/ids['motion_warm_id'],
        'motion_oracle_id':Path('/dev/shm/task09e_plan_b/motion_oracle')/ids['motion_oracle_id']}
    root.joinpath('sources').mkdir(exist_ok=True);remap={}
    for key,path in paths.items():
        meta=json.loads((path/'metadata.json').read_text())
        if meta['status']!='finished':raise ValueError('unfinished evidence '+key)
        dest=root/'sources'/ids[key]
        if not dest.exists():dest.symlink_to(path,target_is_directory=True)
        remap[str(path)]=str(dest.relative_to(root))
    # Retain the earlier, no-prior-support-bookkeeping evaluation as superseded.
    old=Path('/dev/shm/task09e_plan_b/motion/ae7a081beae0439ba6c83555bec78260')
    dest=root/'sources'/old.name
    if old.is_dir() and not dest.exists():dest.symlink_to(old,target_is_directory=True);remap[str(old)]=str(dest.relative_to(root))
    source_capture=BASE/'inputs/26c2b180003c417d8464ee60314c95fc'
    root.joinpath('inputs').mkdir(exist_ok=True);dest=root/'inputs'/source_capture.name
    if not dest.exists():dest.symlink_to(source_capture,target_is_directory=True)
    remap[str(source_capture)]=str(dest.relative_to(root));remap[str(root)]='.'
    write_json(root/'path_remap.json',remap)
    shutil.copytree(BASE/'logs',root/'logs',dirs_exist_ok=True)
    code=Path(__file__).resolve().parents[2]
    shutil.copytree(code,root/'source_snapshot_final',ignore=shutil.ignore_patterns('__pycache__','checkpoints'),dirs_exist_ok=True)
    hashes={str(p.relative_to(code)):sha256(p) for p in code.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in str(p)}
    initial=json.loads((root/'source_hashes.json').read_text())
    changed=[k for k,v in initial.items() if hashes.get(k)!=v]
    if changed:raise ValueError('initial implementation/dependency changed '+str(changed))
    root.joinpath('audits').mkdir(exist_ok=True)
    write_json(root/'audits/source_identity.json',dict(final_source_hashes=hashes,changed_initial_files=changed,
        source_head='0498a64faf0f4baac3ba4d7aa5846d1e1b84f356',dirty_source=True,git_commit_created=False,
        added_postprocessing_files=[k for k in hashes if k not in initial],
        map_and_gate_rules_unchanged=True,source_snapshot='source_snapshot_final',
        motion_support_amendment='separate baseline-initialized t0 diagnostic, after original t0/25/65 lacked an observable moved Box with prior map; same rendered data and fixed memory/Gate rules',
        original_no_history_motion_run=str(old),primary_motion_run=ids['motion_id'],warm_visible_run=ids['motion_warm_id']))
    diff=[]
    for path in sorted(code.joinpath('graph/box_persistence').glob('*')):
        if path.is_file() and path.suffix in ('.py','.json','.md'):
            relative=str(path.relative_to(code));diff.extend(difflib.unified_diff([],path.read_text().splitlines(True),fromfile='/dev/null',tofile='b/'+relative))
    root.joinpath('audits/implementation.diff').write_text(''.join(diff))
    gate=json.loads((root/'summary/gate_decision.json').read_text())
    write_json(root/'pipeline_state.json',dict(**ids,status='geometry_gate_complete',decision=gate['overall'],
        source_runs={k:str(v) for k,v in paths.items()},final_collected=False,gnn_executed=False,
        gate_rules_changed=False,completed_at=datetime.now(timezone.utc).isoformat()))
    phases={
        'Stage0':dict(status='PASS',scope='PlanA selection/source/sensor/split identity and reserved new Final manifest'),
        'Stage1':dict(status='PASS',scope='B0/B1/B2 Box-only implementation and synthetic/shared contracts; no Peg map'),
        'Stage2':dict(status='PASS',scope='12EP all-frame causal update and native/cap128/matched-N geometry measurements; suitability is separate NO-GO'),
        'Stage3':dict(status='PARTIAL',scope='actual visible/partial/full-hidden same-robot motion controls; warm t0 diagnostic; reappearance uses re-presented snapshot, no dynamic recovery/contact-force evidence'),
        'Stage4':dict(status='PASS',scope='preregistered Geometry Gate: B1/B2 NO-GO; no eligible GNN candidate'),
        'Stage5':dict(status='NOT_EXECUTED',scope='E0/E1/probes and independent Final withheld by Geometry Gate; A/B GNN rank NOT_TESTED'),
        'Stage6':dict(status='PARTIAL',scope='report/artifact verification pending; see verification.json')}
    write_json(root/'phase_status.json',phases)
    oracle=paths['motion_oracle_id']/'oracle_motion_metrics.csv';shutil.copy2(oracle,root/'summary/oracle_motion_metrics.csv')
    write_json(root/'runtime_execution.json',dict(host='Vast.ai57640710 / RTX3090',
        capture_backend=json.loads((paths['capture_id']/'collection/seed30002/sensor_contract.json').read_text())['backend'],
        CUDA='DINO/SAM2 inference',CPU='PhysX/SAPIEN capture and geometry/graph arrays',
        not_executed=['CUDA PhysX','macOS simulator/MPS','policy/GRU/BC','GNN training due to Geometry Gate'],
        contract_tests=dict(remote=86,local_new_synthetic=32)))
    for path in [root/'summary/gate_decision.json',root/'phase_status.json',root/'pipeline_state.json',root/'runtime_execution.json']:
        register_existing_output(root,path,kind='metrics')
if __name__=='__main__':main()
