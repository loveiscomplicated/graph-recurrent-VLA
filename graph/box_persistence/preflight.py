"""Fresh UUID, source/dependency/protection audit and preregistration."""
import argparse,json,shutil,subprocess,sys,importlib.metadata
from datetime import datetime,timezone
from pathlib import Path
from graph.integration_baseline.io import sha256,write_json
from utils.experiment_logger import ExperimentLogger

BASE=Path('/workspace/task09e_plan_b')
PROTECTED=[Path('/workspace/task09d_plan_a/v2'),Path('/dev/shm/task09d_plan_a/v2'),Path('/workspace/task09b'),Path('/workspace/task09a')]

def protection():
    result={}
    for folder in PROTECTED:
        for p in folder.rglob('*'):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in str(p):result[str(p)]=dict(size=p.stat().st_size,sha256=sha256(p))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'experiments');a=p.parse_args()
    code=Path(__file__).resolve().parents[2]
    cfg=json.loads(Path(__file__).with_name('protocol.json').read_text())
    cap=BASE/'inputs'/cfg['development_capture_id'];manifest=json.loads((cap/'capture_manifest.json').read_text())
    with ExperimentLogger(team='integration',config=cfg,seed=42,model='task09e_plan_b_box_persistence',job_type='evaluate',root=a.output) as log:
        root=log.directory;print('RUN_DIRECTORY='+str(root),flush=True)
        write_json(BASE/'state.json',dict(run_id=log.run_id,root=str(root),capture=str(cap),status='registered'))
        write_json(root/'preregistered_protocol.json',dict(registered_at=datetime.now(timezone.utc).isoformat(),protocol=cfg))
        write_json(root/'reserved_final_manifest.json',dict(seeds=cfg['final_seeds'],episode_count=2,excluded=cfg['excluded_final_seeds'],collected=False,independent_final_status='NOT_TESTED',access=cfg['final_access']))
        write_json(root/'protection_before.json',protection())
        files={str(p.relative_to(code)):sha256(p) for p in code.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in str(p)}
        write_json(root/'source_hashes.json',files)
        shutil.copytree(code,root/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__','checkpoints'))
        write_json(root/'baseline_identity.json',dict(run='c7c1934765394dc4866e6564a6dbc7ec',candidate='C3_SF',decision='CONDITIONAL',
            protocol_sha256=sha256(code/'graph/plan_a_geometry/protocol.json'),
            suitability_sha256=sha256(Path('/workspace/task09d_plan_a/v2/summary/suitability.json')),
            development_manifest_sha256=sha256(cap/'capture_manifest.json'),
            episodes=[dict(seed=e['seed'],split=e['split'],frames=len(e['frames']),reset_group=e['actual_reset_group']) for e in manifest['episodes']],
            peg_current_only=True,box_main_only=True,robot76=True,temporal_extras_zero=True,production_changed=False))
        write_json(root/'runtime_dependencies.json',dict(python=sys.version,packages={k:importlib.metadata.version(k) for k in ('torch','mani_skill','sapien','numpy','scipy')},
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader']).decode().strip(),
            simulator_execution='pending live controls; geometry reads saved actual CPU PhysX/SAPIEN captures'))
        for path in root.glob('*.json'):log.link_output(path,kind='metrics')
if __name__=='__main__':main()
