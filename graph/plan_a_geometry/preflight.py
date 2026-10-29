"""Read-only source/data/protection identity and fresh-run registration."""
import argparse,hashlib,json,subprocess,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
from utils.experiment_logger import ExperimentLogger
from graph.integration_baseline.io import write_json,sha256
from legacy_fps.integration_guard import validate_source


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    base=Path(__file__).resolve().parents[2]
    protocol=json.loads((base/'graph/plan_a_geometry/protocol.json').read_text())
    historical=Path('/workspace/task09c_legacy_fps/revalidation_runs/bd465b9d57ed4fd6843e87c950896edc/integration_manifest.json')
    old=json.loads(historical.read_text()); mismatches=[];equal=[]
    for name,expected in old['required_dependency_source_hashes'].items():
        path=base/name
        (equal if path.is_file() and sha256(path)==expected else mismatches).append(name)
    with ExperimentLogger(team='integration',config=protocol,seed=42,model='task09d_preflight',job_type='evaluate',root=a.root/'experiments') as log:
        root=log.directory
        print('RUN_DIRECTORY='+str(root),flush=True)
        hashes={str(p.relative_to(base)):sha256(p) for p in base.rglob('*.py') if '__pycache__' not in str(p)}
        dependencies={k:v for k,v in hashes.items() if not k.startswith(('legacy_fps/','graph/plan_a_geometry/'))}
        extension={k:v for k,v in hashes.items() if k.startswith('legacy_fps/')}
        guard=dict(required_dependency_source_hashes=dependencies,extension_source_hashes=extension,
            historical_manifest=str(historical),historical_manifest_sha256=sha256(historical),
            exact_historical_common_files=equal,changed_or_omitted_historical_files=mismatches,
            compatibility='new opt-in current-source adapter; current full snapshot pinned; regression tests mandatory',
            original_manifest_unchanged=True)
        write_json(root/'integration_manifest.json',guard)
        write_json(root/'integration_guard.json',validate_source(base,root/'integration_manifest.json'))
        write_json(root/'preregistered_protocol.json',dict(registered_at=datetime.now(timezone.utc).isoformat(),protocol=protocol))
        capture=Path(protocol['development_capture']);manifest=json.loads((capture/'capture_manifest.json').read_text())
        groups=[e['actual_reset_group'] for e in manifest['episodes']]
        assert len(groups)==len(set(groups))
        assert not set(protocol['final_seeds']) & {e['seed'] for e in manifest['episodes']}
        write_json(root/'development_data_audit.json',dict(episodes=[dict(seed=e['seed'],split=e['split'],frames=len(e['frames']),actual_reset_group=e['actual_reset_group']) for e in manifest['episodes']],
            sensor_roles=[e['roles'] for e in manifest['episodes']],capture_manifest_sha256=sha256(capture/'capture_manifest.json'),final_seeds_not_collected=True))
        protected={}
        for directory in ('/workspace/task09c/benchmark_code','/workspace/task09c_legacy_fps/source',
            '/workspace/task09c/comparison_runs/e7b6f9ac66c848308e28da0716bbce0c',
            '/workspace/task09c_legacy_fps/experiments/8c41299e9d3e4989bbf32ab1b894dc49',str(capture),
            '/workspace/task09c/benchmark_training_runs/99908cb9d49c4b1a868c2e2d86c2701f',
            '/workspace/task09c/benchmark_probe_runs/790dbcb6cc3f42d3b09f52087c6f03c8'):
            for path in Path(directory).rglob('*'):
                if path.is_file() and '__pycache__' not in str(path):protected[str(path)]=dict(sha256=sha256(path),size=path.stat().st_size)
        write_json(root/'protection_before.json',protected)
        write_json(root/'source_hashes.json',hashes)
        shutil.copytree(base,root/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
        write_json(a.root/'state.json',dict(preflight=str(root),status='preflight_registered',protocol_sha256=sha256(base/'graph/plan_a_geometry/protocol.json')))
        for path in root.glob('*.json'):log.link_output(path,kind='metrics')
if __name__=='__main__':main()
