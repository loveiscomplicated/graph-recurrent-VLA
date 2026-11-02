"""Download bounded diagnostic evidence, without raw datasets or model weights.

This is an operational copy, not an experiment or publishing command. Each
snapshot records exact remote paths and file hashes. Run again after completion
to replace the local snapshot with final receipts.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


ROOTS = {
    'completion': '/workspace/task10i_stage4a_completion_20261101_v1/run',
    'finalization': '/workspace/task10i_stage4a_finalization_20261101_v5/run',
    'posthoc': '/workspace/task10i_stage4a_posthoc_20261101_v3/run',
    'oracle_graph': '/workspace/task10i_stage4a_oracle_graph_20261101_v2/run',
    'oracle_grouping': '/workspace/task10i_stage4a_oracle_grouping_20261101_v2/run',
    'detector_outcomes': '/workspace/task10i_stage4a_detector_outcomes_20261101_v1/run',
    'camera_contract': '/workspace/task10i_stage4a_camera_contract_20261101_v2/run',
    'development_temporal': '/workspace/task10i_stage4a_development_temporal_20261101_v1/run',
    'initialization_bottlenecks': '/workspace/task10i_stage4a_initialization_bottlenecks_20261101_v1/run',
    'refinement_retention': '/workspace/task10i_stage4a_refinement_retention_20261101_v3/run',
    'global_missing_oracles': '/workspace/task10i_stage4a_global_missing_oracles_20261101_v1/run',
    'approximate_reference': '/workspace/task10i_stage4a_approximate_reference_20261101_v2/run',
    'restore_probe': '/workspace/task10i_stage4a_gt_restore_probe_20261101_v1',
    'surface_audit': '/workspace/task10i_stage4a_surface_audit_20261101_v1',
    'cache_equivalence': '/workspace/task10i_stage4a_cache_equivalence_20261101_v1',
    'forward_equivalence': '/workspace/task10i_stage4a_forward_equivalence_20261101_v1/run',
    'blas_equivalence': '/workspace/task10i_stage4a_blas_equivalence_20261101_v1/run',
}


def collect(destination, control_socket):
    remote = '''
import hashlib,io,json,os,tarfile
from pathlib import Path
from datetime import datetime,timezone
ROOTS = ROOTS_REPLACEMENT
records=[]
with tarfile.open(fileobj=__import__('sys').stdout.buffer,mode='w|gz') as archive:
    for label,base in ROOTS.items():
        base=Path(base)
        if not base.exists():continue
        candidates=[]
        blocked={'source','traces','perception','references','dino','dino_raw','__pycache__'}
        for folder,dirs,files in os.walk(base,followlinks=True):
            dirs[:]=sorted(d for d in dirs if d not in blocked)
            candidates.extend(Path(folder)/name for name in sorted(files))
        for p in candidates:
            if not p.is_file() or p.is_symlink():continue
            relative=p.relative_to(base)
            if any(part in ('source','traces','perception','references','dino','dino_raw','__pycache__') for part in relative.parts):continue
            if p.name in ('development_representative_points.json','representative_points.json','frame_attribution.json','stage_rows.json','temporal_frame_metrics.json','frame_metrics.json','failure_cases.json'):
                oracle_payload=p.name in ('representative_points.json','stage_rows.json') and (label in ('oracle_grouping','global_missing_oracles') or label=='finalization' and relative.parts[0]=='oracles')
                if not oracle_payload:continue
            if p.suffix not in ('.json','.png','.html','.md','.txt'):continue
            if p.stat().st_size>35_000_000:continue
            data=p.read_bytes();name=str(Path(label)/relative)
            record=dict(local_path=name,remote_path=str(p),remote_resolved_path=str(p.resolve()),size=len(data),sha256=hashlib.sha256(data).hexdigest())
            info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data));records.append(record)
    payload=dict(kind='READ_ONLY_EVIDENCE_SNAPSHOT',snapshot_utc=datetime.now(timezone.utc).isoformat(),roots=ROOTS,files=records,
        full_remote_traces_and_raw_data_not_downloaded=True,not_a_publishing_receipt=True)
    data=json.dumps(payload,indent=2).encode();info=tarfile.TarInfo('download_manifest.json');info.size=len(data);archive.addfile(info,io.BytesIO(data))
'''.replace('ROOTS_REPLACEMENT', repr(ROOTS))
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.tar.gz') as bundle:
        proc = subprocess.run(['ssh', '-S', control_socket, '-p', '26452',
            'root@151.237.25.16', '/workspace/venvs/task10i-stage4a-cu130/bin/python -B -'], input=remote.encode(), stdout=bundle,
            stderr=subprocess.PIPE, check=True)
        bundle.flush()
        with tarfile.open(bundle.name) as archive:
            for member in archive.getmembers():
                path=Path(member.name)
                assert not path.is_absolute() and '..' not in path.parts
                assert member.isfile()
            archive.extractall(destination, filter='data')
    manifest=json.loads((destination/'download_manifest.json').read_text())
    import hashlib
    for record in manifest['files']:
        assert hashlib.sha256((destination/record['local_path']).read_bytes()).hexdigest()==record['sha256']
    print(json.dumps(dict(files=len(manifest['files']), bytes=sum(r['size'] for r in manifest['files']),
        manifest=str(destination/'download_manifest.json'))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--control-socket',default='/tmp/task10i_stage4a_58576054_finish_mux')
    args=parser.parse_args();collect(args.destination,args.control_socket)
