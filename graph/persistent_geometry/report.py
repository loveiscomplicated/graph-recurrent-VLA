"""Build immutable comparison facts/figures for subsequent evidence-based prose."""
import argparse
import hashlib
from importlib import metadata as packages
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from graph.integration_baseline.io import write_json,sha256
from utils.experiment_logger import ExperimentLogger
from .collect import inventory_sources,TASK09B
from .summarize import collect_summary,figures,source_paths
from .render import sequence,temporal,filter_and_aperture
from .cases import analyze
from .verify import verify_run
from .audit import protection_snapshot,verify_protection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();state=json.loads(args.state.read_text())
    if state['status']!='finished':raise ValueError('all frozen evaluation stages must finish before comparison facts')
    paths=source_paths(state)
    protected=protection_snapshot(p for source in paths.values() for p in source.rglob('*') if p.is_file())
    base=args.state.parent
    additional=[]
    for name in ('experiments','preflight_runs','geometry_runs','geometry_revision2_runs','filter_revision_runs',
                 'branch_smoke_runs','final_branch_smoke_runs','final_preflight_runs','final_geometry_smoke_runs',
                 'final_branch_geometry_smoke_runs','training_smoke_runs','camera_probe_runs','render_smoke_runs',
                 'historical_readback_runs','smoke_readback_runs','negative_smoke_runs','cache_branch_smoke_runs','probe_smoke_runs'):
        for entry in (base/name).glob('*/metadata.json'):
            if json.loads(entry.read_text())['status']=='finished':additional.append(entry.parent)
    sources=sorted(set(paths.values())|set(additional))
    with ExperimentLogger(team='integration',config=dict(task='09-C',stage='comparison_facts',
            source_state=str(args.state),no_model_selection=True,report_state='facts_ready_interpretation_pending'),
            seed=42,model='task09c_geometry_comparison',job_type='evaluate',root=args.output) as logger:
        root=logger.directory;print('TASK09C_COMPARISON_DIRECTORY='+str(root),flush=True)
        logger.record(source_run_id=paths['diagnostic_training'].name,source_run_ids=[p.name for p in sources])
        inventory_sources(root);write_json(root/'protection_before.json',protected)
        write_json(root/'pipeline_state.json',state)
        summary=collect_summary(state,root/'summary');figures(summary,paths,root/'figures')
        render=[]
        for seed in (62091,62092):
            destination=root/'figures'/f'seed{seed}'
            render.append(sequence(paths['independent_capture'],paths['independent_branches'],destination,seed))
            temporal(paths['independent_branches'],destination,seed)
            render[-1].update(filter_and_aperture(paths['independent_capture'],paths['independent_branches'],destination,seed))
        write_json(root/'render_verification.json',render)
        for label,cap,geom in (('development','development_capture','development_branches'),
                               ('independent','independent_capture','independent_branches')):
            destination=root/'negative_cases'/label;destination.mkdir(parents=True)
            analyze(paths[cap],paths[geom],destination)
        original=json.loads(next((base/'final_preflight_runs').glob('*/protection_before.json')).read_text())
        protection=verify_protection(original)
        if protection['status']!='PASS':raise ValueError('original frozen source/artifacts changed')
        write_json(root/'original_frozen_protection.json',protection)
        dependencies={}
        for name in ('torch','numpy','scipy','mani_skill','sapien','gymnasium','trimesh','imageio','matplotlib','h5py'):
            try:dependencies[name]=packages.version(name)
            except packages.PackageNotFoundError:dependencies[name]='NOT_FOUND'
        identity=json.dumps(dependencies,sort_keys=True).encode()
        write_json(root/'dependency_identity.json',dict(versions=dependencies,
            canonical_versions_sha256=hashlib.sha256(identity).hexdigest(),
            CPU_quota=(Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None),
            cuda_physx='NOT_EXECUTED',GPU_runtime=subprocess.check_output(
                ['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
            note='package-version identity; imported repository Python source hashes are recorded separately'))
        readback=[]
        for source in sources:
            # Early pilots predate the final snapshot schema. Preserve them as
            # historical failure evidence; the complete final-stage runs must
            # pass the current independent joins and checkpoint checks.
            readback.append(verify_run(source,check_joins=source in paths.values()))
        write_json(root/'source_readback_verification.json',dict(status='PASS',runs=readback))
        mapping={}
        for source in sources:
            destination=root/'sources'/source.name
            # Same-filesystem hard links avoid duplicating immutable raw maps.
            # No contents, permissions, sidecars or timestamps are modified.
            shutil.copytree(source,destination,copy_function=os.link)
            mapping[str(source)]=str(Path('sources')/source.name)
        write_json(root/'path_remap.json',dict(remote_prefix_to_bundle_relative=mapping,
            note='recorded source manifests retain original absolute provenance; resolve prefixes relative to this comparison directory'))
        for filename in ('benchmark_pipeline.py','benchmark_pipeline_state.json','cache_branch_identity.json'):
            if (base/filename).exists():shutil.copyfile(base/filename,root/filename)
        for source_root in ('final_code','benchmark_code'):
            for filename in (base/source_root).glob('task09c*.json'):
                target=root/'deployment'/source_root/filename.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(filename,target)
        audit=verify_protection(protected)
        if audit['status']!='PASS':raise ValueError('comparison generation changed finished inputs')
        write_json(root/'protection_audit.json',audit)
        write_json(root/'facts_ready.json',dict(status='FACTS_READY',human_interpretation_pending=True,
            models=len(summary['models']),source_runs=len(sources),source_readback='PASS',
            original_frozen_protection='PASS',MP4_decodes='PASS'))
        for name in ('summary/benchmark_summary.json','facts_ready.json','render_verification.json',
                     'source_readback_verification.json','original_frozen_protection.json','protection_audit.json'):
            logger.link_output(root/name,kind='metrics')


if __name__=='__main__':main()
