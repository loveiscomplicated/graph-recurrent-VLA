"""Execute existing native camera output plus live wrapper/sensor contracts."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import unittest
import torch
from utils.experiment_logger import ExperimentLogger


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--live-root',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('CONTAINER_ID')=='58752018' and torch.cuda.is_available()
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('another CUDA process active; do not stop it')
    os.environ['TASK10I_LIVE_ROOT']=str(a.live_root);torch.set_num_threads(1)
    a.output.mkdir(parents=True,exist_ok=False);source=Path(__file__).resolve().parents[3]
    manifest=json.loads((source/'source_manifest.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    config=dict(task='10i_live_gpu_observation_output_20261104',instance='58752018',
        action='reset seed42 plus two zero-action CPU physics steps; no detector/API/model calls',
        reuse='shared ObservationDevice and native CUDA renderer; existing TensorCameraObservation live adapter',
        source_manifest_sha256=hashlib.sha256((source/'source_manifest.json').read_bytes()).hexdigest(),
        execution_identity=json.loads((source/'execution_identity.json').read_text()),live_root=str(a.live_root),
        live_controller_reference_sha256=hashlib.sha256((a.live_root/'inputs/task08_reference_conditions.json').read_bytes()).hexdigest())
    logger=ExperimentLogger(team='graph',job_type='diagnostic',seed=42,root=a.output/'experiments',config=config)
    try:
        names=('graph.plan_a_bc.test_live_observation_output','tests.test_observation_device_native_gpu',
            'tests.test_observation_device.OutputDeviceContracts')
        suite=unittest.TestLoader().loadTestsFromNames(names)
        with (a.output/'tests.log').open('w') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
        assert result.wasSuccessful(),(len(result.errors),len(result.failures))
        assert not result.skipped,result.skipped
        from graph.plan_a_bc.test_live_observation_output import NativeLiveObservation
        packages={name:importlib.metadata.version(name) for name in ('torch','numpy','mani_skill','sapien','gymnasium')}
        summary=dict(run_id=logger.run_id,tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),skips=len(result.skipped),
            native_live=NativeLiveObservation.proof,packages=packages,python=platform.python_version(),
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),config=config,source_files_verified=len(manifest))
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        logger.record(summary=summary)
        for path in a.output.iterdir():
            if path.is_file():logger.link_output(path)
        logger.finish();print(json.dumps(summary,indent=2))
    except BaseException:logger.finish(failed=True);raise


if __name__=='__main__':main()
