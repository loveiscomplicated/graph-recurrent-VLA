"""Launcher for isolated effort evaluation; never transfers API credentials or GT."""
from common import *
import argparse
import shlex
import shutil
import subprocess
import tarfile

SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15',
     '-o','ServerAliveCountMax=2','-p','26452','root@151.237.25.16']
PY='/workspace/task10i_stage4a2_qwen35_verifier_20261102_v1/env/bin/python'
REMOTE_CODE='source/graph/experiments/api_terra_effort_grounding_20261103_v1'


def remote_audit():
    code='''import json,hashlib,importlib.metadata,pathlib,subprocess
import sam2
root=pathlib.Path(sam2.__file__).parent
paths=[pathlib.Path('/workspace/graph-recurrent-VLA/visual/checkpoints/sam2.1_hiera_small.pt')]+list(root.rglob('*.py'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
print(json.dumps(dict(files={str(p):sha(p) for p in paths},packages=sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader']).decode().strip(),gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip())))
'''
    result=subprocess.run(SSH+[PY+' -c '+shlex.quote(code)],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,'Remote read-only audit failed'
    return json.loads(next(l for l in reversed(result.stdout.splitlines()) if l.startswith('{')))


def segment():
    v=verify_frozen();assert v['stage']=='API_PREDICTIONS_FROZEN'
    for n,h in {**v['api_files'],**v['raw_files']}.items():assert sha(ROOT/n)==h
    before=remote_audit()
    assert not before['gpu_processes'], 'Existing GPU compute process: do not interfere'
    assert before['files'][SAM_CHECKPOINT]==SAM_SHA
    write(ROOT/'.execution/remote_preservation_before.json',before)
    bundle=ROOT/'.execution/segment_bundle';bundle.mkdir(parents=True,exist_ok=False)
    shutil.copy2(ROOT/'fence_grounding_predictions.json',bundle/'predictions.json')
    images=read(ROOT/'input_manifest.json')['images']
    code=bundle/REMOTE_CODE;code.mkdir(parents=True)
    for name in ['common.py','segment.py']:shutil.copy2(SOURCE/name,code/name)
    (bundle/'images').mkdir()
    for c in images:shutil.copy2(c['image_path'],bundle/'images'/(c['id']+'.png'))
    write(bundle/'segment_manifest.json',dict(images=[{k:c[k] for k in ['id','image_sha256']} for c in images],
                                             predictions_sha256=sha(bundle/'predictions.json')))
    local=ROOT/'sam2';local.mkdir(exist_ok=False)
    valid=sum(p['parsed'][o]['status']=='VALID' for p in read(bundle/'predictions.json')['rows'] for o in ['peg','box'])
    if not valid:
        write(local/'sam2_predictions.json',dict(rows=[],status='NO_VALID_API_BOXES',SAM2_loads=0))
        write(local/'sam2_latency_vram.json',dict(SAM2_loads=0,status='NO_VALID_API_BOXES'))
    else:
        assert not v.get('SAM2_execution_started'),'Refusing second SAM2 load'
        archive=ROOT/'.execution/segment_bundle.tar.gz'
        with tarfile.open(archive,'w:gz') as f:
            for p in bundle.rglob('*'):
                if p.is_file():f.add(p,arcname=p.relative_to(bundle))
        remote_start='test ! -e '+shlex.quote(REMOTE)+' && mkdir -p '+shlex.quote(REMOTE)+' && tar -xzf - -C '+shlex.quote(REMOTE)
        with archive.open('rb') as f:subprocess.run(SSH+[remote_start],stdin=f,check=True)
        v.update(SAM2_execution_started=now(),SAM2_loads=1);write(ROOT/'verification.json',v)
        cmd='cd '+shlex.quote(REMOTE)+' && '+shlex.quote(PY)+' '+shlex.quote(REMOTE_CODE+'/segment.py')+' --root '+shlex.quote(REMOTE)
        with (ROOT/'sam2_execution.log').open('w') as log:
            result=subprocess.run(SSH+[cmd],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            v.update(stage='SAM2_RUNTIME_FAILURE',SAM2_exit_code=result.returncode);write(ROOT/'verification.json',v)
            raise SystemExit('SAM2 failed; inspect remote receipts; no automatic retry')
        download=ROOT/'.execution/sam2_results.tar.gz'
        with download.open('wb') as f:subprocess.run(SSH+['cd '+shlex.quote(REMOTE)+' && tar -czf - sam2_predictions.json sam2_latency_vram.json masks'],stdout=f,check=True)
        with tarfile.open(download) as f:
            for m in f.getmembers():
                assert (local/m.name).resolve().is_relative_to(local.resolve()) and not m.issym() and not m.islnk()
            f.extractall(local)
    receipt=read(local/'sam2_predictions.json')
    assert receipt['status'] in ['COMPLETE','NO_VALID_API_BOXES']
    for row in receipt['rows']:assert sha(local/row['mask_file'])==row['mask_sha256']
    after=remote_audit()
    assert before['files']==after['files'] and before['packages']==after['packages']
    write(ROOT/'remote_preservation_verification.json',dict(status='PASS',before=before,after=after))
    v.update(stage='MASKS_FROZEN',masks_frozen_utc=now(),sam2_masks=len(receipt['rows']),
             sam2_files={str(p.relative_to(ROOT)):sha(p) for p in local.rglob('*') if p.is_file()})
    write(ROOT/'verification.json',v)
    print('MASKS_FROZEN',len(receipt['rows']),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','infer','segment','evaluate','verify','report','complete'])
    ap.add_argument('--dry-run',action='store_true');ap.add_argument('--mode',choices=['strict','fence'],default='strict')
    args=ap.parse_args();guard=None
    if sys.platform=='darwin' and not args.dry_run:
        guard=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        if args.stage=='segment':
            if args.dry_run:print('No network or GPU execution')
            else:segment()
        else:
            command=[sys.executable,str(SOURCE/(args.stage+'.py'))]
            if args.stage in ('evaluate','verify'):command+=['--mode',args.mode]
            if args.dry_run:
                if args.stage not in ('prepare','infer'):raise SystemExit('Dry run only available for prepare/infer/segment')
                command+=['--dry-run']
            subprocess.run(command,check=True,cwd=REPO)
    finally:
        if guard is not None:guard.terminate();guard.wait(timeout=5)


if __name__=='__main__':main()
