"""Stage launcher; network inference is explicit, segmentation transfers no secrets/GT."""
from common import *
import argparse
import shlex
import shutil
import subprocess
import tarfile

SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15',
     '-o','ServerAliveCountMax=2','-p','26452','root@151.237.25.16']
PY='/workspace/task10i_stage4a2_qwen35_verifier_20261102_v1/env/bin/python'

def segment():
    v=verify_frozen()
    assert v['stage']=='API_PREDICTIONS_FROZEN'
    for n,h in {**v['api_files'],**v['raw_files']}.items():assert sha(ROOT/n)==h
    bundle=ROOT/'.execution/segment_bundle'
    bundle.mkdir(parents=True,exist_ok=True)
    predictions=ROOT/'full_image_grounding_predictions.json'
    shutil.copy2(predictions,bundle/'predictions.json')
    images=read(ROOT/'input_manifest.json')['images']
    for name in ['common.py','segment.py']:shutil.copy2(SOURCE/name,bundle/name)
    # Preserve the package depth used to resolve REPO in common.py on the remote host.
    code=bundle/'source/graph/experiments/api_full_image_grounding_20261103_v1'
    code.mkdir(parents=True,exist_ok=True)
    for name in ['common.py','segment.py']:shutil.copy2(SOURCE/name,code/name)
    (bundle/'images').mkdir(exist_ok=True)
    for c in images:shutil.copy2(c['image_path'],bundle/'images'/(c['id']+'.png'))
    write(bundle/'segment_manifest.json',dict(images=[{k:c[k] for k in ['id','image_sha256']} for c in images],
                                             predictions_sha256=sha(bundle/'predictions.json')))
    local=ROOT/'sam2';local.mkdir(exist_ok=True)
    valid=sum(p['parsed'][o]['status']=='VALID' for p in read(predictions)['rows'] for o in ['peg','box'])
    if not valid:
        write(local/'sam2_predictions.json',dict(rows=[],status='NO_VALID_API_BOXES',SAM2_loads=0))
        write(local/'sam2_latency_vram.json',dict(SAM2_loads=0,status='NO_VALID_API_BOXES'))
    else:
        if v.get('SAM2_execution_started'):
            raise SystemExit('Remote SAM2 was already started. Inspect/pull existing artifacts; do not reload.')
        archive=ROOT/'.execution/segment_bundle.tar.gz'
        with tarfile.open(archive,'w:gz') as f:
            for p in bundle.rglob('*'):
                if p.is_file():f.add(p,arcname=p.relative_to(bundle))
        cmd='mkdir -p '+shlex.quote(REMOTE)+' && tar -xzf - -C '+shlex.quote(REMOTE)
        with archive.open('rb') as f:subprocess.run(SSH+[cmd],stdin=f,check=True)
        v['SAM2_execution_started']=now();v['SAM2_loads']=1;write(ROOT/'verification.json',v)
        cmd='cd '+shlex.quote(REMOTE)+' && '+shlex.quote(PY)+' source/graph/experiments/api_full_image_grounding_20261103_v1/segment.py --root '+shlex.quote(REMOTE)
        with (ROOT/'sam2_execution.log').open('w') as log:
            result=subprocess.run(SSH+[cmd],stdout=log,stderr=subprocess.STDOUT)
        # Transfer only new experiment artifacts, not environment files or credentials.
        cmd='cd '+shlex.quote(REMOTE)+' && tar -czf - sam2_predictions.json sam2_latency_vram.json masks'
        downloaded=ROOT/'.execution/sam2_results.tar.gz'
        with downloaded.open('wb') as f:subprocess.run(SSH+[cmd],stdout=f,check=True)
        with tarfile.open(downloaded) as f:
            for member in f.getmembers():
                p=(local/member.name).resolve()
                assert p.is_relative_to(local.resolve()) and not member.issym() and not member.islnk()
            f.extractall(local)
        if result.returncode:
            v['SAM2_runtime_status']='PARTIAL_RUNTIME_FAILURE'
    receipt=read(local/'sam2_predictions.json')
    for r in receipt['rows']:assert sha(local/r['mask_file'])==r['mask_sha256']
    v.update(stage='MASKS_FROZEN',masks_frozen_utc=now(),sam2_masks=len(receipt['rows']),
             sam2_files={str(p.relative_to(ROOT)):sha(p) for p in local.rglob('*') if p.is_file()})
    write(ROOT/'verification.json',v)
    print('MASKS_FROZEN',len(receipt['rows']))

def dispatch(args):
    if args.stage=='segment':
        if args.dry_run: print('No network calls or SAM2 load; segment would use instance 58576054')
        else:segment()
    else:
        if args.dry_run and args.stage not in ['prepare','infer']:
            raise SystemExit('Dry-run is supported for prepare, infer and segment')
        command=[sys.executable,str(SOURCE/(args.stage+'.py'))]
        if args.dry_run:command+=['--dry-run']
        subprocess.run(command,check=True,cwd=REPO)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','infer','segment','evaluate','report'])
    ap.add_argument('--dry-run',action='store_true');args=ap.parse_args()
    guard=None
    if sys.platform=='darwin' and not args.dry_run:
        guard=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        dispatch(args)
    finally:
        if guard is not None:
            guard.terminate()
            guard.wait(timeout=5)

if __name__=='__main__':main()
