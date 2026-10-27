"""Explicit filesystem bridge for CPU stages delegated to the user's Mac."""
import hashlib
import json
from pathlib import Path
import time


def maybe_delegate(module,args):
    marker=Path('/workspace/task09c/local_cpu_handoff.json')
    if not marker.exists():return False
    config=json.loads(marker.read_text())
    output=Path(args.output).resolve()
    if not config.get('enabled') or str(output) not in config.get('outputs',{}).get(module,[]):return False
    values={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    key=hashlib.sha256(json.dumps(dict(module=module,args=values),sort_keys=True).encode()).hexdigest()[:24]
    folder=Path(config['directory']);(folder/'requests').mkdir(parents=True,exist_ok=True)
    (folder/'receipts').mkdir(exist_ok=True)
    request=folder/'requests'/(key+'.json');receipt=folder/'receipts'/(key+'.json')
    request.write_text(json.dumps(dict(request_id=key,module=module,args=values,
        status='AWAITING_LOCAL_CPU',requested_at=time.time(),local_workers=1),indent=2)+'\n')
    print('TASK09C_CPU_HANDOFF_REQUEST='+str(request),flush=True)
    while not receipt.exists():time.sleep(5)
    result=json.loads(receipt.read_text())
    if result.get('status')!='finished':raise RuntimeError('local CPU stage failed; see handoff receipt')
    directory=Path(result['remote_result_directory']).resolve()
    if directory.parent!=output:raise ValueError('delegated output root identity')
    metadata=json.loads((directory/'metadata.json').read_text())
    if metadata['status']!='finished' or metadata['run_id']!=directory.name:raise ValueError('unfinished delegated result')
    for name,digest in result['files'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:raise ValueError('delegated transfer hash mismatch')
    print('TASK09C_CPU_HANDOFF_VERIFIED='+str(directory),flush=True)
    return True
