"""Read-only SAM2 checkpoint/source/package preservation check after segmentation."""
from common import *
import shlex
import subprocess

def main():
    code='''import json,hashlib,importlib.metadata,pathlib,subprocess
import sam2
root=pathlib.Path(sam2.__file__).parent
paths=[pathlib.Path('/workspace/graph-recurrent-VLA/visual/checkpoints/sam2.1_hiera_small.pt')]+list(root.rglob('*.py'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
print(json.dumps(dict(files={str(p):sha(p) for p in paths},packages=sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader']).decode().strip())))
'''
    cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','26452','root@151.237.25.16',
         '/workspace/task10i_stage4a2_qwen35_verifier_20261102_v1/env/bin/python -c '+shlex.quote(code)]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,'Remote read-only preservation audit failed'
    after=json.loads(next(l for l in reversed(result.stdout.splitlines()) if l.startswith('{')))
    before=read(ROOT/'.execution/remote_preservation_before.json')
    assert before['files']==after['files'],'Remote checkpoint or installed SAM2 source changed'
    assert before['packages']==after['packages'],'Remote installed package versions changed'
    write(ROOT/'remote_preservation_verification.json',dict(status='PASS',
        protected_files=len(after['files']),packages=len(after['packages']),before=before,after=after))
    print('REMOTE_PRESERVATION_PASS',len(after['files']),len(after['packages']))

if __name__=='__main__':main()
