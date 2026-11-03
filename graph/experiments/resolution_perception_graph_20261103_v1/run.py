"""Local/SSH stage launcher; explicit allowlists keep credentials on the local host."""
from common import *
import argparse
import io
import shlex
import subprocess
import tarfile
import time


def remote_command(script,*args):
    return 'cd '+shlex.quote(REMOTE)+' && '+ ' '.join([
        'TASK10I_REPO='+shlex.quote(REMOTE_REPO),'TASK10I_ROOT='+shlex.quote(REMOTE),
        'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        shlex.quote(REMOTE_PY),shlex.quote('code/'+script),*(shlex.quote(str(a)) for a in args)])


def sync_code():
    b=io.BytesIO()
    with tarfile.open(fileobj=b,mode='w:gz') as t:
        for p in SOURCE.iterdir():
            if p.suffix in ('.py','.md'):t.add(p,arcname='code/'+p.name)
    subprocess.run(SSH+['tar -xzf - -C '+shlex.quote(REMOTE)],input=b.getvalue(),check=True)


def fetch_inputs():
    ROOT.mkdir(parents=True,exist_ok=True)
    # Only RGB and non-secret protocol metadata leave the server for the API worker.
    probe='''import json,pathlib
r=pathlib.Path(%r);d=json.loads((r/'capture_manifest.json').read_text())
a=[e for e in d['entries'] if e['cohort']!='tracking' or e['timestep']==0]
print(len(a))
''' % REMOTE
    while True:
        r=subprocess.run(SSH+[REMOTE_PY+' -c '+shlex.quote(probe)],capture_output=True,text=True,check=True)
        n=int(r.stdout.splitlines()[-1])
        if n==280:break
        print('WAITING_API_INPUTS',n,'/280',flush=True);time.sleep(10)
    code='''import json,pathlib,tarfile,sys
r=pathlib.Path(%r);d=json.loads((r/'capture_manifest.json').read_text())
a=[e for e in d['entries'] if e['cohort']!='tracking' or e['timestep']==0]
assert len(a)==280
names=['protocol.json','grounding_prompt.txt','smoke_verification.json']+[e['image_file'] for e in a]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for n in names:t.add(r/n,arcname=n)
 # The API worker receives no actor IDs, GT paths or GT arrays.
 keys=['id','scene','size','camera','cohort','timestep','parent','image_file','image_sha256']
 data=json.dumps(dict(entries=[{k:e[k] for k in keys} for e in a],status='API_INPUTS_COMPLETE')).encode()
 import io
 info=tarfile.TarInfo('capture_manifest.json');info.size=len(data);t.addfile(info,io.BytesIO(data))
''' % REMOTE
    archive=ROOT/'api_inputs.tar.gz'
    with archive.open('wb') as f:subprocess.run(SSH+[REMOTE_PY+' -c '+shlex.quote(code)],stdout=f,check=True)
    extract(archive,ROOT)
    for e in read(ROOT/'capture_manifest.json')['entries']:assert sha(ROOT/e['image_file'])==e['image_sha256']
    print('API_INPUTS_LOCAL',280,flush=True)


def extract(archive,dest):
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            assert (dest/m.name).resolve().is_relative_to(dest.resolve()) and not m.issym() and not m.islnk()
        t.extractall(dest)


def transfer_api():
    freeze=read(ROOT/'api_freeze.json');assert sha(ROOT/'api_predictions.json')==freeze['predictions_sha256']
    b=io.BytesIO()
    with tarfile.open(fileobj=b,mode='w:gz') as t:
        for n in ('api_predictions.json','api_freeze.json'):t.add(ROOT/n,arcname=n)
    subprocess.run(SSH+['tar -xzf - -C '+shlex.quote(REMOTE)],input=b.getvalue(),check=True)


def remote_stage(script,*args,log=None):
    log=log or script.replace('.py','')+'_'+'_'.join(str(a).replace('/','_') for a in args)+'.log'
    command=remote_command(script,*args)+' > '+shlex.quote(log)+' 2>&1'
    subprocess.run(SSH+[command],check=True)


def graphs(arm):
    processes=[]
    for shard in range(4):
        command=remote_command('graph_worker.py','quality','--arm',arm,'--shard',shard,'--shards',4)
        command+=' > '+shlex.quote(f'graph_{arm}_{shard}.log')+' 2>&1'
        processes.append(subprocess.Popen(SSH+[command]))
    codes=[p.wait() for p in processes]
    assert codes==[0]*4,codes
    remote_stage('graph_worker.py','merge','--arm',arm,'--shards',4)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['sync-code','fetch-inputs','infer','transfer-api','sam','graphs','oracles'])
    ap.add_argument('--arm',default='actual');a=ap.parse_args()
    if a.stage=='sync-code':sync_code()
    elif a.stage=='fetch-inputs':fetch_inputs()
    elif a.stage=='infer':
        guard=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]) if sys.platform=='darwin' else None
        try:subprocess.run([sys.executable,str(SOURCE/'api.py')],check=True)
        finally:
            if guard is not None:guard.terminate()
    elif a.stage=='transfer-api':transfer_api()
    elif a.stage=='sam':
        for s in ('static','tracking'):remote_stage('sam_worker.py',s,'--arm',a.arm)
        remote_stage('pipeline.py','--freeze-masks',a.arm)
    elif a.stage=='graphs':graphs(a.arm)
    elif a.stage=='oracles':
        remote_stage('graph_worker.py','oracle_jobs')
        for s in ('static','tracking'):remote_stage('sam_worker.py',s,'--arm','gt_box')
        remote_stage('pipeline.py','--freeze-masks','gt_box')
        graphs('gt_box');graphs('gt_mask')


if __name__=='__main__':main()
