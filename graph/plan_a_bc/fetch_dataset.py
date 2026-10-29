"""Retrieve an already-published, pinned original dataset without publishing.

Authentication stays on the user's host. Only an expiring signed file download
URL is passed over SSH stdin, consumed in memory and never logged or persisted.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

import requests
from huggingface_hub import get_token


REMOTE = r'''
import sys,json,urllib.request,hashlib,zipfile,os
from pathlib import Path
spec=json.load(sys.stdin); root=Path(spec['root']); archive=root/'original_dataset_published.zip'
try:
    if not archive.exists():
        partial=archive.with_suffix('.partial')
        request=urllib.request.urlopen(spec['url'],timeout=60)
        h=hashlib.sha256();total=0
        with partial.open('wb') as f:
            while True:
                b=request.read(4<<20)
                if not b:break
                f.write(b);h.update(b);total+=len(b)
                if total%(64<<20)==0:print('downloaded_bytes',total,flush=True)
        assert h.hexdigest()==spec['archive_sha256'],'published archive hash mismatch'
        os.link(partial,archive);partial.unlink()
    def sha(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for b in iter(lambda:f.read(4<<20),b''):h.update(b)
        return h.hexdigest()
    assert sha(archive)==spec['archive_sha256']
    manifest=json.load(open(root/'dataset/dataset_manifest.json'))
    with zipfile.ZipFile(archive) as z:
        for ep in manifest['episodes']:
            target=root/'dataset'/ep['path']
            if target.exists() and sha(target)==ep['sha256']:continue
            matches=[n for n in z.namelist() if n.endswith('/'+ep['path']) or n==ep['path']]
            assert len(matches)==1,('episode archive cardinality',ep['episode_id'],len(matches))
            target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_suffix('.hf_partial')
            with z.open(matches[0]) as src,partial.open('wb') as dst:
                import shutil;shutil.copyfileobj(src,dst)
            assert sha(partial)==ep['sha256'],'original HDF5 hash mismatch'
            os.replace(partial,target)
    print('ORIGINAL_DATASET_SHA_PASS',len(manifest['episodes']),flush=True)
except BaseException as exc:
    print('dataset_download_failure',type(exc).__name__,flush=True)
    sys.exit(1)
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',required=True)
    p.add_argument('--remote-root',required=True);a=p.parse_args()
    token=get_token()
    if not token:raise RuntimeError('local read credential unavailable')
    archive_sha='a91dcdc7ac20535c7cd2be500b232d76dfed82a66fd1e52ddadbc4296c954d5f'
    url='https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/8352c757fac743fe61695838c2393d2cc32ec96f/integration/b8d6db904ba34f6f8dfefe14c5a90cab/artifacts/reports/'+archive_sha+'/report-'+archive_sha+'.zip'
    try:
        r=requests.get(url,headers={'Authorization':'Bearer '+token},stream=True,timeout=40)
        r.raise_for_status();signed=r.url;r.close()
        if signed==url:raise RuntimeError('expected a signed artifact redirect')
    except BaseException as exc:
        raise RuntimeError('authenticated read URL failed: '+type(exc).__name__) from None
    spec=dict(url=signed,root=a.remote_root,archive_sha256=archive_sha)
    submit="import socket,sys; s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.connect("+repr(str(Path(a.remote_root)/'download.sock'))+"); s.sendall(sys.stdin.buffer.read()); s.shutdown(socket.SHUT_WR); s.close(); print('EPHEMERAL_READ_URL_SUBMITTED')"
    command=['ssh','-o','BatchMode=yes','-o','ControlMaster=auto','-o','ControlPersist=600',
        '-o','ControlPath=/tmp/plan_a_bc_ssh_master','-p',a.port,a.host,
        '/venv/main/bin/python -u -c '+shlex.quote(submit)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,text=True)
    process.stdin.write(json.dumps(spec));process.stdin.close()
    if process.wait():raise RuntimeError('remote pinned dataset download failed (URL omitted)')


if __name__=='__main__':main()
