"""Chunked SSH transfer into a new experiment directory, with SHA verification.

Some hosts close bulk SFTP sessions. Small command-channel chunks avoid that
transport limit. This never transmits credentials or publishes an artifact.
"""
import argparse
import base64
from pathlib import Path
import shlex
import subprocess
from .common import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',required=True)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--remote-root',required=True)
    a=p.parse_args();data=a.archive.read_bytes();chunks=[data[i:i+32768] for i in range(0,len(data),32768)]
    remote=Path(a.remote_root);name=a.archive.name;ident=sha(a.archive)
    def run(script):
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-p',a.port,a.host,
            'python3 -c '+shlex.quote(script)],capture_output=True,text=True,timeout=50)
        if r.returncode:raise RuntimeError('SSH transfer failed: '+r.stderr[-300:])
        return r.stdout
    for i,chunk in enumerate(chunks):
        path=remote/'transfer_parts'/ident/f'{i:05d}'
        script="from pathlib import Path; import base64,hashlib; p=Path("+repr(str(path))+"); b=base64.b64decode("+repr(base64.b64encode(chunk).decode())+"); p.parent.mkdir(parents=True,exist_ok=True); old=p.read_bytes() if p.exists() else None; assert old is None or old==b; p.write_bytes(b)"
        run(script)
        if i%5==0:print(f'transfer {name} {i+1}/{len(chunks)}',flush=True)
    script="from pathlib import Path; import hashlib,tarfile; r=Path("+repr(str(remote))+"); b=b''.join((r/'transfer_parts'/"+repr(ident)+"/f'{i:05d}').read_bytes() for i in range("+str(len(chunks))+")); assert hashlib.sha256(b).hexdigest()=="+repr(ident)+"; p=r/"+repr(name)+"; p.write_bytes(b); t=tarfile.open(p); assert all(not n.startswith('/') and '..' not in Path(n).parts for n in t.getnames()); t.extractall(r,filter='data'); print('TRANSFER_SHA_PASS',hashlib.sha256(b).hexdigest())"
    print(run(script),flush=True)


if __name__=='__main__':main()
