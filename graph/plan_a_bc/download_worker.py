"""One-shot supervised pinned download; signed URL enters an ephemeral socket."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import urllib.request
import zipfile
from .common import sha,read,write,now


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;sock_path=root/'download.sock';server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    server.bind(str(sock_path));os.chmod(sock_path,0o600);server.listen(1)
    print('WAITING_FOR_EPHEMERAL_READ_URL',flush=True)
    try:
        conn,_=server.accept();data=b''
        with conn:
            while True:
                b=conn.recv(16384)
                if not b:break
                data+=b
        spec=json.loads(data);archive=root/'original_dataset_published.zip'
        if not archive.exists():
            partial=root/'original_dataset_published.partial';offset=partial.stat().st_size if partial.exists() else 0
            headers={'Range':f'bytes={offset}-'} if offset else {}
            response=urllib.request.urlopen(urllib.request.Request(spec['url'],headers=headers),timeout=60)
            append=offset and response.status==206
            if not append:offset=0
            with partial.open('ab' if append else 'wb') as f:
                total=offset;next_log=total+(32<<20)
                while True:
                    b=response.read(2<<20)
                    if not b:break
                    f.write(b);total+=len(b)
                    if total>=next_log:
                        print('downloaded_bytes',total,flush=True);next_log=total+(32<<20)
            if sha(partial)!=spec['archive_sha256']:raise ValueError('published archive SHA mismatch')
            os.link(partial,archive);partial.unlink()
        if sha(archive)!=spec['archive_sha256']:raise ValueError('published archive changed')
        manifest=read(root/'dataset/dataset_manifest.json')
        with zipfile.ZipFile(archive) as z:
            for ep in manifest['episodes']:
                target=root/'dataset'/ep['path']
                if target.exists() and sha(target)==ep['sha256']:continue
                # The published bundle explicitly uses episodes/<basename>;
                # the canonical local manifest uses collection/episodes/.
                published='episodes/'+Path(ep['path']).name
                matches=[n for n in z.namelist() if n==published]
                if len(matches)!=1:raise ValueError('episode archive path ambiguity')
                target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_suffix('.hf_partial')
                with z.open(matches[0]) as src,partial.open('wb') as dst:shutil.copyfileobj(src,dst)
                if sha(partial)!=ep['sha256']:raise ValueError('original HDF5 SHA mismatch')
                os.replace(partial,target)
        write(root/'dataset_transfer_verification.json',dict(status='PASS',episodes=len(manifest['episodes']),
            archive_sha256=spec['archive_sha256'],dataset_sha256=sha(root/'dataset/dataset_manifest.json'),
            source='existing published original dataset; authenticated read only; no publishing',completed_at=now()),immutable=True)
        print('ORIGINAL_120_EPISODES_SHA_PASS',flush=True)
    except BaseException as exc:
        # Exception URLs can contain signatures; keep only the exception class.
        write(root/'dataset_transfer_failure.json',dict(status='FAIL',error_type=type(exc).__name__,time=now()))
        print('DOWNLOAD_FAILED',type(exc).__name__,flush=True);raise SystemExit(1)
    finally:server.close();sock_path.unlink(missing_ok=True)


if __name__=='__main__':main()
