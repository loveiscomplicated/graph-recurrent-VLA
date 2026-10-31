"""Read-only SSH collection and local SHA readback. No remote worker launch."""
import argparse,json,hashlib,subprocess,time,re
from pathlib import Path

HOST='root@104.233.209.157';PORT='13159'
BASE='/workspace/task10h_phase_handoff_20261031_v5'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ssh(command):
    try:return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p',PORT,HOST,command],text=True,capture_output=True,timeout=40)
    except subprocess.TimeoutExpired:return subprocess.CompletedProcess([],255,stdout='',stderr='SSH timeout')

def collect(local,doc):
    local.mkdir(parents=True,exist_ok=True)
    r=subprocess.run(['rsync','-az','-e','ssh -p '+PORT+' -o BatchMode=yes -o ConnectTimeout=20',HOST+':'+BASE+'/',str(local)+'/'],capture_output=True,text=True,timeout=600)
    if r.returncode:raise RuntimeError(r.stderr[-1000:])
    audit=json.loads((local/'completion_audit.json').read_text())
    if audit['status']!='COMPLETE':raise ValueError('remote completion absent')
    artifacts=json.loads((local/'artifact_index.json').read_text())
    for row in artifacts:
        rel=Path(row['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        p=local/rel;assert p.stat().st_size==row['size'] and sha(p)==row['sha256'],row['path']
    source=local/'10h_phase_conditioned_handoff_diagnosis.md'
    text=source.read_text()
    prefix='../../outputs/task10h/final_results/'
    text=re.sub(r'\]\((?!https?:|#)([^)]+)\)',lambda m:']('+prefix+m.group(1)+')',text)
    doc.parent.mkdir(parents=True,exist_ok=True);doc.write_text(text)
    (local.parent/'independent_completion_audit.json').write_text(json.dumps(dict(status='PASS',files=len(artifacts),sha256_readback=True,report_source_sha256=sha(source),document_sha256=sha(doc),document=str(doc)),indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[2];dest=root/'outputs/task10h/final_results';doc=root/'docs/visual-graph-integration/10h_phase_conditioned_handoff_diagnosis.md'
    while True:
        r=ssh('cat '+BASE+'/progress.json; test ! -f '+BASE+'/execution_failure.json')
        print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),r.stdout.strip(),flush=True)
        if r.returncode==0 and '"COMPLETE"' in r.stdout:
            collect(dest,doc);print('COLLECTED_AND_SHA_VERIFIED',flush=True);return
        if r.returncode==1:
            raise RuntimeError('remote failure: inspect execution_failure.json; no completion claimed')
        if not a.wait:return
        time.sleep(50)

if __name__=='__main__':main()
