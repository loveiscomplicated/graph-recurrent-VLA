"""Independent supervised collection/cache/export; never controls original jobs."""
import argparse,subprocess,sys,time,shutil,traceback
from pathlib import Path
from graph.plan_a_bc.common import read,write,sha,now

def run(root,source,dino,sam):
    children=[];streams=[]
    def launch(name,module,args):
        out=(root/(name+'.log')).open('a');streams.append(out)
        p=subprocess.Popen([sys.executable,'-u','-B','-m',module,*args],stdout=out,stderr=subprocess.STDOUT)
        children.append((name,p));write(root/(name+'_pid.json'),dict(pid=p.pid,at=now()))
    def collect(w):launch(f'collect_{w}','graph.gt_pretrain10f1.remediation_collect',['--root',str(root),'--worker',str(w),'--workers','4'])
    try:
        for w in [1,2,3]:collect(w)
        started0=False;pilot=root/'engineering_pilot';cache=False
        while True:
            for name,p in children:
                rc=p.poll()
                if rc is not None and rc!=0:
                    write(root/(name+'.failure.json'),dict(returncode=rc,at=now()),immutable=True)
                    raise RuntimeError(name+' failed; original jobs remain untouched')
            if not started0 and (pilot/'collection_worker0.json').exists():
                assert read(pilot/'collection_worker0.json')['status']=='PASS'
                # Adopt only completed, hash-verified direct-capture pilot episodes.
                for receipt in (pilot/'raw').glob('*/receipt.json'):
                    r=read(receipt)
                    for n,h in r['hashes'].items():assert sha(receipt.parent/n)==h
                    shutil.copytree(receipt.parent,root/'raw'/receipt.parent.name)
                shutil.copytree(pilot/'snapshots/30000',root/'snapshots/30000',dirs_exist_ok=True)
                write(root/'engineering_adoption.json',dict(status='PASS',episodes=len(list((pilot/'raw').glob('*/receipt.json'))),
                    pilot=str(pilot),all_hash_readback=True,GT_captured_directly=True),immutable=True)
                started0=True;collect(0)
            if not cache:
                launch('cache','graph.gt_pretrain10f1.remediation_cache',['--root',str(root),'--source',str(source),'--dino',str(dino),'--sam',str(sam)]);cache=True
            if started0 and all(p.poll()==0 for _,p in children):break
            write(root/'pipeline_progress.json',dict(status='RUNNING',completed_raw=len(list((root/'raw').glob('*/receipt.json'))),
                completed_graph=len(list((root/'cache').glob('*/receipt.json'))),pilot_handoff=started0,at=now()))
            time.sleep(10)
        subprocess.run([sys.executable,'-u','-B','-m','graph.gt_pretrain10f1.remediation_export','--root',str(root),'--source',str(source),'--dino',str(dino),'--sam',str(sam)],check=True)
        write(root/'pipeline_complete.json',dict(status='PASS',DATA_READY=True,at=now()),immutable=True)
    except BaseException:
        write(root/'pipeline_failure.json',dict(status='FAIL',traceback=traceback.format_exc(),at=now()))
        raise
    finally:
        # Only this independent pipeline's own children, never any existing job.
        for _,p in children:
            if p.poll() is None:p.terminate()
        for s in streams:s.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--dino',type=Path,required=True);p.add_argument('--sam',type=Path,required=True)
    a=p.parse_args();run(a.root,a.source,a.dino,a.sam)
