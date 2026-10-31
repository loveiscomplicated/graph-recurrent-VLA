"""Evidence media from accepted raw episodes and frozen paired measurements."""
import argparse,csv
from pathlib import Path
import cv2,h5py,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from graph.plan_a_bc.common import read,write,sha


def run(root):
    pilot=root/'matched_hold_v4';branch='30000_t10_5_0';out=root/'pilot_media';out.mkdir(exist_ok=False)
    receipts={}
    for tag in ['N','P']:
        folder=pilot/'raw'/branch/tag;receipt=read(folder/'receipt.json');assert receipt['approved'] and sha(folder/'episode.h5')==receipt['sha256']
        trace=read(folder/'trace.json');target=np.asarray(receipt['target_refresh']['grasp'][:3]);
        with h5py.File(folder/'episode.h5') as f:
            seg=f['segment'][:];tcp=f['tcp_pose'][:];rgb1=f['base_camera/rgb'][:];rgb2=f['hand_camera/rgb'][:]
        path=out/f'expert_{tag}_{branch}.mp4';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),20,(640,300));assert writer.isOpened()
        for t in range(len(seg)):
            images=[cv2.resize(x,(320,240),interpolation=cv2.INTER_AREA) for x in [rgb1[t],rgb2[t]]]
            image=np.zeros((300,640,3),np.uint8);image[:240]=np.concatenate(images,1)
            stage=['Expert prefix','External hold / disturbance','Expert recovery'][int(seg[t])]
            error=np.linalg.norm(tcp[t,:3]-target)*1000
            cv2.putText(image,f'{tag}: {stage}, t={t}',(10,260),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(image,f'TCP to target {error:.1f} mm',(10,284),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA)
            writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        writer.release();receipts[tag]=dict(path=str(path),sha256=sha(path),frames=len(seg),readback='PASS')
    fig,ax=plt.subplots(figsize=(8,4))
    for tag,color in [('N','#3264a8'),('P','#c94a3b')]:
        folder=pilot/'raw'/branch/tag;receipt=read(folder/'receipt.json')
        with h5py.File(folder/'episode.h5') as f:tcp=f['tcp_pose'][:];segment=f['segment'][:]
        target=np.asarray(receipt['target_refresh']['grasp'][:3]);error=np.linalg.norm(tcp[:,:3]-target,axis=1)*1000
        ax.plot(np.arange(len(error))/20,error,label=tag,color=color)
        prefix=int((segment==0).sum());disturbance=int((segment==1).sum())
    ax.axvspan(prefix/20,(prefix+disturbance)/20,color='#c8c8c8',alpha=.4,label='matched duration')
    ax.set(xlabel='Simulator time (s)',ylabel='TCP to reach target (mm)',title='Same-anchor expert correction');ax.legend();fig.tight_layout();fig.savefig(out/'pilot_recovery_error_curves.png',dpi=160);plt.close(fig)
    rows=list(csv.DictReader(open(pilot/'paired_expert_correction.csv')));fig,ax=plt.subplots(figsize=(8,4))
    groups={}
    for r in rows:
        offset=read(pilot/'raw'/r['branch']/'P/receipt.json')['offset_mm'];label=('X' if offset[0] else 'Y')+('+' if max(offset)>0 else '-')
        groups.setdefault(label,[]).append(float(r['direction_cosine']))
    for i,(label,values) in enumerate(sorted(groups.items())):ax.scatter(np.full(len(values),i),values,label=label,alpha=.75)
    ax.set(xticks=range(len(groups)),xticklabels=sorted(groups),ylim=(0,1.05),ylabel='First expert correction direction cosine',title='Measured P vs matched N correction');fig.tight_layout();fig.savefig(out/'correction_direction.png',dpi=160);plt.close(fig)
    bench=read(root/'parallel_gate/parallel_gate.json')['results'];fig,ax=plt.subplots(figsize=(6,3.5));ax.bar(['serial','worker 1','worker 2'],[x['graphs_per_second'] for x in bench],color=['#888888','#517dba','#2759a0']);ax.set(ylabel='Graphs/s, same 51-frame episode',title='CUDA vision + CPU geometry end-to-end');fig.tight_layout();fig.savefig(out/'gpu_cpu_throughput.png',dpi=160);plt.close(fig)
    receipts['figures']={p.name:dict(sha256=sha(p),readback='PASS') for p in out.glob('*.png')}
    write(out/'media_manifest.json',dict(status='PASS',source_pilot_uuid=read(pilot/'identity.json')['run_id'],branch=branch,receipts=receipts),immutable=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
