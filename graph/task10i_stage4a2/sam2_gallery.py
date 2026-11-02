"""Scientific frame strips and full-prefix videos from saved control outputs only."""
import argparse
from pathlib import Path
import cv2
import h5py
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
from .sam2_controls import CONDITIONS
from .prompt_audit import AUDITS
from graph.task10i_stage4a.run_pipeline import read,write,sha,STAGE3,CAMERAS


def panel(rgb,mask,target,title):
    image=rgb.copy()
    if mask is not None:
        color=np.zeros_like(image);color[mask&target]=[0,255,0];color[mask&~target]=[255,40,40]
        image[mask]=(image[mask]*.55+color[mask]*.45).astype(np.uint8)
    edge=target&~binary_erosion(target);image[edge]=[40,220,255]
    image=Image.fromarray(image).resize((256,256),Image.Resampling.NEAREST)
    canvas=Image.new('RGB',(256,290),'white');canvas.paste(image,(0,34))
    draw=ImageDraw.Draw(canvas);draw.text((4,3),title,fill='black')
    if mask is not None:
        tp=int((mask&target).sum());fp=int((mask&~target).sum());fn=int((~mask&target).sum())
        draw.text((4,17),f'TP {tp} FP {fp} IoU {tp/(tp+fp+fn):.3f}' if tp+fp+fn else 'IoU undefined',fill='black')
    return canvas


def run(output,controls):
    output.mkdir(parents=True,exist_ok=False)
    assert read(controls/'scope.json')['status']=='SAM2_ORACLE_MEMORY_CONTROLS_COMPLETE'
    windows=read(controls/'registered_windows.json')['windows']
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    gallery=[]
    names={'B0':'B0 original','ORACLE_CORRECT_INIT':'GT box at t0','ORACLE_WRONG_INIT':'Wrong actor init',
        'ORACLE_CORRECTION_EXISTING_MEMORY':'Naive existing-memory correction',
        'ORACLE_CORRECTION_FRESH_MEMORY':'Fresh memory GT box'}
    for w in windows:
        eid=w['episode'];e=entries[eid];ci=CAMERAS.index(w['target_camera']);oid=w['target_oid']
        rrpath=Path(read(AUDITS/'corrected/perception'/eid/'receipt.json')['corrected_reference']);rr=read(rrpath)
        with np.load(rrpath.parent/'segmentation.npz') as z:
            seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
        masks={}
        for condition in CONDITIONS:
            with np.load(controls/'episodes'/eid/(condition+'.npz')) as z:
                masks[condition]=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
        timesteps=sorted({0,max(0,w['correction_t']-1),w['correction_t'],
            min(w['last_t'],w['correction_t']+1),w['last_t']//2,w['last_t']})
        strip=Image.new('RGB',(256*7,320*len(timesteps)),'#dddddd')
        video=output/(eid+'.mp4')
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),5,(256*7,320))
        assert writer.isOpened()
        with h5py.File(e['raw_path']) as f:
            try:
                for t in range(w['last_t']+1):
                    rgb=f[w['target_camera']+'/rgb'][t];target=np.isin(seg[t,ci],rr['object_ids'][('peg','box')[oid]])
                    wrist=f[('hand_camera' if ci==0 else 'base_camera')+'/rgb'][t]
                    tiles=[panel(rgb,None,target,'Raw target camera / cyan GT edge'),
                           panel(wrist,None,np.zeros((128,128),bool),'Other camera raw')]
                    tiles.extend(panel(rgb,masks[c][t,ci,oid],target,names[c]) for c in CONDITIONS)
                    frame=Image.new('RGB',(256*7,320),'white');draw=ImageDraw.Draw(frame)
                    draw.text((5,5),f'{eid} t={t} target={w["target_camera"]}/{oid} correction={w["correction_t"]} ref={"EXACT_GT" if exact[t] else "GT_UNSUPPORTED"} -- ORACLE DIAGNOSTIC',fill='black')
                    for i,tile in enumerate(tiles):frame.paste(tile,(256*i,30))
                    writer.write(cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR))
                    if t in timesteps:strip.paste(frame,(0,320*timesteps.index(t)))
            finally:writer.release()
        path=output/(eid+'.png');strip.save(path)
        # Reopen both formats rather than trusting successful encoder creation.
        with Image.open(path) as image:assert image.size==(256*7,320*len(timesteps))
        cap=cv2.VideoCapture(str(video));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==w['last_t']+1
        gallery.append(dict(episode=eid,png=str(path),png_sha256=sha(path),mp4=str(video),
            mp4_sha256=sha(video),frames=count,timesteps=timesteps,playback_fps=5,
            not_20hz_runtime=True,oracle_not_practical=True))
    write(output/'gallery_index.json',dict(cases=gallery,controls_scope_sha256=sha(controls/'scope.json'),
        successful_and_unresolved_cases_retained=True,graph_panels_pending=True,dinov2_rank_panels_pending=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('controls',type=Path)
    a=p.parse_args();run(a.output,a.controls)
