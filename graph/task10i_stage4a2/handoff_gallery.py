"""Saved0250 partial-prompt failure, current-event handoff and Oracle control."""
import argparse
from pathlib import Path
import cv2
import h5py
import numpy as np
from PIL import Image,ImageDraw
from .practical_gallery import tile
from .reliability_gallery import graph_tile
from .sam2_gallery import panel
from graph.task10i_stage4a.run_pipeline import read,write,sha,STAGE3,CAMERAS


def run(root,development,handoff,partial,references):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False);eid='10i_stage3_10i_s1_0250_full_task';candidate='BC_WB_HANDOFF'
    e=next(v for v in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values() if str(v['episode_id'])==eid)
    folder=development/'development/perception'/eid
    masks={}
    for label,p in [('B0',folder/'B0.npz'),(candidate,folder/(candidate+'.npz')),
                    ('old_BC_KEEP',Path('/workspace/task10i_stage4a2_newdev_C_KEEP_20261101_v1/run/development/perception')/eid/'BC_KEEP.npz')]:
        with np.load(p) as z:masks[label]=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
    with np.load(partial/'ORACLE_COMPLETE_BOX_T6.npz') as z:
        oracle=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,128,128).astype(bool);start=int(z['start_timestep'])
    ref=references/'references'/eid;rr=read(ref/'receipt.json')
    with np.load(ref/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
    events=read(folder/candidate/'baseline_handoff_events.json');accepted=[r for r in events if r['accepted']]
    assert accepted and accepted[0]['timestep']==15
    protocol=read(partial/'protocol.json');times=[0,6,14,15,16,95,e['actions']]
    config=dict(source_sha256=sha(Path(__file__)),historical_Development_only=True,oracle_label='ORACLE_DIAGNOSTIC complete GT box at6, never a practical trigger',
        practical_handoff='current new original initialization plus support and frozen past-reference comparison',
        handoff_proof_sha256=sha(handoff/'scope.json'),partial_proof_sha256=sha(partial/'scope.json'))
    size=(256*10,310*2+64);video=root/'0250_partial_prompt_handoff.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),5,size);assert writer.isOpened()
    strip=Image.new('RGB',(size[0],size[1]*len(times)),'#eeeeee')
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_partial_prompt_handoff_gallery',seed=42,
            root=root/'experiments',job_type='analysis',config=config) as logger:
        try:
            with h5py.File(e['raw_path']) as raw:
                for t in range(e['actions']+1):
                    frame=Image.new('RGB',size,'#eeeeee');draw=ImageDraw.Draw(frame)
                    kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED: no GT error colors/metrics'
                    draw.text((4,3),f'0250 t={t} {kind}: failed partial anchor vs current-event handoff; original Wrist initialization15, not GT error trigger16',fill='black')
                    draw.text((4,19),'GT/correct box branch is ORACLE_DIAGNOSTIC only. Practical candidate never reads it. Original/full failed BC result preserved.',fill='black')
                    draw.text((4,35),'Event at15 resumes already-running original SAM memory. Similarity to past partial Peg .823931; Robot negative .613650. Video5fps only.',fill='black')
                    for ci,camera in enumerate(CAMERAS):
                        rgb=raw[camera+'/rgb'][t];target=np.isin(seg[t,ci],rr['object_ids']['peg']) if exact[t] else np.zeros((128,128),bool)
                        gt=rgb.copy();gt[target]=[30,190,255]
                        dp=tile(rgb,'Actual partial prompt at6' if ci==1 else 'Main Peg remains B0',['No prompt drawn at other frames'])
                        if ci==1 and t==6:
                            b=protocol['practical_box'];ImageDraw.Draw(dp).rectangle([2*b[0],54+2*b[1],2*b[2],54+2*b[3]],outline='yellow',width=3)
                        mp=tile(rgb,'Actual Re-ID event now',['Peg .823931 / Robot .613650','ACCEPT resume B0 memory'] if ci==1 and t==15 else ['NOT_EXECUTED at current t'])
                        sam=[]
                        for label in ('B0','old_BC_KEEP',candidate):
                            if exact[t]:
                                im=Image.new('RGB',(256,310),'white');im.paste(panel(rgb,masks[label][t,ci,0],target,label),(0,20))
                            else:
                                preview=rgb.copy();preview[masks[label][t,ci,0]]=[150,150,255]
                                im=tile(preview,label,['GT_UNSUPPORTED','blue prediction; no error classification'])
                            sam.append(im)
                        om=Image.new('RGB',(256,310),'white')
                        if ci==1 and t>=start and exact[t]:om.paste(panel(rgb,oracle[t-start],target,'ORACLE complete GT box6'),(0,20))
                        else:om=tile(rgb,'ORACLE_DIAGNOSTIC',['Wrist branch starts6; no claim here'])
                        graphs=[graph_tile(development/'paired_graph',eid,label,t,camera,0,seg[t,ci],rr,rgb) if exact[t] else
                            tile(rgb,label+' Graph',['GT_UNSUPPORTED','No semantic point colors']) for label in ('B0',candidate)]
                        tiles=[tile(rgb,'Raw '+camera),tile(gt,'GT Peg display only'),dp,mp,*sam,om,
                            *graphs]
                        for x,im in enumerate(tiles):frame.paste(im,(256*x,64+310*ci))
                    writer.write(cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR))
                    if t in times:strip.paste(frame,(0,size[1]*times.index(t)))
        finally:writer.release()
        png=root/'0250_partial_prompt_handoff.png';strip.save(png)
        cap=cv2.VideoCapture(str(video));count=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape[:2]==(size[1],size[0]);count+=1
        cap.release();assert count==e['actions']+1
        with Image.open(png) as im:assert im.size==(size[0],size[1]*len(times))
        write(root/'gallery_index.json',dict(status='PARTIAL_PROMPT_FAILURE_HANDOFF_ORACLE_GALLERY_COMPLETE',run_id=logger.run_id,
            **config,times=times,decoded_frames=count,exact_frames=int(exact.sum()),unsupported_frames=int((~exact).sum()),
            unsupported_frames_never_GT_error_colored=True,png=png.name,png_sha256=sha(png),mp4=video.name,mp4_sha256=sha(video),
            handoff_events=events,whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','development','handoff','partial','references'):p.add_argument(n,type=Path)
    run(**vars(p.parse_args()))
