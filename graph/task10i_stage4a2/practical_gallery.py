"""Paired practical success/failure strips and decoded full-prefix MP4s."""
import argparse
from pathlib import Path
import cv2
import h5py
import numpy as np
from PIL import Image,ImageDraw
from .sam2_gallery import panel
from .prompt_audit import AUDITS
from graph.task10i_stage4a.run_pipeline import read,write,sha,STAGE3,CAMERAS

CASES=[dict(episode='10i_stage3_10i_s1_0924_full_task',oid=0,last=20,times=[0,1,5,6,10,11,20]),
       dict(episode='10i_stage3_10i_s1_0121_full_task',oid=1,last=106,times=[0,1,5,6,94,95,106]),
       dict(episode='2000003',oid=1,last=70,times=[0,17,20,54,61,62,70])]


def tile(rgb,title,lines=()):
    canvas=Image.new('RGB',(256,310),'white');canvas.paste(Image.fromarray(rgb).resize((256,256),Image.Resampling.NEAREST),(0,54))
    draw=ImageDraw.Draw(canvas);draw.text((3,2),title,fill='black')
    for i,line in enumerate(lines[:3]):draw.text((3,15+i*12),str(line),fill='black')
    return canvas


def graph_pixels(root,eid,label,t,camera,oid,seg,rr,rgb):
    path=root/'graphs'/eid/label/f't{t:04d}.npz'
    with np.load(path) as z:objects=z['object_id'].copy()
    provenance=read(path.with_name(path.stem+'_provenance.json'))
    assert len(objects)==len(provenance)
    image=rgb.copy();counts=dict(correct=0,robot=0,wrong_object=0,background=0)
    for actor,sources in zip(objects,provenance):
        if actor!=oid:continue
        for episode,step,c,pixel in sources:
            assert str(episode)==eid and step==t
            if c!=camera:continue
            y,x=divmod(pixel,128);value=int(seg[y,x])
            name='correct' if value in rr['object_ids'][('peg','box')[oid]] else 'robot' if value in rr['robot_ids'] else 'wrong_object' if value in rr['object_ids'][('box','peg')[oid]] else 'background'
            counts[name]+=1;image[y,x]=dict(correct=[20,255,20],robot=[255,0,0],wrong_object=[255,0,255],background=[255,160,0])[name]
    return tile(image,label+' Graph current source pixels',[str(counts),'green correct/red robot/purple other'])


def run(output,keep,suppress):
    from utils.experiment_logger import ExperimentLogger
    output.mkdir(parents=True,exist_ok=False)
    assert read(keep/'scope.json')['status']=='HISTORICAL_C_BC_ABLATION_COMPLETE'
    rows=read('/workspace/task10i_stage4a_initialization_bottlenecks_20261101_v1/run/t0_reconstruction_rows.json')
    initial={(r['episode'],r['camera'],r['object_id']):r for r in rows}
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    config=dict(source_sha256=sha(Path(__file__)),cases=CASES,GT_posthoc_only=True,
        keep_scope_sha256=sha(keep/'scope.json'),suppress_scope_sha256=sha(suppress/'scope.json'),
        selection='fixed historical causal controls, both initial miss and contaminated initialization and healthy suppression failure',
        condition='C_KEEP=BC_KEEP for these windows; actual past-reference comparisons0, never invent similarity ranks')
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_practical_gallery',seed=42,
            root=output/'experiments',job_type='analysis',config=config) as logger:
        cases=[]
        for case in CASES:
            eid=case['episode'];entry=entries[eid];oid=case['oid'];folder=keep/'episodes'/eid
            assert sha(entry['raw_path'])==entry['raw_sha256']
            ref=Path(read(AUDITS/'corrected/perception'/eid/'receipt.json')['corrected_reference']);rr=read(ref)
            with np.load(ref.parent/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
            masks={}
            for label,path in [('B0',folder/'B0.npz'),('C_KEEP',folder/'C.npz'),('C_SUPPRESS',suppress/'episodes'/eid/'C.npz')]:
                with np.load(path) as z:masks[label]=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
            events=read(folder/'C/events.json');eventmap={(r['timestep'],r['camera'],r['object_id']):r for r in events}
            dimensions=(256*9,310*2+62)
            video=output/(eid+'.mp4');writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),5,dimensions);assert writer.isOpened()
            strip=Image.new('RGB',(dimensions[0],dimensions[1]*len(case['times'])),'#dddddd')
            frame_data=[]
            with h5py.File(entry['raw_path']) as raw:
                try:
                    for t in range(case['last']+1):
                        assert exact[t],'this fixed gallery requires exact historical frames'
                        frame=Image.new('RGB',dimensions,'#eeeeee');draw=ImageDraw.Draw(frame)
                        accepted=[(r['camera'],r['object_id']) for r in events if r['timestep']==t and r['accepted']]
                        draw.text((5,3),f'{eid} t={t} EXACT_GT; C_KEEP accepted now={accepted}; RGB-D128; video5fps is presentation only',fill='black')
                        draw.text((5,18),'DINOv2: actual past-reference similarity NOT_EXECUTED in these historical windows (no similarity rank). No GT used by C_KEEP.',fill='black')
                        draw.text((5,33),'SAM overlays: green TP / red FP / cyan GT edge. C_SUPPRESS is preserved failed negative control, not the challenger.',fill='black')
                        record=dict(timestep=t,accepted_now=accepted,cameras=[])
                        for ci,camera in enumerate(CAMERAS):
                            rgb=raw[camera+'/rgb'][t];target=np.isin(seg[t,ci],rr['object_ids'][('peg','box')[oid]])
                            gt=rgb.copy()
                            for actor,color in [('robot',[150,150,150]),('box',[250,200,20]),('peg',[30,190,255])]:
                                gt[np.isin(seg[t,ci],rr['robot_ids'] if actor=='robot' else rr['object_ids'][actor])]=color
                            event=eventmap[t,camera,oid] if (t,camera,oid) in eventmap else None
                            chosen=None;rawpath=None;source='No DINO request at this t'
                            if t==0:
                                row=initial[eid,camera,oid];rawpath=Path(row['cached_forward_source']);chosen=row['chosen_box'];source='B0 actual t0 prompt'
                            elif event and event['detection_requested']:
                                rawpath=Path(event['raw_query_artifact']);chosen=event.get('chosen',{}).get('box');source='C redetection (cache allowed)'
                            detector=tile(rgb.copy(),'DINO raw postprocessed Top5',[source])
                            prompt=tile(rgb.copy(),'SAM actual selected prompt',[source,'None unless selected now'])
                            if rawpath:
                                with np.load(rawpath) as z:boxes=z['selected_boxes'][:5].copy();scores=z['selected_scores'][:5].copy()
                                dd=ImageDraw.Draw(detector)
                                for rank,(box,score) in enumerate(zip(boxes,scores)):
                                    rectangle=[2*box[0],54+2*box[1],2*box[2],54+2*box[3]]
                                    dd.rectangle(rectangle,outline=['yellow','orange','magenta','cyan','white'][rank],width=2)
                                    dd.text((rectangle[0],rectangle[1]),f'{rank}:{score:.2f}',fill='black')
                                if chosen:
                                    ImageDraw.Draw(prompt).rectangle([2*chosen[0],54+2*chosen[1],2*chosen[2],54+2*chosen[3]],outline='lime',width=3)
                                else:ImageDraw.Draw(prompt).text((3,41),'NULL / no accepted prompt',fill='red')
                            sam=[]
                            for label in ['B0','C_KEEP','C_SUPPRESS']:
                                p=panel(rgb,masks[label][t,ci,oid],target,label+' '+camera)
                                canvas=Image.new('RGB',(256,310),'white');canvas.paste(p,(0,20));sam.append(canvas)
                            tiles=[tile(rgb,'Raw '+camera),tile(gt,'GT sidecar only',['cyan Peg/yellow Box/gray Robot']),detector,prompt,*sam,
                                graph_pixels(keep,eid,'B0',t,camera,oid,seg[t,ci],rr,rgb),
                                graph_pixels(keep,eid,'C',t,camera,oid,seg[t,ci],rr,rgb)]
                            for i,p in enumerate(tiles):frame.paste(p,(256*i,62+310*ci))
                            record['cameras'].append(dict(camera=camera,DINO_output_for_current_frame=rawpath is not None,
                                raw_query_artifact=str(rawpath) if rawpath else None,
                                raw_query_sha256=sha(rawpath) if rawpath else None,chosen_box=chosen,
                                Wrist_Box_is_not_graph_source=camera=='hand_camera' and oid==1))
                        writer.write(cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR));frame_data.append(record)
                        if t in case['times']:strip.paste(frame,(0,dimensions[1]*case['times'].index(t)))
                finally:writer.release()
            png=output/(eid+'.png');strip.save(png)
            with Image.open(png) as image:assert image.size==(dimensions[0],dimensions[1]*len(case['times']))
            cap=cv2.VideoCapture(str(video));decoded=0
            while True:
                ok,image=cap.read()
                if not ok:break
                assert image.shape[:2]==(dimensions[1],dimensions[0]);decoded+=1
            cap.release();assert decoded==case['last']+1
            write(output/(eid+'_frames.json'),frame_data)
            cases.append(dict(**case,png=png.name,png_sha256=sha(png),mp4=video.name,mp4_sha256=sha(video),decoded_frames=decoded))
        write(output/'gallery_index.json',dict(status='PRACTICAL_CONTROL_GALLERY_COMPLETE',run_id=logger.run_id,
            **{k:v for k,v in config.items() if k!='cases'},registered_cases=CASES,cases=cases,
            success_and_unresolved_failures_retained=True,GT_inference_input=False,
            false_reanchor_case='2000003 has no accepted reanchor; failed control is uncertainty suppression, not an invented reinitialization',
            whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','keep','suppress'):p.add_argument(name,type=Path)
    a=p.parse_args();run(a.output,a.keep,a.suppress)
