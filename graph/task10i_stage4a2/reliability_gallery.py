"""Post-evaluation new-benchmark galleries including gains and failures.

Selection is deterministic from the completed evidence, not used by inference
or the benchmark eligibility gate. Videos retain every current observation.
"""
import argparse
import hashlib
from pathlib import Path
import cv2
import h5py
import numpy as np
from PIL import Image,ImageDraw
from .aggregate_evaluation import csv_rows
from .practical_gallery import tile
from .sam2_gallery import panel
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS


def graph_tile(paired,eid,condition,t,camera,oid,seg,rr,rgb):
    folder=paired/'episodes'/eid/condition/f't{t:04d}'
    with np.load(folder/'graph.npz') as z:objects=z['object_id']
    source=read(folder/'provenance.json');assert source['current_only'] and source['box_main_only']
    image=rgb.copy();counts=dict(correct=0,robot=0,wrong=0,background=0)
    assert len(objects)==len(source['sources'])
    for actor,provenance in zip(objects,source['sources']):
        if actor!=oid:continue
        for episode,step,c,pixel in provenance['sources']:
            assert str(episode)==eid and step==t
            if c!=camera:continue
            y,x=divmod(pixel,128);value=int(seg[y,x])
            label='correct' if value in rr['object_ids'][('peg','box')[oid]] else 'robot' if value in rr['robot_ids'] else 'wrong' if value in rr['object_ids'][('box','peg')[oid]] else 'background'
            counts[label]+=1;image[y,x]=dict(correct=[20,255,20],robot=[255,0,0],wrong=[255,0,255],background=[255,160,0])[label]
    return tile(image,condition+' current Graph',[str(counts),'Wrist Box excluded' if camera=='hand_camera' and oid==1 else 'green correct/red robot/purple other'])


def choose(final,candidate):
    records=[]
    expected=read(final/'expected_counts.json')
    for e in expected['entries']:
        eid=str(e['episode_id']);rows=csv_rows(final/'paired/episodes'/eid/'pixel_metrics.csv')
        lookup={(r['timestep'],r['camera'],r['object_id'],r['condition']):r for r in rows}
        for camera,oid in [('base_camera',0),('hand_camera',0),('base_camera',1)]:
            pair=[(lookup[t,camera,oid,'B0'],lookup[t,camera,oid,candidate]) for t in range(e['actions']+1)]
            pair=[(b,c) for b,c in pair if b['reference_kind']=='EXACT_GT' and b['visible_pixels']>=8]
            if not pair:continue
            gain=float(np.mean([c['correct_mask_availability']-b['correct_mask_availability'] for b,c in pair]))
            damage=sum(b['correct_mask_availability']==1 and c['correct_mask_availability']!=1 for b,c in pair)
            worst=min(pair,key=lambda bc:(bc[1]['iou']-bc[0]['iou'],bc[0]['timestep']))[0]['timestep']
            records.append(dict(episode=eid,parent_id=e['parent_id'],stratum=e['stratum'],camera=camera,object_id=oid,
                availability_gain=gain,healthy_damage_frames=damage,worst_delta_t=int(worst)))
    normal=[r for r in records if r['stratum']=='normal'];stress=[r for r in records if r['stratum']=='stress']
    result=[]
    selectors=[('normal_largest_observed_gain',normal,lambda r:(-r['availability_gain'],r['episode'],r['camera'],r['object_id'])),
        ('normal_largest_healthy_damage',normal,lambda r:(-r['healthy_damage_frames'],r['availability_gain'],r['episode'],r['camera'],r['object_id'])),
        ('stress_largest_healthy_damage_or_worst_gain',stress,lambda r:(-r['healthy_damage_frames'],r['availability_gain'],r['episode'],r['camera'],r['object_id'])),
        ('normal_first_unchanged_slot_control',[r for r in normal if r['availability_gain']==0 and r['healthy_damage_frames']==0],lambda r:(r['episode'],r['camera'],r['object_id']))]
    for label,rows,key in selectors:
        if rows:result.append(dict(selection=label,**sorted(rows,key=key)[0]))
    return result


def run(root,evaluation,candidate):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False);final=evaluation/'final';cases=choose(final,candidate)
    expected=read(final/'expected_counts.json');entries={str(e['episode_id']):e for e in expected['entries']}
    write(root/'case_selection_frozen.json',dict(source_sha256=sha(Path(__file__)),cases=cases,
        selection_after_completed_evaluation=True,GT_not_inference_input=True,not_population_examples=True))
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_new_independent_gallery',seed=42,
            root=root/'experiments',job_type='analysis',config=dict(candidate=candidate,cases=cases,source_sha256=sha(Path(__file__)))) as logger:
        completed=[]
        for i,case in enumerate(cases):
            eid=case['episode'];e=entries[eid];oid=case['object_id'];cpath=final/'candidate/perception'/eid
            ref=final/'references/references'/eid;rr=read(ref/'receipt.json')
            with np.load(ref/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();assert z['exact_frame'].all()
            masks={}
            for name,folder in [('B0',final/'baseline/perception'/eid),(candidate,cpath)]:
                with np.load(folder/(name+'.npz')) as z:masks[name]=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
            events=read(cpath/'events.json');handoffs=read(cpath/'baseline_handoff_events.json')
            prompts=read(cpath/'original_initialization_prompts.json')
            dino=read(cpath/'detector_timing.json')
            times={0,8,19,20,e['actions']//2,e['actions'],case['worst_delta_t']}
            intervention=sorted({r['timestep'] for r in events+handoffs if r['accepted'] and r['object_id']==oid})
            if intervention:times.update([max(0,intervention[0]-1),intervention[0],min(e['actions'],intervention[0]+1)])
            times=sorted(times);size=(256*9,310*2+64)
            basename=f'{i:02d}_{case["selection"]}_{eid}_o{oid}'
            video=root/(basename+'.mp4');writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),5,size);assert writer.isOpened()
            strip=Image.new('RGB',(size[0],size[1]*len(times)),'#eeeeee');trace=[]
            with h5py.File(e['raw_path']) as raw:
                try:
                    for t in range(e['actions']+1):
                        frame=Image.new('RGB',size,'#eeeeee');draw=ImageDraw.Draw(frame)
                        now=[r for r in events+handoffs if r['timestep']==t and r['object_id']==oid and r['accepted']]
                        draw.text((4,3),f'{case["selection"]}: {eid} t={t}, object={oid}; EXACT_GT only for display; accepted current events={len(now)}',fill='black')
                        draw.text((4,19),'Raw128. DINO panel uses actual current-call postprocessed outputs; raw900/token tensors are in separate Development audits.',fill='black')
                        draw.text((4,35),'No request / no reference are explicit. Video5fps is presentation only. Green TP, red FP, cyan GT boundary. Graph is current source.',fill='black')
                        row=dict(timestep=t,current_events=now,cameras=[])
                        for ci,camera in enumerate(CAMERAS):
                            rgb=raw[camera+'/rgb'][t];gtmask=np.isin(seg[t,ci],rr['object_ids'][('peg','box')[oid]])
                            gt=rgb.copy()
                            for ids,color in [(rr['robot_ids'],[150,150,150]),(rr['object_ids']['box'],[250,200,20]),(rr['object_ids']['peg'],[30,190,255])]:gt[np.isin(seg[t,ci],ids)]=color
                            rgbhash=hashlib.sha256(rgb.tobytes()).hexdigest()
                            calls=[r for r in dino if r['decision_timestep']==t and r['input_rgb_sha256']==rgbhash]
                            dp=tile(rgb,'DINO current actual calls',[f'actual forwards={len(calls)}','No call != no raw proposal'])
                            dd=ImageDraw.Draw(dp)
                            for call in calls:
                                for rank,(box,score) in enumerate(zip(call['postprocessed_boxes'][:5],call['postprocessed_scores'][:5])):
                                    rect=[2*box[0],54+2*box[1],2*box[2],54+2*box[3]];dd.rectangle(rect,outline=['yellow','orange','magenta','cyan','white'][rank],width=2)
                                    dd.text((rect[0],rect[1]),f'{rank}:{score:.2f}',fill='black')
                            pe=[r for r in events if r['timestep']==t and r['camera']==camera and r['object_id']==oid]
                            ini=[r for r in prompts if r['timestep']==t and r['camera']==camera and r['object_id']==oid]
                            box=next((r['chosen']['box'] for r in pe if r['accepted']),None)
                            if box is None and ini:box=ini[0].get('box')
                            pp=tile(rgb,'Prompt / handoff at current t',[f'new original init={len(ini)}',f'accepted correction={sum(r["accepted"] for r in pe)}'])
                            if box is not None:ImageDraw.Draw(pp).rectangle([2*box[0],54+2*box[1],2*box[2],54+2*box[3]],outline='lime',width=3)
                            matches=[p for event in pe for p in event['proposals'] if p['similarity'] is not None]
                            matches+= [r for r in handoffs if r['timestep']==t and r['camera']==camera and r['object_id']==oid and r['similarity'] is not None]
                            texts=[f'cos={r["similarity"]:.3f} pass={r["reid_pass"]}' for r in sorted(matches,key=lambda r:-r['similarity'])]
                            mp=tile(rgb,'DINOv2 actual similarity rank',texts if texts else ['NOT_EXECUTED at this t','No future reference fill'])
                            sam=[]
                            for label in ('B0',candidate):
                                canvas=Image.new('RGB',(256,310),'white');canvas.paste(panel(rgb,masks[label][t,ci,oid],gtmask,label),(0,20));sam.append(canvas)
                            tiles=[tile(rgb,'Raw '+camera),tile(gt,'GT display only'),dp,pp,mp,*sam,
                                graph_tile(final/'paired',eid,'B0',t,camera,oid,seg[t,ci],rr,rgb),
                                graph_tile(final/'paired',eid,candidate,t,camera,oid,seg[t,ci],rr,rgb)]
                            for x,im in enumerate(tiles):frame.paste(im,(256*x,64+310*ci))
                            row['cameras'].append(dict(camera=camera,DINO_actual_current_forward_calls=len(calls),similarity_comparisons=matches,
                                actual_prompt=box,new_original_initializations=ini))
                        writer.write(cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR));trace.append(row)
                        if t in times:strip.paste(frame,(0,size[1]*times.index(t)))
                finally:writer.release()
            png=root/(basename+'.png');strip.save(png)
            with Image.open(png) as im:assert im.size==(size[0],size[1]*len(times))
            cap=cv2.VideoCapture(str(video));count=0
            while True:
                ok,im=cap.read()
                if not ok:break
                assert im.shape[:2]==(size[1],size[0]);count+=1
            cap.release();assert count==e['actions']+1
            write(root/(basename+'_frames.json'),trace)
            completed.append(dict(**case,times=times,png=png.name,png_sha256=sha(png),mp4=video.name,mp4_sha256=sha(video),decoded_frames=count))
        write(root/'gallery_index.json',dict(status='NEW_INDEPENDENT_GAIN_DAMAGE_STRESS_CONTROL_GALLERY_COMPLETE',run_id=logger.run_id,
            candidate=candidate,cases=completed,source_sha256=sha(Path(__file__)),GT_posthoc_only=True,
            registered_selection_sha256=sha(root/'case_selection_frozen.json'),success_only_selection=False,
            no_inference_or_selection_rule_changed=True,whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','evaluation'):p.add_argument(n,type=Path)
    p.add_argument('--candidate',required=True);run(**vars(p.parse_args()))
