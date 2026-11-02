"""Broad actual Development cases; absent failure categories stay explicitly absent."""
import argparse
from collections import defaultdict
from pathlib import Path
import html
import h5py
import numpy as np
from PIL import Image,ImageDraw
from .run_pipeline import CAMERAS,STAGE3,read,write
from .gallery import overlay


def run(output,audits,candidate):
    output.mkdir(parents=True,exist_ok=False)
    baseline=audits/'baseline';corrected=audits/'corrected';categories=defaultdict(list)
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    temporal=read(audits/'extended/temporal_frame_metrics.json')
    byseq=defaultdict(list)
    for r in temporal:
        eid=r['episode'];m=r['metrics'];case=(eid,r['timestep'],r['parent_id'])
        byseq[eid,r['camera'],r['object_id']].append(r)
        if r['object_id']==0 and m['robot_pixels']>=16:categories['Peg Robot contamination'].append(case)
        if r['object_id']==1 and r['camera']=='base_camera' and m['background_pixels']+m['robot_pixels']>=16:categories['Main Box contamination'].append(case)
        if m['input_gt_support']==0 and m['output_support']>0:categories['Temporal ghost or wrong object while target absent'].append(case)
    for key,seq in byseq.items():
        for a,b in zip(seq,seq[1:]):
            case=(b['episode'],b['timestep'],b['parent_id'])
            if a['metrics']['true_positive']>0 and b['metrics']['true_positive']==0 and b['metrics']['input_gt_support']>=8:categories['Sudden loss of correct mask'].append(case)
            if a['dominant_source']=='correct' and b['dominant_source'] in ('robot','other_object'):categories['Dominant actor identity change'].append(case)
    for r in read(audits/'filters/wrist_gate_frame_metrics.json'):
        if r['quality']=='good_2d' and not r['accepted']:categories['Good Wrist mask rejected by Gate'].append((r['episode'],r['timestep'],r['parent_id']))
    for r in read(corrected/'development_representative_points.json'):
        if r.get('visible_reference_points',0)>=8 and r['correct']==0:
            categories[('Peg' if r['object_id']==0 else 'Box')+' visible but Graph has no correct point'].append((r['episode'],r['timestep'],r['parent_id']))
    for p in (corrected/'perception').glob('*/stage_rows.json'):
        grouped=defaultdict(dict)
        for r in read(p):grouped[r['episode'],r['timestep'],r['camera'],r['object_id']][r['stage']]=r
        for key,rr in grouped.items():
            if key[2]=='hand_camera' and key[3]==1:continue
            if rr['sam2']['metrics']['true_positive']>=32 and rr['final_graph_sources']['metrics']['true_positive']<6:
                categories['Good 2D support but sparse final Graph'].append((key[0],key[1],rr['sam2']['parent_id']))
    for r in read(audits/'extended/temporal_episode_metrics.json'):
        for event in r['reappearances']:
            if not event['recovered']:categories['Unrecovered reappearance'].append((r['episode'],event['start'],r['parent_id']))
    required=['Peg visible but Graph has no correct point','Box visible but Graph has no correct point',
        'Peg Robot contamination','Main Box contamination','Sudden loss of correct mask','Dominant actor identity change',
        'Good Wrist mask rejected by Gate','Good 2D support but sparse final Graph',
        'Temporal ghost or wrong object while target absent','Unrecovered reappearance']
    selected=[];parents=set();sections=[]
    for category in required:
        cases=sorted(set(categories[category]));fresh=[c for c in cases if c[2] not in parents]
        if not cases:
            selected.append(dict(category=category,status='NOT_OBSERVED_UNDER_REGISTERED_DEFINITION',available=0));continue
        eid,t,parent=(fresh or cases)[0];parents.add(parent);e=entries[eid]
        rrpath=Path(read(corrected/'perception'/eid/'receipt.json')['corrected_reference']);rr=read(rrpath)
        with h5py.File(e['raw_path'],'r') as f,np.load(rrpath.parent/'segmentation.npz',allow_pickle=False) as gt,np.load(baseline/'perception'/eid/'masks.npz',allow_pickle=False) as b,np.load(candidate/'perception'/eid/'masks.npz',allow_pickle=False) as c:
            assert gt['exact_frame'][t]
            bm=np.unpackbits(b['mask_bits'][t]).reshape(2,2,128,128).astype(bool)
            cm=np.unpackbits(c['mask_bits'][t]).reshape(2,2,128,128).astype(bool)
            image=Image.new('RGB',(6*256,2*294+48),(24,24,24));draw=ImageDraw.Draw(image)
            titles=['Raw RGB','EXACT_GT Peg/Box','Baseline SAM2','Candidate mask','Baseline Graph sources','Candidate Graph sources']
            for j,title in enumerate(titles):draw.text((j*256+4,8),title,fill='white')
            counts=[]
            for ci,camera in enumerate(CAMERAS):
                truth=np.stack([np.isin(gt['actor_segmentation'][t,ci],rr['object_ids'][name]) for name in ('peg','box')])
                graphs=[]
                for root in (baseline,candidate):
                    with np.load(root/'traces'/eid/f't{t:04d}/support.npz',allow_pickle=False) as z:
                        mask=np.zeros((2,128,128),bool)
                        for oid in (0,1):mask[oid].ravel()[z[camera+'__'+str(oid)+'__final_graph_sources']]=True
                        graphs.append(mask)
                rgb=f[camera+'/rgb'][t]
                panels=[Image.fromarray(rgb).resize((256,256),Image.Resampling.NEAREST)]+[overlay(rgb,x) for x in (truth,bm[ci],cm[ci],*graphs)]
                for j,panel in enumerate(panels):image.paste(panel,(j*256,40+ci*294))
                counts.append(dict(camera=camera,GT=truth.sum((1,2)).tolist(),baseline_mask=bm[ci].sum((1,2)).tolist(),candidate_mask=cm[ci].sum((1,2)).tolist(),baseline_graph_sources=graphs[0].sum((1,2)).tolist(),candidate_graph_sources=graphs[1].sum((1,2)).tolist()))
                draw.text((5,40+ci*294+260),f'{camera}: t={t}, red=Peg / blue=Box; Wrist Box Graph excluded by contract',fill='white')
        name=f'case{len(sections):02d}.png';image.save(output/name)
        selected.append(dict(category=category,status='EXACT_GT_CASE',episode=eid,parent_id=parent,timestep=t,available=len(cases),image=name,counts=counts))
        sections.append(f'<h2>{html.escape(category)}</h2><p>{html.escape(eid)} / {html.escape(parent)} / t={t}; EXACT_GT; available cases={len(cases)}</p><img src="{name}" style="width:100%;max-width:1536px">')
    missing=[r['category'] for r in selected if r['available']==0]
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Stage4A actual failures</title><body style="background:#181818;color:white;font-family:sans-serif"><h1>Development failure evidence</h1><p>Source-pixel Graph support is distinct from spatial surface coverage. Cases are diagnostic, not independent samples.</p>'+''.join(sections)+'<h2>Not observed</h2><p>'+html.escape(', '.join(missing))+'</p></body>')
    write(output/'case_manifest.json',dict(status='BROAD_FAILURE_GALLERY_FINISHED',cases=selected,distinct_parents=len(parents),validation_used=False,missing_categories_not_fabricated=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('audits',type=Path);p.add_argument('candidate',type=Path)
    a=p.parse_args();run(a.output,a.audits,a.candidate)
