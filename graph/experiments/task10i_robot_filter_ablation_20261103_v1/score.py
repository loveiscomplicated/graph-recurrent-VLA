"""Evaluate frozen predictions only; GT never enters the Graph or filter."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT=Path(os.environ['TASK10I_ROOT']);sys.path.insert(0,str(ROOT/'code'))
import common as C
from graph_adapter import geometry,load_inputs
from graph.persistent_geometry.contracts import transform
from utils.experiment_logger import ExperimentLogger


def xyz_at(camera,pixels):
    w=camera.depth.shape[1];v,u=np.divmod(pixels,w);z=camera.depth.ravel()[pixels];k=camera.intrinsic
    xyz=np.column_stack(((u+.5-k[0,2])*z/k[0,0],(v+.5-k[1,2])*z/k[1,1],z))
    return transform(xyz,camera.world_from_camera)


def mask_counts(mask,valid,labels,oid,near):
    m=mask&valid;obj=labels[oid];robot=labels[2];other=labels[1-oid]
    correct=int((m&obj).sum());rb=int((m&robot).sum());ot=int((m&other).sum())
    total=int(m.sum());ref=int((obj&valid).sum())
    return dict(total=total,correct=correct,robot=rb,other_object=ot,background=total-correct-rb-ot,
        gt_visible_valid=ref,precision=correct/total if total else None,recall=correct/ref if ref else None,
        correct_near_finger=int((m&near).sum()),gt_near_finger=int(near.sum()))


def node_info(records,truth,cameras,pads):
    result=[]
    for r in records:
        e,t,cam,pix=r['sources'][0];idx=int(pix);oid=r['object_id'];labels=truth[cam]
        category='correct' if labels[oid].ravel()[idx] else 'robot' if labels[2].ravel()[idx] else 'other_object' if labels[1-oid].ravel()[idx] else 'background'
        point=np.asarray(r['world_xyz'],float);near=bool((np.linalg.norm(pads[1:]-point,axis=1)<.02).any())
        attached=[]
        for _,_,c,p in r['sources']:
            gt=truth[c];i=int(p)
            attached.append('correct' if gt[oid].ravel()[i] else 'robot' if gt[2].ravel()[i] else 'other_object' if gt[1-oid].ravel()[i] else 'background')
        result.append(dict(object_id=oid,xyz=point,category=category,near_finger=near,attached=attached,source=[e,t,cam,idx]))
    return result


def graph_counts(graph,oid):
    nodes=np.flatnonzero(graph['node_type']==oid+3);edge=graph['edge_index'];typ=graph['edge_type']
    keep=(typ==0)&(edge[0]<edge[1])
    if edge.shape[1]:keep&=(graph['node_type'][edge[0]]==oid+3)&(graph['node_type'][edge[1]]==oid+3)
    e=edge[:,keep];lookup={int(v):i for i,v in enumerate(nodes)}
    ij=np.array([(lookup[int(a)],lookup[int(b)]) for a,b in e.T],dtype=int).reshape(-1,2)
    adjacency=coo_matrix((np.ones(len(ij)),(ij[:,0],ij[:,1])),shape=(len(nodes),len(nodes)))
    components=int(connected_components(adjacency,directed=False)[0]) if len(nodes) else 0
    return dict(intra_undirected_edges=int(e.shape[1]),connected_components=components,
        robot_object_directed_edges=int((typ==1).sum()),peg_box_directed_edges=int((typ==2).sum()))


def score_case(folder,scene,size,spec,entries):
    pinned=C.read(folder/'inputs.json')
    for name,digest in pinned['sensor_files'].items():
        assert C.sha(ROOT/'sensors'/scene/str(size)/name)==digest,'sensor changed since Graph freeze'
    inputs=load_inputs(scene,size,geometry());cameras={c.camera:c for c in inputs[0]};pads=inputs[1]['robot_world_geometry']
    truth={};near={};reference={0:[],1:[]}
    for cam,camera in cameras.items():
        e=entries[scene,size,cam];gtfile=ROOT/e['gt_file'];assert C.sha(gtfile)==e['gt_sha256']
        with np.load(gtfile,allow_pickle=False) as z:seg=z['actor_segmentation'].copy()
        truth[cam]=[np.isin(seg,e['object_ids'][key]) for key in ('peg','box','robot')]
        valid=np.isfinite(camera.depth)
        for oid in (0,1):
            m=truth[cam][oid]&valid;pix=np.flatnonzero(m);world=xyz_at(camera,pix)
            if oid==0 or cam=='base_camera':reference[oid].append(world)
            near_mask=np.zeros(m.shape,bool)
            if len(world):near_mask.ravel()[pix]=(np.linalg.norm(world[:,None,:]-pads[None,1:,:],axis=2)<.02).any(1)
            near[cam,oid]=near_mask
    reference={oid:np.concatenate(ref).reshape(-1,3) for oid,ref in reference.items()}
    with np.load(folder/'raw_masks.npz',allow_pickle=False) as z:raw={c:z[c].copy() for c in cameras}
    raw_counts=[]
    for cam,camera in cameras.items():
        for oid in (0,1):
            if cam=='hand_camera' and oid==1:continue
            raw_counts.append(dict(camera=cam,object_id=oid,**mask_counts(raw[cam][oid],np.isfinite(camera.depth),truth[cam],oid,near[cam,oid])))
    results=[]
    for mode in ('full','confirmed_only','off'):
        path=folder/mode;pred=C.read(path/'prediction.json')
        if pred['status']!='PASS':results.append(dict(mode=mode,status='FAILED',failure=pred['failure']));continue
        with np.load(path/'masks.npz',allow_pickle=False) as z:ms={k:z[k].copy() for k in z.files}
        with np.load(path/'graph.npz',allow_pickle=False) as z:graph={k:z[k].copy() for k in z.files}
        masks=[]
        for stage in ('clean','effective'):
            for cam,camera in cameras.items():
                for oid in (0,1):
                    if cam=='hand_camera' and oid==1:continue
                    masks.append(dict(stage=stage,camera=cam,object_id=oid,
                        **mask_counts(ms[stage+'__'+cam][oid],np.isfinite(camera.depth),truth[cam],oid,near[cam,oid])))
        nodes=node_info(pred['records'],truth,cameras,pads);node_scores=[]
        for oid in (0,1):
            ns=[n for n in nodes if n['object_id']==oid];good=[n['xyz'] for n in ns if n['category']=='correct']
            count={k:sum(n['category']==k for n in ns) for k in ('correct','robot','other_object','background')}
            attached=[a for n in ns for a in n['attached']]
            ref=reference[oid]
            if not len(ref):coverage=None
            elif not len(good):coverage=0.
            else:coverage=float((cKDTree(np.asarray(good)).query(ref,workers=1)[0]<.004).mean())
            node_scores.append(dict(object_id=oid,nodes=len(ns),**count,precision=count['correct']/len(ns) if ns else None,
                correct_near_finger=sum(n['category']=='correct' and n['near_finger'] for n in ns),
                attached_sources=len(attached),attached_correct=attached.count('correct'),attached_robot=attached.count('robot'),
                gt_visible_points=len(ref),visible_coverage_4mm=coverage,**graph_counts(graph,oid)))
        results.append(dict(mode=mode,status='PASS',graph_ms=[x['graph_ms'] for x in pred['samples']],
            peak_rss_bytes=max(x['peak_rss_bytes'] for x in pred['samples']),
            gpu_peak_allocated_bytes=max(x['gpu_peak_allocated_bytes'] for x in pred['samples']),
            gate_accepted=pred['gate']['accepted'],gate_reasons=pred['gate']['reasons'],
            selected_sources=[n['source'] for n in nodes],masks=masks,nodes=node_scores))
    return dict(scene=scene,size=size,parent=spec['parent'],cohort=spec['cohort'],timestep=spec['timestep'],raw_masks=raw_counts,modes=results)


def main(args):
    root=Path(args.run);frozen=C.read(root/'prediction_freeze.json');cases=C.read(args.cases)
    for p,h in frozen['receipts'].items():
        assert C.sha(root/p)==h
        receipt=C.read(root/p)
        for name,digest in receipt['files'].items():assert C.sha((root/p).parent/name)==digest
    specs={x['id']:x for x in C.read(ROOT/'protocol.json')['scenes']}
    entries={(x['scene'],x['size'],x['camera']):x for x in C.read(ROOT/'capture_manifest.json')['entries']}
    output=root/'scoring';output.mkdir(exist_ok=False)
    rows=[];fail=[]
    with ExperimentLogger(team='integration',model='task10i_robot_filter_score',job_type='evaluate',seed=42,
        root=output/'experiments',config=dict(prediction_freeze_sha256=C.sha(root/'prediction_freeze.json'),cases_sha256=C.sha(args.cases),GT_used_for_scoring_only=True)) as log:
        for s in cases['scenarios']:
            for size in cases['sizes']:
                folder=root/'cases'/s['scene']/str(size)
                try:
                    if not (folder/'receipt.json').exists():raise FileNotFoundError('no frozen receipt')
                    row=score_case(folder,s['scene'],size,specs[s['scene']],entries);row['stratum']=s['stratum'];rows.append(row)
                    print('score',s['stratum'],s['scene'],size,flush=True)
                except Exception as exc:fail.append(dict(scene=s['scene'],size=size,type=type(exc).__name__,message=str(exc)))
                C.write(output/'scores.json',dict(status='PARTIAL',rows=rows,failures=fail))
                log.log(dict(completed=len(rows),failed=len(fail)),step=len(rows)+len(fail))
        path=output/'scores.json';C.write(path,dict(status='COMPLETE' if not fail else 'PARTIAL',rows=rows,failures=fail,
            prediction_freeze_sha256=C.sha(root/'prediction_freeze.json'),truth_usage='evaluation only after 768/1024 freeze; 512 used beforehand for scenario stratification'))
        log.link_output(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--cases',required=True);main(p.parse_args())
