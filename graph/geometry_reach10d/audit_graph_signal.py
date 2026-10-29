"""Initial same-robot observed geometry response; no model fitting/filtering."""
from pathlib import Path
import numpy as np
from graph.plan_a_bc.common import read,write,append_csv
from visual.runtime.vision_packet import pose_matrix


def run(root):
    if (root/'graph_conditional_signal.json').exists():return
    manifest=read(root/'graph_cache_manifest.json')
    episodes={e['episode_id']:e for e in manifest['episodes']}
    original={e['seed']:e for e in episodes.values() if e['data_kind']=='original'}
    values={}
    for eid,e in episodes.items():
        with np.load(root/e['path'],allow_pickle=False) as z:
            lo,hi=z['node_ptr'][:2];x=z['node_input'][lo:hi,:3];types=z['node_type'][lo:hi]
            values[eid]={k:x[types==k].copy() for k in [3,4]}
    rows=[]
    for eid,e in episodes.items():
        if e['data_kind']!='perturbed':continue
        base=original[e['seed']];r=read(root/'raw'/str(eid)/'receipt.json')
        tcp=pose_matrix(np.array(r['audit']['original']['ee_pose']))
        expected=tcp[:3,:3].T@(np.array(r['spec']['offset_mm'])*.001)
        row=dict(episode_id=eid,parent_seed=e['seed'],variant=e['variant_id'],expected_TCP_delta_mm=(expected*1000).tolist())
        for k,name in [(3,'peg'),(4,'box')]:
            a,b=values[base['episode_id']][k],values[eid][k]
            row[name+'_points_nominal']=len(a);row[name+'_points_perturbed']=len(b)
            delta=b.mean(0)-a.mean(0) if len(a) and len(b) else None
            row[name+'_surface_centroid_delta_mm']=None if delta is None else (delta*1000).tolist()
            if name=='peg':
                row['peg_centroid_response_error_mm']=None if delta is None else float(np.linalg.norm(delta-expected)*1000)
                row['peg_centroid_signed_gain']=None if delta is None else float(delta@expected/(expected@expected))
        rows.append(row)
    write(root/'graph_conditional_signal.json',dict(rows=rows,
        valid_presence_pairs=sum(r['peg_points_nominal']>0 and r['peg_points_perturbed']>0 for r in rows),
        caveat='Observed sampled surface centroid is not actor origin. Visibility/mask/point-selection variation remains; no geometry quality filter is introduced.',
        used_to_change_training_or_dataset=False),immutable=True)
    for row in rows:append_csv(root/'graph_conditional_signal.csv',row)
