"""SF-only work elimination and identical selection packet with one validation.

The general builder still supports L/H. This context is restricted to SF, so
unrequested hybrid pools/audits are absent. Final packet validation and source
alignment remain in complete_vision_packet and receive_vision_packet.
"""
from time import perf_counter
import numpy as np
import native


def selection_packet(source_frame,prepared,selected,feature_cases):
    from visual.runtime.test import run_priority_fps_metric_control as prior
    fields=None;objects=[];compact=[]
    for oid in (0,1):
        prep,s,case=prepared[oid],selected[oid],feature_cases[oid];f=s['F']
        available,positions,features=prior.feature_rows(case,f)
        schema=[dict(name=k,shape=list(v.shape[1:]),dtype=v.dtype.str) for k,v in sorted(case['saved_features'].items())]
        if fields is None:fields=schema
        elif fields!=schema:raise ValueError('feature schema differs across objects')
        widths=[int(np.prod(k['shape'])) if k['shape'] else 1 for k in fields]
        view=np.full((len(f),sum(widths)),np.nan);offset=0
        for field,width in zip(fields,widths):
            view[positions,offset:offset+width]=features[field['name']].reshape(len(positions),width);offset+=width
        evidence=np.column_stack([np.isin(f,prep['C'][n]) for n in ('pca','sil')])
        chosen=np.column_stack([np.isin(f,s['T'][n]) for n in ('pca','sil')])
        surface=np.arange(len(f))>=len(s['P']);ids=np.full(len(f),oid,np.int64)
        objects.append(dict(points=s['xyz'],object_id=ids,geometric_features=view,
            provenance=dict(source_index=f.copy(),object_id=ids.copy(),candidate_evidence=evidence,
                both_evidence=evidence.all(1),selected_by_structural_source=chosen,
                selection_stage=surface.astype(np.uint8),surface_selected=surface,
                feature_available=available,feature_field_available=np.repeat(available[:,None],len(fields),axis=1))))
        compact.append((positions,features))
    packet=dict(episode='ep000',camera='base_camera',source_frame=int(source_frame),coordinate_frame='CV_camera_metres',
        feature_fields=fields,edge_status='not_implemented',intra_object_edge_index=np.empty((2,0),np.int64))
    for k in ('points','object_id','geometric_features'):packet[k]=np.concatenate([o[k] for o in objects])
    packet['provenance']={k:np.concatenate([o['provenance'][k] for o in objects]) for k in objects[0]['provenance']}
    offset=len(selected[0]['F'])
    packet['compact_features']={name:dict(positions=np.r_[compact[0][0],compact[1][0]+offset],
        values=np.concatenate([v[name] for _,v in compact])) for name in compact[0][1]}
    return packet,0.


def sf_represent(original,dedup_fallback):
    def represent(system,cameras,prepared,condition,representations=('SF','L','H')):
        if tuple(representations)!=('SF',):return original(system,cameras,prepared,condition,representations)
        import graph.plan_a_geometry.core as core
        cm=prepared['conditions'][condition];records=[];packets={}
        for camera in sorted(cameras,key=lambda c:c.camera!=system.main):
            packet,_=core.make_packet(camera,cm[camera.camera]);packets[camera.camera]=packet
            obs=core.observations_from_packet(packet,camera);obs,_=core.consistent_views(obs,cameras,cm)
            rows=core.current_records(obs,fuse_views=False)
            structural=set(packet['provenance']['source_index'][~packet['provenance']['surface_selected']].tolist())
            for r in rows:r['structural']=core.key(r)[3] in structural
            records.extend(rows)
        # These records have one private owner. Nothing else holds or mutates
        # their dicts/arrays; preserve packet arrays and caller-owned inputs.
        records,audit=native.deduplicate(records,system.protocol['merge_distance_m'],fallback=dedup_fallback,owned=True)
        return dict(SF=records),dict(SF=audit),{},packets
    return represent


def global_pairs(points,object_id,source_index,M=16,block_rows=64):
    from graph.integration_baseline.graph import _integer
    _integer(M,0,'M');_integer(block_rows,1,'block_rows')
    peg,box=(np.flatnonzero(object_id==oid) for oid in (0,1))
    if not M or not len(peg) or not len(box):return np.empty((0,2),np.int64)
    xyz=points.astype(np.float64);delta=xyz[peg,None]-xyz[box][None]
    d2=np.sum(delta*delta,axis=-1).ravel()
    # All pairs at the cutoff participate in the old total-order tie break.
    take=min(M,len(d2));threshold=np.partition(d2,take-1)[take-1]
    ix=np.flatnonzero(d2<=threshold);a=peg[ix//len(box)];b=box[ix%len(box)]
    order=np.lexsort((b,a,source_index[b],source_index[a],d2[ix]))[:M]
    return np.column_stack([a[order],b[order]])
