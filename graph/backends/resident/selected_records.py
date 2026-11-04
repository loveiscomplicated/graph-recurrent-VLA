"""Budget compact arrays before constructing Python records, same native rules."""
import numpy as np
import graph.backends.native as native
from graph.geometry.contracts import SurfaceObservation
from graph.geometry.map import current_records
from graph.geometry.coordinates import camera_origin


def records_after_budget(host,frame,count=128,distance=.003,*,coordinate_frame=None):
    names=sorted(host);lengths=[len(host[name]['pixels']) for name in names]
    camera=np.repeat(np.arange(len(names)),lengths)
    arrays={k:np.concatenate([host[name][k] for name in names]) for k in next(iter(host.values()))}
    n=len(camera)
    if not n:return []
    # Same (object, structural priority, episode/time/camera/numeric pixel).
    order=np.lexsort((arrays['pixels'],camera,~arrays['structural'],arrays['oid']))
    x=np.ascontiguousarray(arrays['xyz'][order],np.float64);oid=np.ascontiguousarray(arrays['oid'][order],np.int64)
    if not np.isfinite(x).all() or np.max(np.abs(x))/distance>=2.**62:return None
    labels=np.empty(n,np.int64);unknown=np.empty((max(64,n*8),2),np.int64);overrides=np.empty((0,3),np.int64)
    for _ in range(8):
        ambiguity=native.library().dedup(native.pointer(x),native.pointer(oid),n,distance,native.pointer(labels),
            native.pointer(overrides),len(overrides),native.pointer(unknown),len(unknown))
        if ambiguity==0:break
        if ambiguity<0:return None
        extra=np.array([(i,j,int(np.linalg.norm(x[j]-x[i])<distance)) for i,j in unknown[:ambiguity]],np.int64)
        overrides=np.ascontiguousarray(np.concatenate((overrides,extra)))
    else:return None
    reps=np.flatnonzero(labels==np.arange(n));selected=[];ids={}
    first=frame.cameras[0]
    for row in reps:
        source=order[row];name=names[camera[source]];pixel=int(arrays['pixels'][source])
        ids[row]=f'{first.episode}:current{first.timestep}:{name}:{pixel}'
    for object_id in (0,1):
        rows=reps[oid[reps]==object_id]
        if len(rows)>count:rows=rows[native.farthest_rows(x[rows],[ids[i] for i in rows],count)]
        selected.extend(rows.tolist())
    selected_set=set(selected);sources={i:[] for i in selected};structural={i:False for i in selected}
    for i,rep in enumerate(labels):
        if rep not in selected_set:continue
        source=order[i];name=names[camera[source]];c=frame.tensors[name].camera
        sources[rep].append((c.episode,c.timestep,name,int(arrays['pixels'][source])))
        structural[rep]|=bool(arrays['structural'][source])
    # Build/copy only final representatives. Full merged source provenance is
    # attached even when a duplicate did not survive the spatial budget.
    selected_source=np.asarray([order[i] for i in selected],np.int64);made={}
    for name in names:
        c=frame.tensors[name].camera
        for object_id in (0,1):
            rows=selected_source[(camera[selected_source]==names.index(name))&(arrays['oid'][selected_source]==object_id)]
            obs=SurfaceObservation(c.episode,c.timestep,name,object_id,arrays['xyz'][rows],arrays['normals'][rows],
                arrays['valid'][rows,0],arrays['f5'][rows],arrays['valid'][rows],arrays['pixels'][rows],
                camera_origin(c,coordinate_frame or 'world'),coordinate_frame or 'world',
                tuple(c.tcp_pose) if coordinate_frame=='ee' else None).validate()
            for record in current_records([obs],fuse_views=False,coordinate_frame=coordinate_frame):made[record['persistent_node_id']]=record
    result=[]
    for i in selected:
        r=made[ids[i]];r['source_observation_ids']=sources[i];r['structural']=structural[i];result.append(r)
    return result
