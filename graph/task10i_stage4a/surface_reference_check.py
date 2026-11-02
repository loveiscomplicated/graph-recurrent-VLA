"""Separate correct-actor source distance from far contamination tails."""
import argparse
from pathlib import Path
import numpy as np
from .run_pipeline import CAMERAS,STAGE3,read,write


def run(output,baseline,corrected):
    from graph.persistent_geometry.metrics import sample_meshes
    from graph.persistent_geometry.contracts import transform
    from visual.runtime.vision_packet import pose_matrix
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    rows=[];pools={}
    for receipt in sorted((corrected/'perception').glob('*/receipt.json')):
        cr=read(receipt);eid=str(cr['episode_id']);e=entries[eid]
        rrpath=Path(cr['corrected_reference']);rr=read(rrpath)
        meshes=sample_meshes(baseline/'references'/eid/'meshes.npz',samples_per_object=10000)
        with np.load(baseline/'references'/eid/'geometry_reference.npz',allow_pickle=False) as z:poses=z['actor_pose'].copy()
        with np.load(rrpath.parent/'segmentation.npz',allow_pickle=False) as gt:
            for t in sorted({0,e['actions']//2,e['actions']}):
                assert gt['exact_frame'][t]
                folder=baseline/'traces'/eid/f't{t:04d}';trace=read(folder/'trace.json')
                with np.load(folder/'support.npz',allow_pickle=False) as arrays:
                    for oid in (0,1):
                        indices=np.flatnonzero(arrays['point_object_id']==oid)
                        local=transform(arrays['point_world'][indices],np.linalg.inv(pose_matrix(poses[t,oid])))
                        distances=meshes[oid]['tree'].query(local)[0]*1000
                        labels=[]
                        for i in indices:
                            _,_,camera,pixel=trace['representative_sources'][int(i)]['representative']
                            actor=int(gt['actor_segmentation'][t,CAMERAS.index(camera)].ravel()[pixel])
                            labels.append('correct_actor' if actor in rr['object_ids'][('peg','box')[oid]] else 'robot' if actor in rr['robot_ids'] else 'wrong_object' if actor in rr['object_ids'][('box','peg')[oid]] else 'background')
                        for label in ('correct_actor','robot','wrong_object','background'):
                            d=distances[np.asarray(labels)==label]
                            pools.setdefault((oid,label),[]).extend(d.tolist())
                            rows.append(dict(episode=eid,parent_id=e['parent_id'],timestep=t,object_id=oid,source=label,points=len(d),mean_mm=float(d.mean()) if len(d) else None,p95_mm=float(np.percentile(d,95)) if len(d) else None))
    summary=[dict(object_id=oid,source=label,points=len(d),median_mm=float(np.median(d)) if len(d) else None,
        mean_mm=float(np.mean(d)) if len(d) else None,p95_mm=float(np.percentile(d,95)) if len(d) else None) for (oid,label),d in sorted(pools.items())]
    write(output,dict(status='SOURCE_CONDITIONED_SURFACE_DISTANCE_AUDITED',frame_selection='t0, midpoint, terminal for every64 Development sequence; no outcome-dependent frame selection',
        reference='10000 deterministic external collision-union samples; approximate surface distance, not analytic GT',rows=rows,pooled_summary=summary,
        no_GT_student_input=True,not_population_parent_macro=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('baseline',type=Path);p.add_argument('corrected',type=Path)
    a=p.parse_args();run(a.output,a.baseline,a.corrected)
