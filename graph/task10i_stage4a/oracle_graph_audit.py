"""Teacher-only edge/rim diagnosis after O0--O4, without additional model forwards."""
import argparse
from pathlib import Path
import time
import h5py
import numpy as np
from .run_pipeline import CAMERAS,STAGE3,read,write


def run(root,final,audits):
    from utils.experiment_logger import ExperimentLogger
    from graph.persistent_geometry.metrics import sample_meshes,visible_reference
    from graph.persistent_geometry.contracts import CameraObservation
    from graph.persistent_geometry.geometry_audit import aperture_reference,graph_audit
    root.mkdir(parents=True,exist_ok=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_oracle_graph_structure_audit',seed=42,
         root=root/'experiments',job_type='evaluation',config=dict(condition_type='ORACLE_DIAGNOSTIC',additional_model_forwards=0)) as log:
        write(root/'pipeline_state.json',dict(status='RUNNING',stage='WAITING_SCIENTIFIC_EVALUATIONS',run_id=log.run_id))
        while not (final/'handoff.json').exists():
            p=final/'pipeline_state.json'
            if p.exists():assert read(p)['status']!='FAILED',read(p)
            time.sleep(10)
        entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
        windows=read(final/'oracles/registered_windows.json')['windows'];rows=[]
        try:
            for window in windows:
                eid=window['episode'];e=entries[eid]
                rrpath=Path(read(audits/'corrected/perception'/eid/'receipt.json')['corrected_reference']);rr=read(rrpath)
                assert all(len(rr['object_ids'][name])==1 for name in ('peg','box'))
                meshes=sample_meshes(audits/'baseline/references'/eid/'meshes.npz',samples_per_object=10000)
                with np.load(audits/'baseline/references'/eid/'geometry_reference.npz',allow_pickle=False) as z:poses=z['actor_pose'].copy()
                with h5py.File(e['raw_path'],'r') as f,np.load(e['gt_path'],allow_pickle=False) as labels,np.load(rrpath.parent/'segmentation.npz',allow_pickle=False) as reference:
                    for t in range(window['last_timestep']+1):
                        assert reference['exact_frame'][t]
                        cameras=[CameraObservation(eid,t,c,float(f['timestamp'][t]),f[c+'/rgb'][t],f[c+'/depth_raw_mm'][t],f[c+'/intrinsic'][t],f[c+'/extrinsic_cv'][t],f[c+'/cam2world_gl'][t],f['tcp_pose'][t]) for c in CAMERAS]
                        gt=dict(actor_ids=np.array([rr['object_ids']['peg'][0],rr['object_ids']['box'][0]]),actor_pose=poses[t],hole_pose=labels['raw_hole_pose'][t],hole_radius=labels['raw_hole_radius'][t],robot_ids=rr['robot_ids'])
                        gt.update({c+'/segmentation':reference['actor_segmentation'][t,i] for i,c in enumerate(CAMERAS)})
                        visible=visible_reference(cameras,gt)
                        visible[1]=cameras[0].world_points()[(gt['base_camera/segmentation']==gt['actor_ids'][1])&np.isfinite(cameras[0].depth)]
                        aperture=aperture_reference(meshes,gt)
                        for condition in ('O0','O1','O2','O3','O4'):
                            folder=audits/'baseline/traces'/eid/f't{t:04d}' if condition=='O0' else final/'oracles/traces'/condition/eid/f't{t:04d}'
                            with np.load(folder/'support.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files if k in ('node_input','node_type','edge_index','edge_attr','edge_type')}
                            for r in graph_audit(arrays,cameras,gt,meshes,aperture,visible):
                                r.update(episode=eid,parent_id=e['parent_id'],timestep=t,condition=condition,condition_type='ORACLE_DIAGNOSTIC',box_visible_reference='Main only')
                                rows.append(r)
                write(root/'progress.json',dict(episodes_done=len({r['episode'] for r in rows}),rows=len(rows)))
                print('oracle graph',eid,flush=True)
            write(root/'edge_rim_metrics.json',rows)
            write(root/'scope.json',dict(status='ORACLE_GRAPH_STRUCTURE_AUDITED',condition_type='ORACLE_DIAGNOSTIC',new_model_forwards=0,
                mesh_unsupported_edges_have_sampled_surface_floor=True,free_space_edges_use_current_RGBD_only=True,
                aperture_crossing_uses_GT_hole_geometry_evaluation_only=True,not_a_practical_candidate=True))
            log.link_output(root/'scope.json',kind='metrics');log.link_output(root/'edge_rim_metrics.json',kind='metrics')
            write(root/'pipeline_state.json',dict(status='SUBSTAGE_FINISHED',stage='ORACLE_GRAPH_STRUCTURE_AUDITED',run_id=log.run_id))
        except BaseException as exc:
            write(root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),run_id=log.run_id));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('final',type=Path);p.add_argument('audits',type=Path)
    a=p.parse_args();run(a.root,a.final,a.audits)
