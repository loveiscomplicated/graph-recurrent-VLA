"""Post-inference safety denominators and endpoint selection diagnostics for A.

No forward or Graph sampling changes. This examines saved Graph provenance and
uses GT only to label selected endpoint points and wrong-mask events afterward.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
from .aggregate_evaluation import csv_rows
from .evaluation_metrics import hierarchical_paired
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,paired,inference,reference,expected):
    from visual.runtime.vision_packet import pose_matrix
    from graph.persistent_geometry.contracts import transform
    from graph.persistent_geometry.metrics import sample_meshes
    root.mkdir(parents=True,exist_ok=False)
    assert read(paired/'pipeline_state.json')['status']=='PAIRED_NEW_DEVELOPMENT_GRAPH_COMPLETE'
    safety=[];selection=[];inputs=[];parents=set();total=0
    for e in read(expected)['entries']:
        eid=str(e['episode_id']);parents.add(e['parent_id'])
        folder=paired/'episodes'/eid;receipt=read(folder/'receipt.json')
        assert all(sha(folder/p)==h for p,h in receipt['files'].items())
        inputs.append(dict(path=str(folder/'receipt.json'),sha256=sha(folder/'receipt.json')))
        with np.load(inference/'perception'/eid/'B0.npz') as z:b0=z['mask_bits'].copy()
        with np.load(inference/'perception'/eid/'A.npz') as z:a=z['mask_bits'].copy()
        assert np.array_equal(b0[:,:4096],a[:,:4096]),'Main mask changed'
        changed=np.any(b0!=a,axis=1);total+=len(changed)
        for r in csv_rows(folder/'pixel_metrics.csv'):
            if r['reference_kind']!='EXACT_GT':continue
            n=r['predicted_pixels'];v=r['visible_pixels']
            safety.append(dict(**{k:r[k] for k in ('parent_id','episode','timestep','condition','camera','object_id','reference_kind')},
                visible_wrong_mask=float(n>0 and r['precision']<.8) if v>=8 else None,
                false_pixels_per_visible_pixel=r['fp']/v if v>=8 else None,
                robot_pixels_per_visible_pixel=r['robot_pixels']/v if v>=8 else None,
                empty_prediction=float(n==0),precision_defined=float(n>0),
                precision=r['precision'],false_presence=float(n>0) if v==0 else None))
        ref=reference/'references'/eid
        with np.load(ref/'segmentation.npz') as z:exact=z['exact_frame'].copy()
        with np.load(ref/'geometry_reference.npz') as z:poses=z['actor_pose'].copy()
        meshes=sample_meshes(ref/'meshes.npz',samples_per_object=10000)
        axis=int(np.ptp(meshes[0]['points'],axis=0).argmax())
        lo,hi=meshes[0]['points'][:,axis].min(),meshes[0]['points'][:,axis].max()
        metric={(r['condition'],int(r['timestep'])):r for r in csv_rows(folder/'graph_metrics.csv') if r['object_id']==0}
        for t in np.flatnonzero(changed&exact):
            row=dict(episode=eid,parent_id=e['parent_id'],timestep=int(t))
            sets={}
            for condition in ('B0','A'):
                frame=folder/condition/f't{t:04d}'
                with np.load(frame/'graph.npz') as z:
                    peg=z['object_id']==0;xyz=z['world_points'][peg]
                records=[r for r in read(frame/'provenance.json')['sources'] if r['object_id']==0]
                assert len(records)==len(xyz)
                local=transform(xyz,np.linalg.inv(pose_matrix(poses[t,0])))
                endpoint=(local[:,axis]<=lo+.1*(hi-lo))|(local[:,axis]>=hi-.1*(hi-lo))
                sources=[tuple(r['sources'][0]) for r in records]
                main=np.array([s[2]=='base_camera' for s in sources],bool)
                row.update({condition+'_points':len(sources),condition+'_main_points':int(main.sum()),
                    condition+'_wrist_points':int((~main).sum()),condition+'_main_endpoint_points':int((main&endpoint).sum())})
                row[condition+'_ends_coverage']=metric[condition,int(t)]['peg_ends_coverage']
                row[condition+'_surface_coverage']=metric[condition,int(t)]['correct_source_coverage']
                sets[condition]=set(sources)
            row['removed_main_source_pixels']=sum(s[2]=='base_camera' for s in sets['B0']-sets['A'])
            row['added_wrist_source_pixels']=sum(s[2]=='hand_camera' for s in sets['A']-sets['B0'])
            row['ends_delta']=row['A_ends_coverage']-row['B0_ends_coverage'] if row['A_ends_coverage'] is not None and row['B0_ends_coverage'] is not None else None
            row['surface_delta']=row['A_surface_coverage']-row['B0_surface_coverage'] if row['A_surface_coverage'] is not None and row['B0_surface_coverage'] is not None else None
            selection.append(row)
    summaries=[]
    for name in ('visible_wrong_mask','false_pixels_per_visible_pixel','robot_pixels_per_visible_pixel','empty_prediction','false_presence'):
        summaries.extend(hierarchical_paired(safety,name))
    csv_write(root/'false_positive_safety_audit.csv',[{k:v for k,v in r.items() if k!='parent_values'} for r in summaries])
    write(root/'safety_parent_values.json',summaries)
    csv_write(root/'endpoint_selection_diagnostic.csv',selection)
    worst=sorted([r for r in selection if r['ends_delta'] is not None],key=lambda r:r['ends_delta'])[:10]
    write(root/'endpoint_case_manifest.json',dict(selection='ten worst endpoint delta frames among all exact changed frames; failures retained',cases=worst))
    write(root/'scope.json',dict(status='DEVELOPMENT_SAFETY_ENDPOINT_DIAGNOSTIC_COMPLETE',
        source_sha256=sha(Path(__file__)),parents=len(parents),frames=total,changed_exact_frames=len(selection),
        Main_mask_byte_equal_all_frames=True,Graph_builder_changed=False,GT_posthoc_only=True,
        inputs=inputs,precision_empty_imputed=False,visible_wrong_mask_definition='prediction nonempty and precision<.8; GT visible>=8',
        endpoint_point_counts_are_spatial_diagnostic='selected point is in mesh-axis end band; coverage remains correct-source-only',
        independent_test=False,whole_goal_complete=False))


if __name__=='__main__':
    from utils.experiment_logger import ExperimentLogger
    p=argparse.ArgumentParser()
    for name in ('root','paired','inference','reference','expected'):p.add_argument(name,type=Path)
    a=p.parse_args()
    config={name:str(getattr(a,name)) for name in ('paired','inference','reference','expected')}
    config.update(source_sha256=sha(Path(__file__)),expected_sha256=sha(a.expected),GT_posthoc=True)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_development_safety_endpoint_audit',
            seed=42,root=a.root.parent/'experiments',job_type='evaluation',config=config) as logger:
        run(a.root,a.paired,a.inference,a.reference,a.expected)
        scope=read(a.root/'scope.json');scope['run_id']=logger.run_id;write(a.root/'scope.json',scope)
        logger.link_output(a.root/'scope.json',kind='metrics')
