"""Canonical online-adapter GT-box control for the per-object Oracle memory grouping."""
import argparse
from pathlib import Path
import time
import types
import h5py
import numpy as np
import torch
from .run_pipeline import CAMERAS,STAGE3,read,write


def run(root,final,audits,oracle_root=None):
    from utils.experiment_logger import ExperimentLogger
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    from visual.runtime.tracking import SAM2Tracker
    from graph.plan_a_bc.geometry import CurrentGeometry
    from .detector_audit import bbox
    from .stagewise import trace_geometry
    from .reevaluate_references import updated_points
    from graph.persistent_geometry.metrics import sample_meshes
    torch.set_num_threads(1);torch.manual_seed(42);root.mkdir(parents=True,exist_ok=False)
    oracle_root=oracle_root or final/'oracles'
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_native_grouped_GT_box_oracle',seed=42,
        root=root/'experiments',job_type='evaluation',config=dict(condition_type='ORACLE_DIAGNOSTIC / SIM_ONLY_ORACLE',purpose='check independent-object Oracle grouping confound',practical_candidate=False)) as log:
        write(root/'pipeline_state.json',dict(status='RUNNING',stage='WAITING_LATENCY_AND_INDEPENDENT_EVALUATION',run_id=log.run_id))
        while not (final/'handoff.json').exists():
            p=final/'pipeline_state.json'
            if p.exists():assert read(p)['status']!='FAILED',read(p)
            time.sleep(10)
        entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
        rt=Path('/workspace/task10g_f1_abc_20261031/full');schema=read(Path('/workspace/task10f1_gt_pretrain_20261030/normalizer.json'))['schema']
        tracker=SAM2Tracker(device='cuda',checkpoint=read(rt/'runtime_paths.json')['sam']);comparisons=[];stage_rows=[];point_rows=[]
        for window in read(oracle_root/'registered_windows.json')['windows']:
            eid=window['episode'];e=entries[eid];rrpath=Path(read(audits/'corrected/perception'/eid/'receipt.json')['corrected_reference']);rr=read(rrpath)
            with np.load(rrpath.parent/'segmentation.npz',allow_pickle=False) as z:seg,exact=z['actor_segmentation'].copy(),z['exact_frame'].copy()
            with np.load(audits/'baseline/references'/eid/'geometry_reference.npz',allow_pickle=False) as z:poses=z['actor_pose'].copy()
            meshes=sample_meshes(audits/'baseline/references'/eid/'meshes.npz',samples_per_object=10000)
            adapters={}
            for ci,camera in enumerate(CAMERAS):
                def oracle_initial(frames,detector,_ci=ci):
                    t=len(frames)-1;assert exact[t]
                    out={}
                    for name in ('peg','box'):
                        mask=np.isin(seg[t,_ci],rr['object_ids'][name]);box=bbox(mask)
                        out[name]=dict(box=box,frame=t if box is not None else -1,score=1.,prompt='GT_BOX_ORACLE',scans=[])
                    return out
                adapter=OnlineVisionAdapter(None,tracker);adapter.reset(eid)
                original=OnlineVisionAdapter.process.__wrapped__;namespace=original.__globals__.copy()
                namespace['detect_initial']=oracle_initial;namespace['SCAN_EVERY']=1
                native=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
                adapter.process=types.MethodType(torch.inference_mode()(native),adapter);adapters[camera]=adapter
            geometry=CurrentGeometry(rt/'inputs',schema)
            try:
                with h5py.File(e['raw_path'],'r') as f:
                    for t in range(window['last_timestep']+1):
                        cameras,data,surfaces=geometry.from_h5(f,t,eid)
                        masks={c.camera:adapters[c.camera].process(c.rgb,eid,t)[0] for c in cameras}
                        folder=root/'traces'/eid/f't{t:04d}'
                        _,_,rows=trace_geometry(geometry,cameras,data,surfaces,masks,dict(zip(CAMERAS,seg[t])),[rr['object_ids']['peg'],rr['object_ids']['box']],rr['robot_ids'],folder)
                        for r in rows:r.update(parent_id=e['parent_id'],condition='O1_NATIVE_GROUPED_GT_BOX',condition_type='ORACLE_DIAGNOSTIC')
                        stage_rows.extend(rows)
                        with np.load(folder/'support.npz',allow_pickle=False) as actual,np.load(oracle_root/'traces/O1'/eid/f't{t:04d}/support.npz',allow_pickle=False) as old:
                            same_masks=all(np.array_equal(actual[c+'__'+str(oid)+'__sam2'],old[c+'__'+str(oid)+'__sam2']) for c in CAMERAS for oid in (0,1))
                            graph_diff={k:int(np.count_nonzero(actual[k]!=old[k])) if actual[k].shape==old[k].shape else -1 for k in ('node_input','node_type','edge_index','edge_attr','edge_type')}
                            pp=updated_points(actual,read(folder/'trace.json'),dict(zip(CAMERAS,seg[t])),rr,cameras,meshes,poses,t,e)
                        for r in pp:r.update(condition='O1_NATIVE_GROUPED_GT_BOX',condition_type='ORACLE_DIAGNOSTIC')
                        point_rows.extend(pp);comparisons.append(dict(episode=eid,timestep=t,mask_equal=same_masks,graph_difference_values=graph_diff))
                print('grouped GT-box control',eid,flush=True)
            finally:
                for adapter in adapters.values():adapter.close()
        write(root/'stage_rows.json',stage_rows);write(root/'representative_points.json',point_rows)
        write(root/'oracle_grouping_control.json',dict(status='NATIVE_GROUPED_GT_BOX_CONTROL_FINISHED',frames=len(comparisons),
            differing_mask_frames=sum(not r['mask_equal'] for r in comparisons),differing_graph_frames=sum(any(r['graph_difference_values'].values()) for r in comparisons),rows=comparisons,
            canonical_tracking='unchanged original OnlineVisionAdapter process bytecode and per-camera states; simultaneous objects grouped; absolute prefix clock',
            detector_replacement='GT bounding box at first currently GT-visible frame, perfect-GT check every observed frame',
            old_O1='independent-object AnchorState, local reanchor clock; do not attribute differences solely to box localization when grouping control differs',
            practical_improvement_claim=False))
        log.link_output(root/'oracle_grouping_control.json',kind='metrics')
        write(root/'pipeline_state.json',dict(status='SUBSTAGE_FINISHED',stage='NATIVE_GROUPED_GT_BOX_CONTROL_FINISHED',run_id=log.run_id))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('final',type=Path);p.add_argument('audits',type=Path)
    a=p.parse_args();run(a.root,a.final,a.audits)
