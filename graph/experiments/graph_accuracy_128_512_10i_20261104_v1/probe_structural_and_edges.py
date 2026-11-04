"""GT-free audit of structural-point retention and per-object edge rejection."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import CAMERAS,PARENTS,SIZES,entries,read,write,sha,sensor_frame
from .localize_information_loss import Observer
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.persistent_geometry.tensor_camera import tensor_camera
from graph.persistent_geometry.tensor_graph import numpy_graph,FIELDS,surface_edges_tensor
from graph.persistent_geometry.tensor_records import _identity_ranks
from graph.persistent_geometry.tensor_fusion import prepare_tensor_views
from graph.task10i_stage4a2.dinov2_matching_audit import csv_write
from utils.experiment_logger import ExperimentLogger


def main(capture,inference,runtime,output):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    index=entries(capture);schema=read(runtime/'normalizer.json')['schema'];rows=[]
    config=dict(parents=PARENTS,sizes=SIZES,timesteps=list(range(21)),model_forwards=0,GT_arrays_used=False,
        source_sha256=sha(Path(__file__)),inference_receipt_sha256=sha(inference/'inference_complete.json'),
        edge_scope='same current surface_edges_tensor on each object separately; exact original edges required')
    with ExperimentLogger(team='integration',model='task10i_structural_and_edge_loss_probe',seed=42,
            job_type='evaluate',root=output/'experiments',config=config) as log:
        for parent in PARENTS:
            for size in SIZES:
                geo=CurrentGeometry(runtime/'inputs',schema)
                try:
                    for t in range(21):
                        host,data=sensor_frame(capture,index,parent,t,size,f'{parent}:native{size}')
                        cameras=[tensor_camera(c) for c in host];surfaces=geo.fk.surfaces(data['robot76'])
                        data.update(episode=np.array(cameras[0].episode),timestep=np.int64(t),timestamp=np.float64(t/20),
                            qpos=data['robot76'][:9],robot_link_world=np.stack([s.world_from_link for s in surfaces]))
                        with np.load(inference/parent/str(size)/f't{t:04d}.npz') as z:expected={k:z[k].copy() for k in z.files}
                        masks={c:torch.as_tensor(expected['raw_masks'][ci],device='cuda') for ci,c in enumerate(CAMERAS)}
                        with Observer(geo) as observer:graph,records,_=geo.build(cameras,data,surfaces,lambda:masks)
                        saved=numpy_graph(graph)
                        assert all(np.array_equal(getattr(saved,k).numpy(),expected[k]) for k in FIELDS)
                        pool=observer.data['pool']
                        views=prepare_tensor_views(cameras,observer.data['gate_masks'],coordinate_frame='ee')
                        order=torch.argsort(_identity_ranks(records),stable=True)
                        order=order[torch.argsort(records.object_id[order],stable=True)]
                        xyz=records.xyz[order];normals=records.normals[order]
                        valid=records.normal_valid[order];objects=records.object_id[order]
                        for oid in (0,1):
                            selected=torch.nonzero(objects==oid).flatten()
                            local,audit=surface_edges_tensor(xyz[selected],normals[selected],valid[selected],objects[selected],views)
                            mapped=selected[local]+3
                            take=(graph.edge_type==0)&(graph.node_type[graph.edge_index[0]]==oid+3)
                            assert torch.equal(mapped,graph.edge_index[:,take])
                            values={k:int(v) for k,v in audit.items()}
                            kept=int(local.shape[1]//2)
                            assert values['candidates']==sum(values[k] for k in ('rejected_normal','rejected_witness','rejected_free_space'))+kept
                            before=pool.object_id==oid;after=records.object_id==oid
                            nP=int((pool.structural&before).sum());nS=int((~pool.structural&before).sum())
                            finalP=int((records.structural&after).sum());finalS=int((~records.structural&after).sum())
                            rows.append(dict(parent=parent,timestep=t,size=size,object_id=oid,
                                pool_priority=nP,pool_surface=nS,final_priority=finalP,final_surface=finalS,
                                priority_retention=finalP/nP if nP else None,surface_retention=finalS/nS if nS else None,
                                kept_surface_edges_undirected=kept,**values,exact_original_edges=True))
                        if t==0 or t==20:print('EDGE_PROBE',parent,size,t,flush=True)
                finally:geo.close()
        csv_write(output/'structural_and_edges.csv',rows)
        summary=[]
        for size in SIZES:
            for oid in (0,1):
                group=[r for r in rows if r['size']==size and r['object_id']==oid]
                keys=('pool_priority','pool_surface','final_priority','final_surface','candidates',
                      'rejected_normal','rejected_witness','rejected_free_space','kept_surface_edges_undirected')
                summary.append(dict(size=size,object_id=oid,frames=len(group),
                    means={k:float(np.mean([r[k] for r in group])) for k in keys},
                    pooled_priority_retention=sum(r['final_priority'] for r in group)/sum(r['pool_priority'] for r in group),
                    pooled_surface_retention=sum(r['final_surface'] for r in group)/sum(r['pool_surface'] for r in group)))
        write(output/'summary.json',dict(status='COMPLETE',run_id=log.run_id,config=config,summary=summary,
            graph_frames=168,object_edge_audits=len(rows),all_five_graph_tensors_exact=True,all_per_object_edges_exact=True))
        log.log(dict(graph_frames=168,object_edge_audits=len(rows),model_forwards=0));log.link_output(output/'summary.json')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('capture','inference','runtime','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.capture,a.inference,a.runtime,a.output)
