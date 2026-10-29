"""Real RGB-D fault-injection regressions, separate from natural/Final ranking."""
import argparse,json
from copy import deepcopy
from pathlib import Path
import numpy as np,imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graph.persistent_geometry.evaluate import load_cameras,save_graph,snapshot_records
from graph.persistent_geometry.self_filter import LinkSurface
from graph.persistent_geometry.metrics import visible_reference
from graph.persistent_geometry.contracts import transform
from graph.persistent_geometry.render import decode_video
from graph.integration_baseline.io import save_npz,write_json,sha256
from visual.runtime.vision_packet import pose_matrix
from .core import PlanA,budget
from .media import arrays,world_panel

def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(Path(__file__).with_name('protocol.json').read_text());manifest=json.loads((a.capture/'capture_manifest.json').read_text());found=None
    for ep in manifest['episodes']:
        source=a.capture/'collection'/f'seed{ep["seed"]}';geom=arrays(source/'robot_meshes.npz');roles=ep['roles'];system=PlanA(roles['main'],roles['wrist'],protocol)
        for f in ep['frames'][::5]:
            cameras,data=load_cameras(source/(f['stem']+'_input.npz'));masks=arrays(source/(f['stem']+'_masks.npz'))
            surfaces=[LinkSurface(n,geom[f'vertices_{i}'],data['robot_link_world'][i]) for i,n in enumerate(geom['links'])]
            prepared=system.prepare(cameras,data,masks,surfaces)
            if prepared['gate']['accepted'] and prepared['conditions']['C3'][roles['wrist']][0].any() and prepared['filters'][roles['wrist']]['confirmed_robot'].any() and prepared['clean'][roles['main']][0].any():
                main_pools,_,_,_=system.represent(cameras,prepared,'C0',('SF',))
                if not any(r['object_id']==0 for r in main_pools['SF']):continue
                found=(ep,f,source,cameras,data,masks,surfaces,system,prepared);break
        if found:break
    if not found:raise RuntimeError('no sensor-supported real fixture for requested controlled cases')
    ep,f,source,cameras,data,masks,surfaces,system,prepared=found
    cm={k:v.copy() for k,v in masks.items()};cm[roles['main']]=prepared['clean'][roles['main']].copy()
    cases={'CONTROL_PARENT_REAL_RGBD':deepcopy(cm),'CONTROLLED_MAIN_PEG_MASK_DROP':deepcopy(cm),'CONTROLLED_WRIST_CONFIRMED_ROBOT_MASK':deepcopy(cm)}
    cases['CONTROLLED_MAIN_PEG_MASK_DROP'][roles['main']][0]=False
    cases['CONTROLLED_WRIST_CONFIRMED_ROBOT_MASK'][roles['wrist']][0]=prepared['filters'][roles['wrist']]['confirmed_robot'].copy()
    gt=arrays(source/(f['stem']+'_gt.npz'));refs=visible_reference(cameras,gt);cloud=np.concatenate([p for p in refs.values() if len(p)]);bounds=list(zip(cloud.min(0)-.03,cloud.max(0)+.03));rows=[]
    writer=imageio.get_writer(a.output/'controlled_cases.mp4',fps=1,codec='libx264',macro_block_size=2,ffmpeg_params=['-threads','1'])
    for name,mask in cases.items():
        case=a.output/name;case.mkdir();result=system.prepare(cameras,data,mask,surfaces);pools,_,_,_=system.represent(cameras,result,'C3',('SF',));records=budget(pools['SF'],128);graph=system.graph(records,data,cameras)
        save_graph(case/'C3_SF_graph.npz',graph);snapshot_records(case/'current_points.npz',records);save_npz(case/'input_masks.npz',**mask)
        row=dict(condition=name,seed=ep['seed'],timestep=f['timestep'],parent_episode=cameras[0].episode,parent_split=ep['split'],natural_failure=False,
            gate=result['gate'],peg_nodes=sum(r['object_id']==0 for r in records),source_cameras=sorted({s[2] for r in records if r['object_id']==0 for s in r['source_observation_ids']}),
            input_path=str(source/(f['stem']+'_input.npz')),input_sha256=f['input_sha256'],original_mask_sha256=f['mask_sha256'],GT_used_to_select_fixture_or_points=False)
        if name=='CONTROLLED_MAIN_PEG_MASK_DROP':assert row['peg_nodes']>0 and row['source_cameras']==[roles['wrist']]
        if name=='CONTROLLED_WRIST_CONFIRMED_ROBOT_MASK':assert not result['gate']['accepted']
        write_json(case/'trace.json',row);rows.append(row)
        fig=plt.figure(figsize=(12,8),dpi=100)
        for col,c in enumerate(cameras):
            ax=fig.add_subplot(2,2,col+1);image=c.rgb.astype(float)
            for oid,color in [(0,np.array([255,40,40])),(1,np.array([40,100,255]))]:image[mask[c.camera][oid]]=.6*image[mask[c.camera][oid]]+.4*color
            ax.imshow(image.astype('uint8'));ax.set_title(c.camera+' controlled input mask');ax.axis('off')
        xyz=transform(graph.node_input.numpy()[:,:3],pose_matrix(data['tcp_pose']));oid=graph.node_type.numpy()-3
        world_panel(fig.add_subplot(2,2,3,projection='3d'),xyz,oid,bounds,'C3 A-SF points-only',truth=refs)
        edge=graph.edge_index.numpy()[:,graph.edge_type.numpy()==0].T
        world_panel(fig.add_subplot(2,2,4,projection='3d'),xyz,oid,bounds,'Same points + common surface edges',edges=edge,truth=refs)
        fig.suptitle(f"FAULT-INJECTION REGRESSION, not natural occlusion or policy rollout\n{cameras[0].episode} seed={ep['seed']} t={f['timestep']} | {name}\nWrist gate {'ACCEPTED' if result['gate']['accepted'] else 'REJECTED'}; no GT fill; GT gray overlay evaluation only",fontsize=10)
        fig.tight_layout(rect=[0,0,1,.91]);fig.canvas.draw();image=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy();plt.close(fig);imageio.imwrite(case/'comparison.png',image);writer.append_data(image)
    writer.close();n,shape=decode_video(a.output/'controlled_cases.mp4');assert n==3
    write_json(a.output/'case_manifest.json',dict(cases=rows,decoded_frames=n,scope='post-fit development regression only; same parent physical observation with explicit mask faults; excluded from natural condition counts, fitting and Primary Final rankings'))
if __name__=='__main__':main()
