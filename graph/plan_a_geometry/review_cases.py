"""Evaluation-only source/error/rim and camera-control figures from frozen data."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from graph.persistent_geometry.evaluate import load_cameras
from graph.persistent_geometry.metrics import sample_meshes,visible_reference
from graph.persistent_geometry.geometry_audit import aperture_reference
from graph.persistent_geometry.contracts import transform
from visual.runtime.vision_packet import pose_matrix
from graph.integration_baseline.io import write_json
from .media import arrays,world_panel

def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    m=json.loads((a.geometry/'geometry_manifest.json').read_text());metrics=json.loads((a.geometry/'metrics.json').read_text());chosen=[]
    for seed in sorted({f['seed'] for f in m['frames']}):
        ff=[f for f in m['frames'] if f['seed']==seed and f['condition']=='baseline'];chosen.append(ff[0])
        worst=max((r for r in metrics if r['seed']==seed and r['condition']=='baseline' and r['variant']=='C1_SF' and r['budget']=='cap128' and r['object_id']==1),key=lambda r:r['surface_distance_mean_mm'] or -1)
        chosen.append(next(f for f in ff if f['timestep']==worst['timestep']))
    index=[]
    for frame in { (f['seed'],f['timestep']):f for f in chosen}.values():
        folder=a.geometry/frame['folder'];cameras,data=load_cameras(frame['input_path']);gt=arrays(frame['gt_path']);refs=visible_reference(cameras,gt);mesh=sample_meshes(Path(frame['source_folder'])/'evaluation_meshes.npz');rim=aperture_reference(mesh,gt)[-1]
        pointsets={v:arrays(folder/(v+'_cap128_points.npz')) for v in ('C0_SF','C1_SF','C2_SF','C3_SF','C3_L','C3_H')}
        allxyz=np.concatenate([p['world_xyz'] for p in pointsets.values()]+[r for r in refs.values() if len(r)]);bounds=list(zip(allxyz.min(0)-.025,allxyz.max(0)+.025))
        fig=plt.figure(figsize=(16,5),dpi=110)
        for col,v in enumerate(('C0_SF','C1_SF','C2_SF','C3_SF')):
            pp=pointsets[v];ax=fig.add_subplot(1,4,col+1,projection='3d');world_panel(ax,pp['world_xyz'],pp['object_id'],bounds,v+' cap128; Peg red / Box blue',truth=refs)
        fig.suptitle(f"Camera controls | actual Episode {cameras[0].episode} t={frame['timestep']} | same SF and surface-edge policy; points-only")
        fig.tight_layout();name=f'seed{frame["seed"]}_t{frame["timestep"]:04d}_controls.png';fig.savefig(a.output/name);plt.close(fig);index.append(name)
        fig=plt.figure(figsize=(16,10),dpi=110)
        by={c.camera:c for c in cameras}
        for col,(v,label) in enumerate([('C3_SF','A-SF'),('C3_L','A-L'),('C3_H','A-H')]):
            pp=pointsets[v];src=arrays(folder/(v+'_cap128_points_sources.npz'));xyz=pp['world_xyz'];oid=pp['object_id'];dist=np.zeros(len(xyz));classes=np.zeros(len(xyz),int)
            for obj in (0,1):
                selected=oid==obj;dist[selected]=mesh[obj]['tree'].query(transform(xyz[selected],np.linalg.inv(pose_matrix(gt['actor_pose'][obj]))))[0]*1000
            for i,obj in enumerate(oid):
                j=np.flatnonzero(src['node_row']==i)[0];actor=int(gt[str(src['camera'][j])+'/segmentation'].ravel()[int(src['pixel'][j])])
                classes[i]=0 if actor==gt['actor_ids'][obj] else 1 if actor in gt['robot_ids'] else 3 if actor==gt['actor_ids'][1-obj] else 2
            for row in (0,1):
                ax=fig.add_subplot(2,3,row*3+col+1,projection='3d');world_panel(ax,np.empty((0,3)),np.empty(0,int),bounds,label+(' collision distance mm' if row==0 else ' first-source class'),truth=refs)
                if row==0:
                    sc=ax.scatter(*xyz.T,c=dist,s=7,cmap='magma',vmin=0,vmax=50);fig.colorbar(sc,ax=ax,shrink=.55,label='mm (50+ clipped)')
                    if len(rim):ax.scatter(*rim[::8].T,c='gold',s=4,alpha=.5)
                else:
                    for kind,color,title in [(0,'green','Correct object'),(1,'cyan','Robot'),(2,'gray','Background'),(3,'magenta','Other object')]:
                        selected=classes==kind
                        if selected.any():ax.scatter(*xyz[selected].T,c=color,s=6,label=title)
                    ax.legend(fontsize=6)
        fig.suptitle(f"Evaluation-only GT overlay/source labels | {cameras[0].episode} t={frame['timestep']}\nSame World axes. Gold: GT collision aperture rim (not model input). Gray: current GT-visible surface.")
        fig.tight_layout();name=f'seed{frame["seed"]}_t{frame["timestep"]:04d}_surface_purity.png';fig.savefig(a.output/name);plt.close(fig);index.append(name)
    write_json(a.output/'review_case_index.json',dict(images=index,selection='each Final initial frame + worst C1 Box distance frame; diagnostic display only, not model selection',GT_scope='evaluation overlays only'))
if __name__=='__main__':main()
