"""Full-episode observed/estimated/new-surface/ghost visual audit."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from graph.integration_baseline.io import write_json
from graph.persistent_geometry.evaluate import load_cameras
from graph.persistent_geometry.metrics import sample_meshes
from graph.persistent_geometry.contracts import transform
from visual.runtime.vision_packet import pose_matrix
from .render import load,figure_png,video

def main():
    p=argparse.ArgumentParser();p.add_argument('--temporal',type=Path,required=True);p.add_argument('--main-run',type=Path,required=True);args=p.parse_args()
    root=args.temporal;manifest=json.loads((root/'geometry_manifest.json').read_text());capture=Path(manifest['config']['capture'])
    source=capture/'collection'/'seed30002';mesh=sample_meshes(source/'evaluation_meshes.npz')
    out=root/'media';out.mkdir(exist_ok=True)
    names=('G1','L1','G2','L2','G3','L3','G4','L4');bounds=[]
    for f in manifest['frames']:
        if f['timestep']%5:continue
        for name in ('G4','L4'):
            a=load(root/f['folder']/(name+'_map.npz'));bounds.append(a['world_xyz'][a['status']!=3])
    union=np.concatenate(bounds);limits=np.c_[union.min(0)-.02,union.max(0)+.02]
    collections={k:[] for k in ('persistent_full','fusion_full','new_surface_full','ghost_full')};cases=[]
    for f in manifest['frames']:
        t=f['timestep']
        if t%5 and t!=manifest['frames'][-1]['timestep']:continue
        folder=root/f['folder'];data={name:load(folder/(name+'_map.npz')) for name in names}
        with np.load(source/f'f{t:04d}_gt.npz',allow_pickle=False) as a:gt={k:a[k] for k in a.files}
        ghost={};new={}
        for name in names:
            arr=data[name];bad=np.zeros(len(arr['world_xyz']),bool)
            for oid in (0,1):
                selected=arr['object_id']==oid;local=transform(arr['world_xyz'][selected],np.linalg.inv(pose_matrix(gt['actor_pose'][oid])))
                bad[selected]=mesh[oid]['tree'].query(local)[0]>.008
            ghost[name]=bad&arr['estimated_only'];new[name]=arr['first_seen_timestep']==t
            cases.append(dict(timestep=t,phase=f['phase'],variant=name,ghost_count=int((ghost[name]&(arr['status']!=3)).sum()),new_nodes=int((new[name]&(arr['status']!=3)).sum()),currently_observed=int((arr['currently_observed']&(arr['status']!=3)).sum()),estimated=int((arr['estimated_only']&(arr['status']!=3)).sum())))
        panels=[(data[n],n+' measured:red/blue; estimated:orange/teal',{}) for n in ('G1','L1','G2','L2','G4','L4')]
        collections['persistent_full'].append(figure_png(out/f'persistent_full_t{t:04d}.png',panels,limits,f'Full causal episode seed30002 t={t} phase={f["phase"]}'))
        panels=[(data[n],n,{}) for n in ('G1','L1','G3','L3','G4','L4')]
        collections['fusion_full'].append(figure_png(out/f'fusion_full_t{t:04d}.png',panels,limits,f'Main+Wrist actual same-time input t={t}'))
        panels=[(data[n],n+' black X: first seen at current t',dict(highlight=new[n])) for n in ('G1','L1','G2','L2','G4','L4')]
        collections['new_surface_full'].append(figure_png(out/f'new_surface_t{t:04d}.png',panels,limits,f'Newly integrated nodes; not all are correct/novel surface t={t}'))
        panels=[(data[n],n+' black X: estimated >8mm from GT surface',dict(highlight=ghost[n])) for n in ('G1','L1','G2','L2','G4','L4')]
        collections['ghost_full'].append(figure_png(out/f'ghost_t{t:04d}.png',panels,limits,f'GT evaluation-only ghost overlay t={t}'))
    decoded=[video(out/(name+'.mp4'),images,fps=5) for name,images in collections.items()]
    write_json(root/'visual_cases.json',cases);write_json(out/'decode_audit.json',dict(status='PASS',videos=decoded,limits=limits,view=[25,-150],marker_size=2,GT_used_only_for_overlay=True))
if __name__=='__main__':main()
