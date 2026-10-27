"""Same-world/view/limits/marker comparisons, with explicit GT sidecars."""
import argparse,json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from graph.integration_baseline.io import write_json,save_npz
from graph.persistent_geometry.evaluate import load_cameras
from graph.persistent_geometry.contracts import transform
from graph.persistent_geometry.metrics import visible_reference
from graph.persistent_geometry.graph import surface_edges
from visual.runtime.vision_packet import pose_matrix

COLORS={0:'#d94145',1:'#356ed3'}
def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def plot(ax,points,title,limits,*,edges=None,gt=None,highlight=None):
    world=points['world_xyz'];oid=points['object_id'];active=points.get('status',np.zeros(len(world)))!=3
    estimated=points.get('estimated_only',np.zeros(len(world),bool))
    for o in (0,1):
        for hist in (False,True):
            rows=(oid==o)&active&(estimated==hist)
            if rows.any():ax.scatter(*world[rows].T,s=2,c=('#f1a340' if o==0 else '#75c9be') if hist else COLORS[o],depthshade=False,alpha=.85)
    if edges is not None and edges.shape[1]:
        ax.add_collection3d(Line3DCollection(world[edges.T],colors='#637089',linewidths=.35,alpha=.45))
    if gt is not None:
        for o in (0,1):
            p=gt[o]
            if len(p):ax.scatter(*p.T,s=2,c='#333333',alpha=.3,depthshade=False)
    if highlight is not None and np.any(highlight):
        ax.scatter(*world[np.asarray(highlight,bool)&active].T,s=9,c='#111111',marker='x',depthshade=False)
    for axis,(lo,hi) in zip('xyz',limits):getattr(ax,'set_'+axis+'lim')(lo,hi)
    ax.view_init(25,-150);ax.set_box_aspect(np.ptp(limits,axis=1));ax.tick_params(labelsize=5)
    ax.set_title(title,fontsize=8);ax.set_xlabel('world X (m)',fontsize=6);ax.set_ylabel('Y',fontsize=6);ax.set_zlabel('Z',fontsize=6)
def graph_points(graph,tcp):
    rows=np.flatnonzero(graph['node_type']>=3)
    xyz=transform(graph['node_input'][rows,:3],pose_matrix(tcp))
    edges=graph['edge_index'][:,graph['edge_type']==0]
    return dict(world_xyz=xyz,object_id=graph['node_type'][rows]-3),edges-3
def figure_png(path,panels,limits,title):
    fig=plt.figure(figsize=(15,8),dpi=110)
    for i,p in enumerate(panels):
        ax=fig.add_subplot(2,3,i+1,projection='3d');plot(ax,p[0],p[1],limits,**p[2])
    fig.suptitle(title,fontsize=11);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(path)
    fig.canvas.draw();image=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy();plt.close(fig)
    check=cv2.imread(str(path))
    if check is None or check.shape[:2]!=image.shape[:2]:raise ValueError('PNG decode failed')
    return image
def video(path,frames,fps=5):
    h,w=frames[0].shape[:2];writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():raise RuntimeError('MP4 writer unavailable')
    for im in frames:writer.write(cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
    writer.release();cap=cv2.VideoCapture(str(path));n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        n+=1
        if frame.shape[:2]!=(h,w):raise ValueError('MP4 shape')
    cap.release()
    if n!=len(frames):raise ValueError('MP4 frame decode mismatch')
    return dict(file=path.name,decoded_frames=n,fps=fps,width=w,height=h)
def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);args=p.parse_args();root=args.geometry
    manifest=json.loads((root/'geometry_manifest.json').read_text());capture=Path(manifest['config']['capture']);out=root/'media';out.mkdir(exist_ok=True)
    records=[f for f in manifest['frames'] if f['seed']==30002]
    if not records:raise ValueError('fixed seed30002 unavailable')
    folder=root/records[0]['folder'];g=load(folder/'G0_map.npz');l=load(folder/'L0_map.npz')
    cameras0,_=load_cameras(capture/'collection'/'seed30002'/'f0000_input.npz')
    with np.load(capture/'collection'/'seed30002'/'f0000_gt.npz',allow_pickle=False) as a:gt0={k:a[k] for k in a.files}
    visible0=visible_reference(cameras0,gt0)
    with np.load(capture/'collection'/'seed30002'/'f0000_masks.npz',allow_pickle=False) as a:masks0={k:a[k] for k in a.files}
    fig,axes=plt.subplots(2,len(cameras0),figsize=(8,7),squeeze=False)
    for j,c in enumerate(cameras0):
        axes[0,j].imshow(c.rgb);axes[0,j].set_title(c.camera+' current RGB')
        overlay=c.rgb.astype(float).copy()
        for oid,color in ((0,np.array([220,40,40])),(1,np.array([40,90,240]))):
            selected=masks0[c.camera][oid];overlay[selected]=.45*overlay[selected]+.55*color
        axes[1,j].imshow(overlay.astype('uint8'));axes[1,j].set_title('Predicted masks: peg red / box blue')
        for i in (0,1):axes[i,j].axis('off')
    fig.tight_layout();fig.savefig(out/'sensor_input_t0000.png',dpi=130);plt.close(fig)
    union=np.concatenate((g['world_xyz'],l['world_xyz'],visible0[0],visible0[1]));lo=union.min(0)-.02;hi=union.max(0)+.02
    # Include evaluation-visible GT so a missing predicted Peg remains visible
    # in the overlay. Limits never enter sampling, map or learned features.
    limits=np.c_[lo,hi];write_json(out/'view_contract.json',dict(limits=limits,view=[25,-150],marker_size=2,coordinate='world metres',fixed_seed=30002,screenshot_exact_reproduction=False))
    collections={k:[] for k in ('same_view','matched_N','persistent_map','main_wrist','new_surface','ghost_tracking')}
    for f in records:
        t=f['timestep'];folder=root/f['folder'];source=capture/'collection'/'seed30002'/f'f{t:04d}_input.npz'
        cameras,data=load_cameras(source)
        with np.load(source.with_name(f'f{t:04d}_gt.npz'),allow_pickle=False) as a:gt={k:a[k] for k in a.files}
        contract=json.loads((source.parent/'sensor_contract.json').read_text());main=contract['roles']['main']
        visible=visible_reference(cameras,gt,[main]);tcp=data['tcp_pose']
        maps={n:load(folder/(n+'_map.npz')) for n in ('G0','G1','G2','G3','G4','L0','L1','L2','L3','L4')}
        original=load(folder/'G0_original_graph.npz');gp,ge=graph_points(original,tcp)
        sp=maps['L0']
        reference={oid:sp['world_xyz'][sp['object_id']==oid] for oid in (0,1)}
        se,edge_audit=surface_edges(sp['world_xyz'],sp['world_normal'],sp['normal_valid'],sp['object_id'],reference,[c for c in cameras if c.camera==main])
        save_npz(out/f'L0_native_surface_edges_t{t:04d}.npz',edge_index=se,world_xyz=sp['world_xyz'],object_id=sp['object_id'])
        gm=load(folder/'G0_matched_map.npz');lm=load(folder/'L0_matched_map.npz')
        panels=[(maps['G0'],'G0 native points ONLY',{}),(gp,'G0 native points + Raw EDE',dict(edges=ge)),
            (maps['L0'],'L0 native legacy points ONLY',{}),(sp,'Same native FPS points + 3D surface edges (F5)',dict(edges=se)),
            (gm,'Matched-N G0 subset (posthoc)',{}),(lm,'Matched-N FPS subset + GT visible overlay',dict(gt=visible))]
        im=figure_png(out/f'comparison_t{t:04d}.png',panels,limits,f'Seed30002 t={t} | same RGB-D/masks/world view | native and cap128 labelled')
        collections['same_view'].append(im)
        matched=[(gm,'Matched-N G0 points',{}),(lm,'Matched-N L0 points',{}),(gm,'G0 matched + GT visible',dict(gt=visible)),
                 (lm,'L0 matched + GT visible',dict(gt=visible)),(maps['G0'],'G0 native (count reference)',{}),(maps['L0'],'L0 native (count reference)',{})]
        collections['matched_N'].append(figure_png(out/f'matched_t{t:04d}.png',matched,limits,f'Matched-N posthoc diagnostic | seed30002 t={t}'))
        temporal=[(maps[n],n+' observed:red/blue; estimated:orange/teal',{}) for n in ('G1','L1','G2','L2','G4','L4')]
        im=figure_png(out/f'persistent_t{t:04d}.png',temporal,limits,f'Causal prefix | seed30002 t={t} | no future correction')
        collections['persistent_map'].append(im);collections['new_surface'].append(im);collections['ghost_tracking'].append(im)
        dual=[(maps[n],n,{}) for n in ('G1','L1','G3','L3','G4','L4')]
        collections['main_wrist'].append(figure_png(out/f'fusion_t{t:04d}.png',dual,limits,f'Main / Main+Wrist | seed30002 t={t}'))
    videos=[video(out/(name+'.mp4'),ims) for name,ims in collections.items()]
    write_json(out/'decode_audit.json',dict(status='PASS',videos=videos,pngs=len(list(out.glob('*.png'))),
        new_surface_and_ghost_video_scope='same fixed causal prefix; does not assert this prefix contains a full occlusion/reappearance or failed tracking event'))
if __name__=='__main__':main()
