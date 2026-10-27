"""Scientific plots and temporal RGB-D/map video from saved diagnostic files."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import imageio.v2 as imageio
from scipy.spatial import cKDTree
from visual.runtime.vision_packet import pose_matrix
from .contracts import transform
from .metrics import sample_meshes
from .evaluate import load_cameras

COLORS=dict(G0='#596579',G1='#009e73',G2='#e69f00',G3='#0072b2',G4='#cc79a7',O1='#6d8d24',O2='#907a38',O3='#555555')


def read_npz(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def table(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def decode_video(path):
    """Decode every frame with bounded threads, without a second count process."""
    reader=imageio.get_reader(path,input_params=['-threads','1'],
        output_params=['-threads','1','-filter_threads','1'])
    first=last=None;count=0
    try:
        for frame in reader:
            if first is None:first=frame.copy()
            last=frame.copy();count+=1
    finally:reader.close()
    if count<2 or first.shape!=last.shape:raise ValueError('temporal MP4 decode')
    return count,list(first.shape)


def plot_axes(ax,bounds,title):
    ax.set(xlim=bounds[0],ylim=bounds[1],zlim=bounds[2],xlabel='World X (m)',ylabel='Y (m)',zlabel='Z (m)',title=title)
    ax.view_init(elev=24,azim=-65);ax.set_box_aspect([b[1]-b[0] for b in bounds]);ax.tick_params(labelsize=7)


def display_map(ax,data,gt,mesh,*,ids=False):
    active=data['status']!=3
    for oid,marker in ((0,'.'),(1,'^')):
        selected=(data['object_id']==oid)&active
        world=data['world_xyz'][selected];current=data['currently_observed'][selected]
        estimated=data['estimated_only'][selected]
        truth=transform(mesh[oid]['points'][::20],pose_matrix(gt['actor_pose'][oid]))
        ax.scatter(*truth.T,c='#a6a6a6',s=1,alpha=.15,label='GT evaluation overlay' if oid==0 else None)
        local=transform(world,np.linalg.inv(pose_matrix(gt['actor_pose'][oid])))
        ghost=estimated&(mesh[oid]['tree'].query(local)[0]>.008)
        if ids:
            values=[int(s.rsplit(':',1)[-1])%20 if s.rsplit(':',1)[-1].isdigit() else i%20
                    for i,s in enumerate(data['persistent_node_id'][selected])]
            ax.scatter(*world.T,c=values,cmap='tab20',vmin=0,vmax=19,s=6,marker=marker)
        else:
            for subset,color,label in ((current,'#00bcd4','Current measured'),
                    (estimated&~ghost,'#f3a22f','Estimated historical'),(ghost,'#d62728','Historical >8mm GT error')):
                if subset.any():ax.scatter(*world[subset].T,c=color,s=5,marker=marker,label=label if oid==0 else None)


def sensor_panel(ax,camera,masks,filtered,*,depth=False):
    if depth:
        image=ax.imshow(camera.depth,cmap='magma',vmin=.05,vmax=.8)
        ax.set_title(camera.camera+' depth (m)',fontsize=10)
        ax.contour(filtered['confirmed_robot'],levels=[.5],colors=['red'],linewidths=.5)
        ax.contour(filtered['uncertain'],levels=[.5],colors=['yellow'],linewidths=.4)
    else:
        ax.imshow(camera.rgb)
        for oid,color in ((0,'cyan'),(1,'lime')):
            if masks[oid].any():ax.contour(masks[oid],levels=[.5],colors=[color],linewidths=.7)
        ax.set_title(camera.camera+' RGB; predicted Peg cyan / Box green',fontsize=9)
    ax.axis('off')


def sequence(capture,geometry,destination,seed,*,stride=2):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((geometry/'geometry_manifest.json').read_text())
    frames=sorted([f for f in manifest['frames'] if f['seed']==seed and f.get('condition','baseline')=='baseline'],key=lambda f:f['timestep'])
    source=capture/'collection'/f'seed{seed}'
    mesh=sample_meshes(source/'evaluation_meshes.npz')
    clouds=[]
    for frame in frames[::max(1,len(frames)//10)]:
        gt=read_npz(source/f'f{frame["timestep"]:04d}_gt.npz')
        clouds.extend(transform(mesh[oid]['points'][::50],pose_matrix(gt['actor_pose'][oid])) for oid in (0,1))
    clouds=np.concatenate(clouds);lo=clouds.min(0)-.035;hi=clouds.max(0)+.035
    bounds=list(zip(lo,hi))
    visibility=[]
    for frame in frames:
        gt=read_npz(source/f'f{frame["timestep"]:04d}_gt.npz')
        visibility.append(sum(int((gt[camera+'/segmentation']==gt['actor_ids'][1]).sum())
            for camera in [k[:-13] for k in gt if k.endswith('/segmentation')]))
    identity_indices=[0,len(frames)//3,2*len(frames)//3,len(frames)-1]
    hidden=np.flatnonzero(np.array(visibility)==0)
    if len(hidden):
        start=int(hidden[0]);end=start
        while end+1<len(frames) and visibility[end+1]==0:end+=1
        identity_indices=[0,start,end,min(end+1,len(frames)-1)]
    selected_times={frames[i]['timestep'] for i in identity_indices}|{frames[-1]['timestep']}
    video=destination/f'seed{seed}_persistent_geometry.mp4'
    written=[]
    with imageio.get_writer(video,fps=8,codec='libx264',quality=8,macro_block_size=16,
            ffmpeg_params=['-threads','1','-filter_threads','1']) as writer:
        chosen=[frames[i] for i in sorted(set(range(0,len(frames),stride))|set(identity_indices)|{len(frames)-1})]
        for frame in chosen:
            t=frame['timestep'];folder=geometry/frame['folder']
            cameras,data=load_cameras(source/f'f{t:04d}_input.npz')
            masks=read_npz(source/f'f{t:04d}_masks.npz');gt=read_npz(source/f'f{t:04d}_gt.npz')
            filtered=read_npz(folder/'robot_filter.npz')
            fig=plt.figure(figsize=(12.8,8),dpi=100)
            grid=fig.add_gridspec(2,4,height_ratios=[1,2],hspace=.16,wspace=.1)
            for i,camera in enumerate(cameras):
                result={key.split('/',1)[1]:value for key,value in filtered.items() if key.startswith(camera.camera+'/')}
                sensor_panel(fig.add_subplot(grid[0,2*i]),camera,masks[camera.camera],result)
                sensor_panel(fig.add_subplot(grid[0,2*i+1]),camera,masks[camera.camera],result,depth=True)
            for column,name in ((0,'G1'),(2,'G4')):
                ax=fig.add_subplot(grid[1,column:column+2],projection='3d')
                maps=read_npz(folder/(name+'_map.npz'))
                display_map(ax,maps,gt,mesh);plot_axes(ax,bounds,name+(' current Main-only' if name=='G1' else ' persistent Main + Wrist'))
                handles,labels=ax.get_legend_handles_labels()
                if handles:ax.legend(handles,labels,loc='upper left',fontsize=6)
            fig.suptitle(f'Seed {seed} / t={t} / physical time {t/20:.2f}s / {frame.get("phase", "")}\n'
                f'Saved RGB-D; CPU PhysX; stride{stride} + event keyframes at8fps; GT only evaluation overlay',fontsize=11)
            fig.subplots_adjust(left=.025,right=.98,bottom=.05,top=.9)
            fig.canvas.draw();pixels=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy();writer.append_data(pixels)
            if t in selected_times or t==frames[-1]['timestep']:
                path=destination/f'seed{seed}_t{t:04d}_rgbd_map.png';fig.savefig(path,dpi=150);written.append(path.name)
            plt.close(fig)
    # A separate stable-ID montage makes persistent identity visible without
    # conflating it with per-frame pixel indices or rendering identity as input.
    fig=plt.figure(figsize=(14,4),dpi=130)
    for i,index in enumerate(identity_indices):
        frame=frames[index];t=frame['timestep'];folder=geometry/frame['folder']
        gt=read_npz(source/f'f{t:04d}_gt.npz');maps=read_npz(folder/'G4_map.npz')
        ax=fig.add_subplot(1,4,i+1,projection='3d');display_map(ax,maps,gt,mesh,ids=True)
        plot_axes(ax,bounds,f't={t}: Box GT pixels={visibility[index]}\nID mod20')
    fig.suptitle('G4 identity over time; colours derive from persistent allocation counters, not source pixels')
    fig.tight_layout();fig.savefig(destination/'persistent_ids.png');plt.close(fig)
    # Original image-connected Raw EDE and separately versioned surface edges.
    frame=frames[0];t=frame['timestep'];folder=geometry/frame['folder']
    cameras,data=load_cameras(source/f'f{t:04d}_input.npz')
    fig=plt.figure(figsize=(12,5),dpi=130)
    for i,(name,label) in enumerate((('G0','G0 induced original Raw EDE'),('G4','G4 signed-normal / witness surface edges'))):
        a=read_npz(folder/(name+'_graph.npz'));xyz=transform(a['node_input'][:,:3],pose_matrix(data['tcp_pose']))
        ax=fig.add_subplot(1,2,i+1,projection='3d')
        edge=a['edge_index'][:,(a['edge_type']==0)&(a['edge_index'][0]<a['edge_index'][1])]
        if edge.shape[1]:ax.add_collection3d(Line3DCollection(xyz[edge.T],colors=COLORS[name],linewidths=.5,alpha=.5))
        ax.scatter(*xyz[3:].T,s=5,c=COLORS[name]);plot_axes(ax,bounds,label)
    fig.tight_layout();fig.savefig(destination/'raw_ede_vs_surface_edges.png');plt.close(fig)
    # Independent actual decode, rather than assuming a successful writer.
    count,shape=decode_video(video)
    if count!=len(chosen):raise ValueError('MP4 decode/readback mismatch')
    return dict(seed=seed,video=video.name,video_frames=count,decoded_shape=shape,
        physical_observation_count=len(frames),render_stride=stride,pngs=written,
        event_keyframes=[frames[i]['timestep'] for i in identity_indices],
        event_selection='first actual GT-both-camera Box occlusion interval and first reappearance; otherwise equal time intervals')


def temporal(geometry,destination,seed):
    manifest=json.loads((geometry/'geometry_manifest.json').read_text())
    base=Path(manifest['config'].get('geometry',geometry))
    rows=table(base/'geometry_metrics.csv')
    rows=[r for r in rows if int(r['seed'])==seed and r['budget_mode']=='cap128_per_object']
    fig,axes=plt.subplots(2,3,figsize=(14,7),dpi=130)
    fields=(('common_both_visible_coverage_4mm','Both-camera current GT coverage'),
            ('surface_distance_mean_mm','Surface distance (mm)'),('ghost_count','Historical ghost nodes >8mm'),
            ('motion_update_error_mm','Stable-node motion update error (mm)'),
            ('new_correct_previously_unseen_nodes','New correct previously unseen nodes'),
            ('uncertainty_mean_mm','Position uncertainty (mm)'))
    for ax,(field,title) in zip(axes.flat,fields):
        for name in ('G0','G1','G2','G3','G4','O3'):
            rr=sorted([r for r in rows if r['variant']==name and r['object_id']=='0' and r.get(field,'')!=''],key=lambda r:int(r['timestep']))
            ax.plot([int(r['timestep']) for r in rr],[float(r[field]) for r in rr],color=COLORS[name],label=name,lw=1.2,ls='--' if name=='O3' else '-')
        ax.set(title='Peg: '+title,xlabel='Physical timestep');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8,ncol=3);fig.suptitle(f'Seed{seed}; identical cap128/object; empty prediction has zero coverage; Oracle shown dashed')
    fig.tight_layout();fig.savefig(Path(destination)/'temporal_geometry.png');plt.close(fig)


def filter_and_aperture(capture,geometry,destination,seed):
    from .geometry_audit import aperture_reference
    manifest=json.loads((geometry/'geometry_manifest.json').read_text())
    parent=Path(manifest['config'].get('geometry',geometry))
    rows=[r for r in table(parent/'robot_filter_metrics.csv') if int(r['seed'])==seed]
    worst=max(rows,key=lambda r:float(r.get('robot_contamination_before') or 0)*int(r['predicted_before']))
    t=int(worst['timestep']);source=capture/'collection'/f'seed{seed}'
    frame=next(f for f in manifest['frames'] if f['seed']==seed and f['timestep']==t and f.get('condition','baseline')=='baseline')
    folder=geometry/frame['folder'];cameras,data=load_cameras(source/f'f{t:04d}_input.npz')
    masks=read_npz(source/f'f{t:04d}_masks.npz');filtered=read_npz(folder/'robot_filter.npz')
    points=[camera.world_points()[masks[camera.camera].any(0)&np.isfinite(camera.depth)] for camera in cameras]
    allpoints=np.concatenate(points);lo=allpoints.min(0)-.02;hi=allpoints.max(0)+.02;bounds=list(zip(lo,hi))
    fig=plt.figure(figsize=(12,9),dpi=140)
    for row,camera in enumerate(cameras):
        world=camera.world_points();pred=masks[camera.camera].any(0)&np.isfinite(camera.depth)
        removed=filtered[camera.camera+'/confirmed_robot'];uncertain=filtered[camera.camera+'/uncertain']
        for column in (0,1):
            ax=fig.add_subplot(2,2,row*2+column+1,projection='3d')
            for subset,color,label in ((pred&~removed&~uncertain,'#00bcd4','Candidate object RGBD'),
                    (pred&removed,'#d62728','Confirmed robot'),(pred&uncertain,'#e6b800','Uncertain')):
                if column and label!='Candidate object RGBD':continue
                p=world[subset][::max(1,int(subset.sum())//2000)]
                if len(p):ax.scatter(*p.T,c=color,s=2,label=label)
            plot_axes(ax,bounds,camera.camera+(' after removal' if column else ' before removal'))
            ax.legend(fontsize=7)
    fig.suptitle(f'Seed{seed} t={t}: maximum predicted-mask robot-pixel contamination example\nRaw RGBD mask pixels before Shape-First sampling; same per-link filter as graph input')
    fig.tight_layout();fig.savefig(Path(destination)/'robot_filter_before_after.png');plt.close(fig)
    # A fixed evaluation-only aperture region exposes missing rims and crossing
    # edges. It neither samples input nodes nor supplies a task goal to models.
    gt=read_npz(source/f'f{t:04d}_gt.npz');mesh=sample_meshes(source/'evaluation_meshes.npz')
    hole,radius,xlo,xhi,rim=aperture_reference(mesh,gt)
    truth=transform(mesh[1]['points'][::15],pose_matrix(gt['actor_pose'][1]))
    bounds=list(zip(truth.min(0)-.015,truth.max(0)+.015))
    fig=plt.figure(figsize=(14,4.8),dpi=140)
    for i,name in enumerate(('G0','G4','O3')):
        a=read_npz(folder/(name+'_graph.npz'));world=transform(a['node_input'][:,:3],pose_matrix(data['tcp_pose']))
        ax=fig.add_subplot(1,3,i+1,projection='3d');ax.scatter(*truth.T,c='#aaaaaa',s=1,alpha=.15)
        ax.scatter(*rim.T,c='#ab1783',s=2,label='GT aperture rims (evaluation only)')
        selected=a['node_type']==4;ax.scatter(*world[selected].T,c=COLORS[name],s=5,label=name+' Box input')
        edges=a['edge_index'][:,(a['edge_type']==0)&(a['edge_index'][0]<a['edge_index'][1])]
        edges=edges[:,a['node_type'][edges[0]]==4]
        if edges.shape[1]:ax.add_collection3d(Line3DCollection(world[edges.T],colors=COLORS[name],linewidths=.5,alpha=.6))
        plot_axes(ax,bounds,name+' Box boundary and aperture');ax.legend(fontsize=6,loc='upper left')
    fig.suptitle(f'Seed{seed} t={t}; fixed GT Box evaluation region; out-of-region contamination shown in full-scene video')
    fig.tight_layout();fig.savefig(Path(destination)/'aperture_boundary.png');plt.close(fig)
    return dict(filter_example_seed=seed,filter_example_timestep=t,
        selection='maximum predicted-mask robot-pixel contamination, not a best-case filter outcome')
