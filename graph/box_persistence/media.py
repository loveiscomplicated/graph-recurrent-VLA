"""Scientific PNG/MP4 artifacts, fixed world axes and explicit evaluated overlays."""
import argparse,json
from pathlib import Path
import numpy as np,cv2,imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graph.integration_baseline.io import write_json
from graph.persistent_geometry.contracts import transform
from graph.persistent_geometry.evaluate import load_cameras
from visual.runtime.vision_packet import pose_matrix
from .summarize import read_csv


def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def panel(ax,records,bounds,title,truth=None,rim=None,edges=None,graph_xyz=None):
    box=records['object_id']==1;xyz=records['world_xyz'][box];current=records['currently_observed'][box];low=records['status'][box]==2
    for mask,color,label in ((current&~low,'#176bc0','current'),(~current&~low,'#ef941f','estimated'),(low,'#cf4444','low confidence')):
        p=xyz[mask]
        if len(p):ax.scatter(*p.T,s=7,c=color,label=label,alpha=.8)
    if truth is not None and len(truth):ax.scatter(*truth[::3].T,s=2,c='#888888',alpha=.18)
    if rim is not None:ax.scatter(*rim[::4].T,s=4,c='#0fadb0',alpha=.4)
    if edges is not None:
        for a,b in edges.T:
            if a>=3 and b>=3 and a<b:
                line=graph_xyz[[a,b]];ax.plot(*line.T,c='#222222',linewidth=.35,alpha=.4)
    for setter,lim in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),bounds):setter(*lim)
    outside=((xyz<np.array(bounds)[:,0])|(xyz>np.array(bounds)[:,1])).any(1).sum()
    ax.set_title(f'{title}\nN={len(xyz)}, current={current.sum()}, low={low.sum()}, outside view={outside}',fontsize=9)
    ax.view_init(24,-55);ax.set_box_aspect([1,1,1]);ax.set_xlabel('World X (m)',fontsize=8);ax.set_ylabel('Y',fontsize=8);ax.set_zlabel('Z',fontsize=8)
    ax.tick_params(labelsize=6)


def figure_image(fig,path):
    fig.canvas.draw();im=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy();plt.close(fig)
    imageio.imwrite(path,im)
    # Decode the stored file, not only the in-memory canvas.
    assert np.array_equal(imageio.imread(path),im)
    return im


def decode(path):
    c=cv2.VideoCapture(str(path));n=0;shape=None
    while True:
        ok,frame=c.read()
        if not ok:break
        shape=list(frame.shape);n+=1
    c.release()
    if not n:raise ValueError('video decode failed')
    return dict(path=path.name,decoded_frames=n,shape=shape)


def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--motion',type=Path,required=True);p.add_argument('--visible-motion',type=Path);p.add_argument('--control-capture',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    files=[];frames=sorted(p for p in (a.geometry/'geometry/seed30002').glob('f*') if p.is_dir())
    alltruth=np.concatenate([load(f/'evaluation_reference.npz')['main_visible'] for f in frames])
    lo,hi=alltruth.min(0)-.04,alltruth.max(0)+.04;span=max(hi-lo);center=(lo+hi)/2;bounds=np.c_[center-span/2,center+span/2]
    writer=imageio.get_writer(a.output/'temporal_box.mp4',fps=4,codec='libx264',macro_block_size=2,ffmpeg_params=['-threads','1'])
    for index,f in enumerate(frames):
        t=int(f.name[1:]);ref=load(f/'evaluation_reference.npz');source=a.capture/'collection/seed30002'/f'f{t:04d}_input.npz'
        cameras,data=load_cameras(source);fig=plt.figure(figsize=(13.5,8),dpi=100)
        full=[]
        for col,name in enumerate(('B0','B1','B2')):
            rr=load(f/(name+'_native.npz'));full.append(rr['world_xyz'][rr['object_id']==1])
            panel(fig.add_subplot(2,3,col+1,projection='3d'),rr,bounds,name+' native',ref['main_visible'],ref['rim'])
            gr=load(f/(name+'_cap128_graph.npz'));meta=json.loads(f.joinpath(name+'_cap128_graph.json').read_text());ids=meta['node_ids']
            selected={k:(v[[list(rr['persistent_node_id']).index(i) for i in ids]] if len(v)==len(rr['object_id']) else v) for k,v in rr.items()}
            world=transform(gr['node_input'][:,:3],pose_matrix(data['tcp_pose']));ed=gr['edge_index'][:,(gr['edge_type']==0)&(gr['node_type'][gr['edge_index'][0]]==4)]
            panel(fig.add_subplot(2,3,col+4,projection='3d'),selected,bounds,name+' cap128 + common surface edges',ref['main_visible'],ref['rim'],ed,world)
        fig.suptitle(f'DEVELOPMENT seed30002 t={t}; Box only, fixed world view | blue=current, orange=estimated, red=low confidence\nGray=current GT-visible and cyan=rim are EVALUATION ONLY; zoom view annotates excluded display points; full data retained',fontsize=10)
        fig.tight_layout(rect=[0,.06,1,.92]);path=a.output/f'temporal_t{t:04d}.png';im=figure_image(fig,path);writer.append_data(im);files.append(dict(path=path.name,seed=30002,timestep=t,representations=['B0','B1','B2'],budget=['native','cap128']))
        if index in (0,len(frames)//2,len(frames)-1):
            cloud=np.concatenate(full+[ref['main_visible']]);low,high=cloud.min(0)-.01,cloud.max(0)+.01;span=max(high-low);mid=(low+high)/2;fb=np.c_[mid-span/2,mid+span/2]
            fig=plt.figure(figsize=(13.5,4.5),dpi=100)
            for col,name in enumerate(('B0','B1','B2')):panel(fig.add_subplot(1,3,col+1,projection='3d'),load(f/(name+'_native.npz')),fb,name+' full extent',ref['main_visible'],ref['rim'])
            fig.suptitle(f'Same world axes/scale, all native Box points, seed30002 t={t}');fig.tight_layout(rect=[0,.10,1,.92]);figure_image(fig,a.output/f'full_extent_t{t:04d}.png')
    writer.close();videos=[decode(a.output/'temporal_box.mp4')]
    stats=json.loads((a.summary/'geometry_macro.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(14,8),dpi=100)
    for col,mode in enumerate(('native','cap128','matched_N')):
        rr=[next(r for r in stats if r['split']=='validation' and r['variant']==name and r['budget']==mode) for name in ('B0','B1','B2','O1','O2','O3')]
        names=[r['variant'] for r in rr]
        axes[0,col].bar(names,[r['surface_distance_mean_mm'] or 0 for r in rr]);axes[0,col].set_title(mode+' mean distance (mm)')
        x=np.arange(len(rr));axes[1,col].bar(x-.15,[r['current_visible_coverage'] for r in rr],.3,label='current-visible');axes[1,col].bar(x+.15,[r['historical_coverage'] for r in rr],.3,label='historical observed');axes[1,col].set_xticks(x,names);axes[1,col].set_ylim(0,1);axes[1,col].legend(fontsize=8)
    fig.suptitle('Validation episode macro; oracle conditions are diagnostic only; distance to sampled external collision surfaces')
    fig.tight_layout(rect=[0,0,1,.95]);figure_image(fig,a.output/'geometry_validation.png')
    csvrows=read_csv(a.geometry/'geometry/seed30002/metrics.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,7),dpi=100)
    for name in ('B0','B1','B2'):
        rr=[r for r in csvrows if r['variant']==name and r['budget']=='cap128']
        for ax,key,title in zip(axes.flat,('historical_coverage','current_visible_coverage','current_rim_coverage','ghost_nodes'),('Historical observed coverage','Current-visible coverage','Currently observed rim coverage','Estimated ghost nodes >8mm')):
            ax.plot([r['timestep'] for r in rr],[r[key] if r[key] is not None else np.nan for r in rr],label=name);ax.set_title(title);ax.set_xlabel('Physical timestep');ax.legend()
    fig.tight_layout();figure_image(fig,a.output/'boundary_and_ghost.png')
    # Actual occluder RGB-D with independent branch geometry, all at same robot state.
    capmanifest=json.loads((a.control_capture/'capture_manifest.json').read_text());ep=capmanifest['episodes'][0]
    writer=imageio.get_writer(a.output/'motion_controls.mp4',fps=1,codec='libx264',macro_block_size=2,ffmpeg_params=['-threads','1'])
    cases=[(b,a.visible_motion if b['timestep']==0 else a.motion) for b in ep['branches'] if b['timestep'] in ([0,25] if a.visible_motion else [25]) and b['condition'] in ('box_x+10','box_partial_x+10','box_full_x+10')]
    for b,motion_root in cases:
        physical_t=b['timestep']
        source=a.control_capture/'collection/seed30002'/b['source'];inp=load(source/'input.npz');gt=load(source/'gt.npz');cs,_=load_cameras(source/'input.npz');main=next(c for c in cs if c.camera=='base_camera')
        truth=main.world_points()[gt['base_camera/segmentation']==gt['actor_ids'][1]]
        for phase in ('intervention','reappearance') if b.get('occlusion')=='full' else ('intervention',):
            displayed=inp['base_camera/rgb'];shown_truth=truth
            if phase=='reappearance':
                vb=next(x for x in ep['branches'] if x['timestep']==physical_t and x['condition']==b['condition'].replace('box_full_','box_'))
                vfolder=a.control_capture/'collection/seed30002'/vb['source'];displayed=load(vfolder/'input.npz')['base_camera/rgb'];vg=load(vfolder/'gt.npz');vc,_=load_cameras(vfolder/'input.npz');vm=next(c for c in vc if c.camera=='base_camera')
                shown_truth=vm.world_points()[vg['base_camera/segmentation']==vg['actor_ids'][1]]
            fig=plt.figure(figsize=(13.5,4.8),dpi=100);ax=fig.add_subplot(1,3,1);ax.imshow(displayed);ax.axis('off');ax.set_title('Actual Main RGB, '+b['condition']+' '+phase)
            for col,n in enumerate(('B1','B2')):
                parent=motion_root/f't{physical_t:04d}'/b['condition']/n;rr=load(parent/('points.npz' if phase=='intervention' else 'reappearance_points.npz'))
                trace=json.loads((parent/('trace.json' if phase=='intervention' else 'reappearance_trace.json')).read_text())
                panel(fig.add_subplot(1,3,col+2,projection='3d'),rr,bounds,n+' '+trace['state'],shown_truth)
            fig.suptitle(f'MOTION CONTROL, physical t={physical_t}, {phase}, Box X+10mm; no physics step/new contact force\nAt t0 baseline initializes memory before intervention; reappearance re-presents the actual visible moved snapshot at a diagnostic next tick',fontsize=10)
            fig.tight_layout(rect=[0,.06,1,.88]);path=a.output/(f't{physical_t:04d}_'+b['condition']+'_'+phase+'.png');writer.append_data(figure_image(fig,path));files.append(dict(path=path.name,physical_timestep=physical_t,condition=b['condition'],phase=phase))
    writer.close();videos.append(decode(a.output/'motion_controls.mp4'))
    write_json(a.output/'media_index.json',dict(frames=files,videos=videos,view='fixed per-episode world axes; full-extent PNG controls and annotated zoom',no_GNN_plot='no trained GNN/probe comparison when geometry gate fails'))
if __name__=='__main__':main()
