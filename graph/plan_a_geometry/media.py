"""Saved same-time camera/point/edge evidence; GT overlay is evaluation only."""
import argparse,json
from pathlib import Path
import numpy as np,cv2,imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from graph.persistent_geometry.evaluate import load_cameras
from graph.persistent_geometry.metrics import visible_reference
from graph.persistent_geometry.contracts import transform
from graph.persistent_geometry.render import decode_video
from visual.runtime.vision_packet import pose_matrix
from graph.integration_baseline.io import write_json


def arrays(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def world_panel(ax,xyz,oid,bounds,title,edges=None,truth=None):
    if truth is not None:
        for points in truth.values():
            if len(points):ax.scatter(*points[::max(1,len(points)//700)].T,c='.65',s=.6,alpha=.25)
    if edges is not None and len(edges):ax.add_collection3d(Line3DCollection(xyz[edges],colors='#999999',linewidths=.3,alpha=.7))
    for kind,color in ((0,'#ce423c'),(1,'#276ed0')):
        selected=oid==kind
        if selected.any():ax.scatter(*xyz[selected].T,c=color,s=2)
    ax.set(xlim=bounds[0],ylim=bounds[1],zlim=bounds[2],xlabel='World X m',ylabel='Y m',zlabel='Z m',title=title)
    ax.view_init(elev=27,azim=-65);ax.set_box_aspect(np.ptp(np.array(bounds),axis=1));ax.tick_params(labelsize=5)
    ax.xaxis.label.set_size(6);ax.yaxis.label.set_size(6);ax.zaxis.label.set_size(6)


def render(root,frame,bounds):
    folder=root/frame['folder'];cameras,data=load_cameras(frame['input_path']);gt=arrays(frame['gt_path']);masks=arrays(frame['masks_path'])
    filters=arrays(folder/'robot_filter.npz');accepted=arrays(folder/'accepted_masks.npz');gate=json.loads((folder/'gate.json').read_text())
    refs=visible_reference(cameras,gt)
    fig=plt.figure(figsize=(15,18),dpi=90)
    for row,c in enumerate(cameras):
        ax=fig.add_subplot(5,3,row*3+1);im=c.rgb.astype(float)
        for oid,color in ((0,np.array([255,45,45])),(1,np.array([30,100,255]))):im[masks[c.camera][oid]]=.55*im[masks[c.camera][oid]]+.45*color
        ax.imshow(im.astype('uint8'));ax.set_title(c.camera+' RGB + predicted Peg(red) / Box(blue)',fontsize=8);ax.axis('off')
        ax=fig.add_subplot(5,3,row*3+2);ax.imshow(c.depth,cmap='magma',vmin=0,vmax=1)
        confirmed=filters[c.camera+'/confirmed_robot'];uncertain=filters[c.camera+'/uncertain']
        if confirmed.any():ax.contour(confirmed,levels=[.5],colors='cyan',linewidths=.5)
        if uncertain.any():ax.contour(uncertain,levels=[.5],colors='yellow',linewidths=.5)
        ax.set_title('Current depth m; robot cyan / uncertain yellow',fontsize=8);ax.axis('off')
        ax=fig.add_subplot(5,3,row*3+3)
        clean=accepted['C3/'+c.camera]
        out=np.zeros((*c.depth.shape,3),np.uint8);out[masks[c.camera].any(0)]=[90,90,90];out[clean[0]]=[230,55,45];out[clean[1]]=[40,110,235];ax.imshow(out)
        ax.set_title(c.camera+' accepted Plan A support; gray removed',fontsize=8);ax.axis('off')
    for col,(rep,label) in enumerate((('SF','A-SF'),('L','A-L (F5 augmented)'),('H','A-H'))):
        p=arrays(folder/f'C3_{rep}_cap128_points.npz')
        ax=fig.add_subplot(5,3,7+col,projection='3d');world_panel(ax,p['world_xyz'],p['object_id'],bounds,label+' cap128 points-only',truth=refs)
        p2=arrays(folder/f'C3_{rep}_matched_N_points.npz')
        ax=fig.add_subplot(5,3,10+col,projection='3d');world_panel(ax,p2['world_xyz'],p2['object_id'],bounds,label+' exact matched N points-only',truth=refs)
        g=arrays(folder/f'C3_{rep}_graph.npz');xyz=transform(g['node_input'][:,:3],pose_matrix(data['tcp_pose']));e=g['edge_index'][:,g['edge_type']==0].T
        ax=fig.add_subplot(5,3,13+col,projection='3d');world_panel(ax,xyz,g['node_type']-3,bounds,label+' same cap128 points + common surface edges',edges=e,truth=refs)
    fig.suptitle(f"Episode {cameras[0].episode} | t={frame['timestep']} | {frame['condition']} | base_camera + hand_camera\n"
        f"C3 Plan A | Wrist Peg {'ACCEPTED' if gate['accepted'] else 'REJECTED'}: {', '.join(gate['reasons']) or 'valid observed support'} | Wrist Box excluded\n"
        'Gray 3D points: current GT-visible evaluation overlay only. Red Peg / Blue Box. Identical world axes across representations.',fontsize=10)
    fig.tight_layout(rect=[0,0,1,.955]);fig.canvas.draw();image=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy();plt.close(fig)
    return image


def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.geometry/'geometry_manifest.json').read_text());index=[]
    for seed in sorted({f['seed'] for f in manifest['frames']}):
        frames=sorted([f for f in manifest['frames'] if f['seed']==seed and f['condition']=='baseline'],key=lambda f:f['timestep'])
        # Complete evaluated sequence, not a temporal upsample or generated observation.
        allpoints=[]
        for f in frames:
            refs=visible_reference(load_cameras(f['input_path'])[0],arrays(f['gt_path']))
            allpoints.extend(p for p in refs.values() if len(p))
            for rep in ('SF','L','H'):
                observed=arrays(a.geometry/f['folder']/f'C3_{rep}_native.npz')['world_xyz']
                if len(observed):allpoints.append(observed)
        points=np.concatenate(allpoints);lo=points.min(0)-.025;hi=points.max(0)+.025;bounds=list(zip(lo,hi))
        video=a.output/f'seed{seed}_plan_a.mp4';writer=imageio.get_writer(video,fps=4,codec='libx264',quality=7,macro_block_size=2,ffmpeg_params=['-threads','1'])
        for i,f in enumerate(frames):
            image=render(a.geometry,f,bounds);writer.append_data(image)
            if i in {0,len(frames)//3,2*len(frames)//3,len(frames)-1}:
                path=a.output/f'seed{seed}_t{f["timestep"]:04d}.png';imageio.imwrite(path,image);assert imageio.imread(path).shape==image.shape
                refs=visible_reference(load_cameras(f['input_path'])[0],arrays(f['gt_path']))
                fig=plt.figure(figsize=(15,5),dpi=100)
                for col,rep in enumerate(('SF','L','H')):
                    pp=arrays(a.geometry/f['folder']/f'C3_{rep}_native.npz')
                    ax=fig.add_subplot(1,3,col+1,projection='3d')
                    world_panel(ax,pp['world_xyz'],pp['object_id'],bounds,f'A-{rep} native points N={len(pp["world_xyz"])}',truth=refs)
                fig.suptitle(f'Seed{seed} t={f["timestep"]} C3 native policies; same world axes; GT visible overlay gray')
                fig.tight_layout();fig.savefig(a.output/f'seed{seed}_t{f["timestep"]:04d}_native.png');plt.close(fig)
            index.append(dict(seed=seed,timestep=f['timestep'],condition=f['condition'],video=video.name,frame_index=i,cameras=['base_camera','hand_camera'],variants=['C3_SF','C3_L','C3_H'],episode=load_cameras(f['input_path'])[0][0].episode))
        writer.close();n,shape=decode_video(video);assert n==len(frames)
        print(f'media seed={seed} decoded={n}',flush=True)
    write_json(a.output/'media_index.json',dict(frames=index,GT_scope='evaluation overlay only',decoded_all_frames=True))
if __name__=='__main__':main()
