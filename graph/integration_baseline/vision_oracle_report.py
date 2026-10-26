"""Task07-F readback, numerical geometry analysis, and CDN-free static report."""
import argparse
import csv
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from transforms3d.quaternions import quat2mat

from graph.integration_baseline.autonomous_reach_report import load_arrays
from graph.integration_baseline.direct_runtime import read
from graph.integration_baseline.grasp_intervention import atomic
from graph.integration_baseline.graph import build_graph
from graph.integration_baseline.io import read_packet, save_npz
from graph.integration_baseline.vision_oracle_adapter import POLICIES, SCOPE, validate_input
from graph.integration_baseline.vision_oracle_reach import assert_initial, surface_distances
from graph.integration_baseline.rollout_stop_clock import validate_controller_command


def stats(a):
    a=np.asarray(a)
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),
        p95=float(np.quantile(a,.95)),maximum=float(a.max()),minimum=float(a.min())) if len(a) else dict(count=0)


def mask_metrics(p,g):
    both=int(np.sum(p&g));union=int(np.sum(p|g));npix=int(p.sum());ngt=int(g.sum())
    return dict(iou=both/union if union else None,precision=both/npix if npix else None,
        recall=both/ngt if ngt else None,pixels=npix,gt_pixels=ngt,
        false_positive_pixels=int(np.sum(p&~g)),false_negative_pixels=int(np.sum(g&~p)))


def graph_metrics(packet,g):
    edge=g['edge_index'];n=len(g['node_input'])
    adj=coo_matrix((np.ones(edge.shape[1]),(edge[0],edge[1])),shape=(n,n)).tocsr()
    return dict(nodes=n,edges=edge.shape[1],connected_components=int(connected_components(adj,directed=False,return_labels=False)),
        raw_ede_edges=packet['intra_object_edge_index'].shape[1],
        robot_object_edges=int(np.sum(g['edge_type']==1)),peg_box_edges=int(np.sum(g['edge_type']==2)),
        feature_available=packet['feature_field_available'].sum(axis=0),
        feature_valid=packet['feature_field_valid'].sum(axis=0))


def graph_difference(pa,ga,pb,gb):
    def keys(p):return [(int(o),int(s)) for o,s in zip(p['object_id'],p['provenance']['source_index'])]
    ka,kb=keys(pa),keys(pb);sa,sb=set(ka),set(kb)
    def edges(p,g):
        labels=[('robot',i) for i in range(3)]+keys(p)
        return {(labels[int(a)],labels[int(b)],int(k)) for (a,b),k in zip(g['edge_index'].T,g['edge_type'])}
    ea,eb=edges(pa,ga),edges(pb,gb)
    ia,ib={k:i for i,k in enumerate(ka)},{k:i for i,k in enumerate(kb)}
    common=sorted(sa&sb)
    fa=np.array([pa['geometric_features'][ia[k]] for k in common])
    fb=np.array([pb['geometric_features'][ib[k]] for k in common])
    valid=np.isfinite(fa)&np.isfinite(fb)
    return dict(common_source_points=len(common),removed_source_points=len(sa-sb),added_source_points=len(sb-sa),
        removed_edges_by_source_identity=len(ea-eb),added_edges_by_source_identity=len(eb-ea),
        common_source_f5_changed_values=int(np.sum(valid&(fa!=fb))),
        common_source_f5_by_column={name:dict(changed=int(np.sum(valid[:,i]&(fa[:,i]!=fb[:,i]))),
            abs_difference=stats(np.abs(fa[:,i][valid[:,i]]-fb[:,i][valid[:,i]])))
            for i,name in enumerate(('normal_x','normal_y','normal_z','normal_variation_rad','depth_jump_m'))},
        common_source_f5_abs_difference=stats(np.abs(fa[valid]-fb[valid])))


def verify_original(run,original):
    a=run/'trials/A';obs=load_arrays(a/'observations.npz');old=load_arrays(original/'observations.npz')
    mask=load_arrays(a/'masks.npz')['masks'][:,0];oldmask=load_arrays(original/'masks.npz')['masks']
    action=load_arrays(a/'actions.npz')['actions'];oldaction=load_arrays(original/'actions.npz')['actions']
    hidden=load_arrays(a/'hidden.npz')['hidden'];oldhidden=load_arrays(original/'hidden.npz')['hidden']
    frames=min(len(action),len(oldaction));exact={}
    for k in obs:exact[k]=np.array_equal(obs[k][:frames+1],old[k][:frames+1])
    exact['mask']=np.array_equal(mask[:frames],oldmask[:frames])
    graph_exact=True;first_mismatch=None
    for t in range(frames):
        ga=load_arrays(a/f'graph_{t:04d}.npz');gb=load_arrays(original/f'graph_{t:04d}.npz')
        eq=all(np.array_equal(ga[k],gb[k]) for k in ga)
        if not eq and first_mismatch is None:first_mismatch=t
        graph_exact &= eq
    dact=float(np.max(np.abs(action[:frames]-oldaction[:frames])))
    dhid=float(np.max(np.abs(hidden[:frames]-oldhidden[:frames])))
    return dict(status='PASS' if all(exact.values()) and graph_exact and np.allclose(action[:frames],oldaction[:frames],atol=1e-6,rtol=1e-5) and np.allclose(hidden[:frames],oldhidden[:frames],atol=1e-6,rtol=1e-5) else 'DIFFERENCE_OBSERVED',
        frames=frames,observations_exact=exact,graph_source_exact=graph_exact,first_graph_mismatch=first_mismatch,
        maximum_action_difference=dact,maximum_hidden_difference=dhid,
        close27_action_max_difference=float(np.max(np.abs(action[:28]-oldaction[:28]))))


def verify_trial(folder,c):
    trial=read(folder/'trial.json');trace=read(folder/'trace.json');n=len(trace)
    obs=load_arrays(folder/'observations.npz');masks=load_arrays(folder/'masks.npz')['masks']
    side=load_arrays(folder/'gt_segmentation_oracle_audit.npz')
    hidden=load_arrays(folder/'hidden.npz');actions=load_arrays(folder/'actions.npz')
    if (n!=trial['steps'] or trial['physics_steps']!=5*n or trial['hidden_steps']!=n or
        hidden['hidden'].shape!=(n,1,1,144) or obs['robot_state'].shape!=(n+1,76) or
        len(trial['hidden_reset_events'])!=1):raise ValueError('episode clock/hidden/schema contract')
    np.testing.assert_array_equal(actions['actions'],actions['controller_input'])
    np.testing.assert_array_equal(hidden['timestep'],np.arange(n))
    negative=np.flatnonzero(actions['actions'][:,6]<0)
    if trial['first_close']!=(int(negative[0]) if len(negative) else None):raise ValueError('first negative close event')
    ids=side['actor_ids'].reshape(2,1,1)
    np.testing.assert_array_equal(masks[:,2],side['segmentation'][:,None]==ids[None])
    np.testing.assert_array_equal(masks[:,1],masks[:,0]&masks[:,2])
    applied=masks[:,0] if trial['condition']=='A' else masks[:,1] if trial['condition']=='B' else masks[:,2]
    np.testing.assert_array_equal(masks[:,3],applied)
    for t,r in enumerate(trace):
        if (r['physics_steps_before'],r['physics_steps_after'],r['inference_physics_steps'],r['segmentation_physics_steps'])!=(5*t,5*t+5,0,0):raise ValueError('physics contract')
        np.testing.assert_array_equal(r['action'],actions['actions'][t])
        validate_controller_command(actions['actions'][t],r['controller'])
        packet=read_packet(folder/f'packet_{t:04d}.npz');saved=load_arrays(folder/f'graph_{t:04d}.npz')
        row={k:v[t] for k,v in obs.items()}
        graph=build_graph(packet,row['robot_ee_geometry'],**c['graph'])
        validate_input(packet,graph,applied[t],row,str(hidden['episode_identity']),t,trial['input_policy'])
        for k in ('node_input','node_type','edge_index','edge_attr','edge_type'):
            np.testing.assert_array_equal(getattr(graph,k).numpy(),saved[k])
        np.testing.assert_array_equal(packet['provenance']['source_index'],saved['source_index'])
        world=packet['camera_points']@row['cam2world'][:3,:3].T+row['cam2world'][:3,3]
        ee=(world-row['tcp_pose'][:3])@quat2mat(row['tcp_pose'][3:])
        np.testing.assert_allclose(packet['points'],ee,atol=1e-6,rtol=1e-6)
        # The network consumes the established finite 14D graph and robot-only 76D state.
        if not np.isfinite(saved['node_input']).all():raise ValueError('nonfinite model graph')
        if trial['condition']!='A':
            source=packet['provenance']['source_index'];obj=packet['object_id']
            if not np.all(masks[t,2].reshape(2,-1)[obj,source]):raise ValueError('oracle source contamination')
    return trial,trace,obs,masks,hidden['hidden']


def montage(run,data,event,filename):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,6,figsize=(18,9),squeeze=False)
    for i,condition in enumerate(POLICIES):
        trial,trace,obs,masks,hidden=data[condition]
        t=trial['first_close'] if event=='close' else 27
        if t is None:t=0
        rgb=obs['rgb'][t];p=read_packet(run/'trials'/condition/f'packet_{t:04d}.npz')
        g=load_arrays(run/'trials'/condition/f'graph_{t:04d}.npz')
        axes[i,0].imshow(rgb)
        for col,j in ((1,0),(2,2),(3,1),(4,3)):
            image=rgb.copy()
            for oid,color in enumerate(([240,80,80],[60,160,255])):
                m=masks[t,j,oid];image[m]=(image[m]*.4+np.array(color)*.6).astype(np.uint8)
            axes[i,col].imshow(image)
        ax=axes[i,5];ax.imshow(rgb)
        row={k:v[t] for k,v in obs.items()};tcp=row['tcp_pose'];pos=g['node_input'][:,:3]
        world=pos@quat2mat(tcp[3:]).T+tcp[:3]
        cam=(world-row['cam2world'][:3,3])@row['cam2world'][:3,:3]
        uv=cam[:,:2]/cam[:,2,None]*np.array([row['intrinsic'][0,0],row['intrinsic'][1,1]])+row['intrinsic'][:2,2]-.5
        for a,b in g['edge_index'].T:
            if a<b:ax.plot(uv[[a,b],0],uv[[a,b],1],color='cyan',alpha=.18,lw=.4)
        source=p['provenance']['source_index'];correct=masks[t,2].reshape(2,-1)[p['object_id'],source]
        ax.scatter(uv[3:,0],uv[3:,1],c=np.where(correct,'lime','magenta'),s=3)
        for j,ax in enumerate(axes[i]):
            ax.set_xlim(-.5,127.5);ax.set_ylim(127.5,-.5);ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(f'{condition} t={t} '+['RGB','Predicted','GT visible','Cleaned','Applied','Actual graph'][j],fontsize=10)
    fig.suptitle('Each row is its own real state; green=correct source, magenta=non-target. '+('First-close events differ in time.' if event=='close' else 'Same absolute t does not mean same state.'))
    fig.tight_layout();fig.savefig(run/filename);plt.close(fig)


def compare_video(run,data,aligned=False):
    name='close_aligned_after_action.mp4' if aligned else 'rollout_comparison_after_action.mp4'
    writer=cv2.VideoWriter(str(run/name),cv2.VideoWriter_fourcc(*'mp4v'),20,(1536,512))
    if not writer.isOpened():raise ValueError('comparison video writer')
    caps={k:cv2.VideoCapture(str(run/'trials'/k/'rollout_after_action.mp4')) for k in POLICIES}
    try:
        times=range(-10,41) if aligned else range(max(x[0]['steps'] for x in data.values()))
        for t in times:
            images=[]
            for k in POLICIES:
                s=data[k][0]['first_close'];index=(s+t if s is not None else -1) if aligned else t
                if 0<=index<data[k][0]['steps']:
                    caps[k].set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=caps[k].read()
                    if not ok:raise ValueError('source video decode')
                else:image=np.zeros((512,512,3),np.uint8)
                if aligned:cv2.putText(image,f'close offset={t}; each trial own state',(8,492),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1)
                images.append(image)
            writer.write(np.concatenate(images,axis=1))
    finally:
        writer.release()
        for cap in caps.values():cap.release()


def training_distribution(bundle,data):
    split=read(bundle/'split_manifest.json')['splits']['train']
    manifest=read(bundle/'vision_manifest.json')
    counts=np.asarray([p['object_counts'] for e in manifest['episodes'] if e['episode_id'] in split
        for p in e['packets'] if p.get('has_action_target')])
    out=dict(train_episode_count=len(split),train_frames=len(counts),source='immutable train split + original vision manifest',
        train={name:dict(point_count=stats(counts[:,i]),presence_fraction=float(np.mean(counts[:,i]>0))) for i,name in enumerate(('peg','box'))},trials={})
    for k,(_,trace,*_) in data.items():
        values=np.asarray([r['point_counts'] for r in trace]);out['trials'][k]={}
        for i,name in enumerate(('peg','box')):
            low,high=np.quantile(counts[:,i],[.01,.99])
            out['trials'][k][name]=dict(point_count=stats(values[:,i]),presence_fraction=float(np.mean(values[:,i]>0)),
                outside_train_p01_p99_fraction=float(np.mean((values[:,i]<low)|(values[:,i]>high))),train_p01=float(low),train_p99=float(high))
    return out


def verify_files(run):
    counts=dict(json=0,npz=0,png=0,mp4=0,mp4_frames=0,csv=0,relative_links=0)
    for p in run.rglob('*'):
        if not p.is_file():continue
        if p.suffix=='.json':json.loads(p.read_text());counts['json']+=1
        elif p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as z:
                for k in z.files:
                    a=z[k]
                    if 'packet' not in p.name and a.dtype.kind=='f' and not np.isfinite(a).all():raise ValueError('nonfinite NPZ '+str(p))
            if 'packet' in p.name:
                from graph.vision_input import receive_vision_packet
                receive_vision_packet(read_packet(p))
            counts['npz']+=1
        elif p.suffix=='.png':
            if cv2.imread(str(p)) is None:raise ValueError('PNG decode')
            counts['png']+=1
        elif p.suffix=='.mp4':
            cap=cv2.VideoCapture(str(p));n=0
            if not np.isclose(cap.get(cv2.CAP_PROP_FPS),20):raise ValueError('video FPS')
            while True:
                ok,_=cap.read()
                if not ok:break
                n+=1
            cap.release()
            if not n:raise ValueError('empty MP4')
            counts['mp4']+=1;counts['mp4_frames']+=n
        elif p.suffix=='.csv':
            with p.open() as f:
                if not list(csv.DictReader(f)):raise ValueError('empty CSV')
            counts['csv']+=1
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('href','src'):
                    u=urlsplit(v)
                    if u.scheme or u.netloc:raise ValueError('external HTML resource')
                    if u.path:
                        if not (run/unquote(u.path)).is_file():raise ValueError('broken relative link '+v)
                        counts['relative_links']+=1
    Links().feed((run/'vision_oracle_reach_ablation.html').read_text())
    return counts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--original-episode',type=Path,required=True);parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();report=read(run/'results.json');c=report['protocol']
    if report['status']!='COMPLETE':raise ValueError('cannot analyze incomplete run as PASS')
    if not report['weights_and_normalizer_unchanged'] or read(run/'protection.json')['status']!='PASS':raise ValueError('invariant failure')
    data={k:verify_trial(run/'trials'/k,c) for k in POLICIES}
    for k in ('B','C'):assert_initial(data['A'][0]['initial'],data[k][0]['initial'])
    original=verify_original(run,args.original_episode)
    probe_initial=load_arrays(run/'probes/probe_0000.npz')['actions']
    for i,k in enumerate(POLICIES):
        np.testing.assert_allclose(probe_initial[i],load_arrays(run/'trials'/k/'actions.npz')['actions'][0],atol=c['atol'],rtol=c['rtol'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import trimesh
    plt.rcParams.update({'figure.dpi':130,'font.size':10})
    metrics={};csvrows=[];surface_arrays={}
    fig=plt.figure(figsize=(18,8));ax=fig.add_subplot(121,projection='3d');reachax=fig.add_subplot(122,projection='3d')
    timeline,axes=plt.subplots(5,2,figsize=(16,18))
    colors=dict(A='tab:red',B='tab:orange',C='tab:green')
    for condition,(trial,trace,obs,masks,hidden) in data.items():
        folder=run/'trials'/condition;frames=[]
        meshes={name:trimesh.Trimesh(**load_arrays(folder/(name+'_diagnostic_mesh.npz')),process=False) for name in ('peg','box')}
        for t,r in enumerate(trace):
            p=read_packet(folder/f'packet_{t:04d}.npz');g=load_arrays(folder/f'graph_{t:04d}.npz');row={k:v[t] for k,v in obs.items()}
            v=dict(timestep=t,graph=graph_metrics(p,g),objects={})
            for i,name in enumerate(('peg','box')):
                select=p['object_id']==i;src=p['provenance']['source_index'][select]
                correct=masks[t,2,i].ravel()[src];d=surface_distances(p['camera_points'][select],row,r['before'],name,meshes) if np.any(select) else np.array([])
                v['objects'][name]=dict(predicted=mask_metrics(masks[t,0,i],masks[t,2,i]),
                    applied=mask_metrics(masks[t,3,i],masks[t,2,i]),point_count=len(src),
                    non_target_points=int(np.sum(~correct)),correct_source_fraction=float(correct.mean()) if len(src) else None,
                    surface_abs_distance_mm=stats(d),correct_source_surface_mm=stats(d[correct]),incorrect_source_surface_mm=stats(d[~correct]))
                surface_arrays[f'{condition}_{name}_{t:04d}']=d
            frames.append(v)
            csvrows.append(dict(condition=condition,input_policy=trial['input_policy'],timestep=t,
                target_position_error_mm=r['metric']['reference_position_error_mm'],target_rotation_error_deg=r['metric']['reference_rotation_error_deg'],
                tcp_x_m=r['before']['ee_pose'][0],tcp_y_m=r['before']['ee_pose'][1],tcp_z_m=r['before']['ee_pose'][2],
                left_fingertip_surface_mm=r['metric']['fingertip_centre_to_peg_surface_mm'][0],right_fingertip_surface_mm=r['metric']['fingertip_centre_to_peg_surface_mm'][1],
                opening_mm=r['metric']['gripper_opening_mm'],tcp_speed_m_s=float(np.linalg.norm(r['before']['tcp_linear_velocity'])),peg_speed_m_s=float(np.linalg.norm(r['before']['peg_linear_velocity'])),
                dx=r['action'][0],dy=r['action'][1],dz=r['action'][2],rx=r['action'][3],ry=r['action'][4],rz=r['action'][5],
                gripper=r['action'][6],grasp_after=r['after']['grasp'],bilateral_after=r['after']['bilateral_force'],
                peg_iou=v['objects']['peg']['predicted']['iou'],box_iou=v['objects']['box']['predicted']['iou'],
                peg_applied_iou=v['objects']['peg']['applied']['iou'],box_applied_iou=v['objects']['box']['applied']['iou'],
                peg_points=v['objects']['peg']['point_count'],box_points=v['objects']['box']['point_count'],
                peg_non_target=v['objects']['peg']['non_target_points'],box_non_target=v['objects']['box']['non_target_points'],
                raw_ede=v['graph']['raw_ede_edges'],components=v['graph']['connected_components']))
        s=trial['first_close'];metrics[condition]=dict(trial=trial,frames=frames,
            at_first_close=frames[s] if s is not None else None,
            first_close_next8_actions=np.asarray([r['action'] for r in trace[s:s+8]]) if s is not None else None,
            first_close_next8_grasp=[r['after']['grasp'] for r in trace[s:s+8]] if s is not None else None)
        xyz=np.array([r['before']['ee_pose'][:3] for r in trace]);ax.plot(*xyz.T,label=condition+' '+trial['input_policy'],color=colors[condition])
        end=s+1 if s is not None else len(xyz)
        reachax.plot(*xyz[:end].T,color=colors[condition],label=f'{condition}, first close t{s}')
        if s is not None:
            ax.scatter(*xyz[s],s=65,color=colors[condition],marker='x')
            reachax.scatter(*xyz[s],s=65,color=colors[condition],marker='x')
        distance=np.array([r['metric']['reference_position_error_mm'] for r in trace]);rotation=np.array([r['metric']['reference_rotation_error_deg'] for r in trace])
        values=[distance,rotation,np.array([r['action'][6] for r in trace]),
            np.array([np.linalg.norm(r['action'][:3])*.1*1000 for r in trace]),
            np.array([f['objects']['peg']['predicted']['iou'] or 0 for f in frames]),
            np.array([f['objects']['peg']['surface_abs_distance_mm'].get('p95',0) for f in frames]),
            np.array([f['objects']['peg']['point_count'] for f in frames]),
            np.array([f['graph']['raw_ede_edges'] for f in frames]),np.linalg.norm(hidden.reshape(len(trace),-1),axis=1),
            np.array([r['after']['grasp'] for r in trace],int)]
        titles=['Target position error (mm)','Target rotation error (deg)','Raw gripper action','Translation command norm (mm)','Predicted peg IoU','Peg surface distance p95 (mm)','Peg point count','Raw EDE edges','GRU hidden L2 norm','Installed Panda.is_grasping (after action)']
        for a,y,title in zip(axes.flat,values,titles):
            a.plot(y,label=condition,color=colors[condition]);a.set_title(title);a.set_xlabel('control timestep');a.grid(alpha=.2)
            if s is not None:a.axvline(s,color=colors[condition],ls=':',alpha=.5)
    ref=np.asarray(data['A'][0]['grasp_reference']);ax.scatter(*ref[:3],marker='*',s=180,color='blue',label='diagnostic expert grasp target')
    reachax.scatter(*ref[:3],marker='*',s=180,color='blue',label='diagnostic expert grasp target')
    for a,title in ((ax,'Full autonomous rollout'),(reachax,'Approach through each first close')):
        a.set(xlabel='World X (m)',ylabel='World Y (m)',zlabel='World Z (m)',title=title);a.legend(fontsize=9)
    fig.tight_layout();fig.savefig(run/'reach_trajectory_3d.png');plt.close(fig)
    for a in axes.flat:a.legend()
    timeline.tight_layout();timeline.savefig(run/'timelines.png');plt.close(timeline)
    atomic(run/'analysis.json',metrics);save_npz(run/'surface_distances.npz',**surface_arrays)
    with (run/'frames.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    distribution=training_distribution(args.bundle,data);atomic(run/'training_distribution.json',distribution)
    differences={}
    for t in c['probe_timesteps']:
        f=run/'probes';pa=read_packet(f/f'A_packet_{t:04d}.npz');ga=load_arrays(f/f'A_graph_{t:04d}.npz')
        differences[str(t)]={k:graph_difference(pa,ga,read_packet(f/f'{k}_packet_{t:04d}.npz'),load_arrays(f/f'{k}_graph_{t:04d}.npz')) for k in ('B','C')}
    atomic(run/'same_state_graph_changes.json',differences)
    same_geometry={};same_distances={}
    a_mesh={name:trimesh.Trimesh(**load_arrays(run/'trials/A'/(name+'_diagnostic_mesh.npz')),process=False) for name in ('peg','box')}
    a_trial,a_trace,a_obs,a_masks,_=data['A']
    for t in c['probe_timesteps']:
        same_geometry[str(t)]={};row={k:v[t] for k,v in a_obs.items()}
        for condition in POLICIES:
            p=read_packet(run/'probes'/f'{condition}_packet_{t:04d}.npz');g=load_arrays(run/'probes'/f'{condition}_graph_{t:04d}.npz')
            objects={}
            for i,name in enumerate(('peg','box')):
                selected=p['object_id']==i;source=p['provenance']['source_index'][selected]
                correct=a_masks[t,2,i].ravel()[source]
                d=surface_distances(p['camera_points'][selected],row,a_trace[t]['before'],name,a_mesh) if np.any(selected) else np.array([])
                objects[name]=dict(point_count=len(source),non_target_points=int(np.sum(~correct)),surface_mm=stats(d),
                    correct_source_surface_mm=stats(d[correct]),incorrect_source_surface_mm=stats(d[~correct]))
                same_distances[f'{condition}_{name}_{t:04d}']=d
            same_geometry[str(t)][condition]=dict(graph=graph_metrics(p,g),objects=objects)
    atomic(run/'same_state_geometry.json',same_geometry);save_npz(run/'probe_surface_distances.npz',**same_distances)
    samefig,sameaxes=plt.subplots(2,3,figsize=(16,8),sharex=True,sharey=True)
    for i,name in enumerate(('peg','box')):
        for j,condition in enumerate(POLICIES):
            d=same_distances[f'{condition}_{name}_0027'];ax=sameaxes[i,j]
            ax.hist(d,bins=np.geomspace(.0001,500,80),color=colors[condition])
            ax.set_xscale('log');ax.set_yscale('log');ax.set_title(f'{condition} {name}: same A t27 state, N={len(d)}');ax.set_xlabel('GT surface distance (mm, log scale)')
    samefig.tight_layout();samefig.savefig(run/'same_state_t27_outliers.png');plt.close(samefig)
    # Show both objects, all native action components and the residual geometry.
    extra,extra_axes=plt.subplots(4,2,figsize=(16,14))
    action_fig,action_axes=plt.subplots(7,1,figsize=(14,16),sharex=True)
    hist,hist_axes=plt.subplots(2,3,figsize=(16,8),sharex=True,sharey=True)
    for condition,(trial,trace,obs,masks,hidden) in data.items():
        frames=metrics[condition]['frames'];s=trial['first_close'];color=colors[condition]
        for i,name in enumerate(('peg','box')):
            for key,style in [('predicted','-'),('applied','--')]:
                extra_axes[0,i].plot([v['objects'][name][key]['iou'] for v in frames],style,color=color,label=condition+' '+key)
            extra_axes[0,i].set_title(name+' mask IoU')
            extra_axes[1,i].plot([v['objects'][name]['non_target_points'] for v in frames],color=color,label=condition)
            extra_axes[1,i].set_title(name+' non-target source point count')
            extra_axes[2,i].plot([v['objects'][name]['point_count'] for v in frames],color=color,label=condition)
            extra_axes[2,i].set_title(name+' point count; zero means absent')
            if s is not None:
                d=surface_arrays[f'{condition}_{name}_{s:04d}']
                hist_axes[i,list(POLICIES).index(condition)].hist(d,bins=np.geomspace(.0001,500,80),color=color)
                hist_axes[i,list(POLICIES).index(condition)].set_title(f'{condition} {name}, own close t{s}')
                hist_axes[i,list(POLICIES).index(condition)].set_xlabel('GT surface distance (mm)')
                hist_axes[i,list(POLICIES).index(condition)].set_xscale('log')
                hist_axes[i,list(POLICIES).index(condition)].set_yscale('log')
        extra_axes[3,0].plot([np.linalg.norm(np.asarray(r['after']['peg_relative_tcp_pose'][:3])-trace[0]['before']['peg_relative_tcp_pose'][:3])*1000 for r in trace],color=color,label=condition)
        extra_axes[3,1].plot([r['after']['peg_pose'][2]*1000 for r in trace],color=color,label=condition)
        actions=np.asarray([r['action'] for r in trace])
        for j,a in enumerate(action_axes):
            a.plot(actions[:,j],color=color,label=condition);a.set_ylabel(['dx','dy','dz','rx','ry','rz','gripper'][j]);a.grid(alpha=.2)
            if s is not None:a.axvline(s,color=color,ls=':',alpha=.5)
    extra_axes[3,0].set_title('Peg–TCP relative translation change from reset (mm)')
    extra_axes[3,1].set_title('Peg world Z (mm)')
    for a in extra_axes.flat:a.legend();a.grid(alpha=.2);a.set_xlabel('control timestep')
    for a in action_axes:a.legend()
    extra.tight_layout();extra.savefig(run/'mask_point_geometry.png');plt.close(extra)
    hist.tight_layout();hist.savefig(run/'close_outlier_distributions.png');plt.close(hist)
    action_fig.tight_layout();action_fig.savefig(run/'native_action7.png');plt.close(action_fig)
    montage(run,data,'close','first_close_vision_geometry.png');montage(run,data,'t27','t27_vision_geometry.png')
    compare_video(run,data);compare_video(run,data,True)
    sections=[]
    for condition in POLICIES:
        trial=data[condition][0];summary={k:trial[k] for k in ('input_policy','first_close','close_geometry','minimum_target_error_before_close_mm','grasp_observed','first_grasp_after_action','first_bilateral_force_after_action','peg_lift_observed','max_peg_lift_mm','success','first_success_step','steps')}
        sections.append(f'<h2>{condition} — {trial["input_policy"]}</h2><pre>{escape(json.dumps(summary,indent=2))}</pre><video controls src="trials/{condition}/rollout_after_action.mp4"></video>')
    images=''.join(f'<h2>{label}</h2><img src="{file}">' for label,file in [('Reach trajectory','reach_trajectory_3d.png'),('Errors, topology, hidden and grasp','timelines.png'),('Native Action7, dotted first-close events','native_action7.png'),('Peg/Box masks, sources, presence and relative motion','mask_point_geometry.png'),('Same A t27 state: outlier cleanup','same_state_t27_outliers.png'),('Outlier distributions, each own first close','close_outlier_distributions.png'),('Each condition at its own first close','first_close_vision_geometry.png'),('Absolute t27: trajectories have diverged','t27_vision_geometry.png')])
    links=''.join(f'<li><a href="{p}">{p}</a></li>' for p in ('results.json','analysis.json','frames.csv','reset_preflight.json','fixture_verification.json','one_step_probes.json','same_state_graph_changes.json','training_distribution.json','protection.json','source_hashes.json','verification.json','report_snapshot.md'))
    html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Task07-F Vision Geometry Oracle</title><style>body{{font:16px system-ui;margin:32px auto;max-width:1500px;padding:0 20px;color:#222}}img,video{{max-width:100%}}pre{{white-space:pre-wrap;background:#f3f4f5;padding:16px}}.boundary{{background:#fff2cf;padding:20px}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #aaa}}</style><h1>Task07-F — Vision Geometry Oracle Reach Ablation</h1><p>ExperimentLogger UUID: {report['run_id']}; Seed41005; fixed epoch39 / global step8959.</p><div class="boundary">{SCOPE}. A=PREDICTED; B=predicted AND current visible GT; C=current visible GT. Every condition uses original RGB tracker, depth, Shape-First/PCA5/F5/Raw EDE, Robot76 and its own GRU144 history. All actions are autonomous native Action7. B/C are excluded from standard policy success rate. Single seed; no population success rate or architecture conclusion.</div><!--INTERPRETATION--><h2>Initial state, pixel alignment and physical clock</h2><pre>{escape(json.dumps(read(run/'reset_preflight.json'),indent=2))}</pre><h2>Original41005 reproduction</h2><pre>{escape(json.dumps(original,indent=2))}</pre>{''.join(sections)}{images}<h2>After-action video, absolute timestep</h2><video controls src="rollout_comparison_after_action.mp4"></video><h2>After-action video, aligned to each first close</h2><p>Alignment is for inspection only; these are different physical states and possibly different absolute timesteps.</p><video controls src="close_aligned_after_action.mp4"></video><h2>Same-state read-only action probes</h2><pre>{escape(json.dumps(read(run/'one_step_probes.json'),indent=2))}</pre><h2>Sampling, F5 and edges at the same A state</h2><pre>{escape(json.dumps(differences,indent=2))}</pre><h2>Training distribution and remaining uncertainty</h2><p>Oracle inputs may change point counts, availability and topology relative to training. Mask correctness does not establish graph physical correctness. Depth/feature/topology limits, BC action error and history dependence remain separate hypotheses. No GT object/goal pose, contact or expert stage enters the model; meshes and target are diagnostic only.</p><pre>{escape(json.dumps(distribution,indent=2))}</pre><h2>Conclusions and evidence</h2><p><a href="report_snapshot.md">Korean research report, Q1–Q8 verdicts</a> · <a href="../../docs/visual-graph-integration/07f_vision_oracle_reach_ablation.md">Repository Markdown</a></p><ul>{links}</ul></html>'''
    (run/'vision_oracle_reach_ablation.html').write_text(html)
    # Created by the final report step; keep links valid during iterative verification.
    if not (run/'report_snapshot.md').exists():(run/'report_snapshot.md').write_text('Report synthesis pending. Numerical run is complete.\n')
    atomic(run/'verification.json',dict(status='PASS',trial_contracts=3,physics5_control1=True,
        initial_states_exact=True,oracle_sources_correct=True,graphs_rebuilt_and_readback=True,
        original_41005=original,weights_normalizer_unchanged=True,initial_probe_all_live_actions_match=True))
    # Markdown link is created separately before full file/link readback.
    print('ANALYSIS_COMPLETE '+str(run),flush=True)


if __name__=='__main__':main()
