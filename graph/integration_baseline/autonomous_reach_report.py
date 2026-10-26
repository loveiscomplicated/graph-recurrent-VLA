"""Read-only Task07-E analysis, figures, local HTML and independent readback."""
import argparse
import csv
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urlsplit,unquote

import cv2
import numpy as np
from transforms3d.quaternions import quat2mat

from graph.integration_baseline.autonomous_reach_diagnosis import (
    rotation_distance,relative_pose,box_surface_distance,compare_prefix,treatment_action)
from graph.integration_baseline.grasp_intervention import atomic,retention
from graph.integration_baseline.io import read_packet
from graph.integration_baseline.direct_runtime import read,sha

ORDER=('positive_control','A','B','C')
COLORS=dict(positive_control='gray',A='tab:green',B='tab:orange',C='tab:red')


def load_arrays(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def metric_at(state,trial):
    pose=np.asarray(state['ee_pose']);ref=np.asarray(trial['grasp_reference'])
    relative=np.asarray(state['peg_relative_tcp_pose']);relative_ref=np.asarray(trial['reference_relative'])
    peg_in_tcp=relative_pose(pose,np.asarray(state['peg_pose']))
    pads=np.asarray([state['left_fingertip_pos'],state['right_fingertip_pos']])
    pads_tcp=(pads-pose[:3]) @ quat2mat(pose[3:])
    return dict(peg_centre_in_tcp_mm=(peg_in_tcp[:3]*1000).tolist(),
        pad_centres_in_tcp_mm=(pads_tcp*1000).tolist(),
        reference_position_error_mm=float(np.linalg.norm(pose[:3]-ref[:3])*1000),
        reference_position_delta_mm=((pose[:3]-ref[:3])*1000).tolist(),
        reference_rotation_error_rad=rotation_distance(pose[3:],ref[3:]),
        reference_rotation_error_deg=float(np.degrees(rotation_distance(pose[3:],ref[3:]))),
        peg_relative_position_error_mm=float(np.linalg.norm(relative[:3]-relative_ref[:3])*1000),
        peg_relative_position_delta_mm=((relative[:3]-relative_ref[:3])*1000).tolist(),
        peg_relative_rotation_error_deg=float(np.degrees(rotation_distance(relative[3:],relative_ref[3:]))),
        fingertip_centre_to_peg_surface_mm=(np.asarray(state['fingertip_centre_to_peg_surface_m'])*1000).tolist(),
        pad_centre_segment_intersects_peg=state['pad_centre_segment_intersects_peg'],
        tcp_peg_centre_distance_mm=float(np.linalg.norm(pose[:3]-np.asarray(state['peg_pose'])[:3])*1000),
        gripper_opening_mm=state['gripper_opening_m']*1000)


def vision_metrics(pred,seg,ids,packet,state):
    out={};source=packet['provenance']['source_index'];obj=packet['object_id']
    for i,name in enumerate(('peg','box')):
        gt=seg==ids[i];p=pred[i];intersection=int((p&gt).sum());union=int((p|gt).sum())
        selected=source[obj==i]
        out[name]=dict(gt_visible_pixels=int(gt.sum()),predicted_pixels=int(p.sum()),
            iou=intersection/union if union else None,
            precision=intersection/int(p.sum()) if p.any() else None,
            recall=intersection/int(gt.sum()) if gt.any() else None,
            predicted_on_other_gt_pixels=int((p&(seg==ids[1-i])).sum()),
            selected_point_count=len(selected),
            selected_on_correct_gt_fraction=float(np.mean(gt.ravel()[selected])) if len(selected) else None)
    peg=packet['points'][obj==0]
    if len(peg):
        tcp=np.asarray(state['ee_pose']);world=peg @ quat2mat(tcp[3:]).T+tcp[:3]
        distances=np.abs(box_surface_distance(world,np.asarray(state['peg_pose']),state['peg_half_sizes']))*1000
        correct=(seg==ids[0]).ravel()[source[obj==0]]
        def group(values):
            return dict(count=len(values),mean=float(values.mean()),p95=float(np.quantile(values,.95)),maximum=float(values.max())) if len(values) else None
        out['peg_points_gt_surface_abs_distance_mm']=dict(**group(distances),correct_gt_sources=group(distances[correct]),incorrect_gt_sources=group(distances[~correct]))
    else:out['peg_points_gt_surface_abs_distance_mm']=None
    out['predicted_overlap_pixels']=int((pred[0]&pred[1]).sum())
    out['raw_ede_count']=int(packet['intra_object_edge_index'].shape[1])
    return out


def readback_contracts(run,original=None):
    report=read(run/'results.json');c=report['protocol']
    if report['status']!='COMPLETE' or len(report['trials'])!=4:raise ValueError('incomplete experiment')
    if not report['weights_and_normalizer_unchanged'] or read(run/'protection.json')['status']!='PASS':raise ValueError('protected dependency changed')
    data={}
    for name in ORDER:
        f=run/'trials'/name;trial=read(f/'trial.json');trace=read(f/'trace.json');n=len(trace);s=trial['close_start']
        actions=load_arrays(f/'actions.npz');obs=load_arrays(f/'observations.npz');h=load_arrays(f/'hidden.npz');mask=load_arrays(f/'masks.npz')['masks']
        if n!=s+48 or trial['physics_steps']!=5*n or trial['hidden_steps']!=n or trial['hidden_at_close_reset'] or len(trial['hidden_reset_events'])!=1:
            raise ValueError('clock/hidden lifecycle')
        if h['hidden'].shape!=(n,1,1,144) or obs['robot_state'].shape!=(n+1,76):raise ValueError('Robot76/hidden shape')
        np.testing.assert_array_equal(h['timestep'],np.arange(n));np.testing.assert_array_equal(actions['controller_input'],actions['applied'])
        if name in ('B','C'):
            negative=np.flatnonzero(actions['predicted'][:28,6]<0)
            np.testing.assert_array_equal(negative,[27]);np.testing.assert_array_equal(actions['applied'][:27],actions['predicted'][:27])
        np.testing.assert_array_equal(actions['applied'][s:s+8],np.stack([treatment_action(a,name) for a in actions['predicted'][s:s+8]]))
        np.testing.assert_array_equal(actions['applied'][s+8:,6],-np.ones(40,np.float32))
        np.testing.assert_array_equal(actions['stage'][s:],['close']*8+['lift']*20+['hold']*20)
        # Recompute common servo from each current TCP rather than accepting stage names.
        target=np.asarray(trial['anchor']['ee_pose']);target[:3]+=[0,0,.08]
        from recurrent.expert import rotvec_between
        for t,r in enumerate(trace):
            if (r['physics_steps_before'],r['physics_steps_after'])!=(5*t,5*(t+1)):raise ValueError('physics5/control1')
            np.testing.assert_array_equal(r['predicted_action'],actions['predicted'][t])
            np.testing.assert_array_equal(r['applied_action'],actions['applied'][t])
            if t>=s+8:
                tcp=np.asarray(r['before']['ee_pose']);dp=(target[:3]-tcp[:3])*3
                dp=dp/max(1.,np.linalg.norm(dp)/.02)
                rv=rotvec_between(tcp[3:],target[3:]);rv=rv/max(1.,np.linalg.norm(rv)/.15)
                np.testing.assert_allclose(actions['applied'][t],np.r_[dp/.1,-rv/.1,-1],atol=2e-6,rtol=1e-5)
        outcome=retention(trace,trial['anchor'],c)
        for k in ('retained','lift_execution_valid','close_end_grasp','grasp_all_hold'):
            if outcome[k]!=trial['outcome'][k]:raise ValueError('outcome readback')
        data[name]=(trial,trace,actions,obs,h['hidden'],mask)
    pairs={}
    for first,second in [('positive_control','A'),('B','C')]:
        a,b=data[first],data[second];s=a[0]['close_start']
        pairs[first+'_'+second]=compare_prefix(a[0],b[0],c)
        for k in a[3]:np.testing.assert_array_equal(a[3][k][:s+1],b[3][k][:s+1])
        np.testing.assert_array_equal(a[5][:s+1],b[5][:s+1])
        for t in range(s+1):
            for prefix in ('graph','packet'):
                x=load_arrays(run/'trials'/first/f'{prefix}_{t:04d}.npz');y=load_arrays(run/'trials'/second/f'{prefix}_{t:04d}.npz')
                # Packet metadata embeds episode identity, numerical payload must agree.
                for k in x:
                    if x[k].dtype.kind in 'biufc':np.testing.assert_array_equal(x[k],y[k])
    old={"status":"NOT_TESTED","reason":"original episode not supplied"}
    if original is not None:
        live=data['C'];oldobs=load_arrays(original/'observations.npz');oldact=load_arrays(original/'actions.npz');oldh=load_arrays(original/'hidden.npz');oldmask=load_arrays(original/'masks.npz')
        for k in live[3]:np.testing.assert_array_equal(live[3][k][:28],oldobs[k][:28])
        np.testing.assert_array_equal(live[5][:28],oldmask['masks'][:28])
        np.testing.assert_allclose(live[2]['predicted'][:28],oldact['actions'][:28],atol=c['atol'],rtol=c['rtol'])
        np.testing.assert_allclose(live[4][:28],oldh['hidden'][:28],atol=c['atol'],rtol=c['rtol'])
        for t in range(28):
            x=load_arrays(run/'trials'/'C'/f'graph_{t:04d}.npz');y=load_arrays(original/f'graph_{t:04d}.npz')
            for k in x:np.testing.assert_array_equal(x[k],y[k])
        old=dict(status='PASS',frames=28,observations_masks_graph_source_exact=True,
            maximum_action_difference=float(np.max(np.abs(live[2]['predicted'][:28]-oldact['actions'][:28]))),
            maximum_hidden_difference=float(np.max(np.abs(live[4][:28]-oldh['hidden'][:28]))))
    return data,dict(trial_contracts=4,paired_prefixes=pairs,original_41005_prefix=old,physics5_control1=True,common_servo_recomputed=True)


def verify_files(run):
    counts=dict(json=0,npz=0,png=0,mp4=0,mp4_frames=0,csv=0,relative_links=0)
    for p in run.rglob('*'):
        if not p.is_file():continue
        if p.suffix=='.json':json.loads(p.read_text());counts['json']+=1
        elif p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as f:
                for k in f.files:
                    a=f[k]
                    if not p.name.startswith('packet_') and a.dtype.kind=='f' and not np.isfinite(a).all():raise ValueError('nonfinite model/observation NPZ')
            if p.name.startswith('packet_'):
                from graph.vision_input import receive_vision_packet
                receive_vision_packet(read_packet(p))
            counts['npz']+=1
        elif p.suffix=='.png':
            if cv2.imread(str(p)) is None:raise ValueError('PNG decode')
            counts['png']+=1
        elif p.suffix=='.mp4':
            cap=cv2.VideoCapture(str(p));n=0
            if not np.isclose(cap.get(cv2.CAP_PROP_FPS),20):raise ValueError('20FPS video required')
            while True:
                ok,_=cap.read()
                if not ok:break
                n+=1
            cap.release()
            if not n:raise ValueError('empty video')
            counts['mp4']+=1;counts['mp4_frames']+=n
        elif p.suffix=='.csv':
            with p.open() as f:
                rows=list(csv.DictReader(f))
                if not rows:raise ValueError('empty CSV')
            counts['csv']+=1
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ('src','href'):
                    u=urlsplit(v)
                    if not u.scheme and u.path:
                        if not (run/unquote(u.path)).is_file():raise ValueError('broken link: '+v)
                        counts['relative_links']+=1
    Links().feed((run/'autonomous_reach_diagnosis.html').read_text())
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--original-episode',type=Path)
    args=p.parse_args();run=args.run.resolve();report=read(run/'results.json');c=report['protocol']
    data,checks=readback_contracts(run,args.original_episode)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':130,'font.size':10})
    metrics={};all_csv=[];sections=[]
    fig3=plt.figure(figsize=(10,7));ax3=fig3.add_subplot(111,projection='3d')
    for name in ORDER:
        trial,trace,actions,obs,hidden,masks=data[name];s=trial['close_start'];folder=run/'trials'/name
        gt=load_arrays(folder/'gt_segmentation_diagnostic_only.npz');ids=gt['actor_ids'].reshape(-1)
        distance=np.array([metric_at(r['before'],trial)['reference_position_error_mm'] for r in trace[:s+1]])
        meaningful=np.flatnonzero(distance<=distance[0]-1000*c['meaningful_approach_reduction_m'])
        closest=int(np.argmin([metric_at(r['before'],trial)['tcp_peg_centre_distance_mm'] for r in trace[:s+1]]))
        key_times=dict(episode_start=0,first_meaningful_approach=int(meaningful[0]) if len(meaningful) else None,
            closest_tcp_to_peg_before_close=closest,closest_to_expert_target_before_close=int(np.argmin(distance)),first_negative_gripper=int(np.flatnonzero(actions['predicted'][:,6]<0)[0]) if np.any(actions['predicted'][:,6]<0) else None,
            close_start=s,close_end_observation=s+8,lift_end_observation=s+28,hold_end_observation=s+48)
        key_states={}
        for key,t in key_times.items():
            if t is not None:
                state=trace[t]['before'] if t<len(trace) else trace[-1]['after']
                key_states[key]=dict(timestep=t,**metric_at(state,trial))
        vision=[]
        for t,r in enumerate(trace):
            packet=read_packet(folder/f'packet_{t:04d}.npz')
            v=vision_metrics(masks[t],gt['segmentation'][t],ids,packet,r['before']);vision.append(v)
            m=metric_at(r['before'],trial)
            all_csv.append(dict(condition=name,timestep=t,stage=r['stage'],simulation_time_s=t/20,
                **{k:v for k,v in m.items() if isinstance(v,(float,int,bool))},
                left_fingertip_surface_mm=m['fingertip_centre_to_peg_surface_mm'][0],right_fingertip_surface_mm=m['fingertip_centre_to_peg_surface_mm'][1],
                gripper_predicted=actions['predicted'][t,6],gripper_applied=actions['applied'][t,6],
                grasp_after=r['after']['grasp'],bilateral_force_after=r['after']['bilateral_force'],
                peg_iou=v['peg']['iou'],box_iou=v['box']['iou'],peg_points=v['peg']['selected_point_count'],box_points=v['box']['selected_point_count']))
        approach_vision=vision[:s+1]
        summary={}
        for obj in ('peg','box'):
            def stats(field):
                values=[v[obj][field] for v in approach_vision if v[obj][field] is not None]
                return dict(minimum=min(values),mean=float(np.mean(values)),maximum=max(values)) if values else None
            missing=[t for t,v in enumerate(approach_vision) if v[obj]['selected_point_count']==0]
            summary[obj]=dict(iou=stats('iou'),precision=stats('precision'),recall=stats('recall'),
                correct_selected_fraction=stats('selected_on_correct_gt_fraction'),point_missing_timesteps=missing,
                close_start=vision[s][obj])
        metrics[name]=dict(close_start=s,close_geometry=metric_at(trial['anchor'],trial),key_states=key_states,
            vision_approach=summary,vision_frames=vision,outcome=trial['outcome'],
            first_grasp_after_action_t=trial['first_grasp_after_action_t'],
            first_bilateral_force_after_action_t=trial['first_bilateral_force_after_action_t'],
            first_lift_grasp_after_action_t=trial['first_lift_grasp_after_action_t'])
        atomic(folder/'diagnosis.json',metrics[name])
        xyz=obs['tcp_pose'][:s+1,:3]*1000
        ax3.plot(*xyz.T,label=f'{name} approach (close t{s})',color=COLORS[name]);ax3.scatter(*xyz[-1],color=COLORS[name])
        # Temporal geometry, contact and common lift/hold outcome.
        fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True);tt=np.arange(len(trace))
        axes[0].plot(tt,[metric_at(r['before'],trial)['reference_position_error_mm'] for r in trace],label='TCP to initial Expert target (mm)')
        axes[0].plot(tt,[metric_at(r['before'],trial)['peg_relative_position_error_mm'] for r in trace],label='Peg-relative target error (mm)');axes[0].legend()
        tips=np.array([r['before']['fingertip_centre_to_peg_surface_m'] for r in trace])*1000
        axes[1].plot(tt,tips[:,0],label='left pad centre to Peg surface');axes[1].plot(tt,tips[:,1],label='right');axes[1].set_ylabel('mm (signed)');axes[1].legend()
        anchor=trial['anchor'];peg=np.array([r['after']['peg_pose'][2]-anchor['peg_pose'][2] for r in trace])*1000
        tcp=np.array([r['after']['ee_pose'][2]-anchor['ee_pose'][2] for r in trace])*1000
        axes[2].plot(tt,peg,label='Peg +z');axes[2].plot(tt,tcp,label='TCP +z')
        axes[2].scatter(tt,[5 if r['after']['grasp'] else 0 for r in trace],s=5,label='grasp bool x5')
        axes[2].axhline(40,color='gray',ls=':');axes[2].axhline(60,color='gray',ls=':');axes[2].set_ylabel('mm after action');axes[2].legend()
        for ax in axes:
            for x in (s,s+8,s+28):ax.axvline(x,color='gray',ls='--',lw=.7)
            ax.grid(alpha=.25)
        axes[-1].set_xlabel('Action timestep; dashed: close / lift / hold starts')
        fig.suptitle(f'{name}: geometry and diagnostic retention');fig.tight_layout();fig.savefig(folder/'geometry_retention.png');plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
        for j in range(6):axes[0].plot(np.arange(8),actions['predicted'][s:s+8,j],label=f'pred arm{j}')
        for j in range(6):axes[0].plot(np.arange(8),actions['applied'][s:s+8,j],ls=':',alpha=.6)
        axes[0].legend(ncol=3);axes[0].set_ylabel('native arm6D')
        axes[1].plot(np.arange(8),actions['predicted'][s:s+8,6],label='pred gripper')
        axes[1].plot(np.arange(8),actions['applied'][s:s+8,6],ls='--',label='applied');axes[1].axhline(-1,ls=':',color='gray');axes[1].legend();axes[1].set_xlabel('Close action offset')
        fig.suptitle(name+' actual current-observation model outputs');fig.tight_layout();fig.savefig(folder/'close_actions.png');plt.close(fig)
        selected=sorted(set(t for t in (0,key_times['first_meaningful_approach'],closest,s,s+7) if t is not None))
        canvas=[]
        for t in selected:
            packet=read_packet(folder/f'packet_{t:04d}.npz');panels=[]
            for mode in ('rgb','predicted masks','GT diagnostic only','Shape-First / Raw EDE'):
                image=obs['rgb'][t].copy()
                if mode in ('predicted masks','GT diagnostic only'):
                    for i,color in enumerate(((255,60,60),(60,140,255))):
                        select=masks[t,i] if mode=='predicted masks' else gt['segmentation'][t]==ids[i]
                        image[select]=(image[select]*.5+np.array(color)*.5).astype(np.uint8)
                image=cv2.resize(image,(384,384),interpolation=cv2.INTER_NEAREST)
                if mode=='Shape-First / Raw EDE':
                    src=packet['provenance']['source_index'];uv=np.stack((src%128,src//128),axis=1)*3+1
                    for a,b in packet['intra_object_edge_index'].T:cv2.line(image,tuple(uv[a]),tuple(uv[b]),(100,100,100),1)
                    for xy,oid in zip(uv,packet['object_id']):cv2.circle(image,tuple(xy),2,(255,50,50) if oid==0 else (50,160,255),-1)
                cv2.putText(image,f'{name} t{t} {mode}',(6,20),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1);panels.append(image)
            canvas.append(np.concatenate(panels,axis=1))
        cv2.imwrite(str(folder/'vision_geometry.png'),cv2.cvtColor(np.concatenate(canvas,axis=0),cv2.COLOR_RGB2BGR))
        sections.append(f'<h2>{name}: close t={s}; retained={trial["outcome"]["retained"]}</h2><p>Close8 end grasp={trial["outcome"]["close_end_grasp"]}; first grasp after action t={trial["first_grasp_after_action_t"]}; first bilateral force t={trial["first_bilateral_force_after_action_t"]}</p><video controls src="trials/{name}/rollout.mp4"></video><p><a href="trials/{name}/trace.json">Trace</a> · <a href="trials/{name}/diagnosis.json">Geometry / Vision metrics</a> · <a href="trials/{name}/observations.npz">RGB-D / Robot76</a> · <a href="trials/{name}/actions.npz">Action / controller</a> · <a href="trials/{name}/hidden.npz">Hidden</a></p><img src="trials/{name}/geometry_retention.png"><img src="trials/{name}/close_actions.png"><img src="trials/{name}/vision_geometry.png">')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,('reference_position_error_mm','reference_rotation_error_deg','peg_relative_position_error_mm'),('World target error (mm)','Rotation error (degrees)','Peg-relative target error (mm)')):
        ax.bar(ORDER,[metrics[n]['close_geometry'][key] for n in ORDER],color=[COLORS[n] for n in ORDER]);ax.set_title(title);ax.grid(axis='y',alpha=.25)
    fig.suptitle('Each condition at its own close start: PC/A t40; B/C t27');fig.tight_layout();fig.savefig(run/'close_geometry_comparison.png');plt.close(fig)
    ref=np.asarray(data['A'][0]['grasp_reference'][:3])*1000;ax3.scatter(*ref,marker='*',s=160,color='blue',label='Initial Expert grasp target')
    peg=np.asarray(data['A'][0]['initial_diagnostic']['peg_pose'][:3])*1000;ax3.scatter(*peg,marker='s',s=70,color='black',label='Initial Peg centre')
    ax3.set(xlabel='World X mm',ylabel='World Y mm',zlabel='World Z mm');ax3.legend();fig3.tight_layout();fig3.savefig(run/'reach_trajectory_3d.png');plt.close(fig3)
    # Close-aligned comparison: every image is the actual state AFTER the labeled action.
    writer=cv2.VideoWriter(str(run/'close_lift_hold_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1024,1024))
    if not writer.isOpened():raise RuntimeError('comparison video writer')
    montage=[]
    for offset in range(48):
        panels=[]
        for name in ORDER:
            trial,trace,actions,obs,_,_=data[name];t=trial['close_start']+offset
            im=cv2.resize(obs['rgb'][t+1],(512,512),interpolation=cv2.INTER_NEAREST)
            cv2.putText(im,f'{name} after t{t} / {trace[t]["stage"]}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),1)
            cv2.putText(im,f'grasp={trace[t]["after"]["grasp"]}',(12,48),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),1);panels.append(im)
        image=np.concatenate((np.concatenate(panels[:2],axis=1),np.concatenate(panels[2:],axis=1)),axis=0)
        writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        if offset in (7,27,47):montage.append(cv2.resize(image,(768,768)))
    writer.release();cv2.imwrite(str(run/'close_lift_hold_montage.png'),cv2.cvtColor(np.concatenate(montage,axis=1),cv2.COLOR_RGB2BGR))
    with (run/'diagnostic_frames.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_csv[0]));w.writeheader();w.writerows(all_csv)
    atomic(run/'analysis.json',metrics)
    rows=''.join(f'<tr><td>{n}</td><td>{metrics[n]["close_start"]}</td><td>{metrics[n]["outcome"]["close_end_grasp"]}</td><td>{metrics[n]["outcome"]["retained"]}</td><td>{metrics[n]["close_geometry"]["reference_position_error_mm"]:.3f}</td><td>{metrics[n]["close_geometry"]["reference_rotation_error_deg"]:.3f}</td><td>{metrics[n]["close_geometry"]["fingertip_centre_to_peg_surface_mm"]}</td></tr>' for n in ORDER)
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Task07-E Autonomous Reach Diagnosis</title><style>body{{font:16px system-ui;max-width:1240px;margin:32px auto;padding:0 20px;color:#20252b}}img{{max-width:100%;height:auto}}video{{max-width:100%;width:620px}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:8px}}pre{{white-space:pre-wrap;background:#f4f6f7;padding:16px}}</style><h1>Task07-E: autonomous reach / descend causal diagnosis</h1><p>Run {report['run_id']} · Seed 41005 only · {escape(report['scope'])}. Expert approach and common Lift/Hold are diagnostic interventions. No policy success rate is computed.</p><p>FP32 CUDA Vision/GNN/GRU; CPU PhysX; 20Hz control / 100Hz physics; 5 substeps; fresh reset and hidden for each trial. B/C fixed first negative gripper at t27. GT never enters Student input.</p><table><tr><th>Condition</th><th>Close t</th><th>Grasp after close8</th><th>Retained all hold</th><th>Target position mm</th><th>Rotation deg</th><th>Pad-centre surface distances mm</th></tr>{rows}</table><p>Reference is the same-seed initial Expert grasp target. A and B/C close at different times. Rotation uses 2 acos(abs(normalized quaternion dot)). Surface distances use pad centres and the GT Peg cuboid; the pad-centre segment indicator is not a sufficient physical grasp criterion.</p><p>Retention requires all hold20 frames: Panda.is_grasping, Peg rise≥40mm, TCP rise≥60mm. Lift/hold gripper=-1 can complete an imperfect close: first contact and close-end results remain separate.</p><img src="reach_trajectory_3d.png"><img src="close_geometry_comparison.png"><h2>Actual after-action comparison</h2><p>Panels: positive control / A / B / C. Columns in still montage: close8 end / lift end / hold end. Each trial is aligned to its own close start.</p><video controls src="close_lift_hold_comparison.mp4"></video><img src="close_lift_hold_montage.png"><p><a href="analysis.json">All quantitative diagnoses</a> · <a href="diagnostic_frames.csv">Frame CSV</a> · <a href="results.json">Raw trial results</a> · <a href="verification.json">Independent readback</a> · <a href="source_hashes.json">Source hashes</a> · <a href="implementation.diff">Implementation diff</a> · <a href="protection.json">Immutability audit</a></p><h2>Execution contracts</h2><pre>{escape(json.dumps(checks,indent=2))}</pre>'''+''.join(sections)+'''<h2>Interpretation limits</h2><p>Presence=True means nonempty observed point support. Mask IoU and correct-object source membership are evaluated independently against current rendered GT segmentation. GT masks do not correct predicted masks or generate points. Vision metrics and the actual reach trajectory can show coincident errors, but no GT-corrected-vision intervention was performed; Vision versus Action Learning causality cannot be fully separated by this comparison. One seed cannot establish a population success rate or general causal effect.</p></html>'''
    if (run/'interpretation.json').exists():
        interpretation=read(run/'interpretation.json')
        conclusion='<h2>Questions and evidence</h2>'+''.join('<p><strong>'+escape(k+' — '+v['status'])+'</strong> '+escape(v['answer'])+'</p>' for k,v in interpretation.items())
        body=body.replace('<h2>Execution contracts</h2>',conclusion+'<h2>Execution contracts</h2>')
    (run/'autonomous_reach_diagnosis.html').write_text(body)
    atomic(run/'verification.json',dict(status='VERIFYING'))
    counts=verify_files(run)
    atomic(run/'verification.json',dict(status='PASS',**checks,decoded=counts,scope='actual execution trace + independent artifact readback; not a performance metric'))
    print(json.dumps(dict(run_id=report['run_id'],results={n:dict(geometry=metrics[n]['close_geometry'],outcome=metrics[n]['outcome'],vision=metrics[n]['vision_approach']) for n in ORDER},verification=checks),indent=2))


if __name__=='__main__':main()
