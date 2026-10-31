"""Frozen Task10-F live recovery evaluation with causal current RGB-D history."""
import argparse,os,time,traceback
from pathlib import Path

import cv2,numpy as np,sapien,torch
from graph.corrective_reach10f.recovery_audit import Audit,bounded_action
from graph.corrective_reach10f.recovery_collect import FEEDBACK,SAFETY
from graph.corrective_reach10f.safety_v2 import PhysicalSafety
from graph.geometry_reach10d.runtime import environment
from graph.integration_baseline.geometry09a import physical_state
from graph.integration_baseline.graph import pack_graphs
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
from graph.integration_baseline.bc08_codec import ActionCodec
from graph.plan_a_bc.rollout import sensors
from graph.plan_a_bc.geometry import CurrentGeometry
from graph.plan_a_bc.model import CachedGraphPolicy,normalizer
from graph.plan_a_bc.cache import make_vision
from graph.plan_a_bc.common import read,write,sha,digest,npz,append_csv,checkpoint_load,tensor_identity
from recurrent.expert import ScriptedExpert,rotvec_between
from utils.experiment_logger import ExperimentLogger


def measure(env,initial):
    s=physical_state(env)
    # Evaluation-only current GT target. Nothing here enters the student.
    goal=ScriptedExpert(env).grasp;approach=goal*sapien.Pose([0,0,-.05])
    p=float(np.linalg.norm(s['ee_pose'][:3]-goal.p));r=float(np.linalg.norm(rotvec_between(s['ee_pose'][3:],goal.q)))
    ap=float(np.linalg.norm(s['ee_pose'][:3]-approach.p));ar=float(np.linalg.norm(rotvec_between(s['ee_pose'][3:],approach.q)))
    return dict(target_position_error_m=p,target_orientation_error_rad=r,approach_position_error_m=ap,
                approach_orientation_error_rad=ar,opening_m=float(s['qpos'][-2:].sum()),
                peg_translation_m=float(np.linalg.norm(s['peg_pose'][:3]-initial['peg_pose'][:3])),
                tcp_pose=s['ee_pose'].tolist(),max_qvel=float(abs(s['qvel']).max()))


def frame(cameras,masks,condition,kind,seed,t,row):
    images=[]
    for c in cameras:
        rgb=c.rgb.copy()
        for oid,color in [(0,np.array([255,70,70])),(1,np.array([70,255,90]))]:
            selected=masks[c.camera][oid];rgb[selected]=(rgb[selected]*.6+color*.4).astype(np.uint8)
        images.append(cv2.resize(rgb,(320,240),interpolation=cv2.INTER_AREA))
    out=np.zeros((300,640,3),np.uint8);out[:240]=np.concatenate(images,axis=1)
    cv2.putText(out,f'{condition} {kind} seed={seed} t={t} {row["segment"]}',(8,260),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,f'post-step pos={row["target_position_error_m"]*1000:.1f}mm rot={row["target_orientation_error_rad"]:.3f} safe={int(row["physical_safety"])}',(8,284),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1,cv2.LINE_AA)
    return cv2.cvtColor(out,cv2.COLOR_RGB2BGR)


def episode(root,folder,model,norm,codec,detector,tracker,condition,kind,spec,horizon=130):
    folder.mkdir(parents=True,exist_ok=False)
    env=environment(root/'pilot03');seed=spec['seed'];obs,_=env.reset(seed=seed)
    expert=ScriptedExpert(env);initial=physical_state(env)
    reset_sha=digest({k:v.tolist() for k,v in initial.items() if isinstance(v,np.ndarray)})
    audit=Audit(env);safety=PhysicalSafety(env,SAFETY);geometry=CurrentGeometry(root/'full_cache/inputs',norm.schema)
    identity=f'10F-final-{condition}-{kind}-{seed}-{spec.get("offset_mm",[0,0,0])}'
    adapters={n:OnlineVisionAdapter(detector,tracker) for n in ('base_camera','hand_camera')}
    for adapter in adapters.values():adapter.reset(identity)
    writer=cv2.VideoWriter(str(folder/'rollout.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(640,300))
    if not writer.isOpened():raise RuntimeError('MP4 writer unavailable')
    hidden=None;rows=[];actions=[];robots=[];gates=[];first_approach=None;completion=None
    anchor=spec.get('anchor',25);disturbed=False;disturbance_ok=True;disturbance_steps=0;actual_offset=None
    before_disturbance=None;after_disturbance=None;recovery_start_t=None;safe=True;stable=0;started=time.perf_counter()
    try:
        while int(env.unwrapped.elapsed_steps.item())<horizon:
            t=int(env.unwrapped.elapsed_steps.item())
            cameras,data,surfaces=sensors(env,obs,identity,t,norm.schema)
            masks={c.camera:adapters[c.camera].process(c.rgb,identity,t)[0] for c in cameras}
            graph,_,gate=geometry.build(cameras,data,surfaces,masks)
            pg=pack_graphs([graph],[(0,0)]).to(next(model.parameters()).device)
            robot=torch.from_numpy(data['robot76'][None]).to(next(model.parameters()).device)
            with torch.inference_mode():
                raw,hidden=model.step(pg,robot,norm,hidden)
                command=codec.decode(codec.tag(raw))
            if not torch.isfinite(command).all():raise FloatingPointError('nonfinite policy command')
            proposed=command[0].detach().cpu().numpy().astype(np.float32)
            if kind=='handoff' and t<anchor:
                executed=expert.act();segment='expert_prefix'
            elif kind!='nominal' and t==anchor and not disturbed:
                # The model has consumed the anchor observation before external
                # disturbance. Every subsequent disturbance observation is also
                # processed before its control action.
                fixed=physical_state(env)['ee_pose'].copy();fixed[:3]+=np.asarray(spec['offset_mm'])*.001
                before_disturbance=measure(env,initial)
                integral=np.zeros(3);stable_disturbance=0;disturbed=True
                executed,integral=bounded_action(env,fixed,integral,FEEDBACK)
                segment='disturbance'
            elif kind!='nominal' and disturbed and recovery_start_t is None:
                current=physical_state(env);error=float(np.linalg.norm(current['ee_pose'][:3]-fixed[:3]))
                angle=float(np.linalg.norm(rotvec_between(current['ee_pose'][3:],fixed[3:])))
                velocity=float(abs(current['qvel']).max())
                stable_disturbance=stable_disturbance+1 if error<.001 and angle<.01 and velocity<.03 else 0
                if stable_disturbance>=FEEDBACK['stable_steps']:
                    after_disturbance=measure(env,initial);actual_offset=((current['ee_pose'][:3]-initial_anchor[:3])*1000).tolist()
                    recovery_start_t=t;executed=proposed;segment='student_recovery'
                elif disturbance_steps>=FEEDBACK['max_steps']:
                    disturbance_ok=False;break
                else:
                    executed,integral=bounded_action(env,fixed,integral,FEEDBACK);segment='disturbance'
            else:
                executed=proposed;segment='student_nominal' if kind=='nominal' else 'student_recovery'
            if segment=='disturbance' and disturbance_steps==0:initial_anchor=physical_state(env)['ee_pose'].copy()
            obs,step=audit.step(np.asarray(executed,np.float32));physical=safety.evaluate(initial,step['substep_contacts'])
            if segment=='disturbance':disturbance_steps+=1
            m=measure(env,initial)
            if m['approach_position_error_m']<.010 and m['approach_orientation_error_rad']<.05 and first_approach is None:first_approach=t+1
            final=first_approach is not None and m['target_position_error_m']<.004 and m['target_orientation_error_rad']<.03 and m['opening_m']>=.065 and m['peg_translation_m']<.002
            stable=stable+1 if final and segment.startswith('student') and physical['valid'] else 0
            row=dict(timestep=t,simulator_time_s=(t+1)/20,segment=segment,proposed_action7=proposed.tolist(),executed_action7=np.asarray(executed).tolist(),
                     controller_scaled_action=step['scaled_action'],controller_target_pose=step['controller_target_pose'],
                     controller_target_qpos=step['controller_target_qpos'],ik=step['ik'],
                     graph_nodes=len(graph.node_type),graph_edges=graph.edge_index.shape[1],presence2=[bool((graph.node_type==k).any()) for k in (3,4)],gate=gate,
                     physical_safety=physical['valid'],physical_details=physical,**m)
            rows.append(row);actions.append(proposed);robots.append(data['robot76'].copy());gates.append(gate)
            writer.write(frame(cameras,masks,condition,kind,seed,t,row))
            if t in (0,anchor) or t%25==0:npz(folder/f'graph_t{t:04d}.npz',{k:getattr(graph,k).numpy() for k in ('node_input','node_type','edge_index','edge_attr','edge_type')})
            if not physical['valid']:
                safe=False;break
            if stable>=3:
                completion=t+1;break
        if kind!='nominal' and not disturbed:disturbance_ok=False
        if kind!='nominal' and recovery_start_t is None:disturbance_ok=False
        final_measure=measure(env,initial)
        reach_success=bool(completion is not None and safe and disturbance_ok)
        response=dict(first_student_translation_command_m=None,offset_correction_cosine=None,
                      offset_correction_signed_gain=None,first_student_action7=None)
        if kind!='nominal' and recovery_start_t is not None:
            first=next((r for r in rows if r['segment']=='student_recovery'),None)
            if first:
                correction=np.asarray(first['proposed_action7'][:3])*.1;reference=-np.asarray(actual_offset)*.001
                norm_ref=np.linalg.norm(reference);norm_c=np.linalg.norm(correction)
                response=dict(first_student_translation_command_m=correction.tolist(),offset_correction_cosine=float(correction@reference/(norm_c*norm_ref)) if norm_c>1e-8 and norm_ref>.0001 else None,
                              offset_correction_signed_gain=float(correction@reference/(norm_ref**2)) if norm_ref>.0001 else None,
                              first_student_action7=first['proposed_action7'])
        recovery_errors=[r['target_position_error_m'] for r in rows if r['segment']=='student_recovery']
        result=dict(status='COMPLETE',condition=condition,kind=kind,seed=seed,reset_sha256=reset_sha,anchor=anchor,requested_offset_mm=spec.get('offset_mm'),actual_offset_mm=actual_offset,
                    disturbance_steps=disturbance_steps,disturbance_ok=disturbance_ok,first_approach_t=first_approach,recovery_start_t=recovery_start_t,
                    completion_t=completion,recovery_steps=None if completion is None or recovery_start_t is None else completion-recovery_start_t,
                    corrective_reach_success=reach_success,physical_safety=safe,position_error_m=final_measure['target_position_error_m'],
                    orientation_error_rad=final_measure['target_orientation_error_rad'],opening_m=final_measure['opening_m'],
                    peg_translation_m=final_measure['peg_translation_m'],error_at_disturbance_start_m=None if before_disturbance is None else before_disturbance['target_position_error_m'],
                    error_at_recovery_start_m=None if after_disturbance is None else after_disturbance['target_position_error_m'],
                    target_error_reduction_m=None if after_disturbance is None else after_disturbance['target_position_error_m']-final_measure['target_position_error_m'],
                    recovery_error_overshoot_m=None if after_disturbance is None or not recovery_errors else max(0.,max(recovery_errors)-after_disturbance['target_position_error_m']),
                    peg_presence_fraction=float(np.mean([r['presence2'][0] for r in rows])),hole_presence_fraction=float(np.mean([r['presence2'][1] for r in rows])),
                    graph_gate_accepted_fraction=float(np.mean([bool(r['gate'].get('accepted')) for r in rows])),
                    actions=len(rows),wall_seconds=time.perf_counter()-started,**response)
        return result,rows,np.asarray(actions),np.asarray(robots)
    finally:
        writer.release()
        for adapter in adapters.values():adapter.close()
        env.close()


def run(root,dino,sam,smoke=False,smoke_horizon=30):
    assert os.getenv('CONTAINER_ID')=='57640710';root=root.resolve();cache=root/'full_cache'
    if not smoke:
        frozen=read(root/'model_selection_frozen.json');assert frozen['no_final_access_yet'] and read(cache/'corrective_cache_manifest.json')['status']=='PASS'
        cases=read(root/'full_experiment_plan.json')['final_cases']
    else:cases=[dict(seed=30000,anchor=10,offset_mm=[5,0,0])]
    torch.set_num_threads(1);detector,tracker=make_vision('cuda',dino,sam)
    output=root/(f'evaluation_smoke_{smoke_horizon}' if smoke else 'final_evaluation');output.mkdir(exist_ok=False)
    norm=normalizer(cache if not smoke else root/'cache_gate/original')
    checkpoint_by_condition={'smoke':str(root/'initial_B_step006000.pt')} if smoke else {k:v['path'] for k,v in frozen['selected'].items()}
    config=read((cache if not smoke else root/'cache_gate/original')/'config.json')
    protocol=dict(version='10F_final_live_v1',plan_sha256=sha(root/'full_experiment_plan.json'),cases=cases,
                  horizon=130,disturbance=FEEDBACK,safety=SAFETY,first_approach_position_m=.010,first_approach_orientation_rad=.05,
                  final_position_m=.004,final_orientation_rad=.03,opening_m=.065,stable_student_steps=3,
                  intervention_at_t=25,all_failed_approaches_in_denominator=True,
                  source_sha256={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('recovery_collect.py'),
                                Path(__file__).resolve().parents[1]/'plan_a_bc'/'geometry.py',
                                Path(__file__).resolve().parents[1]/'plan_a_bc'/'model.py']})
    write(output/'evaluation_protocol.json',protocol,immutable=True)
    with ExperimentLogger(team='graph',model='Task10F_live_recovery',seed=42,config=dict(protocol=protocol,smoke=smoke),root=output/'experiments',job_type='evaluate') as log:
        write(output/'identity.json',dict(run_id=log.run_id),immutable=True)
        rows=[];reset_by_case={}
        for condition,checkpoint in checkpoint_by_condition.items():
            log.link_output(checkpoint,checkpoint=True)
            cp=checkpoint_load(checkpoint);assert cp['variant']=='E0'
            if not smoke:assert cp['config_sha256']==sha(cache/'config.json') and cp['cache_manifest_sha256']==sha(cache/'cache_manifest.json')
            codec=ActionCodec(cp['codec']);model=CachedGraphPolicy(norm.schema,'E0',**config['model']).cuda();model.load_state_dict(cp['model'],strict=True);model.eval()
            assert tensor_identity(model.state_dict())==tensor_identity(cp['model'])
            for kind in ['handoff','autonomous','nominal']:
                specs=cases if kind!='nominal' else [dict(seed=s,anchor=25) for s in sorted({c['seed'] for c in cases})]
                for spec in specs:
                    tag=f'{condition}_{kind}_{spec["seed"]}_{spec.get("offset_mm",[0,0,0])[0]}_{spec.get("offset_mm",[0,0,0])[1]}'
                    folder=output/'rollouts'/tag
                    try:
                        result,trace,actions,robots=episode(root,folder,model,norm,codec,detector,tracker,condition,kind,spec,horizon=smoke_horizon if smoke else 130)
                        result['video_sha256']=sha(folder/'rollout.mp4')
                        write(folder/'trace.json',trace,immutable=True);npz(folder/'trajectory.npz',dict(proposed_action7=actions,robot76=robots));write(folder/'episode.json',result,immutable=True)
                        key=(kind,spec['seed'],tuple(spec.get('offset_mm',[0,0,0])))
                        if key in reset_by_case:assert reset_by_case[key]==result['reset_sha256'],'R0/R1 final reset mismatch'
                        else:reset_by_case[key]=result['reset_sha256']
                        rows.append(result);append_csv(output/({'handoff':'expert_handoff_rollouts.csv','autonomous':'autonomous_recovery_rollouts.csv','nominal':'nominal_reach_rollouts.csv'}[kind]),result)
                        print('EVAL',condition,kind,spec,result['corrective_reach_success'],result['position_error_m'],flush=True)
                    except Exception as exc:
                        write(folder/'failure.json',dict(status='FAIL',error=str(exc),traceback=traceback.format_exc()),immutable=True)
                        raise
            del model;torch.cuda.empty_cache()
        if not smoke:
            write(output/'evaluation_complete.json',dict(status='PASS',episodes=len(rows),by_condition={c:sum(r['condition']==c for r in rows) for c in checkpoint_by_condition},
                 checkpoint_sha256={c:sha(p) for c,p in checkpoint_by_condition.items()},plan_sha256=sha(root/'full_experiment_plan.json')),immutable=True)
            log.link_output(output/'evaluation_complete.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dino',type=Path,required=True);p.add_argument('--sam',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-horizon',type=int,default=30);a=p.parse_args();run(a.root,a.dino,a.sam,a.smoke,a.smoke_horizon)
