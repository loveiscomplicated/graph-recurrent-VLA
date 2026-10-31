"""Task10-F v2 physical-gated matched-duration recovery pilot."""
import argparse,copy,os,time,traceback
from pathlib import Path
import h5py,numpy as np,sapien,torch
from graph.corrective_reach10f.pilot import snapshot,restore,deltas
from graph.corrective_reach10f.recovery_audit import Audit,bounded_action,PARENT
from graph.corrective_reach10f.safety_v2 import PhysicalSafety
from graph.geometry_reach10d.runtime import environment
from graph.geometry_reach10d.profile import hardware,Resources
from graph.integration_baseline.collect_recurrent_expert import capture,append_h5
from graph.integration_baseline.geometry09a import physical_state
from graph.integration_baseline.robot import robot_observation,state_schema
from graph.plan_a_bc.common import read,write,sha,append_csv
from recurrent.expert import ScriptedExpert,rotvec_between
from utils.experiment_logger import ExperimentLogger

SAFETY=dict(version='10F_safety_v2',penetration_tolerance_m=.0002,moving_external_clearance_m=.0002,contact_impulse_threshold=1e-5,max_joint_velocity_rad_s=5,
    height_surrogate_removed=True,mount='panda_link0-table has fixed mounting overlap32.5um; same inherited200um penetration tolerance, no dynamic contact exception',
    self_collision='all enabled nonadjacent pairs SAT; URDF fixed components, parent-child and PhysX collision-group exclusions recorded',
    observations='all5 physics substeps contact/penetration; control-boundary collision-shape SAT; no claim of continuous swept-volume proof')
FEEDBACK=dict(version='10F_feedback_v2',max_steps=60,max_translation_command_m=.006,max_rotation_command_rad=.01,position_gain=1.5,integral_gain=.2,integral_cap_m=.004,
    position_error_m=.001,orthogonal_error_m=.001,orientation_error_rad=.01,max_qvel=.03,stable_steps=3,
    calibration='seed30000 anchors10/25 all8 directions converge7..14steps; max residual0.748mm, no IK failure; freeze before formal pilot')
NOMINAL_HOLD={**FEEDBACK,'integral_gain':0.,'integral_cap_m':0.,'version':'10F_matched_hold_v4',
    'calibration':'same-bound P-only hold removes PI windup; seed30000 t10/25 duration7 error0.504/0.503mm, qvel0.0078; position/orientation/velocity tolerances unchanged'}

class Recorder:
    def __init__(self,folder,env,expert,audit,safety,initial,schema,prefix):
        folder.mkdir(parents=True,exist_ok=False);self.folder=folder;self.env=env;self.expert=expert;self.audit=audit;self.safety=safety;self.initial=initial;self.schema=schema;self.trace=[]
        self.f=h5py.File(folder/'episode.h5','x')
        for row,action in prefix:
            self.addrow(row);append_h5(self.f,'actions',action);append_h5(self.f,'executed_action',action);append_h5(self.f,'supervision_mask',False);append_h5(self.f,'segment',0)
    def addrow(self,row):
        for k,v in row.items():append_h5(self.f,k,v)
    def step(self,obs,action,segment,supervised):
        row,_=capture(self.env,obs,self.schema);self.addrow(row)
        # External control belongs only to executed_action; masked labels zero.
        append_h5(self.f,'actions',action if supervised else np.zeros(7,np.float32));append_h5(self.f,'executed_action',action)
        append_h5(self.f,'supervision_mask',supervised);append_h5(self.f,'segment',segment)
        dp,dr=self.expert._err(self.expert.grasp);obs,a=self.audit.step(action);a['physical_safety']=self.safety.evaluate(self.initial,a['substep_contacts'])
        a.update(position_error_before_m=float(dp),orientation_error_before_rad=float(dr),segment=segment)
        dp,dr=self.expert._err(self.expert.grasp);a.update(position_error_after_m=float(dp),orientation_error_after_rad=float(dr));self.trace.append(a)
        return obs
    def close(self,obs):
        row,_=capture(self.env,obs,self.schema);self.addrow(row);self.f.flush();self.f.close()
        with h5py.File(self.folder/'episode.h5') as f:
            n=len(f['actions']);assert len(f['robot_state'])==n+1
            np.testing.assert_array_equal(f['timestep'][:],np.arange(n+1));np.testing.assert_array_equal(f['timestamp'][:],np.arange(n+1)/20)
            mask=f['supervision_mask'][:];np.testing.assert_array_equal(mask,f['segment'][:]==2)
            np.testing.assert_array_equal(f['actions'][:][mask],f['executed_action'][:][mask])
            assert f['robot_state'].shape==(n+1,76) and np.isfinite(f['robot_state'][:]).all()
        write(self.folder/'trace.json',self.trace,immutable=True)
        return dict(actions=n,sha256=sha(self.folder/'episode.h5'),readback='PASS')

def refresh(env,expert,initial_peg):
    # The unchanged teacher reads CURRENT Peg pose. Compare against the parent
    # target transported by measured Peg translation/yaw, not a stationary-Peg
    # assertion at a stricter 2um tolerance than the physical 2mm contract.
    from scipy.spatial.transform import Rotation
    current=physical_state(env)['peg_pose']
    oldR=Rotation.from_quat(np.asarray(initial_peg)[[4,5,6,3]])
    newR=Rotation.from_quat(current[[4,5,6,3]])
    yaw=(newR*oldR.inv()).as_euler('ZYX')[0];delta=Rotation.from_euler('z',yaw)
    expected_p=current[:3]+delta.apply(expert.grasp.p-np.asarray(initial_peg)[:3])
    q=expert.grasp.q;expected_q=(delta*Rotation.from_quat(q[[1,2,3,0]])).as_quat()[[3,0,1,2]]
    fresh=ScriptedExpert(env);dp=float(np.linalg.norm(fresh.grasp.p-expected_p));dr=float(np.linalg.norm(rotvec_between(fresh.grasp.q,expected_q)))
    assert dp<2e-6 and dr<1e-5,'teacher target changed'
    expert.grasp=fresh.grasp;expert.phase='reach';expert.counter=0;expert.lift_target=None
    return dict(position_delta_m=dp,rotation_delta_rad=dr,grasp=np.r_[fresh.grasp.p,fresh.grasp.q].tolist(),phase='reach',reference='parent target transported by measured Peg translation/yaw; same2um/10urad tolerances')

def recover(rec,obs):
    audit=refresh(rec.env,rec.expert,rec.initial['peg_pose']);steps=0
    while rec.expert.phase!='close' and steps<130:
        obs=rec.step(obs,rec.expert.act(),2,True);steps+=1
        if not rec.trace[-1]['physical_safety']['valid']:break
    dp,dr=rec.expert._err(rec.expert.grasp)
    s=physical_state(rec.env);pose=rec.expert.phase=='close' and dp<.004 and dr<.03
    result=dict(target_refresh=audit,pose_recovery=bool(pose),position_error_m=float(dp),orientation_error_rad=float(dr),recovery_steps=steps,
        physical_safety=all(r['physical_safety']['valid'] for r in rec.trace),opening_m=float(s['qpos'][-2:].sum()))
    result['approved']=result['pose_recovery'] and result['physical_safety'] and result['opening_m']>=.065
    return obs,result

def collect(root,base,adopt=None,manifest=None):
    assert os.getenv('CONTAINER_ID')=='57640710';torch.set_num_threads(1);root.mkdir(parents=True,exist_ok=False)
    for n in ['inputs','config.json','normalizer.json','action_codec.json','renderer_gpu_gate.json','collection_candidates.json']:(root/n).symlink_to(base/n)
    parents=sorted({e['spec']['seed'] for e in read(base/'collection_candidates.json') if e['effective_split']=='train'})[:5]
    protocol=dict(parent_reference=PARENT,parents=parents,anchors=[10,25],offsets_mm=[[5,0,0],[-5,0,0],[0,5,0],[0,-5,0]],feedback=FEEDBACK,safety=SAFETY,
        target_validation='v3 measured Peg translation/yaw compensated reference; same unchanged fresh ScriptedExpert labels and2um/10urad tolerances',adopted_valid_root=str(adopt) if adopt else None,
        nominal_hold_feedback=NOMINAL_HOLD,
        order='nominal safety, then complete8-condition seed30000 engineering; no failed-direction exclusions; fail closes expansion',
        nominal='same fixed anchor TCP; bounded P-only external hold for exactly P duration; unchanged pd_ee_delta_pose and same refresh/recovery rule',history='prefix burnin; all elapsed disturbance frames; recovery supervised; no hidden stored')
    if manifest:
        frozen=read(manifest)
        assert frozen['feedback']==FEEDBACK and frozen['safety']==SAFETY
        parents=frozen['parents'];protocol.update(parents=parents,anchors=frozen['anchors'],offsets_mm=frozen['offsets_mm'],frozen_manifest_sha256=sha(manifest))
    write(root/'feedback_disturbance_protocol.json',protocol,immutable=True);write(root/'safety_protocol_v2.json',SAFETY,immutable=True)
    gates={};results=[];nominal_results=[]
    with ExperimentLogger(team='graph',model='Task10F_feedback_recovery_pilot',config=protocol,seed=42,root=root/'experiments',job_type='prepare') as log,Resources(root,'v2_pilot'):
        log.record(parent_reference=PARENT);write(root/'identity.json',dict(run_id=log.run_id,parent_reference=PARENT),immutable=True);write(root/'hardware.json',hardware(),immutable=True)
        env=environment(root)
        try:
            for seed in parents:
                obs,_=env.reset(seed=seed);expert=ScriptedExpert(env);audit=Audit(env);safety=PhysicalSafety(env,SAFETY);initial=physical_state(env)
                schema=state_schema(robot_observation(env)[0]);assert schema==read(root/'normalizer.json')['schema'];prefix=[];anchors={};nomtrace=[]
                while expert.phase!='close' and len(prefix)<130:
                    t=len(prefix)
                    if t in protocol['anchors']:anchors[t]=dict(state=snapshot(env,expert),prefix=copy.deepcopy(prefix),physical=physical_state(env),phase=expert.phase)
                    row,_=capture(env,obs,schema);action=expert.act();prefix.append((row,action));obs,a=audit.step(action)
                    a['physical_safety']=safety.evaluate(initial,a['substep_contacts']);a['legacy_height_valid']=a['tcp_after'][2]>.02;nomtrace.append(a)
                nominal_ok=expert.phase=='close' and all(r['physical_safety']['valid'] for r in nomtrace)
                nominal_results.append(dict(seed=seed,approved=nominal_ok,phase=expert.phase,trace=nomtrace,self_excluded_pairs=[sorted(p) for p in safety.excluded]))
                write(root/'nominal_safety_audit.json',nominal_results)
                if not nominal_ok:
                    gates['nominal_physical_safety']='FAIL';raise RuntimeError('nominal physical safety validation failed; no threshold relaxation')
                gates['nominal_physical_safety']='PASS'
                for anchor in protocol['anchors']:
                    a=anchors[anchor];assert a['phase'] in ('reach','descend')
                    trials=[]
                    for repeat in range(2):
                        restore(env,expert,a['state']);states=[];commands=[]
                        for _ in range(5):
                            action=expert.act();audit.step(action);commands.append(action);states.append(physical_state(env))
                        trials.append((states,commands))
                    err=max(max(deltas(x,y).values()) for x,y in zip(trials[0][0],trials[1][0]));assert err<=1e-6
                    write(root/f'replay_{seed}_{anchor}.json',dict(physical_max=err,action_max=float(np.max(abs(np.array(trials[0][1])-np.array(trials[1][1]))))),immutable=True)
                    gates['snapshot_replay']='PASS'
                    for offset in protocol['offsets_mm']:
                        name=f'{seed}_t{anchor}_{offset[0]}_{offset[1]}';folder=root/'raw'/name
                        if adopt and (adopt/'raw'/name/'N/receipt.json').exists() and (adopt/'raw'/name/'P/receipt.json').exists():
                            old=read(adopt/'raw'/name/'P/receipt.json');nr=read(adopt/'raw'/name/'N/receipt.json')
                            assert old['approved'] and nr['approved']
                            for tag,r in [('P',old),('N',nr)]:assert sha(adopt/'raw'/name/tag/'episode.h5')==r['sha256']
                            with h5py.File(adopt/'raw'/name/'P/episode.h5') as f:
                                for k in ['robot_state','base_camera/rgb','hand_camera/rgb']:
                                    np.testing.assert_array_equal(f[k][:anchor],np.stack([row[k] for row,_ in a['prefix']]))
                            folder.mkdir(parents=True,exist_ok=False)
                            for tag in ['N','P']:(folder/tag).symlink_to(adopt/'raw'/name/tag)
                            old=dict(old,adopted_from=str(adopt/'raw'/name),adoption='unchanged fresh-teacher action generation; validation reference correction only')
                            results.append(old);write(root/'recovery_pilot_manifest.json',results)
                            import csv
                            for pair in csv.DictReader((adopt/'paired_expert_correction.csv').open()):
                                if pair['branch']==name:append_csv(root/'paired_expert_correction.csv',pair)
                            print('ADOPT',name,flush=True);continue
                        obs=restore(env,expert,a['state']);np.testing.assert_allclose(physical_state(env)['qpos'],a['physical']['qpos'],atol=1e-6,rtol=0)
                        start=physical_state(env);pose=start['ee_pose'].copy();pose[:3]+=np.asarray(offset)*.001
                        rec=Recorder(folder/'P',env,expert,audit,safety,initial,schema,a['prefix']);integral=np.zeros(3);stable=0
                        for step in range(FEEDBACK['max_steps']):
                            action,integral=bounded_action(env,pose,integral,FEEDBACK);obs=rec.step(obs,action,1,False);s=physical_state(env)
                            error=float(np.linalg.norm(s['ee_pose'][:3]-pose[:3]));angle=float(np.linalg.norm(rotvec_between(s['ee_pose'][3:],pose[3:])));velocity=float(abs(s['qvel']).max())
                            stable=stable+1 if error<.001 and angle<.01 and velocity<.03 else 0
                            if not rec.trace[-1]['physical_safety']['valid'] or stable>=3:break
                        steps=step+1;actual=(s['ee_pose'][:3]-start['ee_pose'][:3])*1000;disturbance_ok=stable>=3 and all(r['physical_safety']['valid'] for r in rec.trace)
                        append_csv(root/'feedback_disturbance_metrics.csv',dict(branch=name,seed=seed,anchor=anchor,requested_mm=offset,actual_mm=actual.tolist(),residual_m=error,orientation_rad=angle,max_qvel=velocity,steps=steps,passed=disturbance_ok))
                        if disturbance_ok:obs,result=recover(rec,obs)
                        else:result=dict(approved=False,pose_recovery=False,physical_safety=all(r['physical_safety']['valid'] for r in rec.trace),reason='disturbance_convergence_or_safety')
                        result.update(branch=name,seed=seed,anchor=anchor,offset_mm=offset,actual_offset_mm=actual.tolist(),disturbance_steps=steps,disturbance_ok=disturbance_ok,**rec.close(obs))
                        write(folder/'P/receipt.json',result,immutable=True);results.append(result);write(root/'recovery_pilot_manifest.json',results)
                        print('PILOT',name,result['approved'],steps,error,flush=True)
                        if not result['approved']:
                            gates['perturbed_recovery']='FAIL'
                            # Remaining directions for this anchor are diagnostic only;
                            # stop expansion, preserve failure instead of filtering it.
                            raise RuntimeError('perturbed branch required physical/recovery gate failed')
                        obs=restore(env,expert,a['state']);recN=Recorder(folder/'N',env,expert,audit,safety,initial,schema,a['prefix']);integral=np.zeros(3);hold=start['ee_pose'].copy()
                        for _ in range(steps):
                            action,integral=bounded_action(env,hold,integral,NOMINAL_HOLD);obs=recN.step(obs,action,1,False)
                            if not recN.trace[-1]['physical_safety']['valid']:break
                        nstate=physical_state(env);hold_error=float(np.linalg.norm(nstate['ee_pose'][:3]-hold[:3]));hold_angle=float(np.linalg.norm(rotvec_between(hold[3:],nstate['ee_pose'][3:])));hold_velocity=float(abs(nstate['qvel']).max())
                        if hold_error>=.001 or hold_angle>=.01 or hold_velocity>=.03:
                            recN.close(obs);gates['nominal_matched_hold_accuracy']='FAIL';raise RuntimeError('matched nominal hold did not maintain anchor within frozen1mm/.01rad/.03 velocity criteria')
                        obs,nresult=recover(recN,obs);nresult.update(**recN.close(obs));write(folder/'N/receipt.json',nresult,immutable=True)
                        pseq=[r for r in rec.trace if r['segment']==2];nseq=[r for r in recN.trace if r['segment']==2];delta=np.asarray(pseq[0]['scaled_action'][:3])-np.asarray(nseq[0]['scaled_action'][:3]);reference=-actual*.001
                        magnitude=float(np.linalg.norm(delta));den=float(np.linalg.norm(reference));cos=float(delta@reference/(magnitude*den)) if magnitude>1e-8 and den>.0001 else None
                        pair=dict(branch=name,nominal_approved=nresult['approved'],perturbed_approved=result['approved'],duration=steps,actual_offset_mm=actual.tolist(),first_command_delta_m=delta.tolist(),correction_magnitude_m=magnitude,direction_cosine=cos,signed_gain=float(delta@reference/(reference@reference)) if den>.0001 else None)
                        for k in [3,5,10]:
                            for tag,seq in [('P',pseq),('N',nseq)]:
                                seq=seq[:k];pair[f'{tag}_first{k}_command_sum_m']=np.asarray([r['scaled_action'][:3] for r in seq]).sum(0).tolist();pair[f'{tag}_first{k}_actual_motion_m']=(np.asarray(seq[-1]['tcp_after'][:3])-np.asarray(seq[0]['tcp_before'][:3])).tolist()
                        append_csv(root/'paired_expert_correction.csv',pair)
                        if not nresult['approved']:gates['nominal_matched_control']='FAIL';raise RuntimeError('matched nominal recovery failed')
                # Controller audit wrappers belong to one reset; next reset
                # replaces controller; avoid nesting scene.step observers.
                env.unwrapped.scene.step=audit.original_step
            gates.update(perturbed_recovery='PASS',nominal_matched_control='PASS',alignment='PASS',supervision_mask='PASS')
        except Exception as exc:
            gates['execution']='FAIL'
            traceback.print_exc();write(root/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()),immutable=True)
        finally:
            env.close()
            for k in ['requested_actual_offset','controller_scaling','physical_collision_joint_limit','target_refresh','action_correction','causal_vision_gru','cache_hash_readback','source_protection','training','closed_loop']:gates.setdefault(k,'NOT_TESTED')
            write(root/'verification.json',dict(status='FAIL' if 'FAIL' in gates.values() else 'DATA_PILOT_COMPLETE_CACHE_GATE_PENDING',checks=gates,completed_branches=len(results),planned=40),immutable=True)
            for p in root.glob('*.json'):
                if not p.is_symlink():log.link_output(p,kind='reports')
            log.record(verification=gates)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,default=Path('/workspace/task10f_corrective_reach'));p.add_argument('--adopt',type=Path);p.add_argument('--manifest',type=Path);a=p.parse_args();collect(a.root,a.base,a.adopt,a.manifest)
