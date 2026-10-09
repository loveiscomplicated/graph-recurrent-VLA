"""Static timeline/geometry figures, plus real prefix-replay boundary renders."""
import argparse
import json
from itertools import product
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from .dataset import load_segments,load_audit
from .metrics import rotation
from .specs import NAMES
from graph.prototype.common import write_json

COLORS=dict(zip(NAMES,plt.get_cmap('tab10').colors[:5]))
DEFAULT_EPISODES=[0,1,2,13,29,38,54,90]


def _phase_at(ep, t):
    for segment in ep['segments']:
        if segment['status'] == 'detected' and segment['start_t'] <= t < segment['end_t']:
            return segment['subtask']
    return 'done' if t == ep['total_actions'] else 'tail'


def _annotate_frame(frame, ep, metrics, audit, t):
    image = Image.fromarray(frame).convert('RGB')
    draw = ImageDraw.Draw(image)
    phase = _phase_at(ep, t)
    color = tuple(int(255*x) for x in COLORS.get(phase, (.25, .25, .25)))
    draw.rectangle((8, 8, 475, 105), fill=(0, 0, 0), outline=color, width=3)
    command = float(audit['actions'][min(t, len(audit['actions'])-1), -1]) if len(audit['actions']) else 0.0
    lines = [
        f"episode {ep['episode_id']}  state {t}/{ep['total_actions']}",
        f"phase: {phase.upper()}   gripper command: {command:+.1f}",
        f"contact grasp: {bool(audit['grasp'][t])}   official success: {bool(audit['success'][t])}",
        f"head lateral: {metrics['head_lateral']*1000:.1f} mm   depth: {metrics['insertion_depth']*1000:.1f} mm",
    ]
    for row, line in enumerate(lines):
        draw.text((18, 14 + row*21), line, fill='white')
    return np.asarray(image)


def plots(index_path,output,ids=DEFAULT_EPISODES):
    data=load_segments(index_path);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(13,20))
    for row,ep in enumerate(data['episodes']):
        ax.barh(row,ep['total_actions'],color='.9')
        for s in ep['segments']:
            if s['status']=='detected':ax.barh(row,s['end_t']-s['start_t'],left=s['start_t'],color=COLORS[s['subtask']])
    ax.set_yticks(range(len(data['episodes'])));ax.set_yticklabels([e['episode_id'] for e in data['episodes']],fontsize=6)
    ax.set(xlabel='Original action index (grey = uncovered)',ylabel='Episode',title='91 expert episodes: geometric milestone partitions')
    fig.tight_layout();fig.savefig(output/'all_durations.png',dpi=150);plt.close(fig)
    for ep in data['episodes']:
        if ep['episode_id'] not in ids:continue
        eid=ep['episode_id'];audit=load_audit(ep['audit_file'])
        metrics=json.loads((Path(index_path).parent/f'metrics_{eid}.json').read_text())['metrics'];t=np.arange(len(metrics))
        fig,axs=plt.subplots(5,1,figsize=(12,10),sharex=True)
        for s in ep['segments']:
            if s['status']=='detected':
                i,j=s['start_t'],s['end_t'];axs[0].barh(0,j-i,left=i,color=COLORS[s['subtask']]);axs[0].text((i+j)/2,0,s['subtask'],ha='center',va='center',fontsize=9)
        axs[0].set_yticks([]);axs[0].set_title(f'Episode {eid} ({ep["split"]}), T={ep["total_actions"]}; estimated geometric boundaries')
        axs[1].step(t[:-1],audit['actions'][:,-1],label='gripper command');axs[1].step(t,audit['grasp'],label='real contact grasp');axs[1].legend()
        for key in ('position_error','transport_error'):axs[2].plot(t,[m[key] for m in metrics],label=key)
        axs[2].set_ylabel('metres');axs[2].legend()
        for key in ('head_lateral','center_lateral','surface_clearance'):axs[3].plot(t,[m[key]*1000 for m in metrics],label=key)
        axs[3].axhline(3,ls='--',color='.5');axs[3].set_ylabel('mm');axs[3].legend()
        axs[4].plot(t,[m['insertion_depth'] for m in metrics],label='insertion depth (m)')
        axs[4].plot(t,[m['relative_rotation_error'] for m in metrics],label='rotation (rad)');axs[4].axhline(0,color='.5',ls='--');axs[4].legend();axs[4].set_xlabel('State timestep; action t maps s[t] to s[t+1]')
        for ax in axs[1:]:
            for s in ep['segments']:
                if s['end_t'] is not None:ax.axvline(s['end_t'],color=COLORS[s['subtask']],alpha=.35)
        fig.tight_layout();fig.savefig(output/f'timeline_{eid}.png',dpi=140);plt.close(fig)
        fig=plt.figure(figsize=(16,7))
        for col,s in enumerate(ep['segments']):
            for row,key in enumerate(('start_t','end_t')):
                ax=fig.add_subplot(2,5,row*5+col+1,projection='3d');idx=s[key]
                ax.set_title(f"{s['subtask']} {key}: {idx}",fontsize=9)
                if idx is None:continue
                peg=audit['peg_pose'][idx];hole=audit['hole_pose'][idx];half=audit['peg_half_sizes'][idx]
                corners=np.asarray(list(product((-1,1),repeat=3)))*half
                local=(peg[:3]-hole[:3])@rotation(hole)+corners@rotation(peg).T@rotation(hole)
                for u in range(8):
                    for v in range(u+1,8):
                        if np.count_nonzero(corners[u]!=corners[v])==1:ax.plot(*local[[u,v]].T,color='tab:orange')
                r=audit['hole_radius'][idx];L=half[0]
                for x in (-L,L):
                    square=np.array([[x,-r,-r],[x,r,-r],[x,r,r],[x,-r,r],[x,-r,-r]])
                    ax.plot(*square.T,color='tab:blue')
                ee=(audit['ee_pose'][idx,:3]-hole[:3])@rotation(hole);ax.scatter(*ee,color='black',s=10)
                ax.set(xlabel='hole X',ylabel='Y',zlabel='Z');ax.set_box_aspect((2,1,1))
        fig.suptitle(f'Episode {eid}: hole-local peg box, aperture planes and TCP (black)')
        fig.tight_layout();fig.savefig(output/f'geometry_{eid}.png',dpi=130);plt.close(fig)


def render_boundaries(index_path,output,ids):
    import h5py
    import torch
    from envs.maniskill import make_peg_insertion_env
    from mani_skill.utils.registration import TimeLimitWrapper
    from graph.prototype.data import reset_episode,state_at
    data=load_segments(index_path);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    env=make_peg_insertion_env(control_mode=data['control_mode'],render_mode='rgb_array')
    checks=[]
    try:
        with h5py.File(data['source']) as source,h5py.File(data['initial_state_source']) as original:
            for ep in data['episodes']:
                eid=ep['episode_id']
                if eid not in ids:continue
                audit=load_audit(ep['audit_file']);ref=ep['segments'][0]['initial_state_ref'];T=ep['total_actions']
                env=TimeLimitWrapper(env.unwrapped,T+1)
                reset_episode(env,dict(episode_seed=ref['episode_seed'],reset_kwargs=ref['reset_kwargs']))
                env.unwrapped.set_state_dict(state_at(original[f"traj_{ref['original_episode_id']}/env_states"],0))
                env.unwrapped.agent.controller.reset()
                selected={s[k] for s in ep['segments'] for k in ('start_t','end_t') if s[k] is not None}
                frames={};max_error=0.;grasp_equal=True
                for t in range(T+1):
                    if t in selected:
                        frames[t]=Image.fromarray(env.render()[0].cpu().numpy()).convert('RGB')
                        if t:grasp_equal &= bool(env.unwrapped.agent.is_grasping(env.unwrapped.peg).item())==bool(audit['grasp'][t])
                    if t<T:
                        obs,*_=env.step(torch.from_numpy(audit['actions'][t])[None])
                        max_error=max(max_error,float(np.max(np.abs(obs.cpu().numpy()[0]-source[f'traj_{eid}/obs'][t+1]))))
                sheet=Image.new('RGB',(5*320,2*345),'white');draw=ImageDraw.Draw(sheet)
                for col,s in enumerate(ep['segments']):
                    for row,k in enumerate(('start_t','end_t')):
                        t=s[k]
                        if t is not None:sheet.paste(frames[t].resize((320,320)),(col*320,row*345))
                        draw.text((col*320+5,row*345+322),f"{s['subtask']} {k}={t}",fill='black')
                sheet.save(output/f'replay_boundaries_{eid}.png')
                checks.append(dict(episode_id=eid,max_observation_error=max_error,contact_matches=bool(grasp_equal),boundary_states=sorted(selected)))
                print(checks[-1],flush=True)
        write_json(output/'boundary_replay.json',dict(checks=checks,backend=str(env.unwrapped.backend)))
    finally:env.close()


def render_videos(index_path, output, ids, fps=10, stride=1):
    """Replay unchanged expert actions and write annotated MP4s."""
    import h5py
    import imageio.v2 as imageio
    import torch
    from envs.maniskill import make_peg_insertion_env
    from mani_skill.utils.registration import TimeLimitWrapper
    from graph.prototype.data import reset_episode, state_at
    data=load_segments(index_path);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    env=make_peg_insertion_env(control_mode=data['control_mode'],render_mode='rgb_array')
    records=[]
    try:
        with h5py.File(data['source']) as source,h5py.File(data['initial_state_source']) as original:
            for ep in data['episodes']:
                eid=ep['episode_id']
                if eid not in ids: continue
                audit=load_audit(ep['audit_file'])
                metrics=json.loads((Path(index_path).parent/f'metrics_{eid}.json').read_text())['metrics']
                ref=ep['segments'][0]['initial_state_ref']; T=ep['total_actions']
                env=TimeLimitWrapper(env.unwrapped,T+1)
                reset_episode(env,dict(episode_seed=ref['episode_seed'],reset_kwargs=ref['reset_kwargs']))
                env.unwrapped.set_state_dict(state_at(original[f"traj_{ref['original_episode_id']}/env_states"],0))
                env.unwrapped.agent.controller.reset()
                path=output/f'episode_{eid}_subtasks.mp4'
                max_error=0.0; frame_count=0
                with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
                    for t in range(T+1):
                        if t % stride == 0 or t == T:
                            frame=env.render()[0].cpu().numpy()
                            writer.append_data(_annotate_frame(frame,ep,metrics[t],audit,t))
                            frame_count += 1
                        if t < T:
                            obs,*_=env.step(torch.from_numpy(audit['actions'][t])[None])
                            max_error=max(max_error,float(np.max(np.abs(obs.cpu().numpy()[0]-source[f'traj_{eid}/obs'][t+1]))))
                record=dict(episode_id=eid,video=str(path.resolve()),fps=fps,stride=stride,
                    frames=frame_count,max_observation_error=max_error,source_actions=T,
                    phase_boundaries=[dict(subtask=s['subtask'],start_t=s['start_t'],end_t=s['end_t']) for s in ep['segments']])
                records.append(record);print(record,flush=True)
        write_json(output/'videos.json',dict(videos=records,kind='expert prefix replay with annotated geometric milestones',
            backend=str(env.unwrapped.backend)))
    finally: env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',default='outputs/subtasks/segments/index.json');p.add_argument('--output',default='outputs/subtasks/figures')
    p.add_argument('--episodes',type=int,nargs='+',default=DEFAULT_EPISODES)
    p.add_argument('--render',action='store_true',help='write boundary contact sheets')
    p.add_argument('--video',action='store_true',help='write annotated expert replay MP4s')
    p.add_argument('--fps',type=int,default=10)
    p.add_argument('--stride',type=int,default=1,help='render every Nth state; actions replay at every step')
    a=p.parse_args()
    if a.video:
        if a.fps < 1 or a.stride < 1: p.error('--fps and --stride must be positive')
        render_videos(a.index,a.output,a.episodes,a.fps,a.stride)
    elif a.render:render_boundaries(a.index,a.output,a.episodes)
    else:plots(a.index,a.output,a.episodes)
if __name__=='__main__':main()
