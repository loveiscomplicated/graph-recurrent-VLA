"""Small direct-data reader and optional causal Vision/sequence forward.

This is a diagnostic adapter, not a modification of ReplayDataset/split/reset
policy. No optimizer, backward, learned-policy rollout or GT task input exists.
"""
import argparse
import json
import os
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import torch

from graph.integration_baseline.collect_recurrent_expert import validate_episode
from graph.integration_baseline.io import sha256, write_json, save_npz, read_packet, save_packet
from graph.integration_baseline.data import AlignedFrame, collate_sequences
from graph.integration_baseline.robot import geometry_in_ee, RobotNormalizer
from graph.integration_baseline.model import IntegrationPolicy, EpisodeStream
from graph.prototype.v2.state import StateNormalizer
from graph.vision_input import validate_packet


class DirectExpertReader:
    def __init__(self, run):
        self.run = Path(run)
        self.manifest = json.loads((self.run/'pilot_manifest.json').read_text())
        assert self.manifest['schema_version'] == 'direct_scripted_expert_v1'
        assert self.manifest['run_id'] == self.run.name
        assert self.manifest['config']['control_mode'] == 'pd_ee_delta_pose'
        assert sha256(self.run/'source_snapshot/recurrent/expert.py') == self.manifest['expert_source_sha256']
        self.schema = json.loads((self.run/'robot_state_schema.json').read_text())
        assert self.schema['id'] == 'panda_observable_robot_state_v1' and self.schema['dimension'] == 76
        action_schema=json.loads((self.run/'expert_action_schema.json').read_text())
        assert action_schema['id'] == 'pd_ee_delta_pose_regression7' and action_schema['dimension'] == 7
        self.episodes = {e['episode_id']:e for e in self.manifest['episodes']}
        self.raw = {}
        for eid, e in self.episodes.items():
            assert not e['exceptions'], 'partial collector attempt is ineligible'
            assert e['seed'] == eid and e['control_mode'] == 'pd_ee_delta_pose'
            assert e['action_schema'] == action_schema['id']
            assert e['expert_source_sha256'] == self.manifest['expert_source_sha256']
            path = self.run/e['path']; assert sha256(path) == e['sha256']
            assert sha256(path.with_suffix('.h5')) == e['h5_sha256']
            with np.load(path, allow_pickle=False) as loaded:
                a = {k: loaded[k].copy() for k in loaded.files}
            validate_episode(a, self.schema, eid, path.with_suffix('.h5'))
            assert e['actions'] == len(a['actions']) and e['observations'] == len(a['robot_state'])
            assert e['success'] == bool(a['success'].any())
            success_step=int(np.flatnonzero(a['success'])[0])+1 if a['success'].any() else None
            assert e['first_success_step'] == success_step
            assert e['terminated'] == bool(a['terminated'][-1]) and e['truncated'] == bool(a['truncated'][-1])
            self.raw[eid] = a

    def frame(self, eid, t, path):
        a = self.raw[eid]
        if not 0 <= t < len(a['actions']): raise ValueError('final observation has no action')
        p = read_packet(path); validate_packet(p)
        provenance = p['observation_provenance']
        assert p['source_frame'] == t and p['frame_idx'] == t and p['timestamp'] == a['timestamp'][t]
        assert p['episode'] == self.manifest['run_id']+f':ep{eid}'
        assert provenance['episode_id'] == eid and provenance['timestep'] == t
        assert provenance['episode_sha256'] == self.episodes[eid]['sha256']
        from graph.integration_baseline.collect_recurrent_expert import digest
        assert provenance['rgb_sha256'] == digest(a['base_camera/rgb'][t])
        np.testing.assert_array_equal(p['observation_ee_pose'], a['tcp_pose'][t].astype(np.float64))
        np.testing.assert_array_equal(p['camera_to_world'], a['base_camera/cam2world'][t].astype(np.float64))
        return AlignedFrame(eid, t, p, a['robot_state'][t].copy(),
                            geometry_in_ee(a['robot_world_geometry'][t], a['tcp_pose'][t]),
                            a['actions'][t].copy(), bool(a['terminated'][t]), bool(a['truncated'][t]))


def sequence_forward(run, reader, records, eid):
    start = next((t for t in range(max(0,len(records)-3)) if all(j in records for j in range(t,t+4))), None)
    if start is None: return dict(status='BLOCKED_SEQUENCE', reason='no four consecutive valid packets')
    seq = [reader.frame(eid,t,run/records[t]['path']) for t in range(start,start+4)]
    g,s,y,mask = collate_sequences([seq], k=4, M=16)
    # Reuse the previously frozen Task06 train-prefix statistics. No new split or
    # normalization policy is chosen from this five-episode feasibility pilot.
    previous = Path('experiments/c184b85dadf14db5953f11cdcacb6674/real_forward_backward.npz')
    with np.load(previous, allow_pickle=False) as a:
        mean = torch.from_numpy(a['normalizer_mean'].copy()); std = torch.from_numpy(a['normalizer_std'].copy())
    normalizer = RobotNormalizer(reader.schema, StateNormalizer(mean,std), [])
    torch.manual_seed(0)
    model = IntegrationPolicy(reader.schema, d=64,L=2,gru_hidden=144,gru_layers=1).eval()
    tick = perf_counter()
    with torch.no_grad():
        initial = torch.randn(1,1,144)
        action, hidden = model(g,s,normalizer,initial)
        assert action.shape == (1,4,7) and torch.isfinite(action).all() and torch.isfinite(hidden).all()
        outputs=[]; h=initial.clone()
        for row in seq:
            one, state, _, _ = collate_sequences([[row]],k=4,M=16)
            out,h = model.step(one,state,normalizer,h); outputs.append(out)
        step = torch.stack(outputs,1)
        torch.testing.assert_close(action,step,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(hidden,h,atol=1e-6,rtol=1e-5)
        stream=EpisodeStream(model,normalizer)
        first=None
        for row in seq:
            one,state,_,_=collate_sequences([[row]],k=4,M=16)
            out=stream.step(one,state,(reader.manifest['run_id'],eid),row.timestep)
            if first is None: first=out
        one,state,_,_=collate_sequences([[seq[0]]],k=4,M=16)
        for kwargs in ({'reset':True}, {'previous_terminated':True}, {'previous_truncated':True}):
            out=stream.step(one,state,(reader.manifest['run_id'],eid),start,**kwargs)
            torch.testing.assert_close(out,first,atol=1e-6,rtol=1e-5)
        out=stream.step(one,state,'explicit-different-episode',start)
        torch.testing.assert_close(out,first,atol=1e-6,rtol=1e-5)
    save_npz(run/'sequence_forward.npz', action=action.numpy(), hidden=hidden.numpy(),
             targets=y.numpy(), target_mask=mask.numpy(), robot_state=s.numpy(),
             node_input=g.node_input.numpy(), edge_index=g.edge_index.numpy(),
             graph_ptr=g.ptr.numpy(), frame_map=g.frame_map.numpy(), normalizer_mean=mean.numpy(), normalizer_std=std.numpy())
    return dict(status='PASS_REAL_SEQUENCE_FORWARD', episode_id=eid, timesteps=list(range(start,start+4)),
                output_shape=list(action.shape), hidden_shape=list(hidden.shape), finite=True,
                node_counts=[len(x.packet['points'])+3 for x in seq],
                target_alignment_exact=bool(np.array_equal(y[0].numpy(),reader.raw[eid]['actions'][start:start+4])),
                hidden_carry_max_error=float((action-step).abs().max()),
                hidden_reset='PASS contract flags on real frames; no new Policy rollout',
                normalizer_source=str(previous), normalizer_sha256=sha256(previous),
                model='existing architecture, random seed0 weights; untrained',
                backward_steps=0, optimizer_steps=0, elapsed_seconds=perf_counter()-tick)


def run(args):
    assert 4 <= args.frames <= 32, 'small diagnostic only, not a dataset builder'
    assert os.environ.get('HF_HUB_OFFLINE') == '1'
    torch.set_num_threads(1)
    run = args.run.resolve(); reader = DirectExpertReader(run)
    eligible = [e for e in reader.manifest['episodes'] if e['success']]
    if not eligible:
        write_json(run/'vision_coverage.json',dict(status='BLOCKED_NO_SUCCESS_EPISODE')); return
    ep = eligible[0]; eid=ep['episode_id']; a=reader.raw[eid]
    folder=run/'vision'; folder.mkdir(exist_ok=False); (folder/'packets').mkdir()
    summary=dict(status='RUNNING',episode_id=eid,frames_requested=args.frames,packet_rows=[],failures=[],
                 episode_observations=len(a['robot_state']),frames_not_executed=list(range(args.frames,len(a['robot_state']))),
                 gt_task_inputs=False,camera='base_camera',keep_ratio=.5,edge_policy='raw_extended_delaunay',
                 edge_validation='none', input_sha256=ep['sha256'])
    write_json(run/'vision_coverage.json',summary)
    tick=perf_counter()
    try:
        from graph.integration_baseline.vision import causal_masks,packet_from_frame
        from visual.runtime.grounding import GDino
        from visual.runtime.tracking import SAM2Tracker
        from visual.runtime.rgbd import TrackedRGBDFrame
        from graph.integration_baseline.collect_recurrent_expert import digest
        detector=GDino(device='cpu'); tracker=SAM2Tracker(device='cpu')
        summary['model_load_seconds']=perf_counter()-tick
        def process(t,masks,timing):
            try:
                frame=TrackedRGBDFrame(t,'base_camera',a['base_camera/rgb'][t],a['base_camera/depth'][t],
                     a['base_camera/intrinsic'][t],a['base_camera/cam2world'][t],a['tcp_pose'][t],
                     {0:masks[0],1:masks[1]},t,float(a['timestamp'][t]))
                p,cost=packet_from_frame(frame,reader.manifest['run_id']+f':ep{eid}',keep_ratio=.5)
                p['observation_provenance']=dict(run_id=reader.manifest['run_id'],episode_id=eid,timestep=t,
                     episode_sha256=ep['sha256'],rgb_sha256=digest(frame.rgb),
                     mask_source='Task06 unchanged prefix-causal RGB DINO + forward SAM2')
                path=folder/'packets'/f'f{t:04d}.npz';save_packet(path,p)
                reader.frame(eid,t,path)  # independent readback + join gate
                summary['packet_rows'].append(dict(timestep=t,path=str(path.relative_to(run)),sha256=sha256(path),
                    point_count=len(p['points']),object_counts=[int((p['object_id']==o).sum()) for o in (0,1)],
                    edges=p['intra_object_edge_index'].shape[1],timings_ms=dict(**cost,mask_ms=timing['elapsed_ms'])))
            except Exception as exc:
                summary['failures'].append(dict(timestep=t,type=type(exc).__name__,reason=str(exc)))
            write_json(run/'vision_coverage.json',summary)
        masks,provenance=causal_masks(a['base_camera/rgb'][:args.frames],detector,tracker,process)
        save_npz(folder/'predicted_masks.npz',masks=masks,timestep=np.arange(args.frames),episode_id=np.array(eid))
        summary.update(mask_provenance=provenance,mask_sha256=sha256(folder/'predicted_masks.npz'))
        records={r['timestep']:r for r in summary['packet_rows']}
        forward=sequence_forward(run,reader,records,eid);write_json(run/'sequence_forward.json',forward)
    except Exception as exc:
        summary['failures'].append(dict(type=type(exc).__name__,reason=str(exc)))
    summary.update(status='PASS_PARTIAL_COVERAGE' if len(summary['packet_rows'])==args.frames else 'BLOCKED_OR_PARTIAL',
                   elapsed_seconds=perf_counter()-tick, process_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   memory_scope='macOS process high-water including imports/model loads; not isolated per stage')
    write_json(run/'vision_coverage.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','elapsed_seconds','failures')}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--frames',type=int,default=16);run(p.parse_args())
