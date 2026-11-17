"""Build provenance-preserving segment indices and condition diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
from .metrics import compose
from graph.prototype.common import write_json
from graph.prototype.data import sha256
from .dataset import load_audit,save_segments
from .segmentation import segment_episode
from .specs import load_config,DEFAULT_CONFIG


def run(manifest_path,audit_dir,output,config_path=DEFAULT_CONFIG):
    manifest=json.loads(Path(manifest_path).read_text());config=load_config(config_path)
    for key in ('source','initial_state_source'):
        if sha256(manifest[key])!=manifest[key+'_sha256']:
            raise ValueError(f'{key} checksum differs')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    payload={k:manifest[k] for k in ('source','source_sha256','initial_state_source','initial_state_source_sha256','splits','control_mode')}
    payload.update(segmentation_version=config['version'],thresholds=config,
        config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        boundary_semantics='geometric milestones; original expert phase labels unavailable',
        undefined_failure_rules=['IK unreachable','severe collision','stalled insertion'],episodes=[])
    for ep in manifest['episodes']:
        eid=ep['episode_id']; path=Path(audit_dir)/f'episode_{eid}.npz'
        data=load_audit(path)
        if (data['record']['episode_id']!=eid or str(data['source_sha256'])!=manifest['source_sha256'] or
                str(data['initial_state_source_sha256'])!=manifest['initial_state_source_sha256']):
            raise ValueError('audit provenance differs')
        segments,metrics,diagnostics=segment_episode(data,config)
        split=next(k for k,v in manifest['splits'].items() if eid in v)
        for s in segments:
            i=s['start_t'];s.update(source_episode_id=eid,split=split,source_dataset=manifest['source'],segmentation_version=config['version'],
                initial_state_ref=dict(method='original_s0_then_expert_prefix',original_source=manifest['initial_state_source'],
                    original_episode_id=data['record']['source_episode_id'],episode_seed=ep['episode_seed'],
                    reset_kwargs=ep['reset_kwargs'],prefix_actions=[0,i],state_index=i,
                    controller_reset_at_s0=True),
                goal_metadata=dict(reach_local=data['reach_local'].tolist(),grasp_local=data['grasp_local'].tolist(),
                    hole_pose=data['hole_pose'][0].tolist(),peg_half_sizes=data['peg_half_sizes'][0].tolist(),
                    preinsert_center_hole=[float(-2*data['peg_half_sizes'][0,0]+config['waiting_depth']),0,0],
                    align_head_hole=[float(-data['peg_half_sizes'][0,0]+config['waiting_depth']),0,0],
                    reach_pose_at_start=compose(data['peg_pose'][i],data['reach_local']).tolist() if i is not None else None,
                    insertion_success_source='live environment evaluate()[success]',
                    waiting_depth=config['waiting_depth'],frame='world wxyz; hole-local +X insertion',
                    dynamic_reach='current peg pose composed with reach_local'))
        write_json(output/f'metrics_{eid}.json',dict(episode_id=eid,metrics=metrics,diagnostics=diagnostics))
        payload['episodes'].append(dict(episode_id=eid,split=split,total_actions=len(data['actions']),
            audit_file=str(path.resolve()),audit_sha256=sha256(path),segments=segments,diagnostics=diagnostics,replay=data['record']))
    save_segments(output/'index.json',payload)
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='outputs/graph/dataset.json');p.add_argument('--audit',default='outputs/subtasks/audit')
    p.add_argument('--output',default='outputs/subtasks/segments');p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    a=p.parse_args();d=run(a.manifest,a.audit,a.output,a.config)
    print(f"saved {len(d['episodes'])} episodes to {a.output}/index.json")
if __name__=='__main__': main()
