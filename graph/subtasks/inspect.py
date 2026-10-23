"""Train-only threshold evidence, written before held-out evaluation."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from .dataset import load_audit
from .segmentation import segment_episode
from .specs import load_config,NAMES,DEFAULT_CONFIG
from .metrics import rotation_error
from graph.prototype.common import write_json


def inspect(manifest_path,audit_dir,output,config_path=DEFAULT_CONFIG):
    manifest=json.loads(Path(manifest_path).read_text());c=load_config(config_path)
    coverage={n:[] for n in NAMES};values={}
    def add(key,value): values.setdefault(key,[]).append(float(value))
    for eid in manifest['splits']['train']:
        d=load_audit(Path(audit_dir)/f'episode_{eid}.npz');ss,ms,_=segment_episode(d,c)
        for s in ss:
            coverage[s['subtask']].append(dict(episode_id=eid,status=s['status']))
            if s['end_t'] is not None:
                mm=ms[s['end_t']]
                for key in ('position_error','rotation_error','transport_error','relative_rotation_error',
                            'head_lateral','center_lateral','surface_clearance','waiting_error'):
                    add(s['subtask']+'_'+key,mm[key])
        for t,m in enumerate(ms):
            if t and m['grasp'] and ms[t-1]['grasp']:
                add('grasp_adjacent_position_drift',np.linalg.norm(np.asarray(m['grasp_relative_position'])-ms[t-1]['grasp_relative_position']))
                add('grasp_adjacent_rotation_drift',rotation_error(np.asarray(ms[t-1]['grasp_relative_rotation']).T@np.asarray(m['grasp_relative_rotation'])))
    distributions={k:dict(n=len(v),quantiles=dict(zip(['min','p50','p90','p95','max'],np.quantile(v,[0,.5,.9,.95,1]).tolist()))) for k,v in values.items()}
    result=dict(split='train only',thresholds=c,config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        decision='retain v0 Reach 2cm/15deg and Transport 5cm/20deg; provisional strict Align/temporal defaults, no test tuning',
        coverage={n:dict(detected=sum(r['status']=='detected' for r in v),unresolved=[r['episode_id'] for r in v if r['status']!='detected']) for n,v in coverage.items()},
        distributions=distributions)
    write_json(output,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',default='outputs/graph/dataset.json')
    p.add_argument('--audit',default='outputs/subtasks/audit');p.add_argument('--output',default='outputs/subtasks/reports/threshold_train.json')
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    a=p.parse_args();inspect(a.manifest,a.audit,a.output,a.config)
if __name__=='__main__':main()
