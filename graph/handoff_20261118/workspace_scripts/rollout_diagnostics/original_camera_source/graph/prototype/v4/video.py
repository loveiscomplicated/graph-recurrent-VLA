"""Replay recorded applied commands and verify poses while rendering video."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch
from ..common import seed_all, write_json
from ..data import load_dataset
from ..v2.diagnostics import ReplaySession


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--stage',default='alignment')
    p.add_argument('--episode-id',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('choose a fresh output file')
    report=json.loads(args.report.read_text());record=next(e for e in report['settings'][args.stage]['episodes'] if e['episode_id']==args.episode_id)
    data,_=load_dataset('outputs/graph/dataset.pt');seed_all(0,4)
    ep=next((e for e in data['manifest']['episodes'] if e['episode_id']==args.episode_id),None)
    if ep is None:ep=dict(episode_id=args.episode_id,episode_seed=record['seed'],reset_kwargs={'seed':record['seed']},new_configuration=True)
    session=ReplaySession(data,horizon=report['horizon'],render_mode='rgb_array');args.output.parent.mkdir(parents=True,exist_ok=True)
    maximum=0.
    try:
        error,fingerprint=session.prepare(ep,record['prefix'])
        if fingerprint!=record['initial_geometry_sha256']:raise AssertionError('initial geometry differs')
        with imageio.get_writer(args.output,fps=5) as writer:
            writer.append_data(session.env.render()[0].numpy())
            for t,row in enumerate(record['trace']):
                session.env.step(torch.tensor(row['applied_action'])[None])
                g=session.adapter(session.env)
                for field,key in [('ee_pose','ee_pose'),('peg_pose','peg_pose')]:
                    maximum=max(maximum,float(np.max(np.abs(getattr(g,field)[0].numpy()-np.array(row['after'][key])))))
                if t%4==0:writer.append_data(session.env.render()[0].numpy())
        if maximum>2e-5:raise AssertionError(f'replay poses differ: {maximum}')
        write_json(args.output.with_suffix('.json'),dict(source_report=str(args.report),stage=args.stage,episode_id=args.episode_id,success=record['success'],failure_stage=record['failure_stage'],replayed_applied_commands=True,maximum_pose_error=maximum,prefix_error=error))
    finally:session.close()


if __name__=='__main__':main()
