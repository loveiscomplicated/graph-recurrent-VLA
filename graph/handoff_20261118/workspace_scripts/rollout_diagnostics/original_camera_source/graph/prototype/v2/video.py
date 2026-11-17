"""Render one actual paired rollout; output includes a repeatable state trace."""
import argparse
from pathlib import Path
import imageio.v2 as imageio
import torch
from ..common import seed_all,write_json
from ..data import load_dataset
from .diagnostics import load_predictor,ReplaySession,run_episode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episode-id',type=int,default=29)
    p.add_argument('--stage',choices=['full','alignment'],default='full')
    args=p.parse_args();seed_all(0,4)
    data,g=load_dataset('outputs/graph/dataset.pt');pred,ck,*_=load_predictor(args.checkpoint,data)
    if args.episode_id not in data['manifest']['splits']['test']:p.error('held-out episode required')
    ep=next(e for e in data['manifest']['episodes'] if e['episode_id']==args.episode_id);prefix=0
    if args.stage=='alignment':prefix=int(data['timestep'][torch.where((data['episode_id']==ep['episode_id'])&(data['phase']==3))[0][0]])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    session=ReplaySession(data,render_mode='rgb_array')
    try:
        with imageio.get_writer(args.output,fps=5) as writer, torch.inference_mode():
            r=run_episode(session,ep,pred,prefix=prefix,frame_callback=lambda env:writer.append_data(env.render()[0].numpy()))
        r['checkpoint']=str(args.checkpoint);r['video']=str(args.output);r['render_stride']=4;write_json(args.output.with_suffix('.json'),r)
        print('success',r['success'],'steps',r['steps'])
    finally:session.close()

if __name__=='__main__':main()
