"""Actual expert-prefix geometry and current task-derived grasp/head targets."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path('outputs/graph/v2/.mplconfig').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from ..common import seed_all
from ..data import load_dataset
from ..geometry import transform
from ..graph_builder import GraphBuilder
from .diagnostics import ReplaySession
from .goals import goal_geometry


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();seed_all(0,4)
    data,g=load_dataset('outputs/graph/dataset.pt');ep=next(e for e in data['manifest']['episodes'] if e['episode_id']==29)
    args.output.mkdir(parents=True,exist_ok=True);session=ReplaySession(data)
    try:
        alignment=int(data['timestep'][torch.where((data['episode_id']==29)&(data['phase']==3))[0][0]])
        for stage,t in [('initial',0),('acquired',56),('transport',72),('alignment',alignment)]:
            session.prepare(ep,t);g=session.adapter(session.env);goal=goal_geometry(g)
            points=torch.cat((g.ee_pose[:,:3,None].transpose(1,2),torch.stack((g.left_fingertip,g.right_fingertip),1),g.peg_points,g.hole_points),1)[0]
            graph=GraphBuilder()(g);fig=plt.figure(figsize=(11,5))
            for panel in (1,2):
                ax=fig.add_subplot(1,2,panel,projection='3d')
                for a,b,c,label in [(0,1,'red','TCP'),(1,2,'green','Left pad'),(2,3,'blue','Right pad'),(3,67,'orange','Peg surface'),(67,131,'purple','Hollow box surface')]:ax.scatter(*points[a:b].T.numpy(),c=c,s=12,label=label)
                target=goal['grasp_position'][0];ax.scatter(*target.numpy(),marker='x',c='black',s=80,label='Fixed tail grasp target')
                ax.plot(*torch.stack((g.ee_pose[0,:3],target)).T.numpy(),c='black',lw=2)
                local=torch.zeros(1,1,3);local[0,0,0]=g.peg_half_size[0,0];head=transform(local,g.peg_pose)[0,0]
                ax.plot(*torch.stack((head,g.hole_pose[0,:3])).T.numpy(),c='teal',lw=2,label='Head → hole centre')
                if panel==2:
                    for i,(a,b) in enumerate(graph.edge_index.T):
                        if a<b:ax.plot(*points[[int(a),int(b)]].T.numpy(),c=['black','silver','green','teal'][graph.edge_type[i]],alpha=.12,lw=.5)
                ax.set(xlabel='World x (m)',ylabel='World y (m)',zlabel='World z (m)');ax.view_init(25,-60 if panel==1 else 20)
            fig.axes[0].legend(fontsize=7);fig.suptitle(f'Actual episode 29 / {stage} / expert prefix {t} steps');fig.tight_layout();fig.savefig(args.output/f'goals_{stage}.png',dpi=150);plt.close(fig)
    finally:session.close()

if __name__=='__main__':main()
