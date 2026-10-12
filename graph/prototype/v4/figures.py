"""Static plots from completed measured comparison only."""
import os
from pathlib import Path
import json
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path('outputs/graph/v4/.mplconfig').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('outputs/graph/v4');r=json.loads((root/'comparison/results.json').read_text());rows=r['rows']
    if not all(v['offline'] for v in rows):raise ValueError('all offline results required')
    (root/'figures').mkdir(exist_ok=True)
    labels=['Current','Action','State','Both'];x=np.arange(4)
    fig,ax=plt.subplots(figsize=(7,4))
    for offset,name,color in [(-.18,'flat','#466caa'),(.18,'surface','#cf8a38')]:
        values=[v['offline']['action_mse'] for v in rows if v['encoder']==name]
        bars=ax.bar(x+offset,values,.36,label=name,color=color)
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v,f'{v:.5f}',ha='center',va='bottom',fontsize=8)
    ax.set_xticks(x,labels);ax.set_ylabel('Held-out normalized action MSE');ax.legend();ax.set_ylim(0,max(v['offline']['action_mse'] for v in rows)*1.18)
    fig.tight_layout();fig.savefig(root/'figures/offline_mse.png',dpi=180);plt.close(fig)
    conditions=['heldout_full','fresh_full','acquired','transport','alignment']
    data=np.array([[int(v[c].split('/')[0])/int(v[c].split('/')[1]) for c in conditions] for v in rows])
    fig,ax=plt.subplots(figsize=(8,5));im=ax.imshow(data,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    for i,v in enumerate(rows):
        for j,c in enumerate(conditions):ax.text(j,i,v[c],ha='center',va='center',color='white' if data[i,j]>.5 else 'black')
    ax.set_yticks(range(8),[v['encoder']+'/'+v['history'] for v in rows]);ax.set_xticks(range(5),['Full held-out','Full fresh','Acquired start','Transport start','Alignment start'],rotation=15)
    ax.set_title('Successes / evaluated episodes; expert-prefix starts are separate conditions');fig.colorbar(im,ax=ax,label='Success rate');fig.tight_layout();fig.savefig(root/'figures/success_rates.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
