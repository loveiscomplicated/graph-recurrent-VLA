"""Measured randomization, segment lengths and collection status figures."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graph.prototype.common import write_json
from .dataset import load_segments
from .specs import NAMES


def run(directory):
    directory=Path(directory)
    records=json.loads((directory/'reports/collection.json').read_text())['records']
    index=load_segments(directory/'segments/index.json')
    output=directory/'figures'
    output.mkdir(exist_ok=True)
    from .generate import versions
    write_json(directory/'reports/report_runtime.json',dict(versions=versions(),
        note='Runtime executing this report; collection versions remain in the source manifests'))
    sampled=[r for r in records if r.get('randomization')]
    distributions={}
    for key in sampled[0]['randomization'] if sampled else ():
        values=np.asarray([r['randomization'][key] for r in sampled])
        distributions[key]=dict(count=len(values),minimum=values.min(axis=0).tolist(),maximum=values.max(axis=0).tolist(),
                                std=values.std(axis=0).tolist(),unique_rows=len(np.unique(values,axis=0)),
                                varies=bool(np.any(np.ptp(values,axis=0)>1e-6)))
    write_json(directory/'reports/randomization.json',dict(samples=len(sampled),distributions=distributions,
               note='Observed sample bounds; fixed seed reset randomization, no custom physical perturbations'))
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    selections=[('peg_pose',0,'Peg x (m)'),('peg_pose',1,'Peg y (m)'),
                ('peg_half_sizes',0,'Peg half length (m)'),('peg_half_sizes',1,'Peg half width (m)'),
                ('box_pose',0,'Box x (m)'),('box_pose',1,'Box y (m)')]
    for ax,(key,col,label) in zip(axes.flat,selections):
        ax.hist([r['randomization'][key][col] for r in sampled],bins=min(12,max(1,len(sampled))),color='tab:blue')
        ax.set(xlabel=label,ylabel='Native episodes')
    fig.tight_layout();fig.savefig(output/'randomization.png',dpi=140);plt.close(fig)
    lengths={n:[s['end_t']-s['start_t'] for ep in index['episodes'] for s in ep['segments'] if s['subtask']==n and s['status']=='detected'] for n in NAMES}
    write_json(directory/'reports/lengths.json',{n:dict(count=len(v),mean=float(np.mean(v)) if v else None,
                minimum=min(v) if v else None,maximum=max(v) if v else None,p95=float(np.percentile(v,95)) if v else None) for n,v in lengths.items()})
    fig,axes=plt.subplots(1,5,figsize=(15,3))
    for ax,n in zip(axes,NAMES):
        ax.hist(lengths[n],bins=min(12,max(1,len(lengths[n]))))
        ax.set(title=f'{n} (n={len(lengths[n])})',xlabel='Segment actions',ylabel='Segments')
    fig.tight_layout();fig.savefig(output/'lengths.png',dpi=140);plt.close(fig)
    summary=json.loads((directory/'reports/summary.json').read_text())
    labels=['Source success','Native valid','Conversion success','Converted replay valid','Converted full success']
    keys=['source_success','native_replay_valid','conversion_success','converted_replay_valid','converted_replay_success']
    fig,ax=plt.subplots(figsize=(11,4));ax.bar(labels,[summary[k] for k in keys])
    ax.set(ylabel='Episodes',title=f'Collection stages; {summary["attempted"]} attempted seeds')
    ax.tick_params(axis='x',rotation=12);fig.tight_layout();fig.savefig(output/'collection.png',dpi=140);plt.close(fig)
    return distributions


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');a=p.parse_args();run(a.directory)
