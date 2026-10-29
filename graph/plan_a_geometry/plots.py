"""Scientific figures from saved metric tables/predictions; no fitting."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graph.persistent_geometry.render import read_npz
from graph.plan_a_geometry.summarize import table,numeric
from graph.integration_baseline.io import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state=json.loads(a.state.read_text());runs={k:Path(v) for k,v in state['completed'].items()};images=[]
    geometry=json.loads((a.summary/'final.json').read_text())['geometry']
    labels=['C0_SF','C1_SF','C2_SF','C3_SF','C3_L','C3_H'];fig,axes=plt.subplots(2,4,figsize=(16,7))
    for oid in (0,1):
        for col,(metric,title) in enumerate([('current_both_visible_coverage_4mm','Current-visible coverage'),('source_object_purity','Source purity'),('surface_distance_mean_mm','Collision surface distance mm'),('boundary_coverage_4mm','Visible boundary coverage')]):
            values=[next(r[metric] for r in geometry if r['variant']==v and r['budget']=='cap128' and r['object_id']==oid) for v in labels]
            axes[oid,col].bar(labels,[v if v is not None else np.nan for v in values],color=['#888888','#cc7777','#55aaaa','#4477aa','#eeaa44','#66aa66'])
            axes[oid,col].set_title(('Peg' if oid==0 else 'Box')+' '+title,fontsize=10);axes[oid,col].tick_params(axis='x',labelrotation=45)
            if 'distance' not in metric:axes[oid,col].set_ylim(0,1)
    fig.suptitle('Independent Final | cap128/object | episode macro means | same observations and common edges');fig.tight_layout();fig.savefig(a.output/'final_geometry.png',dpi=130);plt.close(fig);images.append('final_geometry.png')
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for oid in (0,1):
        for col,mode in enumerate(('native','cap128','matched_N')):
            for metric,marker in [('source_object_purity','o'),('boundary_coverage_4mm','s'),('current_both_visible_coverage_4mm','^')]:
                rr=[next(r for r in geometry if r['variant']==v and r['budget']==mode and r['object_id']==oid) for v in ['C3_SF','C3_L','C3_H']]
                axes[oid,col].plot(['A-SF','A-L','A-H'],[r[metric] for r in rr],marker=marker,label=metric.replace('_4mm',''))
            axes[oid,col].set_ylim(0,1);axes[oid,col].set_title(('Peg ' if oid==0 else 'Box ')+mode)
    axes[0,0].legend(fontsize=6);fig.suptitle('Native vs cap128 vs exact matched N | Final episode macro means');fig.tight_layout();fig.savefig(a.output/'budget_comparison.png',dpi=130);plt.close(fig);images.append('budget_comparison.png')
    for stage,proberoot in [('validation',runs['probes']),('evaluation',runs['final_apply'])]:
        if stage=='validation':manifest=json.loads((runs['probes']/'probe_manifest.json').read_text())
        else:manifest=json.loads((runs['probes']/'probe_manifest.json').read_text())
        fig,axes=plt.subplots(2,2,figsize=(13,7))
        rr=table(proberoot/'absolute_metrics.csv');dd=table(proberoot/'paired_delta_metrics.csv')
        for oid in (0,1):
            for enc in ('E0','E1'):
                vals=[];deltas=[]
                for v in ('C0_SF','C3_SF','C3_L','C3_H'):
                    candidates=[r for r in rr if r['split']==stage and r['condition_group']=='baseline' and r['variant']==v and r['encoder']==enc and r['probe']=='P1_readout' and r['decoder']=='linear' and r['object_id']==str(oid)]
                    vals.append(numeric(candidates[0]['mean_mm']) if candidates else np.nan)
                    pairs=[numeric(r['delta_error_mm']) for r in dd if r['split']==stage and r['variant']==v and r['encoder']==enc and r['probe']=='P1_readout' and r['decoder']=='linear' and r['object_id']==str(oid) and r['physical_valid']=='True' and r['target_moved']=='True']
                    deltas.append(np.mean(pairs) if pairs else np.nan)
                axes[oid,0].plot(['C0_SF','C3_SF','C3_L','C3_H'],vals,marker='o',label=enc)
                axes[oid,1].plot(['C0_SF','C3_SF','C3_L','C3_H'],deltas,marker='o',label=enc)
            axes[oid,0].set_title(('Peg' if oid==0 else 'Box')+' linear readout position mm');axes[oid,1].set_title(('Peg' if oid==0 else 'Box')+' moved-object paired delta mm')
            axes[oid,0].legend();axes[oid,1].legend()
        fig.suptitle(stage+' frozen probes | reported tables additionally provide episode macro values');fig.tight_layout();name=stage+'_probes.png';fig.savefig(a.output/name,dpi=130);plt.close(fig);images.append(name)
    # Actual readout heatmaps and relative XYZ reconstruction, first Final episode.
    pm=json.loads((runs['probes']/'probe_manifest.json').read_text())['probes']
    fig,axes=plt.subplots(3,2,figsize=(14,9))
    for col,enc in enumerate(('E0','E1')):
        data=read_npz(runs['final_apply']/'models'/('C3_SF_'+enc)/'frozen_features.npz')
        selected=(data['seed']==np.min(data['seed']))&(data['condition']=='baseline');t=data['timestep'][selected];z=data['readout'][selected]
        axes[0,col].imshow(z.T,aspect='auto',origin='lower',cmap='coolwarm');axes[0,col].set_title(enc+' actual320-D readout, first Final episode');axes[0,col].set_ylabel('Readout feature');axes[0,col].set_xlabel('Evaluated frame index')
        for oid in (0,1):
            probe=next(p for p in pm if p['variant']=='C3_SF' and p['encoder']==enc and p['probe']=='P1_readout' and p['decoder']=='linear' and p['object_id']==oid)
            pred=read_npz(runs['final_apply']/probe['directory']/'predictions.npz')
            for axis,color in enumerate(('red','green','blue')):
                axes[oid+1,col].plot(t,pred['target'][selected,axis]*1000,c=color,label='GT '+str(axis))
                axes[oid+1,col].plot(t,pred['prediction'][selected,axis]*1000,c=color,ls='--',label='decoded '+str(axis))
            axes[oid+1,col].set_title(enc+(' Peg' if oid==0 else ' Box')+' TCP-relative XYZ mm');axes[oid+1,col].legend(ncol=3,fontsize=6);axes[oid+1,col].set_xlabel('Actual timestep')
    fig.tight_layout();fig.savefig(a.output/'readout_reconstruction.png',dpi=130);plt.close(fig);images.append('readout_reconstruction.png')
    write_json(a.output/'figure_index.json',dict(images=images,source='saved metrics and predictions only; no fits'))
if __name__=='__main__':main()
