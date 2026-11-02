"""Static scientific ROC/PR figures from completed Development diagnostics."""
import argparse
import csv
from pathlib import Path
import numpy as np
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root.mkdir(parents=True,exist_ok=False)
    with metrics.open() as stream:rows=list(csv.DictReader(stream))
    modes={'full_patch448':'Full image, mask-weighted','bbox_patch448':'Box, stretched',
           'padded_patch448':'Padded box','masked_padded_patch448':'Masked padded box'}
    branches=[('ORACLE_DIAGNOSTIC','Oracle mask diagnostic'),('PREDICTION_CROP_DIAGNOSTIC','Predicted mask diagnostic')]
    coverage=[]
    for kind in ('ROC','PR'):
        fig,axes=plt.subplots(2,2,figsize=(10.5,8),sharex=True,sharey=True)
        for row,(branch,label) in enumerate(branches):
            for col,cross in enumerate(('False','True')):
                ax=axes[row,col]
                for mode,legend in modes.items():
                    values=[r for r in rows if r['mode']==mode and r['branch']==branch and r['cross_camera']==cross]
                    assert values,(branch,mode,cross)
                    values=sorted(values,key=lambda r:-float(r['threshold']))
                    coverage.append(dict(kind=kind,branch=branch,mode=mode,cross_camera=cross,
                        positive_pairs=int(values[0]['positives']),negative_pairs=int(values[0]['negatives'])))
                    if kind=='ROC':
                        x=np.array([float(r['fpr']) for r in values]);y=np.array([float(r['tpr']) for r in values])
                    else:
                        values=[r for r in values if r['precision']!='']
                        x=np.array([float(r['tpr']) for r in values]);y=np.array([float(r['precision']) for r in values])
                    assert np.isfinite(x).all() and np.isfinite(y).all()
                    assert (np.diff(x)>=-1e-12).all()
                    if kind=='ROC':assert (np.diff(y)>=-1e-12).all()
                    ax.plot(x,y,label=legend,lw=1.6)
                ax.set_title(label+' / '+('cross camera' if cross=='True' else 'same camera'),fontsize=10)
                ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.2)
                ax.set_xlabel('False positive rate' if kind=='ROC' else 'Recall')
                ax.set_ylabel('True positive rate' if kind=='ROC' else 'Pair-set precision')
                if kind=='ROC':ax.plot([0,1],[0,1],'--',color='gray',lw=.8)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=9)
        fig.suptitle('DINOv2 Small: fixed448 crop/patch diagnostic '+kind,fontsize=13)
        fig.text(.5,.065,'Development only. Correlated physical-actor pairs; descriptive curves, not population CIs. No Final fitting.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.10,1,.95))
        for extension in ('png','svg'):fig.savefig(root/('dinov2_crop_'+kind+'.'+extension),dpi=160)
        plt.close(fig)
    write(root/'verification.json',dict(status='DESCRIPTIVE_DEVELOPMENT_MATCHING_CURVES_RENDERED',
        source_sha256=sha(Path(__file__)),metrics=str(metrics),metrics_sha256=sha(metrics),groups=coverage,
        model_forwards=0,threshold_selected=False,Final_data_read=False,pair_dependence_preserved_as_limitation=True,
        runtime_margin_and_geometry_guards_not_represented=True,files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','metrics'):p.add_argument(name,type=Path)
    run(**vars(p.parse_args()))
