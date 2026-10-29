"""Post-fit evaluation strata: never label contaminated input as clean evidence."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from graph.integration_baseline.io import write_json
from graph.integration_baseline.geometry09b_probe import csv_write
from .summarize import macro

def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state=json.loads(a.state.read_text());runs={k:Path(v) for k,v in state['completed'].items()};manifest=json.loads((runs['probes']/'probe_manifest.json').read_text());rows=[];counts=[]
    for stage,geomstage in [('probes','development_geometry'),('final_apply','final_geometry')]:
        quality=json.loads((runs[geomstage]/'metrics.json').read_text())
        quality={(r['variant'],int(r['seed']),int(r['timestep']),r['condition'],int(r['object_id'])):r for r in quality if r['budget']=='cap128'}
        for probe in manifest['probes']:
            if probe['probe'] not in ('P1_readout','P3_object_slots','P4_centroid','P5_geometry_summary','RawCentroidDirect'):continue
            path=runs[stage]/probe['directory']/'predictions.npz'
            with np.load(path,allow_pickle=False) as a2:pred={k:a2[k].copy() for k in a2.files}
            groups=defaultdict(list)
            for i in range(len(pred['seed'])):
                if pred['condition'][i]!='baseline':continue
                key=(probe['variant'],int(pred['seed'][i]),int(pred['timestep'][i]),str(pred['condition'][i]),probe['object_id']);q=quality[key]
                if q['count']==0:label='EMPTY_OBJECT'
                elif not q['gt_visible_points']:label='GT_HIDDEN_NONEMPTY'
                elif q['source_object_purity']<1-1e-12 or q['fused_source_object_purity']<1-1e-12:label='SOURCE_IMPURE'
                elif q['geometry_outlier_fraction_8mm']>0:label='SOURCE_PURE_GEOMETRY_OUTLIER'
                else:label='SOURCE_PURE_VISIBLE_NO_8MM_OUTLIER'
                groups[(int(pred['seed'][i]),str(pred['split'][i]),label)].append(i)
            for (seed,split,label),indices in groups.items():
                err=np.linalg.norm(pred['prediction'][indices]-pred['target'][indices],axis=1)*1000
                rows.append(dict(stage=stage,seed=seed,split=split,variant=probe['variant'],encoder=probe['encoder'],probe=probe['probe'],decoder=probe['decoder'],object_id=probe['object_id'],input_stratum=label,frames=len(indices),mean_mm=float(err.mean()),median_mm=float(np.median(err)),p95_mm=float(np.percentile(err,95))))
    csv_write(a.output/'per_episode_input_quality_probes.csv',rows)
    aggregated=macro(rows,('stage','split','variant','encoder','probe','decoder','object_id','input_stratum'),('mean_mm','median_mm','p95_mm'))
    write_json(a.output/'input_quality_probes.json',aggregated);csv_write(a.output/'input_quality_probes.csv',aggregated)
    write_json(a.output/'scope.json',dict(stage='post-fit evaluation only',no_fit_or_selection=True,clean_definition='nonempty; GT-visible; all first and merged source labels correct; no >8mm collision outlier',GT_used_by_sampler=False,all_input_primary_tables_include_errors=True))
if __name__=='__main__':main()
