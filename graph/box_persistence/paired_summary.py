"""Post-gate common-support paired accuracy sensitivity; no reselection."""
import argparse,json
from pathlib import Path
from collections import defaultdict
from graph.integration_baseline.io import write_json
from graph.integration_baseline.geometry09b_probe import csv_write
from .summarize import read_csv,macro


def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    groups=defaultdict(dict)
    for path in a.geometry.glob('geometry/seed*/metrics.csv'):
        for r in read_csv(path):
            if r['variant'] in ('B0','B1','B2'):groups[(r['seed'],r['timestep'],r['budget'])][r['variant']]=r
    selected=[];support=[]
    for key,rs in groups.items():
        if set(rs)!=set(('B0','B1','B2')):raise ValueError('missing paired representation')
        eligible=all(rs[n]['count']>0 and rs[n]['surface_distance_mean_mm'] is not None for n in rs)
        support.append(dict(seed=key[0],timestep=key[1],budget=key[2],common_nonempty=eligible))
        if eligible:selected.extend(rs.values())
    result=macro(selected,('split','variant','budget'))
    write_json(a.output/'paired_common_support.json',result);csv_write(a.output/'paired_common_support.csv',result)
    csv_write(a.output/'paired_support_frames.csv',support)
    write_json(a.output/'paired_common_support_scope.json',dict(stage='post-gate descriptive sensitivity, no re-selection or threshold change',
        support='identical episode/timestep/budget with B0/B1/B2 all nonempty; no purity/GT visibility/error filtering',
        primary_gate_unchanged=True,independent_final=False,empty_surface_distance_is_undefined=True))
if __name__=='__main__':main()
