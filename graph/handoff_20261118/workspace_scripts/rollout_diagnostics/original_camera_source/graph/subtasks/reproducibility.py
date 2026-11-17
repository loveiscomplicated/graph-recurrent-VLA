"""Compare actual arrays in two same-seed collections (not container file bytes)."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from graph.prototype.common import write_json


def compare(first,second,output):
    first,second=Path(first),Path(second)
    a=json.loads((first/'reports/collection.json').read_text())['records']
    b=json.loads((second/'reports/collection.json').read_text())['records']
    common=sorted({r['seed'] for r in a}&{r['seed'] for r in b})
    if not common:
        raise ValueError('no shared seeds to compare')
    results=[]
    for seed in common:
        ra=next(r for r in a if r['seed']==seed);rb=next(r for r in b if r['seed']==seed)
        row=dict(seed=seed,randomization_equal=ra.get('randomization')==rb.get('randomization'),modes={})
        for key in ('raw_trajectory','converted_trajectory'):
            if key not in ra or key not in rb:
                row['modes'][key]=dict(available=False)
                continue
            with h5py.File(ra[key]) as ha,h5py.File(rb[key]) as hb:
                arrays=[]
                ha['traj_0'].visititems(lambda name,obj:arrays.append(name) if isinstance(obj,h5py.Dataset) else None)
                other_arrays=[]
                hb['traj_0'].visititems(lambda name,obj:other_arrays.append(name) if isinstance(obj,h5py.Dataset) else None)
                errors={};equal=set(arrays)==set(other_arrays)
                for name in arrays:
                    if name not in other_arrays:
                        errors[name]=None
                        continue
                    x,y=ha[f'traj_0/{name}'][:],hb[f'traj_0/{name}'][:]
                    matches=x.shape==y.shape and np.array_equal(x,y)
                    equal &= matches
                    errors[name]=0. if matches else (float(np.max(np.abs(x.astype(float)-y.astype(float)))) if x.shape==y.shape and x.size else None)
                row['modes'][key]=dict(available=True,exactly_equal=bool(equal),array_errors=errors)
        row['exactly_reproduced']=row['randomization_equal'] and all(m.get('exactly_equal',False) for m in row['modes'].values())
        results.append(row)
    payload=dict(records=results,all_exact=all(r['exactly_reproduced'] for r in results))
    write_json(output,payload)
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('first');p.add_argument('second');p.add_argument('--output',required=True)
    a=p.parse_args();result=compare(a.first,a.second,a.output);print(result['all_exact'])
    if not result['all_exact']:
        raise SystemExit(1)
