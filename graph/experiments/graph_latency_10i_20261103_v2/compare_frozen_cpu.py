"""Compare saved final Graphs directly with prior frozen original CPU outputs."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(args):
    rows=[]
    for size in (128,512,768,1024):
        old=args.reference/args.scene/str(size)
        new=args.current/'graphs'/f'{args.scene}_{size}_r0'/'s9'
        with np.load(old/'graph.npz',allow_pickle=False) as data:reference={k:data[k].copy() for k in data.files}
        with np.load(new/'graph.npz',allow_pickle=False) as data:actual={k:data[k].copy() for k in data.files}
        for k in ('node_type','edge_index','edge_type','edge_attr'):
            np.testing.assert_array_equal(actual[k],reference[k],err_msg=k)
        a,b=actual['node_input'],reference['node_input']
        np.testing.assert_array_equal(a[:,:3],b[:,:3]);np.testing.assert_array_equal(a[:,8:],b[:,8:])
        np.testing.assert_allclose(a[:,3:8],b[:,3:8],atol=2e-5,rtol=0)
        assert json.loads((new/'sources_gate.json').read_text())['sources']==json.loads((old/'result.json').read_text())['sources']
        rows.append(dict(size=size,source_ids_exact=True,discrete_graph_exact=True,positions_exact=True,
                         feature_max_abs=float(np.max(np.abs(a[:,3:8]-b[:,3:8]),initial=0))))
    args.output.write_text(json.dumps(dict(status='PASS',reference=str(args.reference),rows=rows,feature_atol=2e-5),indent=2)+'\n')
    print(json.dumps(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',default='10i_s1_0686_t000')
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--current',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
