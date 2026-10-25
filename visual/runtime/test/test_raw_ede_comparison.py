"""Raw structural receiver vs separate algorithm oracle; no real feature rerun."""
import ast
from contextlib import ExitStack
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from graph.vision_input import receive_vision_packet, RAW_EDE_POLICY
from visual.runtime.extended_delaunay_edges import candidate_edges, directed_edges
from visual.runtime.test.raw_ede_comparison_adapter import build_raw_ede
from visual.runtime.test.run_raw_ede_comparison import offline_oracle, execute, io, c


def packet():
    return io.read_packet(io.BASE/'packets/f0000_r0.5.npz')


def raw_packet(p=None):
    p=packet() if p is None else copy.deepcopy(p)
    edge,metadata,a=build_raw_ede(p);p.update(metadata,intra_object_edge_index=edge)
    return p,a


def forged(p,pairs,origins):
    p=copy.deepcopy(p);p.update(raw_candidate_pairs=pairs.copy(),raw_candidate_origins=list(origins),
        raw_candidate_source_pairs=p['provenance']['source_index'][pairs].copy(),
        raw_candidate_object_id=p['object_id'][pairs[:,0]].copy(),
        raw_candidate_component_id=p['observed_component_id'][pairs[:,0]].copy(),
        intra_object_edge_index=directed_edges(pairs))
    return p


class RawEDETests(unittest.TestCase):
    def test_policy_explicit_and_no_test_import_in_receiver(self):
        self.assertEqual(RAW_EDE_POLICY,dict(name='raw_extended_delaunay',version=1,
            candidate_mode='extended_delaunay',edge_validation='none',evaluation_only=True))
        tree=ast.parse(Path('graph/vision_input.py').read_text())
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('visual.runtime.test') for n in ast.walk(tree)))

    def test_raw_never_calls_pixel_work_and_receiver_never_generates_candidates(self):
        p=packet()
        with ExitStack() as stack:
            for module,names in [
                ('visual.runtime.extended_delaunay_edges',('supercover','_segment_supercover_single_sat','prepare_validation','validate_cover')),
                ('visual.runtime.triangle_intra_object_edges',('supercover','prepare_validation','validate_cover','adjacent_pairs'))]:
                for name in names:stack.enter_context(patch(module+'.'+name,side_effect=AssertionError(name)))
            raw,a,t,r=execute('Raw',p)
            self.assertNotIn('checks',a);self.assertNotIn('accepted',a)
            with patch('visual.runtime.extended_delaunay_edges.candidate_edges',side_effect=AssertionError('regenerated')), \
                 patch('visual.runtime.extended_delaunay_edges.Delaunay',side_effect=AssertionError('Qhull')), \
                 patch('visual.runtime.extended_delaunay_edges.collinear_order',side_effect=AssertionError('chain')):
                receive_vision_packet(raw)

    def test_deepcopy_all_fields_and_robot_offset(self):
        p,a=raw_packet();before=copy.deepcopy(p);r=receive_vision_packet(p)
        c.exact(r.packet,p);c.exact(r.edge_index,p['intra_object_edge_index'])
        def no_alias(a,b):
            if isinstance(a,np.ndarray):self.assertFalse(np.shares_memory(a,b))
            elif isinstance(a,dict):
                for k in a:no_alias(a[k],b[k])
            elif isinstance(a,list):
                for x,y in zip(a,b):no_alias(x,y)
        no_alias(r.packet,p)
        r.packet['raw_candidate_pairs'][:]=0;r.positions[:]=0;c.exact(p,before)
        robot=np.zeros((3,3),np.float32);rr=receive_vision_packet(p,robot_positions=robot)
        c.exact(rr.edge_index,p['intra_object_edge_index']+3);c.exact(rr.point_node_index,np.arange(len(p['points']),dtype=np.int64)+3)
        for bad in (np.zeros((2,3),np.float32),np.full((3,3),np.nan,np.float32)):
            with self.assertRaises(ValueError):receive_vision_packet(p,robot_positions=bad)

    def test_structural_corruption_rejected(self):
        p,a=raw_packet();edge=p['intra_object_edge_index'];pairs=p['raw_candidate_pairs']
        mutations=[lambda q:q.update(intra_object_edge_index=edge[:,::-1]),
            lambda q:q.update(intra_object_edge_index=edge[:,1:]),
            lambda q:q.update(intra_object_edge_index=np.c_[edge,edge[:,:1]]),
            lambda q:q['intra_object_edge_index'].__setitem__((0,0),len(p['points'])),
            lambda q:q.update(raw_candidate_pairs=pairs[::-1]),
            lambda q:q.update(raw_candidate_pairs=np.r_[pairs,pairs[:1]]),
            lambda q:q['raw_candidate_pairs'].__setitem__((0,1),pairs[0,0]),
            lambda q:q['raw_candidate_origins'].__setitem__(0,'unknown'),
            lambda q:q.update(raw_candidate_origins=[]),
            lambda q:q['raw_candidate_source_pairs'].__setitem__((0,0),0),
            lambda q:q['raw_candidate_object_id'].__setitem__(0,99),
            lambda q:q['raw_candidate_component_id'].__setitem__(0,99),
            lambda q:q['observed_component_id'].__setitem__(pairs[0,0],999),
            lambda q:q['provenance']['object_id'].__setitem__(0,1-int(q['object_id'][0])),
            lambda q:q['points'].__setitem__((0,0),999),
            lambda q:q['edge_policy'].update(evaluation_only=False)]
        for mutate in mutations:
            q=copy.deepcopy(p);mutate(q)
            with self.subTest(mutation=mutations.index(mutate)),self.assertRaises(ValueError):receive_vision_packet(q)
        rows=[np.flatnonzero(p['object_id']==o)[0] for o in (0,1)]
        cross=forged(p,np.sort(np.array([rows],np.int64),axis=1),['delaunay_edge'])
        with self.assertRaises(ValueError):receive_vision_packet(cross)

    def test_receiver_trust_limit_and_offline_missing_chord_origin_chain_oracle(self):
        p,a=raw_packet();pairs=a['candidates'];origins=a['origins'].tolist()
        variants=[forged(p,pairs[1:],origins[1:])]
        # An arbitrary chord in the same saved component is structurally valid.
        for status in a['components']:
            rows=status['rows']
            if len(rows)>3:
                existing=set(map(tuple,pairs.tolist()))
                chord=next(((min(i,j),max(i,j)) for i in rows for j in rows if i!=j and (min(i,j),max(i,j)) not in existing),None)
                if chord:
                    new=np.unique(np.r_[pairs,np.array([chord],np.int64)],axis=0)
                    mapping=dict(zip(map(tuple,pairs.tolist()),origins));variants.append(forged(p,new,[mapping.get(tuple(x),'delaunay_edge') for x in new]));break
        bad=copy.deepcopy(p);bad['raw_candidate_origins'][0]='collinear_chain';variants.append(bad)
        # Actual plateau chain: replace adjacent pair with a shortcut.
        st=next(s for s in a['components'] if s['object_id']==1 and s['component_id']==20)
        chain=np.asarray(st['chain_rows']);shortcut=np.sort(chain[[0,-1]]).reshape(1,2)
        variants.append(forged(p,shortcut,['collinear_chain']))
        self.assertEqual(len(variants),4)
        for bad in variants:
            receive_vision_packet(bad)
            with self.assertRaises(AssertionError):offline_oracle(bad)

    def test_candidate_corner_cases_preserved(self):
        for sources,components,count in [([],[],0),([2],[0],0),([2,5],[0,0],1),
                ([2,5],[0,1],0),([2,3,5],[0,0,0],2),([2,11,18],[0,0,0],3)]:
            src=np.array(sources,np.int64);oid=np.zeros(len(src),np.int64);comp=np.array(components,np.int64)
            a=candidate_edges(oid,src,comp,np.array([9,9],np.int64),extended=True)
            self.assertEqual(len(a['candidates']),count)
            self.assertEqual(directed_edges(a['candidates']).shape,(2,2*count))

    def test_fixed_source_all_16_exact(self):
        with np.load(c.SOURCE/'extended_delaunay_edgewise.npz',allow_pickle=False) as b:
            total=0
            for frame in io.FRAMES:
                for ratio in io.RATIOS:
                    key=f'f{frame:04d}_r{ratio:g}';p=io.read_packet(io.BASE/f'packets/{key}.npz');raw,a=raw_packet(p)
                    offline_oracle(raw,{f:b[key+'/EDE_'+f] for f in ('candidates','origins','triangles')})
                    receive_vision_packet(raw);total+=len(a['candidates'])
            self.assertEqual(total,30143)


if __name__=='__main__':unittest.main()
