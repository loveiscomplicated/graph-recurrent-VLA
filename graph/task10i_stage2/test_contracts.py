import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .core import (descriptor,fit,distances,partition,greedy,random_order,SEEDS,PROTOCOL,
                   WEIGHTS,freeze,canonical_hash,require_frozen)
from .evaluate import csv_write,coverage,distribution
from graph.initial_pool10i.mirror.verify import sha


def fixture(n=340):
    rng=np.random.default_rng(51);rows=[]
    for i in range(n):
        yaw=float(rng.uniform(-np.pi,np.pi));q=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        s=dict(peg_pose=np.array([rng.uniform(-.05,.05),-.04,.02]+q),
            tcp_pose=np.array([0,0,.15,1,0,0,0]),hole_pose=np.array([0,.25,.1,1,0,0,0]),
            peg_half=np.array([rng.uniform(.085,.125),.02,.02]),hole_radius=np.float32(.023),
            intrinsic=np.array([[60,0,64],[0,60,64],[0,0,1]]),extrinsic_cv=np.c_[np.eye(3),[0,0,.8]])
        rows.append(dict(id=f'fixture_{i:04d}',role='training',descriptor=descriptor(s)))
    return rows


class Contracts(unittest.TestCase):
    def setUp(self):
        self.rows=fixture(40);self.scale=fit(self.rows,[r['id'] for r in self.rows])

    def test_distance_identity_symmetry_finite_nonnegative(self):
        d,g=distances(self.rows,self.rows,self.scale)
        np.testing.assert_array_equal(np.diag(d),np.zeros(len(d)))
        np.testing.assert_allclose(d,d.T,atol=1e-14,rtol=0)
        self.assertTrue(np.isfinite(d).all() and (d>=0).all())
        self.assertEqual(g.shape,(40,40,5));self.assertAlmostEqual(WEIGHTS.sum(),1)

    def test_quaternion_sign_wraparound_and_red_white_direction(self):
        from scipy.spatial.transform import Rotation
        a=copy.deepcopy(self.rows[0]);b=copy.deepcopy(a)
        b['descriptor']['g2']=(-np.array(a['descriptor']['g2'])).tolist()
        self.assertEqual(distances([a],[b],self.scale)[0][0,0],0)
        a['descriptor']['g2']=Rotation.from_euler('z',np.pi-1e-7).as_quat().tolist()
        b['descriptor']['g2']=Rotation.from_euler('z',-np.pi+1e-7).as_quat().tolist()
        self.assertLess(distances([a],[b],self.scale)[0][0,0],1e-6)
        b['descriptor']['g2']=Rotation.from_euler('z',0).as_quat().tolist()
        self.assertGreater(distances([a],[b],self.scale)[0][0,0],.1)

    def test_degenerate_feature_scale_floors(self):
        rows=[copy.deepcopy(self.rows[0]) for _ in range(4)]
        for i,r in enumerate(rows):r['id']=str(i)
        scale=fit(rows,[r['id'] for r in rows]);d,_=distances(rows,rows,scale)
        self.assertTrue(np.isfinite(d).all());self.assertTrue((d==0).all())
        self.assertEqual(scale['g1']['scale'],[.01]*3)
        self.assertEqual(scale['g2']['scale'],PROTOCOL['floors']['g2_rad'])

    def test_holdout_positive_control_fitting_and_greedy_block(self):
        hold=copy.deepcopy(self.rows[0]);hold['role']='holdout';hold['id']='not_training'
        with self.assertRaises(ValueError):fit(self.rows+[hold],self.scale['fit_ids'])
        with self.assertRaises(ValueError):fit([hold],['not_training'])
        with self.assertRaises(ValueError):greedy(self.rows+[hold],self.rows[:2],self.scale,2)
        with self.assertRaises(ValueError):greedy(self.rows,[hold],self.scale,2)
        hold['role']='training'  # Relabeling alone cannot bypass frozen fit membership.
        with self.assertRaises(ValueError):greedy(self.rows+[hold],self.rows[:2],self.scale,2)

    def test_uniform_split_and_random_without_replacement(self):
        ids=[r['id'] for r in fixture(300)];s=partition(ids)
        self.assertEqual(s,partition(list(reversed(ids))))
        self.assertEqual([len(s[k]) for k in ['validation','holdout','training']],[20,60,220])
        self.assertEqual(len(set(s['full_sampling_trace'])),300)
        order=random_order(s['training'],SEEDS['common'])
        self.assertEqual(order,random_order(s['training'],SEEDS['common']))
        self.assertEqual(len(set(order)),220)

    def test_greedy_monotonic_tie_break_and_reserve(self):
        a=greedy(self.rows,self.rows[:4],self.scale)
        self.assertEqual(a,greedy(self.rows[::-1],self.rows[:4],self.scale))
        self.assertTrue(all(t['objective_after']<=t['objective_before'] for t in a))
        self.assertEqual(len(a),36);self.assertEqual(len({t['id'] for t in a}),36)
        rows=copy.deepcopy(self.rows[:5])
        for r in rows:r['descriptor']=copy.deepcopy(rows[0]['descriptor'])
        self.assertEqual(greedy(rows,rows[:1],self.scale,2)[0]['id'],rows[1]['id'])

    def test_full_dataset_counts_and_common_identity(self):
        rows=fixture();new=rows[:300];old=rows[300:333];val0=rows[333:]
        for r in old:r['role']='existing_train'
        s=partition([r['id'] for r in new]);lookup={r['id']:r for r in rows}
        tr=[lookup[i] for i in sorted(s['training'])];scale=fit(old+tr,[r['id'] for r in old+tr])
        common=random_order(s['training'],SEEDS['common'])[:44]
        rt=random_order(set(s['training'])-set(common),SEEDS['random'])[:19]
        base=old+[lookup[i] for i in common];trace=greedy(tr,base,scale)
        d={r['id'] for r in base}|{t['id'] for t in trace[:19]};rr={r['id'] for r in base}|set(rt)
        v={r['id'] for r in val0}|set(s['validation'][:9])
        self.assertEqual((len(rr),len(d),len(v)),(96,96,16))
        self.assertFalse((rr|d)&(v|set(s['holdout'])))
        self.assertEqual(len(trace[19:]),157)

    def test_miniature_end_to_end_load_freeze_report_readback(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);freeze(out/'input.json',self.rows)
            rows=json.loads((out/'input.json').read_text());s=partition([r['id'] for r in rows],3,6)
            lookup={r['id']:r for r in rows};tr=[lookup[i] for i in s['training']]
            scale=fit(tr,[r['id'] for r in tr]);base=tr[:4]
            trace=greedy(tr,base,scale,3);R=base+tr[4:7];D=base+[lookup[t['id']] for t in trace]
            self.assertEqual(len({r['id'] for r in D}),7)
            freeze(out/'manifest.json',dict(R=[r['id'] for r in R],D=[r['id'] for r in D],split=s))
            freeze(out/'selection_identity.json',dict(files={'manifest.json':sha(out/'manifest.json')}))
            result,groups,nearest=coverage(out,[lookup[i] for i in s['holdout']],dict(R=R,D=D),scale,'holdout')
            csv_write(out/'metrics.csv',result)
            (out/'report.md').write_text(json.dumps(result))
            (out/'report.html').write_text('<pre>'+json.dumps(result)+'</pre>')
            require_frozen(out)
            reread=json.loads((out/'manifest.json').read_text())
            self.assertEqual(reread['D'],[r['id'] for r in D])
            freeze(out/'manifest.json',reread)
            with self.assertRaises(ValueError):freeze(out/'manifest.json',dict(changed=True))
            (out/'manifest.json').write_text('{}')
            with self.assertRaises(ValueError):require_frozen(out)

    def test_coverage_forbidden_before_manifest_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):coverage(Path(temp),self.rows,dict(R=self.rows),self.scale,'holdout')

    def test_distribution_full_output_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            boundary,extreme=distribution(out,dict(candidate300=self.rows,R=self.rows[:19],D=self.rows[19:]),self.rows)
            self.assertEqual(len(boundary),3)
            self.assertTrue(all(r['n']>0 for r in boundary))
            for name in ['initial_distribution_summary.csv','distribution_shift_metrics.csv','boundary_margin_comparison.csv']:
                import csv
                with (out/name).open() as f:self.assertTrue(list(csv.DictReader(f)))

    def test_production_report_and_gallery_with_miniature_fixture(self):
        from .report import generate
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);rows=fixture(16)
            for i,r in enumerate(rows):
                r.update(kind='new' if i<12 else 'existing',seed=i,expert_status='NOT_TESTED',
                    source=dict(pool='/fixture',export=dict(initial_npz='raw.npz',gt_visibility_evidence='evidence.npz',snapshot='snapshot.pkl'),
                        first_observations='/fixture/old.npz',visibility_evidence='/fixture/old_evidence.npz'))
            for r in rows[12:15]:r['role']='existing_train'
            rows[15]['role']='existing_validation'
            byid={r['id']:r for r in rows};states={r['id']:dict(rgb=np.zeros((128,128,3),np.uint8)) for r in rows}
            sets=dict(existing33=rows[12:15],base77=rows[12:15]+rows[:4],R=rows[12:15]+rows[:7],D=rows[12:15]+rows[:4]+rows[7:10])
            csv_write(out/'common_random44.csv',[dict(parent_id=r['id']) for r in rows[:4]])
            freeze(out/'selection_identity.json',dict(files={'common_random44.csv':sha(out/'common_random44.csv')}))
            metrics,groups,near=coverage(out,rows[:12],sets,self.scale,'selection')
            csv_write(out/'nearest_set_distances.csv',near+[dict(r,cohort='holdout') for r in near])
            csv_write(out/'random_monte_carlo_metrics.csv',[dict(cohort=k,mean=.2) for k in ['selection','holdout']])
            names=['candidate300','existing33','R_tail19','D_tail19','R','D']
            boundary,extreme=distribution(out,{k:rows[:8] for k in names},rows[:12])
            mc={k:dict(mean=.2,median=.2,variance=.01,p5=.1,p95=.3,datasets={n:dict(score=.2,percentile_lower_is_better=50) for n in ['R','D']}) for k in ['selection','holdout']}
            freeze(out/'stage3_demonstration_plan.json',dict(structure={}))
            freeze(out/'existing_episode_preservation_manifest.json',dict(episodes=[]))
            summary=dict(overlap=dict(shared=7,R_exclusive=3,D_exclusive=3,tail_overlap=[],new_collection_union=10),
                selection=metrics,holdout=[dict(r,cohort='holdout') for r in metrics],monte_carlo=mc,boundary=boundary,extreme=extreme,duplicates={})
            generate(out,byid,states,sets,dict(validation=[r['id'] for r in rows[10:12]],training=[r['id'] for r in rows[:10]]),summary,'MINIATURE_ONLY')
            self.assertEqual(len(list((out/'figures').glob('*.png'))),12)
            self.assertEqual(len(list((out/'gallery').glob('*_raw.png'))),16)
            self.assertIn('MINIATURE_ONLY',(out/'10i_stage2_diversity_aware_dataset_selection.html').read_text())

    def test_resume_preserves_indexed_output_identity(self):
        import os
        from unittest.mock import patch
        from utils.experiment_logger import ExperimentLogger
        from .run_pipeline import link_result
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,WANDB_MODE='disabled'):
            root=Path(temp);p=root/'report.md';p.write_text('fixed result')
            with ExperimentLogger(team='graph',root=root/'experiments') as original:
                link_result(original,p,'reports');original_id=original.run_id
            sidecar=p.with_name(p.name+'.tracking.json');before=sha(sidecar)
            with ExperimentLogger(team='graph',root=root/'experiments') as resumed:
                link_result(resumed,p,'reports')
                self.assertEqual(resumed.metadata['source_run_id'],original_id)
            self.assertEqual(sha(sidecar),before)


if __name__=='__main__':unittest.main()
