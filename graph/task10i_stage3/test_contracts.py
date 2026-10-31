"""Positive and negative tests for immutable selection and episode transactions."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from .core import read,write,freeze,sha,lock,begin,commit,verified,branch_spec,allocations,verify_receipt

class Contracts(unittest.TestCase):
    def test_branch_protocol_rejects_unfrozen_offsets(self):
        self.assertEqual(branch_spec('t25_-10_0'),dict(kind='recovery',anchor=25,offset_mm=[-10,0,0]))
        for bad in ('t20_5_0','t10_20_0','t10_5_5'):
            with self.assertRaises(AssertionError):branch_spec(bad)

    def test_partial_is_never_approved_and_completed_data_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'episode';partial=begin(dest)
            (partial/'raw').write_bytes(b'partial')
            self.assertIsNone(verified(dest))
            partial=begin(dest)
            self.assertEqual(len(list((dest.parent/'interrupted').iterdir())),1)
            (partial/'raw').write_bytes(b'complete')
            receipt=commit(partial,dest,dict(approved=True))
            self.assertEqual(verified(dest),receipt)
            with self.assertRaises(AssertionError):commit(begin(Path(tmp)/'other'),dest,{})
            (dest/'raw').write_bytes(b'corruption')
            with self.assertRaises(AssertionError):verified(dest)

    def test_actual_process_crash_preserves_uncommitted_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            code='from pathlib import Path; import os; from graph.task10i_stage3.core import begin; p=begin(Path('+repr(tmp)+')/"episode"); (p/"frame").write_bytes(b"uncommitted"); os._exit(91)'
            p=subprocess.run([sys.executable,'-c',code])
            self.assertEqual(p.returncode,91)
            self.assertIsNone(verified(Path(tmp)/'episode'))
            begin(Path(tmp)/'episode')
            self.assertEqual(len(list((Path(tmp)/'interrupted').iterdir())),1)

    def test_freeze_and_exclusive_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'identity.json';freeze(p,{'a':1});freeze(p,{'a':1})
            with self.assertRaises(AssertionError):freeze(p,{'a':2})
            with lock(Path(tmp)/'lock'):
                with self.assertRaises(BlockingIOError):
                    with lock(Path(tmp)/'lock'):pass

    def test_real_stage2_manifest_templates(self):
        root=Path(__file__).resolve().parents[2]/'outputs/task10i/stage2/run'
        if not root.exists():
            from .core import STAGE2
            root=STAGE2
        receipt=verify_receipt(root);plan=allocations(root)
        self.assertEqual(len(plan),90)
        for dataset,count in [('R',190),('D',190),('validation',25)]:
            rows=[a for p in plan.values() for a in p['allocations'] if a['dataset']==dataset]
            self.assertEqual(sum(len(a['branch_templates']) for a in rows),count)
        # R/D intersection can have distinct templates; preserve both memberships.
        for p in plan.values():
            self.assertEqual(set(p['branches']),{b['branch'] for a in p['allocations'] for b in a['branch_templates']})
            for b in p['branches']:branch_spec(b)

    def test_frozen_reserve_order_shared_common_and_tail_roles(self):
        from . import run_pipeline as runner
        stage2=Path(__file__).resolve().parents[2]/'outputs/task10i/stage2/run'
        if not stage2.exists():
            from .core import STAGE2
            stage2=STAGE2
        plan=allocations(stage2);frozen=read(stage2/'stage3_demonstration_plan.json')['per_parent_allocations']
        common=read(stage2/'sampling_trace.json')['common'][:44]
        R={a['new_parent'] for a in frozen if a['dataset']=='R'};D={a['new_parent'] for a in frozen if a['dataset']=='D'}
        shared_tail=next(iter((R&D)-set(common)))
        val=next(a['new_parent'] for a in frozen if a['dataset']=='validation')
        failures={common[0],shared_tail,val}
        queues=read(stage2/'stage3_collection_reserve_protocol.json')['queues']
        catalog={cid:dict(id=cid,seed=i) for i,cid in enumerate(set(sum(queues.values(),[]))|set(plan))}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def fake_request(out,parents,workers,label,crash=None):
                for p in parents:
                    cid=p['candidate']['id'];dest=out/'parents'/cid/'parent_result.json'
                    if dest.exists():continue
                    write(dest,dict(status='FAIL' if cid in failures else 'PASS',candidate_id=cid,seed=p['candidate']['seed'],
                        reason='EXPERT_INSERTION_FAILURE' if cid in failures else None,branches=p['branches']))
                return {}
            with patch.object(runner,'STAGE2',stage2),patch.object(runner,'request',fake_request),patch.object(runner,'candidate_catalog',lambda:catalog):
                mapping=runner.collect(root,plan,3)
                self.assertEqual(mapping['R'][common[0]],mapping['D'][common[0]])
                self.assertNotIn(mapping['R'][common[0]],R|D)
                for ds,n in [('R',63),('D',63),('validation',9)]:
                    self.assertEqual(len(set(mapping[ds].values())),n)
                    self.assertFalse(set(mapping[ds].values())&failures)
                self.assertIn(mapping['validation'][val],queues['validation'])
                self.assertIn(mapping['R'][shared_tail],queues['R'])
                self.assertIn(mapping['D'][shared_tail],queues['D'])
                events=read(root/'reserve_events.json')
                assigned=[e for e in events if e['event']=='ASSIGNED']
                self.assertEqual([e['role'] for e in assigned],['validation','common','R','D'])
                # Replay durable assignments, preserving precisely the same mappings.
                self.assertEqual(runner.collect(root,plan,3),mapping)

if __name__=='__main__':unittest.main()
