"""Negative controls for independent receipt validation; no remote access."""
from pathlib import Path
import tempfile
import unittest
from .collect_results import boolean, checked_path, verify_grid


class ReceiptTests(unittest.TestCase):
    def test_duplicate_episode_cannot_replace_missing_parent(self):
        rows=[dict(condition='A',step='2',seed='10')]*2
        with self.assertRaises(ValueError):verify_grid(rows,['A'],[2],[10,11])

    def test_manifest_must_stay_within_collected_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):checked_path(root,'../outside.pt')
            with self.assertRaises(ValueError):checked_path(root,'/outside.pt')
            (root/'escape').symlink_to(root.parent)
            with self.assertRaises(ValueError):checked_path(root,'escape/outside.pt')

    def test_csv_false_is_not_truthy_string(self):
        self.assertFalse(boolean('False'));self.assertTrue(boolean('True'))
        with self.assertRaises(ValueError):boolean('missing')

    def test_grid_checks_all_conditions_steps_parents(self):
        rows=[dict(condition=c,step=str(s),seed=str(p)) for c in 'AB' for s in (2,4) for p in (10,11)]
        verify_grid(rows,'AB',[2,4],[10,11])
        with self.assertRaises(ValueError):verify_grid(rows[:-1],'AB',[2,4],[10,11])


if __name__=='__main__':unittest.main()
