"""Verify scoring categories and denominators, not implementation parity."""
import os
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
os.environ.setdefault('TASK10I_ROOT',str(ROOT/'outputs/task10i/stage4a2/resolution_perception_graph_20261103_v1'))
sys.path.insert(0,str(ROOT/'graph/experiments/resolution_perception_graph_20261103_v1'))
sys.path.insert(0,str(Path(__file__).parent))
from score import mask_counts,node_info,graph_counts


class Scores(unittest.TestCase):
    def setUp(self):
        self.gt=[np.array([[1,1],[0,0]],bool),np.zeros((2,2),bool),np.array([[0,0],[1,0]],bool)]

    def test_valid_mask_correct_robot_and_near_denominators(self):
        values=mask_counts(np.ones((2,2),bool),np.ones((2,2),bool),self.gt,0,np.array([[1,0],[0,0]],bool))
        self.assertEqual((values['total'],values['correct'],values['robot'],values['background']),(4,2,1,1))
        self.assertEqual((values['precision'],values['recall'],values['correct_near_finger']),(.5,1.,1))
        empty=mask_counts(np.zeros((2,2),bool),np.ones((2,2),bool),self.gt,0,np.zeros((2,2),bool))
        self.assertIsNone(empty['precision']);self.assertEqual(empty['recall'],0.)

    def test_representative_and_attached_sources_are_separate(self):
        records=[dict(object_id=0,world_xyz=[0.,0.,0.],sources=[['e',0,'base',0],['e',0,'base',2]])]
        result=node_info(records,{'base':self.gt},{},np.array([[1,1,1],[0,0,.01],[2,2,2]]))
        self.assertEqual(result[0]['category'],'correct');self.assertEqual(result[0]['attached'],['correct','robot'])
        self.assertTrue(result[0]['near_finger'])

    def test_undirected_edges_and_isolated_nodes(self):
        graph=dict(node_type=np.array([0,1,2,3,3,3,4]),
            edge_index=np.array([[3,4,0,3],[4,3,3,0]]),edge_type=np.array([0,0,1,1]))
        out=graph_counts(graph,0)
        self.assertEqual(out['intra_undirected_edges'],1);self.assertEqual(out['connected_components'],2)
        self.assertEqual(out['robot_object_directed_edges'],2)


if __name__=='__main__':unittest.main()
