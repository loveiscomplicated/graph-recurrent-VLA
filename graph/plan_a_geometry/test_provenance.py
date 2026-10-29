import unittest,numpy as np
from graph.plan_a_geometry.provenance import unique_sources
class Provenance(unittest.TestCase):
 def source(self):return dict(node_row=np.array([0,0]),episode=np.array(['e','e']),timestep=np.array([0,0]),camera=np.array(['main','main']),pixel=np.array([7,7]),structural=np.array([True]))
 def test_same_source_support_is_unique(self):
  source=self.source();result,n=unique_sources(source,np.array([0]));self.assertEqual(n,1);self.assertEqual(result['pixel'].tolist(),[7]);self.assertEqual(result['node_row'].tolist(),[0]);self.assertEqual(source['pixel'].tolist(),[7,7])
 def test_cross_node_source_conflict_rejected(self):
  source=self.source();source['node_row'][1]=1
  with self.assertRaises(ValueError):unique_sources(source,np.array([0,0]))
 def test_shared_auxiliary_reserved_for_actual_primary(self):
  source=dict(node_row=np.array([0,0,1]),episode=np.array(['e','e','e']),timestep=np.array([0,0,0]),camera=np.array(['main','main','main']),pixel=np.array([7,8,8]),structural=np.array([True,False]))
  result,n=unique_sources(source,np.array([0,0]));self.assertEqual(n,1);self.assertEqual(result['node_row'].tolist(),[0,1]);self.assertEqual(result['pixel'].tolist(),[7,8])
if __name__=='__main__':unittest.main()
