"""Synthetic artifact readback test. Results are not research evidence."""
import tempfile,unittest,json
from pathlib import Path
import cv2
import numpy as np
from graph.handoff10h.analysis10h import analyze
from graph.plan_a_bc.common import write,sha

class NoPublishingLog:
    run_id='SYNTHETIC_FIXTURE_ONLY'
    def link_output(self,*args,**kwargs):
        if kwargs.get('kind') not in ('metrics','reports'):raise ValueError('unknown artifact kind')

class AnalysisTests(unittest.TestCase):
    def test_complete_grid_readback_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write(root/'evaluation_seed_manifest.json',dict(final=list(range(108200,108212))))
            rows=[]
            for seed in range(108200,108212):
                for c in ('E','B0','H1','H2'):
                    relative=f'rollouts/final/seed{seed}/{c}';folder=root/relative;folder.mkdir(parents=True)
                    tr=[]
                    for t in range(2):
                        tr.append(dict(timestep=t,phase='insert',control='B' if c!='E' else 'Expert',tip_hole_error_m=.01,peg_hole_orientation_error_rad=.02,missing_objects=[False,False],action_difference_mae=.1,action_direction_error=False,arm_saturation=False,action=[0,0,0,0,0,0,-1],tcp_pose=[0,0,0,1,0,0,0],inference_seconds=.1,policy_seconds=.01,gnn_encoder_readout_seconds=.001,simulator_step_wall_seconds=.002,target_error_change_m=-.001,pre=dict(grasp=True),grasp=True))
                    write(folder/'trace.json',tr);write(folder/'anchors.json',[])
                    v=cv2.VideoWriter(str(folder/'rollout.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(64,64))
                    for _ in range(2):v.write(np.zeros((64,64,3),np.uint8))
                    v.release()
                    rows.append(dict(stage='final',seed=seed,condition=c,valid_anchor=c in ('H1','H2'),grasp=True,lift=True,alignment=True,insertion=False,folder=relative,actions=2,trace_sha256=sha(folder/'trace.json'),video_sha256=sha(folder/'rollout.mp4'),failure_labels=['TIMEOUT'],wall_seconds=.3))
            analyze(root,rows,NoPublishingLog())
            self.assertEqual(json.loads((root/'completion_audit.json').read_text())['final_episodes'],48)
            report=(root/'10h_phase_conditioned_handoff_diagnosis.md').read_text()
            self.assertIn('Q10.',report)
            self.assertTrue((root/'handoff_first10_action_metrics.csv').exists())

if __name__=='__main__':unittest.main()
