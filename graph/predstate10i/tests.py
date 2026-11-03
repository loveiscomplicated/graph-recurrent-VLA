"""Synthetic negative controls and report execution, excluded from research data."""
import tempfile
from pathlib import Path
import unittest
import cv2
import numpy as np
import torch
from graph.plan_a_bc.common import write,read,sha
from .model import fit_normalizer


class Contracts(unittest.TestCase):
    def test_train_constant_scales_finite(self):
        x=np.zeros((4,38),'f');x[:,36]=1
        s=fit_normalizer(x)
        z=(x-np.array(s['mean'],'f'))/np.array(s['std'],'f')
        self.assertTrue(np.isfinite(z).all())
        np.testing.assert_array_equal(z[:,36:],np.zeros((4,2)))
        self.assertEqual(s['fitted_frames'],4)

    def test_report_full_grid_execution(self):
        # Tests report code on explicitly synthetic traces, not experimental outcomes.
        from .report import summarize
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);labels=['E']+[f+v for f in 'SRJ' for v in '012'];seeds=list(range(12))
            write(root/'config.json',dict(task='SYNTHETIC_REPORT_FIXTURE_ONLY'))
            write(root/'final_seed_manifest.json',dict(conditions=labels,parents=seeds,episodes=120,representative_parent=0))
            write(root/'development_selection.json',dict(selected_step=1))
            checkpoints={}
            for f in 'SRJ':
                p=root/(f+'.pt');torch.save(dict(model={'policy.encoder.weight':torch.zeros(2,38 if f=='S' else 398)}),p)
                checkpoints[f]=dict(path=str(p))
            import shutil
            shutil.copy2(root/'S.pt',root/'best.pt')
            write(root/'comparison_policy_manifest.json',{f:checkpoints[f] for f in 'RJ'})
            for seed in seeds:
                for c in labels:
                    folder=root/'rollouts/final'/f'seed{seed}'/c;folder.mkdir(parents=True)
                    tr=[];handoff=1 if c.endswith(('1','2')) else None
                    for t in range(2):
                        teacher=c=='E' or (handoff is not None and t<handoff)
                        tr.append(dict(timestep=t,phase='insert',control='Expert' if teacher else 'B',
                            handoff_relative_step=t-handoff if handoff is not None else None,
                            peg_head_pos_at_hole=[-.1,0,0],pre=dict(peg_head_pos_at_hole=[-.1,0,0],grasp=True,tcp_peg_distance_m=.01),
                            grasp=True,arm_saturation=False,action=[0,0,0,0,0,0,-1],action_difference_mae=.1,
                            action_direction_error=False,missing_objects=[False,False],success=False,
                            inference_seconds=.1,policy_seconds=.01,gnn_encoder_readout_seconds=.001,simulator_step_wall_seconds=.002,
                            state_prediction_errors={'peg_position':1.,'hole_position':2.,'peg_tip_hole':3.} if c.startswith('S') or c=='E' else {}))
                    write(folder/'trace.json',tr)
                    v=cv2.VideoWriter(str(folder/'rollout.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(64,64))
                    for _ in tr:v.write(np.zeros((64,64,3),np.uint8))
                    v.release()
                    write(folder/'episode.json',dict(seed=seed,condition=c,folder=str(folder.relative_to(root)),
                        grasp=True,lift=True,alignment=False,insertion=False,actions=2,handoff_timestep=handoff,
                        valid_anchor=handoff is not None,completion_steps=None,failure_labels=['TIMEOUT'],
                        trace_sha256=sha(folder/'trace.json'),video_sha256=sha(folder/'rollout.mp4'),hole_radius=[.023]))
            summarize(root)
            self.assertTrue((root/'report.html').is_file())
            self.assertTrue((root/'docs/visual-graph-integration/10i_predicted_state38_graph_recurrent_bc.md').is_file())
            self.assertEqual(read(root/'paired_summary.json')['denominator_per_condition'],12)
            self.assertFalse((root/'PIPELINE_COMPLETE').exists())


if __name__=='__main__':unittest.main()
