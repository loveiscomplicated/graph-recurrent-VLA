"""Synthetic integration test, never opens the real evaluation GT or APIs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import common
import evaluate
import verify

class Log:
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def log(self,*a):pass
    def link_output(self,*a):pass

class EvaluationPipeline(unittest.TestCase):
    def test_perfect_wrong_robot_and_invalid_output_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            r=Path(folder)
            seg=np.zeros((512,512),np.uint16)
            seg[10:14,20:24]=18;seg[30:34,40:44]=19;seg[50:54,60:64]=1
            gt=r/'gt.npz';np.savez_compressed(gt,actor_segmentation=seg)
            c=dict(id='synthetic',parent_id='synthetic_parent',camera='base_camera',frame_kind='F0',timestep=0,
                   GT_path=str(gt),GT_sha256=common.sha(gt),object_ids={'peg':[18],'box':[19],'robot':[1]})
            common.write(r/'evaluation_manifest.json',{'images':[c]})
            predictions=[];masks=[];jobs={}
            for p in common.MODELS:
                if p=='openai':
                    parsed=common.strict_parse('{"peg":[39.0625,19.53125,46.875,27.34375],"box":[78.125,58.59375,85.9375,66.40625]}')
                elif p=='zai':
                    parsed=common.strict_parse('{"peg":[117.1875,97.65625,125,105.46875],"box":null}')
                else:parsed=common.strict_parse('invalid output')
                predictions.append(dict(provider=p,image_id=c['id'],status='OK',attempted=True,parsed=parsed))
                jobs[p+'__synthetic']=dict(image_id=c['id'],status='OK',parsed=parsed,accounted_usd=.001)
                for obj in ['peg','box']:
                    if parsed[obj]['status']!='VALID':continue
                    a=(seg==({'peg':18,'box':19}[obj] if p=='openai' else 1))
                    path=r/'sam2/masks'/(p+'_'+obj+'.png');path.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(a.astype(np.uint8)*255).save(path)
                    masks.append(dict(provider=p,image_id=c['id'],object=obj,
                                      mask_file='masks/'+path.name,mask_sha256=common.sha(path)))
            common.write(r/'full_image_grounding_predictions.json',{'rows':predictions})
            common.write(r/'sam2/sam2_predictions.json',{'rows':masks})
            common.write(r/'request_ledger.json',{'jobs':jobs})
            common.write(r/'verification.json',dict(stage='MASKS_FROZEN',protocol_files={},source_before={},
                api_files={'full_image_grounding_predictions.json':common.sha(r/'full_image_grounding_predictions.json')},
                raw_files={},sam2_files={'sam2/sam2_predictions.json':common.sha(r/'sam2/sam2_predictions.json')},
                protected_before={str(gt):common.sha(gt)}))
            with patch.object(common,'ROOT',r),patch.object(evaluate,'ROOT',r),\
                 patch.object(verify,'ROOT',r),patch.object(evaluate,'logger',lambda s:Log()):
                evaluate.main();verify.main()
                result=common.read(r/'grounding_evaluation_metrics.json')['models']
                self.assertEqual(result['openai']['overall']['scene']['both_mask08']['rate'],1)
                self.assertEqual(result['zai']['overall']['peg']['wrong_object']['rate'],1)
                self.assertEqual(result['zai']['overall']['scene']['both_mask08']['rate'],0)
                self.assertEqual(result['ollama']['overall']['peg']['missing']['rate'],1)
                self.assertEqual(common.read(r/'independent_verification.json')['object_metrics_recomputed'],6)

if __name__=='__main__':unittest.main()
