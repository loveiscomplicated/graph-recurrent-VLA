"""Synthetic GT only: check strict gating of masks obtained under fence mode."""
import sys
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
    def test_strict_fence_gating_with_same_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);seg=np.zeros((512,512),np.uint16)
            seg[10:14,20:24]=18;seg[30:34,40:44]=19;seg[50:54,60:64]=1
            gt=root/'gt.npz';np.savez_compressed(gt,actor_segmentation=seg)
            c=dict(id='synthetic',parent_id='parent',camera='base_camera',frame_kind='F0',timestep=0,
                   GT_path=str(gt),GT_sha256=common.sha(gt),object_ids={'peg':[18],'box':[19],'robot':[1]})
            common.write(root/'evaluation_manifest.json',dict(images=[c]))
            good=common.strict_parse('{"peg":[39.0625,19.53125,46.875,27.34375],"box":[78.125,58.59375,85.9375,66.40625]}')
            wrong=common.strict_parse('{"peg":[117.1875,97.65625,125,105.46875],"box":null}')
            invalid=common.strict_parse('invalid')
            strict=[];fence=[];masks=[];jobs={}
            for provider,sp,fp in [('openai',good,good),('ollama',wrong,wrong),('gemma_baseline',invalid,good)]:
                base=dict(provider=provider,image_id='synthetic',status='OK',attempted=True)
                strict.append(dict(base,parsed=sp));fence.append(dict(base,parsed=fp))
                if provider!='gemma_baseline':jobs[provider]=dict(image_id='synthetic',status='OK',parsed=sp,accounted_usd=.001)
                for obj in ['peg','box']:
                    if fp[obj]['status']!='VALID':continue
                    mask=seg==({'peg':18,'box':19}[obj] if provider!='ollama' else 1)
                    path=root/'sam2/masks'/(provider+'_'+obj+'.png');path.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(mask.astype(np.uint8)*255).save(path)
                    masks.append(dict(provider=provider,image_id='synthetic',object=obj,mask_file='masks/'+path.name,mask_sha256=common.sha(path)))
            common.write(root/'full_image_grounding_predictions.json',dict(rows=strict[:2]))
            common.write(root/'fence_grounding_predictions.json',dict(rows=fence[:2]))
            common.write(root/'baseline_replay_predictions.json',dict(strict=strict[2:],fence=fence[2:]))
            common.write(root/'sam2/sam2_predictions.json',dict(rows=masks))
            common.write(root/'request_ledger.json',dict(jobs=jobs))
            common.write(root/'verification.json',dict(stage='MASKS_FROZEN',protocol_files={},source_before={},
                api_files={},raw_files={},sam2_files={},segmentation_inputs={},protected_before={str(gt):common.sha(gt)}))
            with patch.object(common,'ROOT',root),patch.object(evaluate,'ROOT',root),patch.object(verify,'ROOT',root),patch.object(evaluate,'logger',lambda s:Log()):
                for mode in ['strict','fence']:
                    with patch.object(sys,'argv',['evaluate.py','--mode',mode]):evaluate.main();verify.main()
                    prefix='fence_' if mode=='fence' else ''
                    result=common.read(root/(prefix+'grounding_evaluation_metrics.json'))['models']
                    self.assertEqual(result['openai']['overall']['scene']['both_mask08']['rate'],1)
                    self.assertEqual(result['ollama']['overall']['peg']['wrong_object']['rate'],1)
                    self.assertEqual(result['gemma_baseline']['overall']['scene']['both_mask08']['rate'],int(mode=='fence'))

if __name__=='__main__':unittest.main()
