import unittest
from unittest.mock import patch
from common import *
from adapters import normalize, payload, call

class Contracts(unittest.TestCase):
    def test_coordinates_and_strict_format(self):
        p = strict_parse('{"peg":[0,100,1000,900],"box":null}')
        self.assertEqual(p['peg']['pixels'],[0,51.2,512,460.8])
        self.assertEqual(p['box']['status'],'NULL')
        for b in [[False,1,2,3],[-1,0,2,3],[0,0,float('inf'),3],[1,0,1,3],[3,4,2,5]]:
            self.assertEqual(validate_box(b)['status'],'INVALID_BOX')
        for text in ['```json\n{"peg":null,"box":null}\n```',
                     '{"peg":null,"peg":null,"box":null}', '{"peg":null}',
                     '{"peg":[0,0,NaN,2],"box":null}']:
            self.assertFalse(strict_parse(text)['json_valid'])
        self.assertTrue(diagnostic_parse('```json\n{"peg":[0,0,3,4],"box":null}\n```')['json_valid'])

    def test_final_answer_not_reasoning(self):
        good='{"peg":[0,0,10,10],"box":null}'
        a=normalize('openai',dict(model='gpt-6-astra',status='completed',output=[
            {'type':'reasoning','summary':[{'text':'wrong'}]},
            {'type':'message','content':[{'type':'output_text','text':good}]}],
            usage={'input_tokens':20,'output_tokens':200,'output_tokens_details':{'reasoning_tokens':180}}),200)
        self.assertEqual(a['final_text'],good)
        z=normalize('zai',{'choices':[{'message':{'content':good,'reasoning_content':'wrong'},'finish_reason':'stop'}]},200)
        o=normalize('ollama',{'model':'gemma4:31b','message':{'content':good,'thinking':'wrong'},'done':True},200)
        self.assertEqual(z['final_text'],good)
        self.assertEqual(o['final_text'],good)
        self.assertEqual(normalize('openai',{'status':'incomplete','incomplete_details':{'reason':'max_output_tokens'}},200)['status'],'TOKEN_LIMIT')
        self.assertEqual(normalize('ollama',{'error':'model not found'},404)['status'],'MODEL_UNAVAILABLE')

    def test_payload_is_single_rgb_no_privilege(self):
        for p in MODELS:
            data=payload(p,PROMPT,b'png')
            self.assertEqual(len(data.get('input',data.get('messages'))),1)
            encoded=json.dumps(data)
            for forbidden in ['GT_path','object_ids','DINO','parent_id','tools','previous_response_id']:
                self.assertNotIn(forbidden,encoded)
            self.assertNotIn('format',data)
            self.assertEqual(data['model'],MODELS[p]['model'])

    def test_budget_reservation_and_unknown_usage(self):
        maximum=sum(budget_reservation(p,PROMPT)*60 for p in MODELS)
        self.assertLessEqual(maximum,20)
        self.assertIsNone(used_cost('openai',None))
        jobs={'a':{'reserved_usd':.3},'b':{'reserved_usd':.3,'accounted_usd':.01}}
        self.assertAlmostEqual(ledger_total(jobs),.31)
        # Durable RESERVED entries are already issued intents, not retry candidates.
        self.assertIn('a',jobs)

    def test_single_network_attempt(self):
        import requests
        with patch('adapters.requests.post',side_effect=requests.Timeout()) as post:
            result=call('openai',{},'test-not-a-real-key')
        self.assertEqual(post.call_count,1)
        self.assertEqual(result['status'],'TIMEOUT')
        self.assertNotIn('test-not-a-real-key',json.dumps(result))

    def test_mask_success_wrong_actor_and_missing_are_distinct(self):
        import numpy as np
        from evaluate import score_mask, box_iou
        gt=np.zeros((10,10),bool);gt[1:5,1:5]=True
        robot=np.zeros_like(gt);robot[6:10,6:10]=True
        truth={'peg':gt,'box':np.zeros_like(gt),'robot':robot,'background':~(gt|robot)}
        perfect=score_mask(gt,truth,'peg')
        self.assertTrue(perfect['mask08']);self.assertFalse(perfect['wrong_object'])
        missing=score_mask(np.zeros_like(gt),truth,'peg')
        self.assertFalse(missing['mask08']);self.assertFalse(missing['wrong_object'])
        wrong=score_mask(robot,truth,'peg')
        self.assertTrue(wrong['wrong_object']);self.assertEqual(wrong['mask_IoU'],0)
        self.assertEqual(box_iou([0,0,2,2],[1,1,3,3]),1/7)

if __name__ == '__main__': unittest.main()
