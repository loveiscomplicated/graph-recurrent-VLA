import unittest
from unittest.mock import patch
from common import *
from adapters import payload,normalize,call

class Contracts(unittest.TestCase):
    def test_same_openai_settings_and_isolation(self):
        text='{"peg":[0,0,100,100],"box":null}'
        for provider in MODELS:
            d=payload(provider,PROMPT,b'png')
            self.assertEqual(d['reasoning'],{'effort':'high'})
            self.assertEqual(d['max_output_tokens'],5120)
            self.assertEqual(d['input'][0]['content'][0]['detail'],'original')
            self.assertEqual(len(d['input']),1)
            for forbidden in ['GT_path','object_ids','DINO','parent_id','tools','previous_response_id']:
                self.assertNotIn(forbidden,json.dumps(d))
            body=dict(model=MODELS[provider]['model'],status='completed',output=[
                {'type':'reasoning','summary':[{'text':'wrong'}]},
                {'type':'message','content':[{'type':'output_text','text':text}]}])
            self.assertEqual(normalize(provider,body,200)['final_text'],text)
    def test_budget_and_retry(self):
        self.assertLess(PRIOR_ACCOUNTED_USD+sum(budget_reservation(p,PROMPT)*60 for p in MODELS),20)
        import requests
        with patch('adapters.requests.post',side_effect=requests.Timeout()) as post:result=call('terra',{},'fake')
        self.assertEqual(post.call_count,1);self.assertEqual(result['status'],'TIMEOUT')
    def test_strict_fence_coordinates(self):
        text='{"peg":[0,100,1000,900],"box":null}'
        self.assertEqual(strict_parse(text)['peg']['pixels'],[0,51.2,512,460.8])
        self.assertTrue(fence_parse('```json\n'+text+'\n```')['json_valid'])
        self.assertFalse(strict_parse('```json\n'+text+'\n```')['json_valid'])
        self.assertFalse(fence_parse('Answer: '+text)['json_valid'])
        for value in [[True,0,2,3],[-1,0,2,3],[2,0,1,3],[0,0,1001,3]]:
            self.assertEqual(validate_box(value)['status'],'INVALID_BOX')

if __name__=='__main__':unittest.main()
