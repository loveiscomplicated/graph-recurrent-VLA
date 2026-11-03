import unittest
from unittest.mock import patch
from common import *
from adapters import payload, normalize, call

GOOD='{"peg":[0,100,1000,900],"box":null}'

class Contracts(unittest.TestCase):
    def test_fence_rule_is_bounded_and_does_not_repair(self):
        self.assertTrue(fence_parse(GOOD)['json_valid'])
        self.assertTrue(fence_parse(' ```json\n'+GOOD+'\n```\n')['json_valid'])
        for text in ['Answer: '+GOOD,'reason\n```json\n'+GOOD+'\n```',
                     '```python\n'+GOOD+'\n```','```json\n'+GOOD+'\n```\nmore',
                     GOOD+GOOD,'```json\n{"peg":null}\n```']:
            self.assertFalse(fence_parse(text)['json_valid'])
        self.assertFalse(strict_parse('```json\n'+GOOD+'\n```')['json_valid'])
        self.assertEqual(fence_parse('```json\n{"peg":[-1,0,20,30],"box":null}\n```')['peg']['status'],'INVALID_BOX')

    def test_coordinates(self):
        self.assertEqual(strict_parse(GOOD)['peg']['pixels'],[0,51.2,512,460.8])
        for b in [[False,0,5,6],[-1,0,4,5],[0,0,1001,5],[0,0,0,10],[3,4,2,5],float('nan')]:
            self.assertEqual(validate_box(b)['status'],'INVALID_BOX')

    def test_payload_and_final_text(self):
        for p in MODELS:
            data=payload(p,PROMPT,b'png')
            self.assertEqual(len(data.get('input',data.get('messages'))),1)
            for forbidden in ['GT_path','object_ids','DINO','parent_id','tools','previous_response_id']:
                self.assertNotIn(forbidden,json.dumps(data))
            self.assertEqual(data['model'],MODELS[p]['model'])
        self.assertEqual(payload('openai',PROMPT,b'png')['reasoning'],{'effort':'high'})
        a=normalize('openai',dict(model='gpt-6.1-sol',status='completed',output=[
            {'type':'reasoning','summary':[{'text':'wrong'}]},
            {'type':'message','content':[{'type':'output_text','text':GOOD}]}]),200)
        k=normalize('ollama',dict(model='kimi-k3',done=True,message={'content':GOOD,'thinking':'wrong'}),200)
        self.assertEqual(a['final_text'],GOOD);self.assertEqual(k['final_text'],GOOD)

    def test_budget_and_no_retry(self):
        self.assertLess(PRIOR_ACCOUNTED_USD+sum(budget_reservation(p,PROMPT)*60 for p in MODELS),20)
        self.assertIsNone(used_cost('ollama',None))
        import requests
        with patch('adapters.requests.post',side_effect=requests.Timeout()) as post:
            result=call('openai',{},'fake-key')
        self.assertEqual(post.call_count,1);self.assertEqual(result['status'],'TIMEOUT')
        self.assertNotIn('fake-key',json.dumps(result))

if __name__=='__main__':unittest.main()
