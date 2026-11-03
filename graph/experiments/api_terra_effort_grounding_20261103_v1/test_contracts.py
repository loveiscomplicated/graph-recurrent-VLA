"""Offline contract checks; no real API, simulator, GT or GPU calls."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import requests
import adapters
import common
import infer


class Log:
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def log(self,*args): pass
    def link_output(self,*args): pass


class Contracts(unittest.TestCase):
    def test_effort_is_only_request_difference(self):
        payloads=[adapters.payload(p,'prompt',b'png') for p in common.MODELS]
        for p,body in zip(common.MODELS,payloads):
            self.assertEqual(body.pop('reasoning'),{'effort':p})
            self.assertEqual(body['service_tier'],'default')
            self.assertEqual(body['max_output_tokens'],5120)
            self.assertEqual(body['input'][0]['content'][0]['detail'],'original')
            self.assertEqual(len(body['input']),1)
        self.assertTrue(all(x==payloads[0] for x in payloads))

    def test_parser_and_no_coordinate_repair(self):
        text='{"peg":[0,100,1000,900],"box":null}'
        self.assertEqual(common.strict_parse(text)['peg']['pixels'],[0,51.2,512,460.8])
        self.assertFalse(common.strict_parse('```json\n'+text+'\n```')['json_valid'])
        self.assertTrue(common.fence_parse('```json\n'+text+'\n```')['json_valid'])
        for x in [[True,0,2,3],[-1,0,2,3],[2,0,1,3],[0,0,1001,3]]:
            self.assertEqual(common.validate_box(x)['status'],'INVALID_BOX')

    def test_adapter_does_not_retry(self):
        with patch('adapters.requests.post',side_effect=requests.Timeout()) as p:
            result=adapters.call('low',{},'fake')
        self.assertEqual(p.call_count,1);self.assertEqual(result['status'],'TIMEOUT')
        self.assertFalse(p.call_args.kwargs['allow_redirects'])

    def test_reservations_fit_in_original_aggregate_cap(self):
        self.assertLess(common.PRIOR_ACCOUNTED_USD+sum(common.budget_reservation(p,common.PROMPT)*60 for p in common.MODELS),20)

    def run_loop(self,status='OK',tier='default',initial=None):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);image=root/'image.png';image.write_bytes(b'fake-png')
            images=[dict(id='frame'+str(i),image_path=str(image),image_sha256=common.sha(image)) for i in range(4)]
            common.write(root/'input_manifest.json',{'images':images})
            (root/'grounding_prompt.txt').write_text('prompt')
            common.write(root/'verification.json',{'stage':'PROTOCOL_FROZEN'})
            if initial:common.write(root/'request_ledger.json',initial)
            calls=[]
            def fake(provider,data,key):
                calls.append((provider,data['reasoning']['effort']))
                return dict(status=status,final_text='{"peg":[0,0,10,10],"box":null}',latency_ms=1,
                    returned_model='gpt-6.1-terra',request_id='fake',http_status=200,finish_reason='completed',
                    usage=dict(input_tokens=10,output_tokens=10),raw=dict(service_tier=tier))
            with patch.object(infer,'ROOT',root),patch.object(infer,'verify_frozen',lambda:None),\
                 patch.object(infer,'logger',lambda _:Log()),patch.object(infer,'call',fake),\
                 patch.object(infer,'load_dotenv',lambda *a,**k:False),\
                 patch.dict(os.environ,{'OPENAI_API_KEY':'fake'},clear=True),\
                 patch.object(sys,'argv',['infer.py']),contextlib.redirect_stdout(io.StringIO()):
                infer.main(); count=len(calls);infer.main();self.assertEqual(len(calls),count)
            return calls,common.read(root/'request_ledger.json'),common.read(root/'full_image_grounding_predictions.json')

    def test_all_requests_once_and_resume_no_network(self):
        calls,ledger,pred=self.run_loop()
        self.assertEqual(len(calls),12);self.assertEqual(len(pred['rows']),12)
        self.assertTrue(all(a==b for a,b in calls));self.assertFalse(ledger['stopped'])

    def test_reserved_intent_is_never_retried(self):
        initial=dict(jobs={'none__frame0':dict(provider='none',image_id='frame0',status='RESERVED',reserved_usd=.03,accounted_usd=.03)},
                     stopped={},consecutive_errors={p:0 for p in common.MODELS})
        calls,ledger,pred=self.run_loop(initial=initial)
        self.assertEqual(len(calls),11);self.assertEqual(ledger['jobs']['none__frame0']['status'],'RESERVED')

    def test_unexpected_tier_stops_each_effort(self):
        calls,ledger,pred=self.run_loop(tier='fast')
        self.assertEqual(len(calls),3)
        self.assertTrue(all(x=='TIER_MISMATCH' for x in ledger['stopped'].values()))
        self.assertTrue(all(not r['parsed']['json_valid'] for r in pred['rows']))
        self.assertAlmostEqual(sum(j['accounted_usd'] for j in ledger['jobs'].values()),3*2*(10+5*10)/1e6)


if __name__=='__main__':unittest.main()
