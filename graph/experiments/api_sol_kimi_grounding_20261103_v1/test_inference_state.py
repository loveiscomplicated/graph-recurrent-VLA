"""Offline behavioral tests of the actual inference loop; no credentials or network."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import infer
from common import write, read, sha, MODELS

class Log:
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def log(self,*args):pass
    def link_output(self,*args):pass

class StateMachine(unittest.TestCase):
    def run_case(self,result,initial=None):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);image=root/'rgb.png';image.write_bytes(b'offline-image')
            images=[dict(id='frame'+str(i),image_path=str(image),image_sha256=sha(image)) for i in range(4)]
            write(root/'input_manifest.json',{'images':images})
            (root/'grounding_prompt.txt').write_text('fixed prompt')
            write(root/'verification.json',{'stage':'PROTOCOL_FROZEN'})
            if initial:write(root/'request_ledger.json',initial)
            calls=[]
            def fake_call(provider,data,key):
                self.assertEqual(key,'offline-'+provider)
                calls.append(provider)
                return dict(result,latency_ms=1,returned_model=MODELS[provider]['model'],
                            request_id='offline',usage={'input_tokens':10,'output_tokens':10},raw={})
            environment={MODELS[p]['key']:'offline-'+p for p in MODELS}
            with patch.object(infer,'ROOT',root),patch.object(infer,'verify_frozen',lambda:None),\
                 patch.object(infer,'logger',lambda s:Log()),patch.object(infer,'call',fake_call),\
                 patch.object(infer,'load_dotenv',lambda *a,**k:False),\
                 patch.dict(os.environ,environment,clear=True),patch.object(sys,'argv',['infer.py']),\
                 contextlib.redirect_stdout(io.StringIO()):
                infer.main()
                before=len(calls)
                infer.main()
                self.assertEqual(len(calls),before,'Completed inference must not resubmit')
            return calls,read(root/'request_ledger.json'),read(root/'full_image_grounding_predictions.json')

    def test_success_and_resume(self):
        calls,ledger,predictions=self.run_case(dict(status='OK',final_text='{"peg":[0,0,10,10],"box":null}'))
        self.assertEqual(len(calls),8)
        self.assertEqual(len(predictions['rows']),8)
        self.assertFalse(ledger['stopped'])

    def test_reserved_intent_is_never_retried(self):
        initial=dict(jobs={'openai__frame0':dict(provider='openai',image_id='frame0',status='RESERVED',
                                              reserved_usd=.3,accounted_usd=.3)},stopped={},
                     consecutive_errors={p:0 for p in MODELS})
        calls,ledger,predictions=self.run_case(dict(status='OK',final_text='{"peg":[0,0,10,10],"box":null}'),initial)
        self.assertEqual(len(calls),7)
        self.assertEqual(ledger['jobs']['openai__frame0']['status'],'RESERVED')
        self.assertEqual(len(predictions['rows']),8)

    def test_auth_error_stops_provider_without_dropping_assigned_images(self):
        calls,ledger,predictions=self.run_case(dict(status='AUTH_ERROR',final_text=''))
        self.assertEqual(len(calls),2)
        self.assertTrue(all(s=='AUTH_ERROR' for s in ledger['stopped'].values()))
        self.assertEqual(sum(r['attempted'] for r in predictions['rows']),2)
        self.assertEqual(len(predictions['rows']),8)

    def test_three_format_failures_stop_without_prompt_repair(self):
        calls,ledger,_=self.run_case(dict(status='OK',final_text='no coordinates'))
        self.assertEqual(len(calls),6)
        self.assertTrue(all(s=='GROUNDING_FORMAT_BLOCKED' for s in ledger['stopped'].values()))

    def test_diagnostic_coordinates_do_not_become_primary(self):
        calls,ledger,predictions=self.run_case(dict(status='OK',final_text='```json\n{"peg":[0,0,10,10],"box":null}\n```'))
        self.assertEqual(len(calls),8)
        self.assertTrue(all(j['mechanical_coordinates'] for j in ledger['jobs'].values()))
        self.assertTrue(all(not r['parsed']['json_valid'] for r in predictions['rows']))

    def test_overflowing_coordinate_is_invalid_and_serializable(self):
        from common import strict_parse
        parsed=strict_parse('{"peg":[0,0,1e999,10],"box":null}')
        self.assertEqual(parsed['peg']['status'],'INVALID_BOX')
        self.assertIsNone(parsed['peg']['pixels'])
        json.dumps(parsed,allow_nan=False)

    def test_mac_idle_sleep_guard_is_scoped_and_dry_run_does_not_acquire_it(self):
        import run
        with patch.object(run.sys,'platform','darwin'),patch.object(run,'dispatch'),\
             patch.object(run.subprocess,'Popen') as launch,patch.object(sys,'argv',['run.py','infer']):
            run.main()
            self.assertEqual(launch.call_args.args[0][:3],['/usr/bin/caffeinate','-i','-w'])
            launch.return_value.terminate.assert_called_once()
            launch.return_value.wait.assert_called_once()
        with patch.object(run.sys,'platform','darwin'),patch.object(run,'dispatch'),\
             patch.object(run.subprocess,'Popen') as launch,patch.object(sys,'argv',['run.py','infer','--dry-run']):
            run.main();launch.assert_not_called()

if __name__=='__main__':unittest.main()
