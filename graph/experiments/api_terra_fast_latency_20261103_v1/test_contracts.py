"""Offline guards for a paid latency experiment; no real API requests."""
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import run


class Contracts(unittest.TestCase):
    def test_only_tier_changes(self):
        baseline = run.payload('terra', run.base.PROMPT, b'png')
        fast = run.request_for(run.base.PROMPT, b'png')
        self.assertEqual(fast.pop('service_tier'), 'fast')
        self.assertEqual(fast, baseline)

    def test_unknown_usage_and_tier_accounting(self):
        self.assertIsNone(run.response_cost({'usage':None, 'raw':{}}))
        r = dict(usage=dict(input_tokens=1000, output_tokens=500), raw=dict(service_tier='fast'))
        self.assertAlmostEqual(run.response_cost(r), .007)
        r['raw']['service_tier'] = 'default'
        self.assertAlmostEqual(run.response_cost(r), .0035)
        r['raw'].clear()
        self.assertAlmostEqual(run.response_cost(r), .007)

    def test_no_retry_for_durable_intent_or_stop(self):
        state = dict(jobs={'image':{'status':'RESERVED'}}, stopped=None)
        self.assertFalse(run.may_send(state, 'image'))
        self.assertTrue(run.may_send(state, 'new'))
        state['stopped'] = 'FAST_TIER_NOT_CONFIRMED'
        self.assertFalse(run.may_send(state, 'new'))

    def test_transport_is_single_attempt(self):
        import requests
        with patch('adapters.requests.post', side_effect=requests.Timeout()) as post:
            result = run.call('terra', {}, 'fake')
        self.assertEqual(post.call_count, 1)
        self.assertEqual(result['status'], 'TIMEOUT')
        self.assertFalse(post.call_args.kwargs['allow_redirects'])

    def test_resume_reserved_never_calls_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run.write(root/'preparation_freeze.json', {})
            run.write(root/'protected_before.json', {})
            run.write(root/'input_manifest.json', {'images':[{'id':'image'}]})
            run.write(root/'protocol.json', {})
            (root/'grounding_prompt.txt').write_text('prompt')
            run.write(root/'request_ledger.json', dict(jobs={'image':dict(status='RESERVED',reserved_usd=.1,accounted_usd=.1)},stopped=None))
            with patch.object(run,'ROOT',root), patch.dict(run.os.environ,{'OPENAI_API_KEY':'fake'}), patch.object(run,'call') as call:
                run.infer()
                call.assert_not_called()
                self.assertTrue((root/'api_freeze.json').exists())


if __name__ == '__main__': unittest.main()
