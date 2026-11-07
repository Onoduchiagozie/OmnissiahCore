import importlib.util
import json
import os
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('provider','/opt/omnissiah/staging/openai-migration/openai_provider.py')
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


class Response:
    headers={}
    def __init__(self,status=200,data=None,events=None):
        self.status_code=status
        self.data=data or {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'A complete answer.'}]}]}
        self.events=events or []
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def json(self): return self.data
    def iter_lines(self,**kwargs):
        for event in self.events:
            if isinstance(event,Exception): raise event
            yield b'data: '+json.dumps(event).encode()


class Tests(unittest.TestCase):
    def setUp(self):
        os.environ['OPENAI_API_KEY']='mock-only'
        self.config={'model':'primary','fallback_models':['backup'],'retries_per_model':0}
    def test_sync_fallback(self):
        with patch.object(p.requests,'post',side_effect=[Response(503),Response()]) as post:
            self.assertEqual(''.join(p.generate(self.config,'sys','user')),'A complete answer.')
            self.assertEqual(post.call_args.kwargs['json']['model'],'backup')
        self.assertEqual(self.config['model'],'primary')
    def test_fatal_auth(self):
        with patch.object(p.requests,'post',return_value=Response(401)) as post:
            with self.assertRaises(p.OpenAIProviderError): list(p.generate(self.config,'sys','user'))
            self.assertEqual(post.call_count,1)
    def test_account_quota_not_retried(self):
        with patch.object(p.requests,'post',return_value=Response(429,{'error':{'code':'insufficient_quota'}})) as post:
            with self.assertRaisesRegex(p.OpenAIProviderError,'billing'): list(p.generate(self.config,'sys','user'))
            self.assertEqual(post.call_count,1)
    def test_timeout_fallback(self):
        with patch.object(p.requests,'post',side_effect=[p.requests.Timeout(),Response()]):
            self.assertEqual(''.join(p.generate(self.config,'sys','user')),'A complete answer.')
    def test_sync_incomplete(self):
        with patch.object(p.requests,'post',return_value=Response(data={'status':'incomplete','incomplete_details':{'reason':'max_output_tokens'}})):
            with self.assertRaisesRegex(p.OpenAIProviderError,'max_output_tokens'): list(p.generate(self.config,'sys','user'))
    def test_stream_success_with_fallback(self):
        events=[{'type':'response.output_text.delta','delta':'first\n\nsecond.'},{'type':'response.completed','response':{'status':'completed'}}]
        with patch.object(p.requests,'post',side_effect=[Response(503),Response(events=events)]):
            self.assertEqual(''.join(p.generate(self.config,'sys','user',True)),'first\n\nsecond.')
    def test_partial_stream_never_switches(self):
        events=[{'type':'response.output_text.delta','delta':'partial'},p.requests.ConnectionError()]
        with patch.object(p.requests,'post',return_value=Response(events=events)) as post:
            stream=p.generate(self.config,'sys','user',True)
            self.assertEqual(next(stream),'partial')
            with self.assertRaises(p.OpenAIProviderError): next(stream)
            self.assertEqual(post.call_count,1)
    def test_stream_incomplete_and_missing_end(self):
        for tail in ([],[{'type':'response.incomplete'}]):
            events=[{'type':'response.output_text.delta','delta':'partial'}]+tail
            with patch.object(p.requests,'post',return_value=Response(events=events)):
                with self.assertRaises(p.OpenAIProviderError): list(p.generate(self.config,'sys','user',True))
    def test_bounded_exhaustion(self):
        with patch.object(p.requests,'post',return_value=Response(503)) as post:
            with self.assertRaises(p.OpenAIProviderError): list(p.generate(self.config,'sys','user'))
            self.assertEqual(post.call_count,2)


unittest.main()
