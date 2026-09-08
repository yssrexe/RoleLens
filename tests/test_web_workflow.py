import json
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
import web_app
from src.graph.workflow import analyze, build_workflow
from test_workflow import FakeLLM

class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), web_app.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, data):
        return urlopen(Request(self.url+'/api/analyze', data=json.dumps(data).encode(), headers={'Content-Type':'application/json'}))

    def test_page_and_analysis(self):
        with urlopen(self.url) as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b'Run three agents', response.read())
        graph=build_workflow(FakeLLM(), lambda job, resumes: [.5])
        with patch.object(web_app, 'analyze', side_effect=lambda data: analyze(data,graph)):
            with self.post({'job_description':'Python Docker engineer with five years experience', 'resumes':[{'label':'Test','text':'weak candidate resume text'}]}) as response:
                result=json.load(response)
        self.assertEqual(len(result['stages']),3)
        self.assertTrue(result['candidates'][0]['questions'])

    def test_invalid_and_blank_input(self):
        for payload in ({}, {'job_description':' '*20,'resumes':[{'label':'Test','text':' '*20}]}):
            with self.assertRaises(HTTPError) as error:
                self.post(payload)
            self.assertEqual(error.exception.code,400)

    def test_model_error_is_visible(self):
        with patch.object(web_app,'analyze',side_effect=RuntimeError('unavailable')), patch.object(web_app.logging,'exception'):
            with self.assertRaises(HTTPError) as error:
                self.post({'job_description':'Python Docker engineer with five years experience','resumes':[{'label':'Test','text':'weak candidate resume text'}]})
            self.assertEqual(error.exception.code,502)
