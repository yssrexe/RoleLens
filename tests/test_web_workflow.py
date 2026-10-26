import json
import io
import tempfile
from pathlib import Path
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from src.loaders import uploads
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import web_app
from src.graph.workflow import analyze, build_workflow
from test_workflow import FakeLLM
from rag_fakes import fake_retrieve

class WebTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(web_app.app)
        self.addCleanup(self.client.close)

    def post(self, data):
        return self.client.post('/api/analyze', json=data)

    def test_page_and_analysis(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Analyze candidates', response.content)
        graph=build_workflow(FakeLLM(), fake_retrieve)
        with patch.object(web_app, 'analyze', side_effect=lambda data: analyze(data,graph)):
            response = self.post({'job_description':'Python Docker engineer with five years experience', 'resumes':[{'label':'Test','text':'weak candidate resume text'}]})
            self.assertEqual(response.status_code, 200)
            result = response.json()
        self.assertEqual(len(result['stages']),4)
        self.assertEqual(result['retrieval']['ranking_method'], 'cross_encoder')
        self.assertTrue(result['candidates'][0]['retrieved_evidence'])
        self.assertTrue(result['candidates'][0]['questions'])

    def test_invalid_and_blank_input(self):
        for payload in ({}, {'job_description':' '*20,'resumes':[{'label':'Test','text':' '*20}]}):
            self.assertEqual(self.post(payload).status_code, 400)

    def test_model_error_is_visible(self):
        with patch.object(web_app,'analyze',side_effect=RuntimeError('unavailable')), patch.object(web_app.logging,'exception'):
            response = self.post({'job_description':'Python Docker engineer with five years experience','resumes':[{'label':'Test','text':'weak candidate resume text'}]})
            self.assertEqual(response.status_code, 502)


class UploadTests(WebTests):
    def multipart(self, pdf, filename='resume.pdf', job=None):
        job = job or '  Python engineer with Docker experience.\n'
        boundary = 'test-upload-boundary'
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="job_description"\r\n\r\n{job}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="resumes"; filename="{filename}"\r\nContent-Type: application/pdf\r\n\r\n').encode() + pdf + f'\r\n--{boundary}--\r\n'.encode()
        return self.client.post('/api/analyze', content=body, headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})

    def pdf(self, text='Python engineer with five years of Docker experience.'):
        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject('/Type'):NameObject('/Font'), NameObject('/Subtype'):NameObject('/Type1'), NameObject('/BaseFont'):NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 50 700 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
        output = io.BytesIO()
        writer.write(output)
        return output.getvalue()

    def test_saved_paths_and_exact_job(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(uploads, 'DATA_DIR', Path(tmp)/'data'):
            pdf = self.pdf()
            with patch.object(web_app, 'analyze', return_value={'candidates':[]}) as analyze_mock:
                response = self.multipart(pdf, '../../resume.pdf')
                self.assertEqual(response.status_code, 200)
                result = response.json()
            saved = result['saved_files']
            self.assertEqual((Path(tmp)/saved['job_path']).read_text(), '  Python engineer with Docker experience.\n')
            self.assertEqual((Path(tmp)/saved['resumes'][0]['path']).read_bytes(), pdf)
            self.assertIn('Python engineer', analyze_mock.call_args.args[0]['resumes'][0]['text'])
            self.assertEqual(saved['resumes'][0]['label'], 'resume.pdf')
            self.assertEqual(Path(saved['resumes'][0]['path']).parent.name, saved['batch_id'])

    def test_invalid_pdf_and_scanned_pdf_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(uploads, 'DATA_DIR', Path(tmp)/'data'), patch.object(web_app, 'analyze') as analyze_mock:
            for pdf in (b'not a pdf', b'%PDF-broken', self.pdf('')):
                self.assertEqual(self.multipart(pdf).status_code, 400)
            self.assertEqual(list(Path(tmp).rglob('*.pdf')), [])
            self.assertEqual(list(Path(tmp).rglob('*.txt')), [])
            analyze_mock.assert_not_called()

    def test_saved_files_survive_model_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(uploads, 'DATA_DIR', Path(tmp)/'data'), patch.object(web_app, 'analyze', side_effect=RuntimeError('offline')), patch.object(web_app.logging, 'exception'):
            response = self.multipart(self.pdf())
            self.assertEqual(response.status_code, 502)
            result = response.json()
            self.assertTrue((Path(tmp)/result['saved_files']['job_path']).exists())


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(web_app.app)
        self.addCleanup(self.client.close)

    def test_docs_health_and_timing(self):
        self.assertEqual(self.client.get('/docs').status_code, 200)
        schema = self.client.get('/openapi.json').json()
        self.assertIn('/api/analyze', schema['paths'])
        self.assertIn('ResumeInput', schema['components']['schemas'])
        response = self.client.get('/api/health')
        self.assertEqual(response.json(), {'status': 'ok'})
        self.assertGreaterEqual(float(response.headers['X-Response-Time-Ms']), 0)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')

    def test_malformed_unsupported_and_oversized_requests(self):
        for body, headers, status in [
            (b'{', {'Content-Type': 'application/json'}, 400),
            (b'hello', {'Content-Type': 'text/plain'}, 415),
            (b'{}', {'Content-Type': 'application/json', 'Content-Length': '1000001'}, 413),
            (b'{}', {'Content-Type': 'application/json', 'Content-Length': 'bad'}, 400),
            (b'{}', {'Content-Type': 'application/json', 'Content-Length': '3'}, 400),
        ]:
            with self.subTest(status=status, headers=headers):
                self.assertEqual(self.client.post('/api/analyze', content=body, headers=headers).status_code, status)
        with patch.object(web_app, 'analyze') as model:
            response = self.client.post('/api/analyze', content=iter([b'x' * 600000, b'x' * 600000]),
                                        headers={'Content-Type': 'application/json'})
            self.assertEqual(response.status_code, 413)
            model.assert_not_called()

    def test_busy_analysis_keeps_health_responsive_and_releases_slot(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        entered, release = Event(), Event()
        payload = {'job_description': 'Python Docker engineer with five years experience',
                   'resumes': [{'label': 'Test', 'text': 'weak candidate resume text'}]}

        def slow_model(data):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test timed out')
            return {'candidates': []}

        with patch.object(web_app, 'analyze', side_effect=slow_model), ThreadPoolExecutor() as pool:
            pending = pool.submit(self.client.post, '/api/analyze', json=payload)
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.client.get('/api/health').status_code, 200)
                self.assertEqual(self.client.post('/api/analyze', json=payload).status_code, 429)
            finally:
                release.set()
            self.assertEqual(pending.result(timeout=5).status_code, 200)
        with patch.object(web_app, 'analyze', return_value={'candidates': []}):
            self.assertEqual(self.client.post('/api/analyze', json=payload).status_code, 200)
