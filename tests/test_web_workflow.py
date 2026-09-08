import json
import io
import tempfile
from pathlib import Path
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from src.loaders import uploads
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
            self.assertIn(b'Analyze candidates', response.read())
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


class UploadTests(WebTests):
    def multipart(self, pdf, filename='resume.pdf', job=None):
        job = job or '  Python engineer with Docker experience.\n'
        boundary = 'test-upload-boundary'
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="job_description"\r\n\r\n{job}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="resumes"; filename="{filename}"\r\nContent-Type: application/pdf\r\n\r\n').encode() + pdf + f'\r\n--{boundary}--\r\n'.encode()
        return Request(self.url+'/api/analyze', data=body, headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})

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
                with urlopen(self.multipart(pdf, '../../resume.pdf')) as response:
                    result = json.load(response)
            saved = result['saved_files']
            self.assertEqual((Path(tmp)/saved['job_path']).read_text(), '  Python engineer with Docker experience.\n')
            self.assertEqual((Path(tmp)/saved['resumes'][0]['path']).read_bytes(), pdf)
            self.assertIn('Python engineer', analyze_mock.call_args.args[0]['resumes'][0]['text'])
            self.assertEqual(saved['resumes'][0]['label'], 'resume.pdf')
            self.assertEqual(Path(saved['resumes'][0]['path']).parent.name, saved['batch_id'])

    def test_invalid_pdf_and_scanned_pdf_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(uploads, 'DATA_DIR', Path(tmp)/'data'), patch.object(web_app, 'analyze') as analyze_mock:
            for pdf in (b'not a pdf', b'%PDF-broken', self.pdf('')):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self.multipart(pdf))
                self.assertEqual(error.exception.code, 400)
            self.assertEqual(list(Path(tmp).rglob('*.pdf')), [])
            self.assertEqual(list(Path(tmp).rglob('*.txt')), [])
            analyze_mock.assert_not_called()

    def test_saved_files_survive_model_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(uploads, 'DATA_DIR', Path(tmp)/'data'), patch.object(web_app, 'analyze', side_effect=RuntimeError('offline')), patch.object(web_app.logging, 'exception'):
            with self.assertRaises(HTTPError) as error:
                urlopen(self.multipart(self.pdf()))
            self.assertEqual(error.exception.code, 502)
            result = json.load(error.exception)
            self.assertTrue((Path(tmp)/result['saved_files']['job_path']).exists())
