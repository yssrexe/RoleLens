"""Persist upload batches and extract complete resumes from their saved paths."""
from email import policy
from email.parser import BytesParser
from pathlib import Path
import shutil
from uuid import uuid4

from pypdf import PdfReader
from src.graph.models import AnalysisRequest

DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_REQUEST_BYTES = 51 * 1024 * 1024


def save_uploads(content_type, body):
    message = BytesParser(policy=policy.default).parsebytes(
        b'Content-Type: ' + content_type.encode('ascii') + b'\r\nMIME-Version: 1.0\r\n\r\n' + body
    )
    if not message.is_multipart() or message.defects:
        raise ValueError('Invalid upload form.')
    jobs, files = [], []
    for part in message.iter_parts():
        name = part.get_param('name', header='content-disposition')
        raw = part.get_payload(decode=True)
        if raw is None or part.defects:
            raise ValueError('Invalid upload form.')
        if name == 'job_description' and part.get_filename() is None:
            jobs.append(raw.decode('utf-8'))
        elif name == 'resumes' and part.get_filename():
            label = part.get_filename().replace('\\', '/').split('/')[-1]
            if not label.lower().endswith('.pdf') or not raw.startswith(b'%PDF-'):
                raise ValueError('Upload valid PDF files only.')
            if not 0 < len(raw) <= MAX_FILE_BYTES:
                raise ValueError('Each PDF must be 5 MB or smaller.')
            files.append((label[:200], raw))
        else:
            raise ValueError('Unexpected upload field.')
    if len(jobs) != 1 or not 20 <= len(jobs[0].strip()) <= 20000:
        raise ValueError('Enter a job description of 20–20,000 characters.')
    if not 1 <= len(files) <= 10:
        raise ValueError('Upload between 1 and 10 PDF resumes.')

    batch = uuid4().hex
    folder = DATA_DIR / 'resumes' / batch
    job_path = DATA_DIR / 'jobs' / f'{batch}.txt'
    folder.mkdir(parents=True, exist_ok=False)
    try:
        job_path.parent.mkdir(parents=True, exist_ok=True)
        job_path.write_text(jobs[0], encoding='utf-8')
        resumes, paths = [], []
        for label, raw in files:
            path = folder / f'{uuid4().hex}.pdf'
            path.write_bytes(raw)
            try:
                reader = PdfReader(path)
                if reader.is_encrypted:
                    raise ValueError('Password-protected PDF')
                if len(reader.pages) > 30:
                    raise ValueError('More than 30 pages')
                text = '\n'.join(page.extract_text() or '' for page in reader.pages).strip()
            except Exception as exc:
                raise ValueError(f'{label}: cannot read PDF. Use an unencrypted PDF with at most 30 pages.') from exc
            if not 20 <= len(text) <= 50000:
                raise ValueError(f'{label}: PDF must contain 20–50,000 readable characters. Scanned images need OCR first.')
            resumes.append({'label': label, 'text': text})
            paths.append({'label': label, 'path': str(path.relative_to(DATA_DIR.parent))})
        payload = {'job_description': job_path.read_text(encoding='utf-8'), 'resumes': resumes}
        AnalysisRequest.model_validate(payload)
        return payload, {'batch_id': batch, 'job_path': str(job_path.relative_to(DATA_DIR.parent)), 'resumes': paths}
    except Exception:
        shutil.rmtree(folder)
        job_path.unlink(missing_ok=True)
        raise
