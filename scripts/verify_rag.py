"""Live check: real PGVector, embeddings, cross-encoder, Ollama, PDF upload and API.

Run inside the app container (or a configured local environment). This indexes
synthetic resumes in two distinct batches and leaves them available for inspection.
"""
import io
import json
import argparse
from pathlib import Path
from uuid import uuid4
import requests
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from sqlalchemy import create_engine, text
from src.config import DB_CONNECTION
from src.embeddings.embed_store import get_vector_store, store_documents
from src.retrievers.resume_search import retrieve_resumes
from langchain_core.documents import Document


def pdf(resume):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                             NameObject('/Subtype'): NameObject('/Type1'),
                             NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({
        NameObject('/F1'): writer._add_object(font)})})
    stream = DecodedStreamObject()
    safe = resume.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
    stream.set_data(f'BT /F1 10 Tf 40 700 Td ({safe}) Tj ET'.encode())
    page[NameObject('/Contents')] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrieval-only', action='store_true',
                        help='Check real indexing, reranking, isolation and persistence without another LLM request')
    args = parser.parse_args()
    store = get_vector_store()
    old_batch = 'verification-' + uuid4().hex
    sentinel_id = old_batch + ':sentinel'
    store_documents([Document(page_content='Python Docker PostgreSQL backend expert', metadata={
        'batch_id': old_batch, 'doc_type': 'resume', 'source': sentinel_id, 'candidate_id': 99,
        'file_name': 'Old candidate must not appear.pdf', 'chunk_id': 0})], ids=[sentinel_id], store=store)
    print('Indexed separate-batch sentinel', flush=True)
    resumes = [
        {'label': 'Python developer.pdf', 'text': 'Backend developer with six years of Python, PostgreSQL and Docker experience. Built REST APIs and debugged production database queries.'},
        {'label': 'Designer.pdf', 'text': 'Graphic designer with two years of experience creating visual identities and illustrations using Photoshop and Illustrator.'},
        {'label': 'Junior developer.pdf', 'text': 'Junior developer with one year of Python experience building small REST APIs and automated tests. Learning PostgreSQL and Docker.'},
    ]
    job = 'Backend engineer requiring Python, PostgreSQL, Docker and five years of professional experience.'
    engine = create_engine(DB_CONNECTION, pool_pre_ping=True)
    if args.retrieval_only:
        first = retrieve_resumes(job, resumes)
        second = retrieve_resumes(job, [resumes[1]])
        assert len(first['selected_resumes']) == 2
        assert [r['label'] for r in second['selected_resumes']] == ['Designer.pdf']
        with engine.connect() as connection:
            assert connection.execute(text('SELECT count(*) FROM langchain_pg_embedding WHERE id = :id'),
                                      {'id': sentinel_id}).scalar_one() == 1
            for batch in (first, second):
                assert connection.execute(text("SELECT count(*) FROM langchain_pg_embedding WHERE cmetadata->>'batch_id' = :batch"),
                                          {'batch': batch['retrieval']['batch_id']}).scalar_one() == batch['retrieval']['indexed_chunks']
        report = {'status': 'passed', 'mode': 'retrieval_only', 'first_batch': first['retrieval'],
                  'second_batch': second['retrieval'], 'existing_records_preserved': True,
                  'database_batch_isolation': True,
                  'candidates': [{key: value for key, value in r.items() if key != 'text'}
                                 for r in first['selected_resumes']]}
        Path('/tmp/rolelens-rag-retrieval-verification.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
        return
    response = requests.post('http://127.0.0.1:8000/api/analyze',
                             data={'job_description': job},
                             files=[('resumes', (r['label'], pdf(r['text']), 'application/pdf')) for r in resumes],
                             timeout=900)
    print('API status:', response.status_code, flush=True)
    response.raise_for_status()
    result = response.json()
    # Preserve a successful API result even if a later infrastructure check fails.
    Path('/tmp/rolelens-rag-api-response.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['stages'] == ['retrieve_candidates', 'resume_parser', 'scorer_ranker', 'interview_q_generator']
    assert result['retrieval']['ranking_method'] == 'cross_encoder'
    assert result['retrieval']['scope'] == 'current_submission'
    assert result['retrieval']['indexed_resumes'] == 3
    candidates = result['candidates']
    assert len(candidates) == 2
    assert [c['rerank_score'] for c in candidates] == sorted([c['rerank_score'] for c in candidates], reverse=True)
    for candidate in candidates:
        assert candidate['candidate_id'] in (1, 2, 3)
        assert candidate['retrieved_evidence']
        assert candidate['questions']
        assert all(q['gap'].startswith(('Skill assessment:', 'Skill not evidenced:')) for q in candidate['questions'])
    with engine.connect() as connection:
        assert connection.execute(text('SELECT count(*) FROM langchain_pg_embedding WHERE id = :id'),
                                  {'id': sentinel_id}).scalar_one() == 1
        indexed = connection.execute(text("SELECT count(*) FROM langchain_pg_embedding WHERE cmetadata->>'batch_id' = :batch"),
                                     {'batch': result['retrieval']['batch_id']}).scalar_one()
        assert indexed == result['retrieval']['indexed_chunks']
    # A second real retrieval proves the original batch survives and never leaks.
    second = retrieve_resumes(job, [resumes[1]])
    assert second['retrieval']['batch_id'] != result['retrieval']['batch_id']
    assert [r['label'] for r in second['selected_resumes']] == ['Designer.pdf']
    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM langchain_pg_embedding WHERE cmetadata->>'batch_id' = :batch"),
                                  {'batch': result['retrieval']['batch_id']}).scalar_one() == indexed
    report = {'status': 'passed', 'stages': result['stages'], 'retrieval': result['retrieval'],
              'database_batch_isolation': True, 'existing_records_preserved': True,
              'saved_files': result['saved_files'], 'candidates': candidates}
    output = Path('/tmp/rolelens-rag-verification.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'retrieval': result['retrieval'],
                      'ranked_labels': [c['label'] for c in candidates],
                      'report': str(output)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
