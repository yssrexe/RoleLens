import unittest
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from src.embeddings.embed_store import store_documents
from src.retrievers.retriever import retrieval_search
from src.retrievers.resume_search import retrieve_resumes
from src.graph.workflow import analyze, build_workflow
from src.graph.nodes.scorer_ranker import scorer_ranker
from src.graph.nodes.interview_q_generator import interview_q_generator
from rag_fakes import MemoryStore, FakeReranker
from test_workflow import FakeLLM


class RetrievalTests(unittest.TestCase):
    def test_batch_filter_and_indexing_preserve_older_records_and_duplicate_names(self):
        store = MemoryStore()
        resumes = [{'label': 'same.pdf', 'text': 'weak candidate Python resume text'},
                   {'label': 'same.pdf', 'text': 'strong candidate Docker Python resume text'}]
        first = retrieve_resumes('Python engineer', resumes, store=store,
                                 reranker=FakeReranker(), batch_id='old')
        count = len(store.records)
        second = retrieve_resumes('Python engineer', resumes, store=store,
                                  reranker=FakeReranker(), batch_id='new')
        self.assertEqual(len(store.records), count * 2)
        self.assertEqual(store.searches[-1][2], {'doc_type': 'resume', 'batch_id': 'new'})
        self.assertEqual([r['candidate_id'] for r in second['selected_resumes']], [2, 1])
        self.assertEqual(first['retrieval']['batch_id'], 'old')
        self.assertTrue(all(doc.metadata['source'].startswith(doc.metadata['batch_id'] + '/')
                            for doc in store.records.values()))

    def test_all_retrieved_chunks_reranked_before_candidate_deduplication(self):
        docs = [Document(page_content=text, metadata={'source': source, 'chunk_id': i})
                for i, (source, text) in enumerate([('a', 'less relevant'), ('b', 'relevant'),
                                                   ('a', 'most relevant')])]
        store, reranker = Mock(), Mock()
        store.similarity_search_with_score.return_value = list(zip(docs, [.1, .2, .3]))
        reranker.predict.return_value = [-2, 1, 5]
        result = retrieval_search('query', store=store, reranker=reranker)
        self.assertEqual(reranker.predict.call_count, 1)
        self.assertEqual(len(reranker.predict.call_args.args[0]), 3)
        self.assertEqual([doc.metadata['source'] for doc in result], ['a', 'b'])
        self.assertEqual(result[0].page_content, 'most relevant')
        self.assertEqual(result[0].metadata['rerank_score'], 5)
        self.assertEqual(len(result[0].metadata['retrieved_evidence']), 2)
        self.assertEqual(result[0].metadata['vector_score'], .3)

    def test_cross_encoder_overrides_vector_order_and_top_k_is_resumes(self):
        store, reranker = Mock(), Mock()
        store.similarity_search_with_score.return_value = [
            (Document(page_content=f'resume {i}', metadata={'source': str(i)}), i / 10)
            for i in range(4)]
        reranker.predict.return_value = [-5, 8, 10, 2]
        result = retrieval_search('job', store=store, reranker=reranker)
        self.assertEqual([d.metadata['source'] for d in result], ['2', '1'])
        self.assertEqual(store.similarity_search_with_score.call_args.kwargs['k'], 4)

    def test_supporting_fit_does_not_change_cross_encoder_order(self):
        candidates = [{'candidate_id': 1, 'profile': {'skills': []},
                       'vector_distance': .8, 'rerank_score': 4},
                      {'candidate_id': 2, 'profile': {'skills': ['Python']},
                       'vector_distance': .1, 'rerank_score': 1}]
        result = scorer_ranker({'requirements': {'required_skills': ['Python']},
                                'candidates': candidates, 'stages': []})
        self.assertEqual([c['candidate_id'] for c in result['candidates']], [1, 2])
        self.assertLess(result['candidates'][0]['fit_score'], result['candidates'][1]['fit_score'])

    def test_default_workflow_uses_pgvector_and_cross_encoder_boundaries(self):
        store, reranker = MemoryStore(), FakeReranker()
        llm = Mock(wraps=FakeLLM())
        with patch('src.retrievers.resume_search.get_vector_store', return_value=store), \
             patch('src.retrievers.retriever.get_reranker', return_value=reranker):
            result = analyze({'job_description': 'Python Docker engineer with five years experience',
                              'resumes': [{'label': 'A', 'text': 'strong candidate Python Docker resume'}]},
                             build_workflow(llm=llm))
        self.assertTrue(store.records)
        self.assertTrue(store.searches)
        self.assertTrue(reranker.pairs)
        question_data = llm.structured.call_args.args[2]
        self.assertTrue(question_data['retrieved_evidence'])
        self.assertNotIn('education', question_data['profile'])
        self.assertEqual(result['stages'][0], 'retrieve_candidates')

    def test_non_destructive_insert_and_no_silent_database_fallback(self):
        store = Mock()
        docs = [Document(page_content='Python backend engineer')]
        store_documents(docs, ids=['one'], store=store)
        store.drop_tables.assert_not_called()
        store.add_documents.assert_called_once_with(docs, ids=['one'])
        with patch('src.retrievers.resume_search.get_vector_store', side_effect=RuntimeError('DB down')):
            with self.assertRaisesRegex(RuntimeError, 'DB down'):
                analyze({'job_description': 'Python engineer with Docker skills',
                         'resumes': [{'label': 'A', 'text': 'A developer with Python experience'}]},
                        build_workflow(llm=FakeLLM()))

    def test_long_resumes_chunked_and_top_two_only_parsed(self):
        store, reranker = MemoryStore(), FakeReranker()
        resumes = [{'label': str(i), 'text': ('Python project experience. ' * 60)} for i in range(3)]
        result = retrieve_resumes('Python', resumes, store=store, reranker=reranker)
        self.assertGreater(result['retrieval']['indexed_chunks'], 3)
        self.assertTrue(all(len(doc.page_content) <= 800 for doc in store.records.values()))
        self.assertLessEqual(len(result['selected_resumes']), 2)
        self.assertEqual(len(reranker.pairs), 4)

    def test_empty_retrieval_and_invalid_scores_fail_clearly(self):
        store, reranker = Mock(), Mock()
        store.similarity_search_with_score.return_value = []
        self.assertEqual(retrieval_search('job', store=store, reranker=reranker), [])
        reranker.predict.assert_not_called()
        store.similarity_search_with_score.return_value = [(Document(page_content='text'), .2)]
        for scores in ([float('nan')], []):
            reranker.predict.return_value = scores
            with self.assertRaises(ValueError):
                retrieval_search('job', store=store, reranker=reranker)
