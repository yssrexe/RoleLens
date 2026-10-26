"""Small doubles at the database/model boundary; production retrieval still runs."""
from src.retrievers.resume_search import retrieve_resumes


class MemoryStore:
    def __init__(self):
        self.records = {}
        self.searches = []

    def add_documents(self, documents, ids=None):
        self.records.update(zip(ids, documents))
        return ids

    def similarity_search_with_score(self, query, k, filter=None):
        self.searches.append((query, k, filter))
        docs = [doc for doc in self.records.values()
                if all(doc.metadata.get(key) == value for key, value in (filter or {}).items())]
        pairs = [(doc, .1 if 'strong' in doc.page_content else .7) for doc in docs]
        return sorted(pairs, key=lambda pair: pair[1])[:k]


class FakeReranker:
    def __init__(self):
        self.pairs = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        return [4.0 if 'strong' in passage else 1.0 for _, passage in pairs]


def fake_retrieve(job, resumes):
    return retrieve_resumes(job, resumes, store=MemoryStore(), reranker=FakeReranker())
