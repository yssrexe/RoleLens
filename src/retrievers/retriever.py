from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL, TOP_K_RERANK, TOP_K_RETRIEVE
from src.embeddings.embed_store import vector_store


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def retrieval_search(query: str, top_k: int = TOP_K_RERANK, doc_type: str = None, candidate_k: int = TOP_K_RETRIEVE):
    search_filter = {"doc_type": doc_type} if doc_type else None
    candidate_pairs = vector_store.similarity_search_with_score(
        query,
        k=candidate_k,
        filter=search_filter,
    )

    if not candidate_pairs:
        return []

    seen_sources = set()
    candidates = []
    for doc, vector_score in candidate_pairs:
        source = doc.metadata.get("source", doc.metadata.get("file_name", "unknown"))
        if source in seen_sources:
            continue
        seen_sources.add(source)
        doc.metadata["vector_score"] = float(vector_score) if vector_score is not None else None
        candidates.append(doc)

    reranker = get_reranker()
    rerank_inputs = [[query, doc.page_content] for doc in candidates]
    rerank_scores = reranker.predict(rerank_inputs)

    ranked_docs = sorted(
        zip(candidates, rerank_scores),
        key=lambda item: item[1],
        reverse=True,
    )

    results = []
    for doc, rerank_score in ranked_docs[:top_k]:
        doc.metadata["rerank_score"] = float(rerank_score)
        results.append(doc)

    return results
