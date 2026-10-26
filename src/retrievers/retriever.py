"""PGVector retrieval followed by chunk-level cross-encoder reranking."""
import math
from functools import lru_cache
from langchain_core.documents import Document
from src.config import RERANKER_MODEL, TOP_K_RERANK, TOP_K_RETRIEVE
from src.embeddings.embed_store import get_vector_store


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL)


def retrieval_search(query: str, top_k: int = TOP_K_RERANK, doc_type: str = None,
                     candidate_k: int = TOP_K_RETRIEVE, *, batch_id=None,
                     store=None, reranker=None):
    if top_k < 1 or candidate_k < 1:
        raise ValueError("Retrieval limits must be positive")
    search_filter = {}
    if doc_type:
        search_filter["doc_type"] = doc_type
    if batch_id is not None:
        search_filter["batch_id"] = batch_id
    store = get_vector_store() if store is None else store
    pairs = store.similarity_search_with_score(query, k=candidate_k,
                                               filter=search_filter or None)
    if not pairs:
        return []
    # Rerank all retrieved chunks before deduplication, so a resume's better
    # supporting passage cannot be discarded before the cross-encoder sees it.
    reranker = get_reranker() if reranker is None else reranker
    scores = reranker.predict([[query, doc.page_content] for doc, _ in pairs])
    if len(scores) != len(pairs):
        raise ValueError("Expected one cross-encoder score per retrieved chunk")
    grouped = {}
    for (doc, distance), score in zip(pairs, scores):
        score, distance = float(score), float(distance)
        if not math.isfinite(score) or not math.isfinite(distance):
            raise ValueError("Retrieval scores must be finite")
        source = doc.metadata.get("source", doc.metadata.get("file_name", "unknown"))
        evidence = {"text": doc.page_content, "chunk_id": doc.metadata.get("chunk_id"),
                    "vector_distance": distance, "rerank_score": score}
        grouped.setdefault(source, []).append((doc, evidence))
    results = []
    for entries in grouped.values():
        entries.sort(key=lambda entry: (-entry[1]["rerank_score"], entry[1]["vector_distance"]))
        best, evidence = entries[0]
        results.append(Document(page_content=best.page_content, metadata={
            **best.metadata, "vector_score": evidence["vector_distance"],
            "rerank_score": evidence["rerank_score"],
            "retrieved_evidence": [entry[1] for entry in entries],
        }))
    results.sort(key=lambda doc: (-doc.metadata["rerank_score"],
                                  doc.metadata["vector_score"], str(doc.metadata.get("source", ""))))
    return results[:top_k]
