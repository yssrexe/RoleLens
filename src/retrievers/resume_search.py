"""Index one submission, retrieve its evidence, and shortlist candidates."""
from uuid import uuid4
from langchain_core.documents import Document
from src.config import EMBEDDING_MODEL, RERANKER_MODEL, TOP_K_RETRIEVE, TOP_K_RERANK
from src.embeddings.embed_store import get_vector_store, store_documents
from src.loaders.chunking import clean_text, chunk_documents
from src.retrievers.retriever import retrieval_search


def retrieve_resumes(job, resumes, *, store=None, reranker=None, batch_id=None):
    batch_id = batch_id or uuid4().hex
    store = get_vector_store() if store is None else store
    documents = [Document(page_content=clean_text(resume["text"]), metadata={
        "doc_type": "resume", "batch_id": batch_id, "candidate_id": index,
        "file_name": resume["label"], "source": f"{batch_id}/{index}",
    }) for index, resume in enumerate(resumes, start=1)]
    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError("Resumes have no indexable text")
    store_documents(chunks, ids=[f'{batch_id}:{chunk.metadata["chunk_id"]}' for chunk in chunks], store=store)
    matches = retrieval_search(job, doc_type="resume", batch_id=batch_id, store=store,
                               reranker=reranker)
    if not matches:
        raise RuntimeError("No resume evidence was retrieved for this submission")
    selected = []
    for match in matches:
        metadata = match.metadata
        candidate_id = metadata["candidate_id"]
        if metadata.get("batch_id") != batch_id or not 1 <= candidate_id <= len(resumes):
            raise ValueError("Retrieved candidate does not belong to this submission")
        selected.append({**resumes[candidate_id - 1], "candidate_id": candidate_id,
                         "vector_distance": metadata["vector_score"],
                         "rerank_score": metadata["rerank_score"],
                         "retrieved_evidence": metadata["retrieved_evidence"]})
    return {"selected_resumes": selected, "retrieval": {
        "batch_id": batch_id, "scope": "current_submission", "indexed_resumes": len(resumes),
        "indexed_chunks": len(chunks), "retrieve_k": TOP_K_RETRIEVE, "rerank_k": TOP_K_RERANK,
        "selected_candidates": len(selected), "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL, "ranking_method": "cross_encoder",
    }}
