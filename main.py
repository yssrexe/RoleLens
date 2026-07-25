from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from src.loaders.ingest import load_resumes, load_jobs, chunk_documents
from src.embeddings.embed_store import store_documents
from src.retrievers.retriever import retrieval_search


if __name__ == "__main__":
    documents = load_resumes(categories=["INFORMATION-TECHNOLOGY"]) + load_jobs()
    chunks = chunk_documents(documents)
    print(f"Loaded {len(chunks)} chunks")
    store_documents(chunks)
    results = retrieval_search("give me a resume with C / C++ (Low-Level) AI / RAG Integration PostgreSQL / MySQL", doc_type="resume")
    for result in results:
        print(
            f"Source: {result.metadata['source']}, "
            f"Vector score: {result.metadata.get('vector_score', 'N/A')}, "
            f"Rerank score: {result.metadata.get('rerank_score', 'N/A')}"
        )
