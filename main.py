from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from src.loaders.ingest import load_resumes, load_jobs, chunk_documents
from src.embeddings.embed_store import store_documents
from src.retrievers.retriever import retrieval_search
from src.prompt_chain.prompt_chain import generate_questions_for_resumes


if __name__ == "__main__":
    # documents = load_resumes(categories=["INFORMATION-TECHNOLOGY"]) #+ load_jobs()
    # chunks = chunk_documents(documents)
    # print(f"Loaded {len(chunks)} chunks")
    # store_documents(chunks)
    query = "give me a resume with C / C++ (Low-Level) AI / RAG Integration PostgreSQL / MySQL"
    results = retrieval_search(query, doc_type="resume")
    outputs = generate_questions_for_resumes(results, query=query)

    for output in outputs:
        print(output["response"])
        print("\n" + "=" * 40 + "\n")
        
        


