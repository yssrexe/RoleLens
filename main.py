from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from src.loaders.ingest import load_resumes, load_jobs, chunk_documents
from src.embeddings.embed_store import store_documents, search_documents



if __name__ == "__main__":
    # documents = load_resumes(categories=["INFORMATION-TECHNOLOGY"]) + load_jobs()
    # chunks = chunk_documents(documents)
    # print(f"Loaded {len(chunks)} chunks")
    # store_documents(chunks)
    search_documents("Strong Â software and application knowledge such as Avaya,Microsoft Office,and Remedy", doc_type="resume")
