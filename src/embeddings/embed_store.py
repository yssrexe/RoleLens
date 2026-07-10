import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DB_CONNECTION, EMBEDDING_MODEL

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{EMBEDDING_MODEL}",
                                   encode_kwargs={"normalize_embeddings": True})



collection_name = "resume_job_matching"

vector_store = PGVector(
    connection=DB_CONNECTION,
    collection_name=collection_name,
    embeddings=embeddings,
    use_jsonb=True
)


def store_documents(chunks):
    vector_store.drop_tables()
    vector_store.create_tables_if_not_exists()
    vector_store.create_collection()
    print(f"Storing {len(chunks)} chunks in the vector store...")
    vector_store.add_documents(chunks)
    print("insert completed")


def get_candidate_info(source_path: str):
    results = vector_store.similarity_search(
        "name contact email phone address",
        k=1,
        filter={"source": source_path}
    )
    return results[0].page_content if results else "No info found"


def search_documents(query: str, top_k: int = 1, doc_type: str = None):
    print(f"search for: {query}\n")
    filter = {"doc_type": doc_type} if doc_type else None
    results = vector_store.similarity_search(query, k=top_k * 3, filter=filter)
    seen = set()
    unique = []
    for result in results:
        src = result.metadata["source"]
        if src not in seen:
            seen.add(src)
            unique.append(result)
        if len(unique) == top_k:
            break
    for i, result in enumerate(unique):
        candidate_info = get_candidate_info(result.metadata["source"])
        print(f"\nmetadata:\n{result.metadata}\n")
