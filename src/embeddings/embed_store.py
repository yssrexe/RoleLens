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


