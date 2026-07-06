import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   encode_kwargs={"normalize_embeddings": True})

CONNECTION_STRING = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

collection_name = "resume_job_matching"

vector_store = PGVector(
    connection=CONNECTION_STRING,
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


def search_documents(query: str, top_k: int = 5, doc_type: str = None):
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
        print(f"--- Result {i+1} ---")
        print(f"File      : {result.metadata.get('file_name')}")
        print(f"Category  : {result.metadata.get('category')}")
        print(f"Candidate :\n{candidate_info[:300]}")
        print(f"Matched   :\n{result.page_content[:300]}")
        print()
