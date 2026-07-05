import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.append(str(Path(__file__).resolve().parents[1] / "loaders"))
from ingest import load_resumes, load_jobs, chunk_documents

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   encode_kwargs={"normalize_embeddings": True})


DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

collection_name = "resume_job_matching"

vector_store = PGVector(
    connection=CONNECTION_STRING,
    collection_name=collection_name,
    embeddings=embeddings,
    use_jsonb=True
)



def store_documents(chunks):
    print(f"Storing {len(chunks)} chunks in the vector store...")
    vector_store.add_documents(chunks)
    print("insert completed")

def search_documents(query : str, top_k : int=5):
    print(f"search for {query}")    
    results = vector_store.similarity_search(query, k=top_k)
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result.metadata['source']}")
        
if __name__ == "__main__":
    documents = load_resumes(categories=["INFORMATION-TECHNOLOGY"]) + load_jobs()
    chunks = chunk_documents(documents)
    print(f"Loaded {len(chunks)} chunks")
    store_documents(chunks)
    search_documents("Python backend engineer with FastAPI experience")
