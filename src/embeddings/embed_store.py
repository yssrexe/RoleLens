"""Lazy PGVector access shared by the web workflow and offline RAG tools."""
from functools import lru_cache
from src.config import DB_CONNECTION, EMBEDDING_MODEL

collection_name = "resume_job_matching"


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{EMBEDDING_MODEL}",
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vector_store():
    from langchain_postgres.vectorstores import PGVector
    return PGVector(connection=DB_CONNECTION, collection_name=collection_name,
                    embeddings=get_embeddings(), use_jsonb=True,
                    engine_args={"pool_pre_ping": True})


def store_documents(chunks, *, ids=None, store=None):
    """Insert/upsert documents without dropping tables or other submissions."""
    store = get_vector_store() if store is None else store
    if not chunks:
        return []
    return store.add_documents(chunks, ids=ids)


def __getattr__(name):
    # Preserve imports used by existing notebooks without eager initialization.
    if name == "vector_store":
        return get_vector_store()
    if name == "embeddings":
        return get_embeddings()
    raise AttributeError(name)
