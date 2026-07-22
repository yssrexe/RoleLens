from src.embeddings.embed_store import vector_store

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

def retrieval_search(query: str, top_k: int = 5, doc_type: str = None):
    if doc_type:
        retriever.search_kwargs["filter"] = {"doc_type": doc_type}
    else:
        retriever.search_kwargs.pop("filter", None)
    docs = retriever.invoke(query)
    seen = set()
    unique = []
    for doc in docs:
        src = doc.metadata["source"]
        if src not in seen:
            seen.add(src)
            unique.append(doc)
        if len(unique) == top_k:
            break
    return unique
