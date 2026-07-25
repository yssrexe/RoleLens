# Resume RAG Project

This project builds a retrieval system for matching resumes with job descriptions using:

- document loading and cleaning
- chunking
- PostgreSQL vector search with embeddings
- cross-encoder reranking for better result quality

The main goal is to take a query like a job requirement and return the most relevant resume chunks, or take a resume query and return the best matching records from the vector store.

## What I changed recently

I updated the retrieval layer so it no longer relies only on raw vector similarity.

Before the change, `retrieval_search()`:

- queried the vector store for similar chunks
- deduplicated results by source file
- returned the first few chunks directly

After the change, `retrieval_search()` now:

- retrieves a larger candidate set from PGVector
- removes duplicates by source
- runs a cross encoder over each query/chunk pair
- sorts results by rerank score
- returns the highest ranked chunks

I also updated `main.py` so the demo prints both:

- the vector similarity score
- the rerank score from the cross encoder

## How retrieval search works now

The retrieval flow is in [src/retrievers/retriever.py](src/retrievers/retriever.py).

### 1. Vector retrieval

The system first gets a candidate list from the vector store:

- it uses `vector_store.similarity_search_with_score()`
- it retrieves `candidate_k` chunks, which defaults to `TOP_K_RETRIEVE`
- it optionally filters by `doc_type`, such as `resume` or `job_description`

This step is fast and gives broad semantic matches.

### 2. Deduplication

Some documents produce multiple chunks, so the same source file can appear more than once.

To avoid returning repeated chunks from the same file, the code keeps only one candidate per source path.

### 3. Cross encoder reranking

The deduplicated candidates are then reranked with a cross encoder.

The reranker is loaded from the model configured in [src/config.py](src/config.py):

- `RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"`

For every candidate, the code builds an input pair:

- `[query, chunk_text]`

The cross encoder scores the pair directly and produces a relevance score.

### 4. Final ordering

The results are sorted by rerank score in descending order and the top `top_k` items are returned.

Each returned document now includes:

- `vector_score`
- `rerank_score`

## Why use a cross encoder

Vector search is good at finding candidates quickly, but it is not always the best final ranker.

A cross encoder is useful because it:

- compares the query and document together
- gives a more precise relevance score
- usually improves ranking quality over pure similarity search

In this project, the vector store does the broad search and the cross encoder does the final ranking.

## Key files

### [src/retrievers/retriever.py](src/retrievers/retriever.py)

This is the main file for retrieval.

It now contains:

- `get_reranker()` to lazily load the cross encoder
- `retrieval_search()` to run vector retrieval and reranking

### [src/embeddings/embed_store.py](src/embeddings/embed_store.py)

This file sets up:

- HuggingFace embeddings
- the `PGVector` vector store
- document storage into PostgreSQL

### [src/config.py](src/config.py)

This file holds the important model and retrieval settings:

- `EMBEDDING_MODEL`
- `RERANKER_MODEL`
- `TOP_K_RETRIEVE`
- `TOP_K_RERANK`

### [main.py](main.py)

This is the demo entrypoint.

It currently:

- loads resumes and jobs
- chunks the documents
- stores them in the vector store
- runs `retrieval_search()`
- prints the source and both scores

## Example usage

```python
from src.retrievers.retriever import retrieval_search

results = retrieval_search(
	"give me a resume with C / C++ (Low-Level) AI / RAG Integration PostgreSQL / MySQL",
	doc_type="resume",
)

for result in results:
	print(result.metadata["source"])
	print(result.metadata.get("vector_score"))
	print(result.metadata.get("rerank_score"))
```

## Current retrieval parameters

The defaults currently come from [src/config.py](src/config.py):

- `TOP_K_RETRIEVE = 10`
- `TOP_K_RERANK = 5`

That means the system:

- fetches 10 candidates from PGVector
- reranks them with the cross encoder
- returns the best 5

## Notes about the current implementation

- The cross encoder is loaded lazily and cached so it is not recreated on every search.
- The demo currently focuses on the `INFORMATION-TECHNOLOGY` resume category.
- `store_documents()` drops and recreates the table before inserting documents, which is useful for development but not ideal for production.

## Next improvements

Possible follow-up work:

- expose `candidate_k` and `top_k` through a CLI or API
- combine vector score and rerank score into one final score
- add tests for reranking order
- add a small API or UI on top of `retrieval_search()`

