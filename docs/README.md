# RoleLens — FastAPI, LangGraph, and local RAG

RoleLens uploads PDF resumes, indexes them in PostgreSQL/PGVector, retrieves passages for a job description, reranks candidates with a cross-encoder, and generates technical interview questions using local Ollama inference.

For the full browser-to-output explanation, read [interview_prep.md](../interview_prep.md).

## Request flow

```text
Browser → FastAPI (web_app.py) → LangGraph (src/graph/workflow.py)
  → clean/chunk resumes → embed and insert into PGVector
  → retrieve top 4 chunks from this submission
  → cross-encoder rerank → select up to 2 distinct candidates
  → Ollama profile extraction → supporting fit explanation
  → Ollama technical questions using retrieved evidence
  → JSON → browser cards
```

The default workflow directly calls the shared [vector store](../src/embeddings/embed_store.py) and [retriever](../src/retrievers/retriever.py). There is no in-memory-only fallback when PostgreSQL is unavailable.

Search is limited to **the current submission**. Indexing preserves older database records. Each submission has its own retrieval batch ID, and duplicate filenames remain distinct through candidate IDs.

## Models and ranking

| Role | Model |
| --- | --- |
| Embeddings | `all-MiniLM-L6-v2` |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Structured extraction and question generation | Ollama `llama3.2:latest` by default |

The cross-encoder reranks every retrieved passage before grouping by candidate. A candidate's best retrieved passage determines its rank. The browser shows the raw cross-encoder score, supporting fit score, profile/gaps, and retrieved evidence. Raw reranker scores are not probabilities. Supporting fit scores do not change the cross-encoder order.

The defaults in [src/config.py](../src/config.py) retrieve **4 chunks**, then keep up to **2 candidates**. Four chunks can represent fewer than four candidates, and not every uploaded resume necessarily reaches the shortlist. All submitted resumes are indexed. The question generator focuses on technical skills, implementation, debugging, and practical experience. Education can remain a supporting profile-scoring signal but is not an interview topic.

## Run with Docker on this Linux machine

Ollama runs on the host; Docker runs the app and PGVector database. Ensure `ollama list` includes `llama3.2:latest`, or download it with `ollama pull llama3.2:latest`. Start `ollama serve` if it is not already running.

Copy `.env.example` to `.env`, then configure it using your own database credentials:

```dotenv
DB_USER=rolelens
DB_PASSWORD=replace-with-your-own-password
DB_NAME=rolelens
OLLAMA_MODEL=llama3.2:latest
APP_PORT=8000
HOST_DB_PORT=5433
```

```bash
docker compose -f compose.yaml -f compose.host-ollama.yaml up -d --build
```

Open [the web page](http://127.0.0.1:8000) or [API documentation](http://127.0.0.1:8000/docs).

The Linux override uses host networking for the app: Ollama is at `127.0.0.1:11434` and PostgreSQL at host port 5433 by default. The app binds to localhost. The base Compose configuration instead uses the Docker bridge, `db:5432`, and `host.docker.internal:11434`; that Ollama address requires a host listener reachable from Docker.

The app image includes the Python dependencies, but model weights may download on first use. The named model-cache volume retains embedding and cross-encoder files. Ollama model storage remains on the host.

## Run Python locally

Use Python 3.12, a running PostgreSQL server with PGVector, and running Ollama:

```bash
python3 -m venv .venv-agents
source .venv-agents/bin/activate
pip install -r requirements-agents.txt
python web_app.py
```

Set `DB_HOST`, `DB_PORT`, and credentials to the database reachable from that Python process. To use the Linux Compose database, the host/port are normally `127.0.0.1:5433`. `python web_app.py` does not start PostgreSQL or Ollama. Do not start it on a port already used by the Docker app; `--port 8080` selects another port.

## Uploads and persistence

Upload 1–10 text-based, unencrypted PDFs, each at most 5 MiB and 30 pages. Job descriptions need 20–20,000 characters. Each extracted resume needs 20–50,000 readable characters. Image-only scans require OCR first.

| Storage | Contents |
| --- | --- |
| `data/jobs/<upload_batch>.txt` | Original entered job description |
| `data/resumes/<upload_batch>/<file_id>.pdf` | Original uploaded PDFs with generated disk filenames |
| PGVector `resume_job_matching` collection | Resume chunks, vectors, and submission/source metadata |
| `postgres_data` named volume | Persistent PostgreSQL data |
| `model_cache` named volume | Hugging Face model cache |

Invalid upload batches are cleaned up. Valid uploads remain saved if later database/model work fails. Indexing adds records and never clears existing tables. Upload and retrieval batches have separate IDs. Retries generate new batches; retention and deduplication across submissions are not implemented. Changing `.env` credentials does not automatically change credentials in an existing initialized database volume.

## API and failure handling

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Browser interface |
| GET | `/api/health` | HTTP liveness only |
| POST | `/api/analyze` | Multipart PDFs or JSON resume text; same RAG workflow |
| GET | `/docs` | Interactive OpenAPI documentation |
| GET | `/openapi.json` | API schema |

Multipart fields are `job_description` and repeated `resumes`. For JSON:

```json
{
  "job_description": "Backend engineer requiring Python and Docker experience.",
  "resumes": [{"label": "Candidate 1", "text": "Developer with two years of Python backend experience."}]
}
```

Responses include `requirements`, `candidates`, `stages`, `retrieval`, and, for saved uploads, `saved_files`. Candidates expose `rerank_score`, `vector_distance`, `retrieved_evidence`, supporting fit information, and questions.

HTTP 400 means invalid input; 413 means too large; 415 means unsupported media; 429 means the single analysis slot is occupied; 500 indicates unexpected upload failure; 502 indicates workflow/dependency/output-validation failure. `/api/health` does not check database or model readiness. The analysis limit is per process, so retain one Uvicorn worker unless concurrency management is redesigned.

```bash
docker compose -f compose.yaml -f compose.host-ollama.yaml logs --tail 100 app
docker compose -f compose.yaml -f compose.host-ollama.yaml ps
```

## Verification

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

These deterministic tests exercise production chunking, filtering, reranking aggregation, workflow/API integration, validation, and evidence delivery using database/model doubles.

For a real integration check against the running deployment:

```bash
docker compose cp scripts/verify_rag.py app:/tmp/verify_rag.py
docker compose exec -T -e PYTHONPATH=/app app python -u /tmp/verify_rag.py
```

This check uses real embeddings, PGVector, the cross-encoder, Ollama, and multipart PDF uploads. It verifies current-batch isolation and existing-record preservation, and leaves synthetic data for inspection. The report is written inside the app container to `/tmp/rolelens-rag-verification.json`. Add `--retrieval-only` to check real indexing, reranking, batch isolation and preservation without repeating LLM inference; that report is `/tmp/rolelens-rag-retrieval-verification.json`. The PGVector connection pool checks connection health before reuse after database restarts.

Full inference can take minutes, particularly on CPU or first model load. Previous timings for the pre-RAG browser path do not establish latency for this connected pipeline. [scripts/benchmark_api.py](../scripts/benchmark_api.py) measures complete JSON API requests with warmup handling; it does not include browser PDF-upload/extraction overhead.
