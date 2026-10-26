# RoleLens

A local resume-analysis app built with FastAPI, LangGraph, PostgreSQL/PGVector, and Ollama. Upload PDF resumes alongside a job description to retrieve relevant candidates and generate technical interview questions.

## How it works

1. Save and extract text from 1–10 PDF resumes.
2. Chunk the text and create embeddings with `all-MiniLM-L6-v2`.
3. Retrieve matching passages from the current submission using PGVector.
4. Rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2` and shortlist up to two candidates.
5. Use Ollama (`llama3.2:latest` by default) to extract profiles and generate questions.

## Quick start on Linux

Requires Docker Compose and Ollama running on the host.

```bash
cp .env.example .env
# Edit .env and set DB_PASSWORD to your own password.
ollama pull llama3.2:latest
# Run ollama serve if Ollama is not already running.
docker compose -f compose.yaml -f compose.host-ollama.yaml up -d --build
```

Open **http://localhost:8000**. API documentation is at **http://localhost:8000/docs**. Model weights download on first use, which can take time.

See the [setup and API guide](docs/README.md) for local Python setup, Docker networking, storage, and integration checks. The [workflow walkthrough](interview_prep.md) explains the code in more detail.

## Tests

With Python 3.12 and `requirements-agents.txt` installed:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The unit tests use database and model doubles; they do not require running Ollama or PostgreSQL.

## Local data

Resumes, job descriptions, generated reports, virtual environments, and `.env` files are excluded from version control. Supply your own PDFs; no resume dataset is required to start the web app. Uploads are created under `data/`, and database records persist in a Docker volume.

This app is intended for local use. Scores support human review; they are not hiring decisions. See the setup guide for upload limits and failure behavior.
