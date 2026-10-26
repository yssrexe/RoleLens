"""FastAPI application. Run locally with: python web_app.py"""
import argparse
import json
import logging
from pathlib import Path
from threading import BoundedSemaphore
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from src.graph.models import AnalysisRequest
from src.graph.workflow import analyze
from src.loaders.uploads import MAX_REQUEST_BYTES, save_uploads

PAGE = Path(__file__).parent / "web" / "index.html"
BUSY = BoundedSemaphore(1)
app = FastAPI(title="RoleLens API", version="1.0.0")


@app.middleware("http")
async def response_headers(request: Request, call_next):
    started = perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-Ms"] = f"{(perf_counter() - started) * 1000:.2f}"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return PAGE.read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    """Liveness only; does not assert that Ollama or its models are ready."""
    return {"status": "ok"}


def process_analysis(content_type, body):
    """Run disk access and blocking model calls outside the event loop."""
    saved = None
    multipart = content_type.split(";")[0].strip().lower() == "multipart/form-data"
    try:
        if multipart:
            payload, saved = save_uploads(content_type, body)
        else:
            payload = AnalysisRequest.model_validate(json.loads(body)).model_dump()
        try:
            result = analyze(payload)
        except Exception:
            logging.exception("Analysis failed")
            return JSONResponse(status_code=502, content={
                "error": "Analysis could not be completed. Check PostgreSQL/PGVector, the embedding and reranker models, Ollama, and the server logs for details.",
                "saved_files": saved,
            })
        if saved:
            result["saved_files"] = saved
        return JSONResponse(content=result)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc) if multipart else
            "Provide a job description (20–20,000 characters) and 1–10 labeled resumes (20–50,000 characters each)."})
    except Exception:
        logging.exception("Upload failed")
        return JSONResponse(status_code=500, content={
            "error": "Could not save uploaded files. Check server logs and available disk space."})


@app.post("/api/analyze", openapi_extra={
    "requestBody": {"required": True, "content": {
        "application/json": {"schema": AnalysisRequest.model_json_schema(ref_template="#/components/schemas/{model}")},
        "multipart/form-data": {"schema": {
            "type": "object", "required": ["job_description", "resumes"], "properties": {
                "job_description": {"type": "string", "minLength": 20, "maxLength": 20000},
                "resumes": {"type": "array", "minItems": 1, "maxItems": 10,
                            "items": {"type": "string", "format": "binary"}},
            }}},
    }},
})
async def analyze_request(request: Request):
    """Analyze JSON resume text or uploaded PDFs with the local Ollama workflow."""
    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";")[0].strip().lower()
    if media_type not in ("application/json", "multipart/form-data"):
        return JSONResponse(status_code=415, content={"error": "Send PDF uploads with multipart/form-data or resume text as JSON."})
    limit = MAX_REQUEST_BYTES if media_type == "multipart/form-data" else 1_000_000
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            size = int(declared)
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "Invalid Content-Length."})
        if not 0 < size <= limit:
            return JSONResponse(status_code=413, content={"error": "Request exceeds the upload size limit."})
    if not BUSY.acquire(blocking=False):
        return JSONResponse(status_code=429, content={"error": "An analysis is running. Try again shortly."})
    try:
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > limit:
                return JSONResponse(status_code=413, content={"error": "Request exceeds the upload size limit."})
            body.extend(chunk)
        if not body or (declared is not None and len(body) != size):
            return JSONResponse(status_code=400, content={"error": "Incomplete upload."})
        return await run_in_threadpool(process_analysis, content_type, bytes(body))
    finally:
        BUSY.release()


# Register models referenced by the manually documented dual-format request body.
from fastapi.openapi.utils import get_openapi


def custom_openapi():
    if app.openapi_schema is None:
        schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
        definitions = AnalysisRequest.model_json_schema(ref_template="#/components/schemas/{model}").get("$defs", {})
        schema.setdefault("components", {}).setdefault("schemas", {}).update(definitions)
        app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
