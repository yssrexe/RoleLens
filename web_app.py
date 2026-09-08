"""Local testing server. Run: python web_app.py"""
import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore
from pydantic import ValidationError
from src.graph.workflow import analyze
from src.loaders.uploads import save_uploads, MAX_REQUEST_BYTES

PAGE = Path(__file__).parent / "web" / "index.html"
BUSY = BoundedSemaphore(1)

class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body, content_type="application/json"):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self.reply(200, PAGE.read_text(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self.reply(200, {"status": "ok"})
        else:
            self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/analyze":
            return self.reply(404, {"error": "Not found"})
        content_type = self.headers.get("Content-Type", "")
        media_type = content_type.split(";")[0].strip().lower()
        if media_type not in ("application/json", "multipart/form-data"):
            return self.reply(415, {"error": "Send PDF uploads with multipart/form-data or resume text as JSON."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            limit = MAX_REQUEST_BYTES if media_type == "multipart/form-data" else 1_000_000
            if not 0 < size <= limit:
                return self.reply(413, {"error": "Request exceeds the upload size limit."})
        except ValueError:
            return self.reply(400, {"error": "Invalid Content-Length."})
        if not BUSY.acquire(blocking=False):
            return self.reply(429, {"error": "An analysis is running. Try again shortly."})
        saved = None
        try:
            body = self.rfile.read(size)
            if len(body) != size:
                raise ValueError("Incomplete upload.")
            if media_type == "multipart/form-data":
                payload, saved = save_uploads(content_type, body)
            else:
                payload = json.loads(body)
                from src.graph.models import AnalysisRequest
                AnalysisRequest.model_validate(payload)
            try:
                result = analyze(payload)
            except Exception:
                logging.exception("Analysis failed")
                return self.reply(502, {"error": "Analysis failed. Check Ollama is running and the configured models are available.", "saved_files": saved})
            if saved:
                result["saved_files"] = saved
            self.reply(200, result)
        except (ValueError, ValidationError) as exc:
            self.reply(400, {"error": str(exc) if media_type == "multipart/form-data" else "Provide a job description (20–20,000 characters) and 1–10 labeled resumes (20–50,000 characters each)."})
        except Exception:
            logging.exception("Upload failed")
            self.reply(500, {"error": "Could not save uploaded files. Check server logs and available disk space."})
        finally:
            BUSY.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"RoleLens: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
