"""Local testing server. Run: python web_app.py"""
import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import BoundedSemaphore
from pydantic import ValidationError
from src.graph.workflow import analyze

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
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.reply(415, {"error": "Send application/json"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 1_000_000:
                return self.reply(413, {"error": "Request must be between 1 byte and 1 MB"})
            payload = json.loads(self.rfile.read(size))
            from src.graph.models import AnalysisRequest
            AnalysisRequest.model_validate(payload)
        except (ValueError, ValidationError):
            return self.reply(400, {"error": "Provide a job description (20–20,000 characters) and 1–10 labeled resumes (20–50,000 characters each)."})
        if not BUSY.acquire(blocking=False):
            return self.reply(429, {"error": "An analysis is running. Try again shortly."})
        try:
            self.reply(200, analyze(payload))
        except Exception:
            logging.exception("Analysis failed")
            self.reply(502, {"error": "Analysis failed. Check Ollama is running, the configured model is pulled, and the embedding model is available. See server logs for details."})
        finally:
            BUSY.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Resume Lab: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
