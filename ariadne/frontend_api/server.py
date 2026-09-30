"""Minimal loopback-only JSON API. Start with python -m ariadne.frontend_api.server."""
import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .service import BusyError, SearchService, validate_request

log = logging.getLogger(__name__)
MAX_BODY = 8192


def create_server(service, port: int = 8765):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def send_json(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/api/health":
                self.send_json(200, service.health())
            else:
                self.send_json(404, {"error": "Route not found."})

        def do_POST(self):
            if self.path != "/api/search":
                self.send_json(404, {"error": "Route not found."})
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json(415, {"error": "Send JSON content."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_BODY:
                self.send_json(413, {"error": "Request body exceeds the allowed size."})
                return
            try:
                payload = json.loads(self.rfile.read(length))
                query, top_k = validate_request(payload)
            except (ValueError, UnicodeDecodeError):
                self.send_json(400, {"error": "Enter a valid query of 1 to 2000 characters and top_k of 1 to 10."})
                return
            try:
                self.send_json(200, service.search(query, top_k))
            except BusyError:
                self.send_json(503, {"error": "Search is busy. Please try again."})
            except Exception:
                log.exception("Retrieval failed")
                self.send_json(500, {"error": "Search is temporarily unavailable. Please try again."})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="dense")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    service = SearchService(repo=args.repo, mode=args.mode)
    server = create_server(service, args.port)
    print(f"Ariadne API ready on http://127.0.0.1:{args.port}; mode={args.mode}; documents={len(service.documents)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
