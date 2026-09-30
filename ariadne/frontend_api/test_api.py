"""Exercise real HTTP handling independently of expensive model initialization."""
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from .corpus import load_functions
from .server import create_server
from .service import BusyError, ROOT, validate_request


class FakeService:
    failure = None

    def health(self):
        return {"api_version": 1, "status": "ready", "mode": "dense", "documents": 1}

    def search(self, query, top_k):
        if self.failure:
            raise self.failure
        return {"api_version": 1, "status": "empty", "mode": "dense", "query": query, "elapsed_ms": 0.1, "results": []}


@pytest.fixture
def api():
    service = FakeService()
    server = create_server(service, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", service
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(api, path="/api/search", data=None, content_type="application/json"):
    endpoint = Request(api[0] + path, data=data, headers={"Content-Type": content_type})
    try:
        response = urlopen(endpoint, timeout=2)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


def test_health(api):
    assert request(api, "/api/health")[1]["documents"] == 1


def test_success_round_trip(api):
    status, payload = request(api, data=json.dumps({"query": " test ", "top_k": 3}).encode())
    assert status == 200 and payload["query"] == "test" and payload["results"] == []


@pytest.mark.parametrize("payload", [None, [], {}, {"query": 2}, {"query": " "}, {"query": "x" * 2001}, {"query": "ok", "top_k": True}, {"query": "ok", "top_k": 0}, {"query": "ok", "top_k": 11}])
def test_invalid_requests(api, payload):
    assert request(api, data=json.dumps(payload).encode())[0] == 400


def test_malformed_json(api):
    assert request(api, data=b"{")[0] == 400


def test_payload_limit(api):
    assert request(api, data=b"x" * 8193)[0] == 413


def test_content_type(api):
    assert request(api, data=b"test", content_type="text/plain")[0] == 415


def test_unknown_route(api):
    assert request(api, "/api/read-file")[0] == 404


@pytest.mark.parametrize("failure,code", [(BusyError("busy"), 503), (RuntimeError("secret exception detail"), 500)])
def test_errors_hide_internals(api, failure, code):
    api[1].failure = failure
    status, payload = request(api, data=b'{"query":"auth"}')
    assert status == code
    assert "secret" not in json.dumps(payload)


def test_function_coordinates_match_source():
    repo = ROOT / "ariadne/data/voice_assistant_js"
    records = load_functions(repo)
    assert len(records) > 10
    for record in records.values():
        lines = (repo / record["file"]).read_text(encoding="utf-8").splitlines()
        assert record["code"] == "\n".join(lines[record["start_line"] - 1:record["end_line"]])
    auth = next(record for record in records.values() if record["title"] == "verifySession")
    assert auth["file"] == "tools/authTool.js" and auth["start_line"] == 5 and auth["end_line"] == 8


def test_control_characters_are_removed():
    assert validate_request({"query": "\x00auth\x01"}) == ("auth", 5)
