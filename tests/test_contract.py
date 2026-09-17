"""Contract server over real sockets, against a fake OpenAI-compatible engine."""

import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contract"))

import server  # noqa: E402
from llm import LlmAdapter  # noqa: E402


class FakeEngine:
    def __init__(self):
        self.up = False
        self.requests = []
        self.reply = (200, {"choices": [{"message": {"role": "assistant", "content": "hi"}}]})
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if not engine.up:
                    self._send(503, {})
                else:
                    self._send(200, {"data": [{"id": "served-model"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                engine.requests.append((self.path, body))
                self._send(*engine.reply)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"


def call(base, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


@pytest.fixture
def box(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DECOMMISSION_DELAY_SECONDS", 0)
    engine = FakeEngine()
    engine_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    marker = tmp_path / "decommissioned"
    image = server.Image(LlmAdapter(engine.url),
                         {"imageId": "img-1", "cliParams": "--x", "capabilities": ["llm"]},
                         f"touch {marker}", engine_proc.pid)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(image))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{httpd.server_port}", "engine": engine,
           "engine_proc": engine_proc, "marker": marker}
    engine_proc.kill()
    httpd.shutdown()
    engine.httpd.shutdown()


def test_health_follows_the_engine_through_boot_serving_and_death(box):
    assert call(box["url"], "GET", "/health")[0] == 503
    box["engine"].up = True
    status, body = call(box["url"], "GET", "/health")
    assert status == 200 and "served-model" in body["reason"]
    box["engine_proc"].kill()
    box["engine_proc"].wait()
    status, body = call(box["url"], "GET", "/health")
    assert status == 500 and "exited" in body["reason"]


def test_metadata_reports_the_build_and_its_capabilities(box):
    assert call(box["url"], "GET", "/metadata") == (
        200, {"imageId": "img-1", "cliParams": "--x", "capabilities": ["llm"]})


def test_job_translates_the_canonical_body_and_returns_the_engine_reply(box):
    box["engine"].up = True
    canonical = {"model": "served-model", "messages": [{"role": "user", "content": "yo"}],
                 "reasoning": "high", "max_tokens": 10, "tools": [{"type": "function"}]}

    status, body = call(box["url"], "POST", "/job",
                        {"jobId": "j1", "type": "llm", "payload": {"kind": "llm",
                                                                   "body": canonical}})

    assert status == 200
    assert body == {"status": "succeeded", "result": box["engine"].reply[1]}
    path, sent = box["engine"].requests[0]
    assert path == "/v1/chat/completions"
    assert sent == {"model": "served-model", "messages": canonical["messages"],
                    "reasoning_effort": "high", "max_tokens": 10,
                    "tools": [{"type": "function"}]}


def test_job_failures_are_5xx_with_the_reason(box):
    box["engine"].reply = (400, {"message": "context length exceeded"})
    status, body = call(box["url"], "POST", "/job",
                        {"jobId": "j1", "type": "llm", "payload": {"body": {}}})
    assert status == 502 and body["status"] == "failed"
    assert body["error"].startswith("Status 400") and "context length" in body["error"]

    status, body = call(box["url"], "POST", "/job",
                        {"jobId": "j2", "type": "comfy_image", "payload": {}})
    assert status == 500 and "comfy_image" in body["error"]


def test_shutdown_turns_health_to_gone_and_runs_the_decommission_once(box):
    box["engine"].up = True
    assert call(box["url"], "POST", "/shutdown")[0] == 202
    assert call(box["url"], "POST", "/shutdown")[0] == 202
    assert call(box["url"], "GET", "/health")[0] == 410
    deadline = time.time() + 5
    while not box["marker"].exists() and time.time() < deadline:
        time.sleep(0.05)
    assert box["marker"].exists()
