"""Shared test doubles: a fake OpenAI-compatible engine, a throwaway process, an HTTP caller."""

import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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


def sleeper():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def health_becomes(base, status, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        got, body = call(base, "GET", "/health")
        if got == status:
            return got, body
        time.sleep(0.01)
    raise AssertionError(f"/health never became {status}, last was {got}: {body}")


def reason_becomes(base, fragment, timeout=5):
    """Poll /health until its reason contains `fragment`; returns the status then."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, body = call(base, "GET", "/health")
        if fragment in (body.get("reason") or ""):
            return status
        time.sleep(0.01)
    raise AssertionError(f"/health never said {fragment!r}, last was {body}")


def call(base, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


class FakeComfy:
    """ComfyUI's surface as the contract uses it. `images(workflow)` decides what a prompt
    outputs: a list of (filename, bytes)."""

    def __init__(self, images=None):
        self.up = True
        self.uploads = {}
        self.prompts = {}
        self.outputs = {}
        self.images = images or (lambda wf: [("out.png", b"\x89PNG fake")])
        self.fail_next = None
        comfy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, status, body, raw=False):
                data = body if raw else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if not comfy.up:
                    self._send(503, {})
                elif self.path == "/system_stats":
                    self._send(200, {"system": {}})
                elif self.path.startswith("/history/"):
                    pid = self.path.rsplit("/", 1)[1]
                    if pid not in comfy.outputs:
                        self._send(200, {})
                        return
                    files = [{"filename": n, "subfolder": "", "type": "output"}
                             for n, _ in comfy.outputs[pid]]
                    images = [f for f in files if not f["filename"].endswith(".glb")]
                    meshes = [f for f in files if f["filename"].endswith(".glb")]
                    self._send(200, {pid: {"outputs": {"9": {"images": images},
                                                       "10": {"3d": meshes}},
                                           "status": {"status_str": "success"}}})
                elif self.path.startswith("/view?"):
                    q = urllib.parse.parse_qs(self.path.split("?", 1)[1])
                    name = q["filename"][0]
                    for outs in comfy.outputs.values():
                        for n, data in outs:
                            if n == name:
                                self._send(200, data, raw=True)
                                return
                    self._send(404, {})
                else:
                    self._send(404, {})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                if self.path == "/upload/image":
                    name = raw.split(b'filename="', 1)[1].split(b'"', 1)[0].decode()
                    comfy.uploads[name] = raw
                    self._send(200, {"name": name})
                elif self.path == "/prompt":
                    if comfy.fail_next:
                        status, body = comfy.fail_next
                        comfy.fail_next = None
                        self._send(status, body)
                        return
                    wf = json.loads(raw)["prompt"]
                    pid = f"p{len(comfy.prompts) + 1}"
                    comfy.prompts[pid] = wf
                    comfy.outputs[pid] = comfy.images(wf)
                    self._send(200, {"prompt_id": pid})
                else:
                    self._send(404, {})

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"


def png_bytes(im) -> bytes:
    import io
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def serve(server_cls, argv, monkeypatch):
    """A contract server on a free port with the probe and decommission delays collapsed."""
    from contract import server as contract_server
    monkeypatch.setattr(contract_server, "DECOMMISSION_DELAY_SECONDS", 0)
    monkeypatch.setattr(contract_server, "PROBE_INTERVAL_SECONDS", 0.01)
    image = server_cls.from_argv([*argv, "--port", "0"])
    httpd = image.start()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return image, httpd, f"http://127.0.0.1:{httpd.server_port}"
