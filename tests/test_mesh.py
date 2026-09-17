"""The mesh column: TRELLIS gates /health on `warm`, a sprite in is a glb out, and a sheet is
rendered beside it and scored cell by cell."""

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from contract import mesh, safety
from runpod.mesh.server import RunpodMeshServer
from support import call, health_becomes, reason_becomes, serve, sleeper


class FakeEngine:
    """One fake for both: TRELLIS (/health, /generate) and the sprite server (/health, /sheet)."""

    def __init__(self):
        self.health = {"status": "ok", "loaded": False, "warm": False}
        self.generate = [(200, b"GLB")]
        self.sheet = (200, {"sheet_b64": "x", "manifest": {}})
        self.calls = []
        engine = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, status, body, headers=()):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                for k, v in headers:
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, engine.health)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                engine.calls.append((self.path, raw))
                if self.path.startswith("/generate"):
                    status, body = engine.generate.pop(0)
                    self._send(status, body, headers=[("X-Generate-Seconds", "7.5"),
                                                      ("X-Load-Seconds", "30")])
                elif self.path == "/sheet":
                    self._send(*engine.sheet)
                else:
                    self._send(404, {})

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"


@pytest.fixture
def box(monkeypatch):
    trellis, sprite = FakeEngine(), FakeEngine()
    sprite.health = {"status": "ok", "loaded": True, "verbs": ["walk"]}
    engine_proc, entrypoint_proc = sleeper(), sleeper()
    image, httpd, url = serve(RunpodMeshServer, [
        "--image-id", "img-1", "--engine-url", trellis.url, "--sprite-url", sprite.url,
        "--engine-pid", str(engine_proc.pid), "--entrypoint-pid", str(entrypoint_proc.pid)],
        monkeypatch)
    yield {"url": url, "trellis": trellis, "sprite": sprite}
    engine_proc.kill()
    entrypoint_proc.kill()
    httpd.shutdown()
    trellis.httpd.shutdown()
    sprite.httpd.shutdown()


def test_health_waits_for_trellis_to_be_warm_not_merely_up(box):
    assert reason_becomes(box["url"], "TRELLIS loading") == 503
    box["trellis"].health.update(loaded=True)
    assert reason_becomes(box["url"], "TRELLIS warming up") == 503
    box["trellis"].health.update(warm=True)
    health_becomes(box["url"], 200)
    assert call(box["url"], "GET", "/metadata")[1]["capabilities"] == ["mesh"]


def test_a_sprite_becomes_a_glb_with_the_servers_timing(box):
    status, body = call(box["url"], "POST", "/job", {
        "jobId": "j1", "type": "trellis_mesh",
        "payload": {"image_b64": base64.b64encode(b"PNG").decode(), "seed": 7}})
    assert status == 200
    assert base64.b64decode(body["result"]["glb_b64"]) == b"GLB"
    assert body["result"]["generate_seconds"] == "7.5"
    path, raw = box["trellis"].calls[0]
    assert path == "/generate?seed=7" and raw == b"PNG"


def test_a_failed_generate_is_retried_once_then_reported(box):
    box["trellis"].generate = [(500, b"CuMesh OOM"), (200, b"GLB2")]
    status, body = call(box["url"], "POST", "/job", {
        "jobId": "j1", "type": "trellis_mesh",
        "payload": {"image_b64": base64.b64encode(b"PNG").decode()}})
    assert status == 200 and base64.b64decode(body["result"]["glb_b64"]) == b"GLB2"
    box["trellis"].generate = [(500, b"a"), (500, b"b")]
    status, body = call(box["url"], "POST", "/job", {
        "jobId": "j2", "type": "trellis_mesh",
        "payload": {"image_b64": base64.b64encode(b"PNG").decode()}})
    assert status == 502 and "TRELLIS failed: Status 500: b" in body["error"]


def _sheet_png(cells_wide=2, cells_high=2, cell=8):
    Image = pytest.importorskip("PIL.Image")
    import io
    buf = io.BytesIO()
    Image.new("RGB", (cells_wide * cell, cells_high * cell), "white").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_a_rendered_sheet_is_scored_cell_by_cell_and_the_worst_cell_speaks(box, monkeypatch):
    box["sprite"].sheet = (200, {"sheet_b64": _sheet_png(2, 2, 8),
                                 "manifest": {"cell": {"w": 8, "h": 8}}})
    seen = []

    def classify(png):
        seen.append(png)
        return {"scores": {"NSFW": 0.9 if len(seen) == 3 else 0.1}}

    monkeypatch.setattr(safety, "classify", classify)
    status, body = call(box["url"], "POST", "/job", {
        "jobId": "j1", "type": "sprite_sheet", "payload": {"glb_b64": "g", "anims": []}})
    assert status == 200 and len(seen) == 4
    assert body["result"]["safety"] == {"scores": {"NSFW": 0.9}}
    assert box["sprite"].calls[0][0] == "/sheet"


def test_a_sheet_that_is_not_an_image_carries_a_verdict_that_refuses_it(box, monkeypatch):
    pytest.importorskip("PIL")
    monkeypatch.setattr(safety, "classify", lambda b: {"scores": {"NSFW": 0.0}})
    status, body = call(box["url"], "POST", "/job", {
        "jobId": "j1", "type": "sprite_sheet", "payload": {"glb_b64": "g", "anims": []}})
    assert status == 200
    assert "scores" not in body["result"]["safety"] and body["result"]["safety"]["error"]


def test_a_fallback_sheet_is_not_scored(box, monkeypatch):
    box["sprite"].sheet = (200, {"fallback": "not a humanoid"})
    monkeypatch.setattr(safety, "classify",
                        lambda b: (_ for _ in ()).throw(AssertionError("scored a fallback")))
    status, body = call(box["url"], "POST", "/job", {
        "jobId": "j1", "type": "sprite_sheet", "payload": {"glb_b64": "g", "anims": []}})
    assert status == 200 and "safety" not in body["result"]
    assert body["result"]["fallback"] == "not a humanoid"


def test_a_dead_sheet_server_is_a_502(box):
    box["sprite"].httpd.shutdown()
    box["sprite"].httpd.server_close()
    status, body = call(box["url"], "POST", "/job", {
        "jobId": "j1", "type": "sprite_sheet", "payload": {"glb_b64": "g", "anims": []}})
    assert status == 502 and "unreachable" in body["error"]


def test_a_box_without_a_sprite_server_refuses_sheet_jobs(monkeypatch):
    trellis = FakeEngine()
    engine_proc, entrypoint_proc = sleeper(), sleeper()
    image, httpd, url = serve(RunpodMeshServer, [
        "--image-id", "img-1", "--engine-url", trellis.url,
        "--engine-pid", str(engine_proc.pid), "--entrypoint-pid", str(entrypoint_proc.pid)],
        monkeypatch)
    try:
        status, body = call(url, "POST", "/job", {
            "jobId": "j1", "type": "sprite_sheet", "payload": {"glb_b64": "g", "anims": []}})
        assert status == 500 and "no sprite server" in body["error"]
    finally:
        engine_proc.kill()
        entrypoint_proc.kill()
        httpd.shutdown()
        trellis.httpd.shutdown()
