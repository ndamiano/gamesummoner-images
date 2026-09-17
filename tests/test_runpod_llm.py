"""The RunPod llm box: the RunPod row over the llm column. Decommission ends the entrypoint and
nothing else; the pod is the control plane's to terminate."""

import threading

import pytest

from contract import server
from runpod.llm.server import RunpodLlmServer
from support import FakeEngine, call, health_becomes, sleeper


@pytest.fixture
def pod(monkeypatch):
    monkeypatch.setattr(server, "DECOMMISSION_DELAY_SECONDS", 0)
    monkeypatch.setattr(server, "PROBE_INTERVAL_SECONDS", 0.01)
    engine = FakeEngine()
    engine_proc, entrypoint_proc = sleeper(), sleeper()
    image = RunpodLlmServer.from_argv([
        "--image-id", "img-1", "--engine-url", engine.url, "--engine-pid", str(engine_proc.pid),
        "--entrypoint-pid", str(entrypoint_proc.pid), "--port", "0"])
    httpd = image.start()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{httpd.server_port}", "engine": engine,
           "entrypoint_proc": entrypoint_proc}
    engine_proc.kill()
    entrypoint_proc.kill()
    httpd.shutdown()
    engine.httpd.shutdown()


def test_the_cell_serves_the_llm_column(pod):
    pod["engine"].up = True
    health_becomes(pod["url"], 200)
    assert call(pod["url"], "GET", "/metadata")[1]["capabilities"] == ["llm"]


def test_shutdown_ends_the_entrypoint_and_holds_no_provider_credential(pod):
    assert call(pod["url"], "POST", "/shutdown")[0] == 202
    assert pod["entrypoint_proc"].wait(timeout=5) == -15
    assert call(pod["url"], "GET", "/health")[0] == 410
    assert "RUNPOD" not in open("runpod/server.py").read()
