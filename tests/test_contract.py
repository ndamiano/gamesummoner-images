"""The local llm box's contract server over real sockets, against a fake OpenAI-compatible
engine and a fake entrypoint process."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from contract import server
from local.llm.server import LocalLlmServer
from support import FakeEngine, call, health_becomes, sleeper

REPO = Path(__file__).resolve().parents[1]


def argv(engine_url, engine_pid, entrypoint_pid, port=0):
    return ["--image-id", "img-1", "--cli-params=--x", "--engine-url", engine_url,
            "--engine-pid", str(engine_pid), "--entrypoint-pid", str(entrypoint_pid),
            "--port", str(port)]


@pytest.fixture
def box(monkeypatch):
    monkeypatch.setattr(server, "DECOMMISSION_DELAY_SECONDS", 0)
    monkeypatch.setattr(server, "PROBE_INTERVAL_SECONDS", 0.01)
    engine = FakeEngine()
    engine_proc, entrypoint_proc = sleeper(), sleeper()
    image = LocalLlmServer.from_argv(argv(engine.url, engine_proc.pid, entrypoint_proc.pid))
    httpd = image.start()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{httpd.server_port}", "engine": engine,
           "engine_proc": engine_proc, "entrypoint_proc": entrypoint_proc, "image": image}
    engine_proc.kill()
    entrypoint_proc.kill()
    httpd.shutdown()
    engine.httpd.shutdown()


def test_health_follows_the_engine_through_boot_serving_and_death(box):
    assert call(box["url"], "GET", "/health")[0] == 503
    box["engine"].up = True
    status, body = health_becomes(box["url"], 200)
    assert "served-model" in body["reason"]
    box["engine_proc"].kill()
    box["engine_proc"].wait()
    status, body = call(box["url"], "GET", "/health")
    assert status == 500 and "exited" in body["reason"]


def test_health_answers_from_the_last_probe_instead_of_asking_the_engine(box):
    box["engine"].up = True
    health_becomes(box["url"], 200)
    box["engine"].httpd.shutdown()
    started = time.perf_counter()
    assert call(box["url"], "GET", "/health")[0] == 200
    assert time.perf_counter() - started < 1


def test_an_image_that_takes_no_work_answers_429_until_it_does(box, monkeypatch):
    verdict = [(429, "queue full")]
    monkeypatch.setattr(box["image"], "ready", lambda: verdict[0])
    assert health_becomes(box["url"], 429) == (429, {"reason": "queue full"})
    verdict[0] = (200, "serving")
    assert health_becomes(box["url"], 200) == (200, {"reason": "serving"})


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


def test_shutdown_turns_health_to_gone_and_terminates_the_entrypoint_once(box):
    box["engine"].up = True
    assert call(box["url"], "POST", "/shutdown")[0] == 202
    assert call(box["url"], "POST", "/shutdown")[0] == 202
    assert call(box["url"], "GET", "/health")[0] == 410
    assert box["entrypoint_proc"].wait(timeout=5) == -15


def test_the_image_runs_as_a_module_the_way_the_entrypoint_starts_it():
    """`python -m local.llm.server` with the repo on PYTHONPATH, as the Dockerfile lays it out."""
    engine, engine_proc, entrypoint_proc = FakeEngine(), sleeper(), sleeper()
    proc = subprocess.Popen(
        [sys.executable, "-m", "local.llm.server",
         *argv(engine.url, engine_proc.pid, entrypoint_proc.pid)],
        env={**os.environ, "PYTHONPATH": str(REPO)}, stderr=subprocess.PIPE, text=True)
    try:
        for line in proc.stderr:
            if "contract server on" in line:
                break
        else:
            raise AssertionError("server never reported its port")
        port = int(line.split("contract server on 127.0.0.1:")[1].split(",")[0])
        url = f"http://127.0.0.1:{port}"
        assert call(url, "GET", "/metadata")[1]["capabilities"] == ["llm"]
        assert call(url, "POST", "/shutdown")[0] == 202
        assert entrypoint_proc.wait(timeout=5) == -15
    finally:
        proc.kill()
        engine_proc.kill()
        entrypoint_proc.kill()
        engine.httpd.shutdown()
