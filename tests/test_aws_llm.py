"""The EC2 llm box: the AWS row over the llm column. Decommission ends the entrypoint and
nothing else; the systemd unit powers the instance off, and the instance terminates itself."""

import threading

import pytest

from aws.llm.server import AwsLlmServer
from contract import server
from support import FakeEngine, call, health_becomes, sleeper


@pytest.fixture
def box(monkeypatch):
    monkeypatch.setattr(server, "DECOMMISSION_DELAY_SECONDS", 0)
    monkeypatch.setattr(server, "PROBE_INTERVAL_SECONDS", 0.01)
    engine = FakeEngine()
    engine_proc, entrypoint_proc = sleeper(), sleeper()
    image = AwsLlmServer.from_argv([
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


def test_the_cell_serves_the_llm_column(box):
    box["engine"].up = True
    health_becomes(box["url"], 200)
    assert call(box["url"], "GET", "/metadata")[1]["capabilities"] == ["llm"]


def test_shutdown_ends_the_entrypoint_and_holds_no_provider_credential(box):
    assert call(box["url"], "POST", "/shutdown")[0] == 202
    assert box["entrypoint_proc"].wait(timeout=5) == -15
    assert call(box["url"], "GET", "/health")[0] == 410
    assert "boto" not in open("aws/server.py").read()
    assert "aws " not in open("aws/llm/entrypoint.sh").read()


def test_the_entrypoint_powers_the_instance_off_and_the_unit_starts_at_boot():
    entrypoint = open("aws/llm/entrypoint.sh").read()
    assert "systemctl poweroff" in entrypoint
    assert "169.254.169.254/latest/user-data" in entrypoint
    unit = open("aws/llm/gamesummoner-llm.service").read()
    assert "ExecStart=/opt/gamesummoner/entrypoint.sh" in unit
    assert "WantedBy=multi-user.target" in unit
    assert "ExecStopPost" not in unit
