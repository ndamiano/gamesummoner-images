"""The image column on ComfyUI: warm before serving, one workflow in, images with verdicts out."""

import base64

import pytest

from contract import safety
from runpod.image.server import RunpodImageServer
from support import FakeComfy, call, health_becomes, reason_becomes, serve, sleeper


@pytest.fixture
def box(monkeypatch):
    monkeypatch.setattr(safety, "classify", lambda data: {"scores": {"NSFW": 0.01}})
    comfy = FakeComfy()
    comfy.up = False
    engine_proc, entrypoint_proc = sleeper(), sleeper()
    image, httpd, url = serve(RunpodImageServer, [
        "--image-id", "img-1", "--engine-url", comfy.url, "--engine-pid", str(engine_proc.pid),
        "--entrypoint-pid", str(entrypoint_proc.pid)], monkeypatch)
    yield {"url": url, "comfy": comfy, "image": image}
    engine_proc.kill()
    entrypoint_proc.kill()
    httpd.shutdown()
    comfy.httpd.shutdown()


def test_health_is_503_until_the_warmup_render_has_run(box):
    assert reason_becomes(box["url"], "ComfyUI not answering") == 503
    box["comfy"].up = True
    health_becomes(box["url"], 200)
    warm = next(iter(box["comfy"].prompts.values()))
    assert warm["m"]["class_type"] == "BiRefNetRMBG", "the warmup is the sprite graph"
    assert warm["k"]["inputs"]["steps"] == 1


def test_metadata_names_the_queue_and_the_kind_separately(box):
    assert call(box["url"], "GET", "/metadata")[1]["capabilities"] == ["image"]
    assert box["image"].kinds == ("comfy_image",)


def test_a_workflow_runs_with_its_uploads_landed_first(box):
    box["comfy"].up = True
    health_becomes(box["url"], 200)
    box["comfy"].images = lambda wf: [("a.png", b"AAA"), ("b.png", b"BBB")]
    status, body = call(box["url"], "POST", "/job", {
        "jobId": "j1", "type": "comfy_image",
        "payload": {"kind": "comfy_image", "workflow": {"1": {"class_type": "X", "inputs": {}}},
                    "uploads": [{"name": "init.png", "b64": base64.b64encode(b"INIT").decode()}]}})
    assert status == 200
    result = body["result"]
    assert [i["filename"] for i in result["images"]] == ["a.png", "b.png"]
    assert base64.b64decode(result["images"][1]["b64"]) == b"BBB"
    assert result["images"][0]["safety"] == {"scores": {"NSFW": 0.01}}
    assert b"INIT" in box["comfy"].uploads["init.png"]
    assert box["comfy"].prompts[result["prompt_id"]] == {"1": {"class_type": "X", "inputs": {}}}


def test_a_refused_workflow_is_a_502_with_comfys_reason(box):
    box["comfy"].up = True
    health_becomes(box["url"], 200)
    box["comfy"].fail_next = (400, {"error": "bad node"})
    status, body = call(box["url"], "POST", "/job", {
        "jobId": "j1", "type": "comfy_image", "payload": {"workflow": {}}})
    assert status == 502 and body["status"] == "failed"
    assert "/prompt status 400" in body["error"] and "bad node" in body["error"]


def test_a_failed_warmup_is_a_500_so_the_worker_kills_the_box(box):
    box["comfy"].fail_next = (500, {"error": "no such model"})
    box["comfy"].up = True
    status, body = health_becomes(box["url"], 500)
    assert "warmup render failed" in body["reason"]
