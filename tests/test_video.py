"""The video column on ComfyUI: a still in, a sprite sheet out, frames turned into cells by
`contract/anim_sheet.py`."""

import base64

import pytest

pytest.importorskip("numpy")
pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

from contract import safety  # noqa: E402
from runpod.video.server import RunpodVideoServer  # noqa: E402
from support import FakeComfy, call, health_becomes, png_bytes, serve, sleeper  # noqa: E402


def blob(cx, size=96):
    im = Image.new("RGB", (size, size), (255, 255, 255))
    ImageDraw.Draw(im).ellipse((cx - 16, 30, cx + 16, 80), fill=(30, 30, 30))
    return im


def moving_clip(wf):
    n = wf["5"]["inputs"]["length"]
    return [(f"f{i}.png", png_bytes(blob(30 + i * 4))) for i in range(n)]


LOOP = {"4": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "5": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"prompt": "", "length": 0}},
        "9": {"class_type": "BasicScheduler", "inputs": {"steps": 20}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}}}


@pytest.fixture
def box(monkeypatch):
    monkeypatch.setattr(safety, "classify", lambda data: {"scores": {"NSFW": 0.02}})
    comfy = FakeComfy(images=moving_clip)
    engine_proc, entrypoint_proc = sleeper(), sleeper()
    image, httpd, url = serve(RunpodVideoServer, [
        "--image-id", "img-1", "--engine-url", comfy.url, "--engine-pid", str(engine_proc.pid),
        "--entrypoint-pid", str(entrypoint_proc.pid)], monkeypatch)
    yield {"url": url, "comfy": comfy}
    engine_proc.kill()
    entrypoint_proc.kill()
    httpd.shutdown()
    comfy.httpd.shutdown()


def test_the_warmup_uploads_a_still_and_runs_the_i2v_graph(box):
    health_becomes(box["url"], 200)
    assert "warmup.png" in box["comfy"].uploads
    warm = next(iter(box["comfy"].prompts.values()))
    assert warm["5"]["class_type"] == "MiniMaxH3ImageToVideo"
    assert warm["4"]["inputs"]["image"] == "warmup.png"


def test_one_still_becomes_a_sheet_of_the_named_animations(box):
    health_becomes(box["url"], 200)
    still = base64.b64encode(png_bytes(blob(48))).decode()
    status, body = call(box["url"], "POST", "/job", {
        "jobId": "j1", "type": "anim_sheet", "payload": {
            "kind": "anim_sheet", "image_b64": still, "turn": None, "dirs": ["front"],
            "loop": {"workflow": LOOP},
            "anims": {"walk": {"prompt": "walks", "frames": 12, "cells": 4, "fps": 12}}}})
    assert status == 200, body
    result = body["result"]
    sheet = Image.open(__import__("io").BytesIO(base64.b64decode(result["sheet_b64"])))
    manifest = result["manifest"]
    assert manifest["anims"]["walk"]["rows"] == {"front": 0}
    assert manifest["anims"]["walk"]["frames"] == 4
    assert sheet.size == (manifest["cell"]["w"] * 4, manifest["cell"]["h"])
    assert result["facings"] == {"front": 0}
    assert result["safety"] == {"scores": {"NSFW": 0.02}}
    assert manifest["warnings"] == []
    submitted = [wf for wf in box["comfy"].prompts.values() if wf.get("5", {}).get("inputs", {}).get("prompt") == "walks"]
    assert len(submitted) == 1 and submitted[0]["5"]["inputs"]["length"] == 12
    assert submitted[0]["4"]["inputs"]["image"].endswith("-front.png")
