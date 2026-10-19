"""The decide column: the step chain over a scripted reader, and the server over real sockets
against a fake OpenAI engine that returns option-token logprobs, read through a synthetic head."""

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from contract import server
from contract.decide import choose, run_chain
from local.decide.server import LocalDecideServer
from runpod.decide.server import RunpodDecideServer
from support import call, health_becomes, sleeper


class Reader:
    """Answers by question text: a choice from a table of option -> weight, a noul from a table of
    question -> P(yes). Records the state each question was asked over."""

    def __init__(self, choices=None, yes=None):
        self.choices, self.yes, self.calls = choices or {}, yes or {}, []

    def __call__(self, kind, state, questions):
        out = []
        for question, options in questions:
            self.calls.append((kind, state, question, tuple(options)))
            if kind == "noul":
                p = self.yes[question]
                out.append([1 - p, p])
            else:
                w = [self.choices[question].get(o, 1.0) for o in options]
                out.append([x / sum(w) for x in w])
        return out


def test_chain_branches_records_and_feeds_kept_items_forward():
    read = Reader(
        choices={"dimension?": {"2d": 9, "3d": 1}, "camera?": {"side": 3, "top-down": 1},
                 "3d style?": {}},
        yes={"makes sense: jump: leap": 0.9, "makes sense: dash: burst": 0.3,
             "makes sense: wall run: run on walls": 0.7,
             "core: jump: leap": 0.95, "core: wall run: run on walls": 0.2,
             "screen: title": 0.4, "screen: pause": 0.1})
    steps = [
        {"key": "dimension", "kind": "choice", "question": "dimension?", "options": ["2d", "3d"]},
        {"key": "camera", "kind": "choice", "question": "camera?",
         "options_by": {"key": "dimension", "map": {"2d": ["side", "top-down"], "3d": ["fps"]}}},
        {"key": "style", "kind": "choice", "when": {"dimension": "3d"}, "question": "3d style?",
         "options": ["low-poly", "voxel"]},
        {"key": "mechanics", "kind": "noul", "question": "makes sense: {item}", "keep_over": 0.6,
         "items": [{"name": "jump", "text": "jump: leap"}, {"name": "dash", "text": "dash: burst"},
                   {"name": "wall run", "text": "wall run: run on walls"}]},
        {"key": "core", "kind": "noul", "record": False, "question": "core: {item}",
         "keep_over": 0.5, "items_from": "mechanics"},
        {"key": "screens", "kind": "noul", "question": "screen: {item}", "keep_over": 0.6,
         "items": [{"name": "title", "text": "title"}, {"name": "pause", "text": "pause"}]},
        {"key": "probe", "kind": "noul", "question": "core: {item}", "items_from": "mechanics"},
    ]
    result = run_chain(read, "Game prompt: a cat", steps)

    assert result["answers"] == {"dimension": "2d", "camera": "side",
                                 "mechanics": "jump, wall run", "core": "jump", "screens": "none"}
    assert result["state"] == ("Game prompt: a cat\ndimension: 2d\ncamera: side"
                               "\nmechanics: jump, wall run\nscreens: none")
    assert [s["key"] for s in result["steps"] if s["skipped"]] == ["style"]
    by_key = {s["key"]: s for s in result["steps"]}
    assert by_key["dimension"]["probabilities"] == {"2d": 0.9, "3d": 0.1}
    assert by_key["mechanics"]["probabilities"] == {"jump": 0.9, "dash": 0.3, "wall run": 0.7}
    assert by_key["probe"]["probabilities"] == {"jump": 0.95, "wall run": 0.2}

    asked = {q: (kind, state, opts) for kind, state, q, opts in read.calls}
    assert asked["camera?"] == ("choice", "Game prompt: a cat\ndimension: 2d", ("side", "top-down"))
    assert "dash" not in " ".join(q for _, _, q, _ in read.calls if q.startswith("core:"))
    assert asked["screen: title"][1].endswith("mechanics: jump, wall run")
    assert asked["makes sense: jump: leap"][2] == ("false", "true")


def test_choice_over_forty_options_runs_a_tournament_that_keeps_the_favourite_and_sums_to_one():
    options = [f"style {i}" for i in range(40)]
    weights = {o: 1.0 for o in options}
    weights["style 29"] = 50.0
    weights["style 3"] = 10.0
    read = Reader(choices={"style?": weights})
    probs = choose(read, "s", "style?", options)

    assert len(probs) == 40 and sum(probs) == pytest.approx(1.0)
    ranked = sorted(range(40), key=lambda i: -probs[i])
    assert ranked[:2] == [29, 3]
    asked = [len(opts) for _, _, _, opts in read.calls]
    assert asked == [14, 13, 13, 16]


BIAS = [0.0, 0.5] + [0.0] * 6 + [0.25, -0.25] + [0.0] * 14


@pytest.fixture
def head_dir(tmp_path):
    (tmp_path / "decision_head.json").write_text(json.dumps({
        "verbalizer_ids": list(range(100, 124)), "bias": BIAS,
        "slots": {"ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]}}}))
    (tmp_path / "calibration.json").write_text(json.dumps(
        {"per_kind": {"noul": 1.0, "score": 1.0, "choice": 2.0}}))
    return tmp_path


def logprobs_for(text, ids):
    """The engine's option-token logprobs: a fixed lean per prompt, so each answer is checkable."""
    lean = 1.5 if "jump" in text else -0.5
    return {t: -0.1 * i + (lean if i == 1 else 0.0) for i, t in enumerate(ids)}


class FakeOpenAI:
    """/v1/models once up; /v1/completions answers one prompt with its option tokens' logprobs,
    keyed "token_id:<id>" as return_tokens_as_token_ids asks, or with `fail` when set."""

    def __init__(self):
        self.up, self.fail, self.requests = False, None, []
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"data": [{"id": "jev-decision"}]}) if engine.up else self._send(503, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                engine.requests.append(body)
                if engine.fail:
                    self._send(*engine.fail)
                    return
                lp = logprobs_for(body["prompt"], body["allowed_token_ids"])
                top = {f"token_id:{t}": v for t, v in lp.items()}
                self._send(200, {"choices": [{"index": 0, "text": "",
                                              "logprobs": {"top_logprobs": [top]}}]})

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"


@pytest.fixture(params=[LocalDecideServer, RunpodDecideServer], ids=["local", "runpod"])
def box(request, monkeypatch, head_dir):
    monkeypatch.setattr(server, "PROBE_INTERVAL_SECONDS", 0.01)
    engine = FakeOpenAI()
    engine_proc, entrypoint_proc = sleeper(), sleeper()
    image = request.param.from_argv([
        "--image-id", "img-1", "--engine-url", engine.url, "--head-dir", str(head_dir),
        "--engine-pid", str(engine_proc.pid), "--entrypoint-pid", str(entrypoint_proc.pid),
        "--port", "0"])
    httpd = image.start()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{httpd.server_port}", "engine": engine}
    engine_proc.kill()
    entrypoint_proc.kill()
    httpd.shutdown()
    engine.httpd.shutdown()


def expected(text, lo, n, temp):
    ids = list(range(100 + lo, 100 + lo + n))
    lp = logprobs_for(text, ids)
    z = [(lp[t] + BIAS[lo + i]) / temp for i, t in enumerate(ids)]
    e = [math.exp(x - max(z)) for x in z]
    return [x / sum(e) for x in e]


def test_a_chain_job_reads_each_decision_from_option_token_logprobs(box):
    assert call(box["url"], "GET", "/metadata")[1]["capabilities"] == ["decide"]
    box["engine"].up = True
    health_becomes(box["url"], 200)
    payload = {"kind": "decide", "state": "Game prompt: frogs", "steps": [
        {"key": "dimension", "kind": "choice", "question": "Which?", "options": ["2d", "3d", "both"]},
        {"key": "mechanics", "kind": "noul", "question": "Fits: {item}?", "keep_over": 0.5,
         "items": [{"name": "jump", "text": "jump"}, {"name": "swim", "text": "swim"}]}]}
    status, body = call(box["url"], "POST", "/job", {"jobId": "j1", "type": "decide",
                                                     "payload": payload})
    assert status == 200 and body["status"] == "succeeded"

    choice = expected("Which?", 8, 3, 2.0)
    yes_jump = expected("jump", 0, 2, 1.0)[1]
    yes_swim = expected("swim", 0, 2, 1.0)[1]
    steps = {s["key"]: s["probabilities"] for s in body["result"]["steps"]}
    assert list(steps["dimension"].values()) == pytest.approx(choice, abs=1e-6)
    assert steps["mechanics"] == pytest.approx({"jump": yes_jump, "swim": yes_swim}, abs=1e-6)
    assert body["result"]["answers"]["dimension"] == ["2d", "3d", "both"][choice.index(max(choice))]

    first, *mechanics = box["engine"].requests
    assert first["model"] == "jev-decision" and first["max_tokens"] == 1
    assert first["allowed_token_ids"] == [108, 109, 110] and first["logprobs"] == 3
    assert first["prompt"] == ("[kind] choice\n[state] Game prompt: frogs\n[question] Which?\n"
                               "[options]\nA) 2d\nB) 3d\nC) both\n[decision]:")
    assert len(mechanics) == 2, "one prompt per request: shared batches move JEV's probabilities"
    assert mechanics[0]["prompt"].startswith(
        f"[kind] noul\n[state] Game prompt: frogs\ndimension: {body['result']['answers']['dimension']}")
    assert mechanics[1]["prompt"].endswith("[question] Fits: swim?\n[options]\nfalse\ntrue\n[decision]:")


def test_an_engine_error_or_a_broken_chain_fails_the_whole_job(box):
    box["engine"].up = True
    box["engine"].fail = (400, {"error": "context too long"})
    step = {"key": "d", "kind": "choice", "question": "q", "options": ["a", "b"]}
    status, body = call(box["url"], "POST", "/job", {"type": "decide", "payload": {"steps": [step]}})
    assert status == 502 and "context too long" in body["error"]

    box["engine"].fail = None
    orphan = {"key": "c", "kind": "noul", "question": "{item}", "items_from": "missing"}
    status, body = call(box["url"], "POST", "/job", {"type": "decide", "payload": {"steps": [orphan]}})
    assert status == 500 and body["status"] == "failed" and "missing" in body["error"]
