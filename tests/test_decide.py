"""The decide column: the step chain over a scripted reader, and the server over real sockets
against a fake SGLang that returns hidden states, read through a synthetic decision head."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("safetensors")
from safetensors.numpy import load_file, save_file  # noqa: E402

from contract import server  # noqa: E402
from contract.decide import choose, run_chain  # noqa: E402
from local.decide.server import LocalDecideServer  # noqa: E402
from support import call, health_becomes, sleeper  # noqa: E402


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


HIDDEN = 4


@pytest.fixture
def jev_dir(tmp_path):
    ids = list(range(100, 124))
    (tmp_path / "decision_head.json").write_text(json.dumps({
        "verbalizer_ids": ids, "bias": [0.0] * 24,
        "slots": {"ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]}}}))
    (tmp_path / "calibration.json").write_text(json.dumps(
        {"per_kind": {"noul": 1.0, "score": 1.0, "choice": 2.0}}))
    rng = np.random.default_rng(0)
    save_file({"ids": np.array(ids, dtype=np.int64),
               "weight": rng.normal(size=(24, HIDDEN)).astype(np.float32),
               "lora_a": rng.normal(size=(2, HIDDEN)).astype(np.float32),
               "lora_b": rng.normal(size=(24, 2)).astype(np.float32)},
              str(tmp_path / "readout.safetensors"))
    return tmp_path


class FakeSglang:
    """/v1/models once up; /generate answers each text with a last hidden state taken from
    `hidden_for(text)`, or with `fail` when set."""

    def __init__(self, hidden_for):
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
                self._send(200, {"data": [{"id": "jev"}]}) if engine.up else self._send(503, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                engine.requests.append(body)
                if engine.fail:
                    self._send(*engine.fail)
                    return
                self._send(200, [{"text": "", "meta_info": {"hidden_states": [hidden_for(t)]}}
                                 for t in body["text"]])

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"


@pytest.fixture
def box(monkeypatch, jev_dir):
    monkeypatch.setattr(server, "PROBE_INTERVAL_SECONDS", 0.01)
    engine = FakeSglang(lambda text: [1.0, -1.0, 0.5, 0.0] if "jump" in text else [0.0, 1.0, 0.0, 2.0])
    engine_proc, entrypoint_proc = sleeper(), sleeper()
    image = LocalDecideServer.from_argv([
        "--image-id", "img-1", "--engine-url", engine.url, "--jev-dir", str(jev_dir),
        "--engine-pid", str(engine_proc.pid), "--entrypoint-pid", str(entrypoint_proc.pid),
        "--port", "0"])
    httpd = image.start()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{httpd.server_port}", "engine": engine, "jev_dir": jev_dir}
    engine_proc.kill()
    entrypoint_proc.kill()
    httpd.shutdown()
    engine.httpd.shutdown()


def expected(jev_dir, hidden, lo, n, temp):
    r = load_file(str(jev_dir / "readout.safetensors"))
    h = np.array(hidden, dtype=np.float32)
    z = (r["weight"][lo:lo + n] @ h + r["lora_b"][lo:lo + n] @ (r["lora_a"] @ h)) / temp
    e = np.exp(z - z.max())
    return e / e.sum()


def test_a_chain_job_reads_the_head_over_one_batched_engine_call_per_step(box):
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

    choice = expected(box["jev_dir"], [0.0, 1.0, 0.0, 2.0], 8, 3, 2.0)
    yes_jump = expected(box["jev_dir"], [1.0, -1.0, 0.5, 0.0], 0, 2, 1.0)[1]
    yes_swim = expected(box["jev_dir"], [0.0, 1.0, 0.0, 2.0], 0, 2, 1.0)[1]
    steps = {s["key"]: s["probabilities"] for s in body["result"]["steps"]}
    assert list(steps["dimension"].values()) == pytest.approx(choice.tolist(), abs=1e-6)
    assert steps["mechanics"] == pytest.approx({"jump": yes_jump, "swim": yes_swim}, abs=1e-6)
    assert body["result"]["answers"]["dimension"] == ["2d", "3d", "both"][int(choice.argmax())]

    first, second = box["engine"].requests
    assert first["return_hidden_states"] == "last" and first["lora_path"] == ["jev-decision"]
    assert first["text"][0] == ("[kind] choice\n[state] Game prompt: frogs\n[question] Which?\n"
                                "[options]\nA) 2d\nB) 3d\nC) both\n[decision]:")
    assert len(second["text"]) == 2 and second["lora_path"] == ["jev-decision"] * 2
    assert second["text"][0].startswith(
        f"[kind] noul\n[state] Game prompt: frogs\ndimension: {body['result']['answers']['dimension']}")
    assert second["text"][1].endswith("[question] Fits: swim?\n[options]\nfalse\ntrue\n[decision]:")


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
