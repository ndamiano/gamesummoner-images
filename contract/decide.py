"""The decide queue: a chain of JEV System 1 decisions, each answer appended to the state the next
step reads. Each decision is one OpenAI-shape completion on the decision LoRA: logprobs over the
option tokens, then JEV's per-slot bias and per-kind temperature (engines/jev/head/).
"""

import argparse
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

from contract.server import Server

HEAD_DIR = Path(__file__).resolve().parent.parent / "engines" / "jev" / "head"
GROUP = 16
LABELS = "ABCDEFGHIJKLMNOP"
FIXED = {"noul": ["false", "true"], "score": [str(i) for i in range(6)]}


class EngineError(Exception):
    pass


class Head:

    def __init__(self, head_dir: Path):
        head = json.loads((head_dir / "decision_head.json").read_text())
        self.ranges = head["slots"]["ranges"]
        self.verbalizer = head["verbalizer_ids"]
        self.bias = head["bias"]
        self.temps = json.loads((head_dir / "calibration.json").read_text())["per_kind"]

    def ids(self, kind: str, n: int) -> list[int]:
        lo = self.ranges[kind][0]
        return self.verbalizer[lo: lo + n]

    def probabilities(self, kind: str, logprobs: dict[int, float], n: int) -> list[float]:
        lo = self.ranges[kind][0]
        z = [(logprobs.get(t, -1e9) + self.bias[lo + i]) / self.temps[kind]
             for i, t in enumerate(self.ids(kind, n))]
        top = max(z)
        e = [math.exp(x - top) for x in z]
        return [x / sum(e) for x in e]


def prompt(kind: str, state: str, question: str, options: list[str]) -> str:
    lines = options if kind != "choice" else [f"{LABELS[i]}) {o}" for i, o in enumerate(options)]
    return (f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n"
            + "\n".join(lines) + "\n[decision]:")


def choose(read, state: str, question: str, options: list[str]) -> list[float]:
    """One choice over any number of options. Past 16: balanced groups, then a final over the
    group winners plus the best runners-up; each option's probability is its final one scaled by
    the group mass the final covers, or its in-group one scaled by its group's final share."""
    if len(options) <= GROUP:
        return read("choice", state, [(question, options)])[0]
    n = len(options)
    g = math.ceil(n / GROUP)
    groups, i = [], 0
    for j in range(g):
        size = n // g + (1 if j < n % g else 0)
        groups.append(list(range(i, i + size)))
        i += size
    p_group = {}
    for grp, probs in zip(groups, read("choice", state,
                                       [(question, [options[k] for k in grp]) for grp in groups])):
        p_group.update(zip(grp, probs))
    winners = [max(grp, key=lambda k: p_group[k]) for grp in groups]
    rest = sorted((k for k in range(n) if k not in winners), key=lambda k: -p_group[k])
    final_ids = sorted(winners + rest[: GROUP - len(winners)])
    p_final = dict(zip(final_ids, read("choice", state,
                                       [(question, [options[k] for k in final_ids])])[0]))
    group_of = {k: gi for gi, grp in enumerate(groups) for k in grp}
    share, cover = [0.0] * g, [0.0] * g
    for k in final_ids:
        share[group_of[k]] += p_final[k]
        cover[group_of[k]] += p_group[k]
    among = sum(s * c for s, c in zip(share, cover))
    p = [p_final[k] * among if k in p_final else share[group_of[k]] * p_group[k] for k in range(n)]
    total = sum(p)
    return [x / total for x in p]


def run_chain(read, state: str, steps: list[dict]) -> dict:
    answers, items_of, kept_of, report = {}, {}, {}, []
    for step in steps:
        key = step["key"]
        if any(answers.get(k) != v for k, v in (step.get("when") or {}).items()):
            report.append({"key": key, "skipped": True})
            continue
        answer = None
        if step["kind"] == "choice":
            by = step.get("options_by")
            options = by["map"][answers[by["key"]]] if by else step["options"]
            probs = choose(read, state, step["question"], options)
            answer = options[max(range(len(options)), key=probs.__getitem__)]
            probabilities = dict(zip(options, probs))
        else:
            if "items" in step:
                items = step["items"]
            else:
                kept = set(kept_of[step["items_from"]])
                items = [i for i in items_of[step["items_from"]] if i["name"] in kept]
            questions = [(step["question"].replace("{item}", i["text"]), FIXED["noul"])
                         for i in items]
            yes = [p[1] for p in read("noul", state, questions)] if questions else []
            probabilities = {i["name"]: p for i, p in zip(items, yes)}
            items_of[key] = items
            if "keep_over" in step:
                kept_of[key] = [i["name"] for i, p in zip(items, yes) if p > step["keep_over"]]
                answer = ", ".join(kept_of[key]) or step.get("empty", "none")
        if answer is not None:
            answers[key] = answer
            if step.get("record", True):
                state += f"\n{key}: {answer}"
        report.append({"key": key, "skipped": False,
                       "probabilities": {k: round(p, 6) for k, p in probabilities.items()}})
    return {"state": state, "answers": answers, "steps": report}


class DecideServer(Server):

    capabilities = ("decide",)
    kinds = ("decide",)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument("--engine-url", required=True,
                            help="base url of the OpenAI-compatible engine serving the decision LoRA")
        parser.add_argument("--model", default="jev-decision",
                            help="the decision LoRA's model name in the engine")
        parser.add_argument("--head-dir", type=Path, default=HEAD_DIR,
                            help="decision_head.json and calibration.json")

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.engine_url = args.engine_url.rstrip("/")
        self.model = args.model
        self.head = Head(args.head_dir)

    def ready(self) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(f"{self.engine_url}/v1/models", timeout=5) as r:
                models = [m.get("id") for m in json.load(r).get("data", [])]
        except (OSError, ValueError) as e:
            return 503, f"engine not answering ({e.__class__.__name__})"
        return (200, f"serving {models}") if models else (503, "engine lists no models")

    def read(self, kind: str, state: str, questions: list[tuple[str, list[str]]]) -> list[list[float]]:
        """The probabilities over each question's options, one prompt per request: JEV's LoRA
        path moves probabilities when sequences share a batch."""
        return [self._one(kind, prompt(kind, state, q, opts), len(opts)) for q, opts in questions]

    def _one(self, kind: str, text: str, n: int) -> list[float]:
        body = {"model": self.model, "prompt": text, "max_tokens": 1, "temperature": 1.0,
                "top_p": 1.0, "top_k": 0, "logprobs": n, "allowed_token_ids": self.head.ids(kind, n),
                "return_tokens_as_token_ids": True, "add_special_tokens": False}
        request = urllib.request.Request(
            f"{self.engine_url}/v1/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request) as r:
                top = json.load(r)["choices"][0]["logprobs"]["top_logprobs"][0]
        except urllib.error.HTTPError as e:
            raise EngineError(f"Status {e.code}: {e.read().decode(errors='replace')[:2000]}") from e
        except (OSError, ValueError, KeyError, IndexError) as e:
            raise EngineError(f"engine unreachable: {e}") from e
        return self.head.probabilities(kind, {int(k.split(":")[1]): v for k, v in top.items()}, n)

    def run(self, job_type: str, payload: dict) -> tuple[int, dict]:
        try:
            result = run_chain(self.read, payload.get("state") or "", payload["steps"])
        except EngineError as e:
            return 502, {"status": "failed", "error": str(e)}
        except (KeyError, IndexError, TypeError, ValueError) as e:
            return 500, {"status": "failed", "error": f"bad chain: {e.__class__.__name__}: {e}"}
        return 200, {"status": "succeeded", "result": result}
