"""The decide queue: a chain of JEV System 1 decisions, each answer appended to the state the next
step reads. The engine is SGLang serving the base model with the decision LoRA and returning the
prompt's last hidden state; the decision head (readout rows of lm_head plus its LoRA, the
per-slot bias and the per-kind temperature) is applied here, because SGLang cannot load the
adapter's lm_head part over an FP8 lm_head. `engines/jev/prep.py` writes what this reads.
"""

import argparse
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from contract.server import Server

GROUP = 16
LABELS = "ABCDEFGHIJKLMNOP"
FIXED = {"noul": ["false", "true"], "score": [str(i) for i in range(6)]}


class EngineError(Exception):
    pass


class Head:

    def __init__(self, jev_dir: Path):
        head = json.loads((jev_dir / "decision_head.json").read_text())
        self.ranges = head["slots"]["ranges"]
        self.verbalizer = head["verbalizer_ids"]
        self.bias = np.array(head["bias"], dtype=np.float32)
        self.temps = json.loads((jev_dir / "calibration.json").read_text())["per_kind"]
        readout = load_file(jev_dir / "readout.safetensors")
        self.slot = {int(t): n for n, t in enumerate(readout["ids"])}
        self.weight, self.lora_a, self.lora_b = readout["weight"], readout["lora_a"], readout["lora_b"]

    def probabilities(self, kind: str, hidden: np.ndarray, n: int) -> list[float]:
        lo = self.ranges[kind][0]
        rows = [self.slot[t] for t in self.verbalizer[lo: lo + n]]
        logits = self.weight[rows] @ hidden + self.lora_b[rows] @ (self.lora_a @ hidden)
        z = (logits + self.bias[lo: lo + n]) / self.temps[kind]
        e = np.exp(z - z.max())
        return (e / e.sum()).tolist()


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
        parser.add_argument("--engine-url", required=True, help="base url of the SGLang engine")
        parser.add_argument("--jev-dir", required=True, type=Path,
                            help="the decision head as engines/jev/prep.py writes it")
        parser.add_argument("--lora", default="jev-decision",
                            help="the decision LoRA's name in the engine")

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.engine_url = args.engine_url.rstrip("/")
        self.head = Head(args.jev_dir)
        self.lora = args.lora

    def ready(self) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(f"{self.engine_url}/v1/models", timeout=5) as r:
                models = [m.get("id") for m in json.load(r).get("data", [])]
        except (OSError, ValueError) as e:
            return 503, f"engine not answering ({e.__class__.__name__})"
        return (200, f"serving {models}") if models else (503, "engine lists no models")

    def read(self, kind: str, state: str, questions: list[tuple[str, list[str]]]) -> list[list[float]]:
        """One batched engine pass: the probabilities over each question's options."""
        texts = [prompt(kind, state, q, opts) for q, opts in questions]
        body = {"text": texts, "lora_path": [self.lora] * len(texts),
                "return_hidden_states": "last",
                "sampling_params": {"max_new_tokens": 1, "temperature": 0}}
        request = urllib.request.Request(
            f"{self.engine_url}/generate", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request) as r:
                outs = json.load(r)
        except urllib.error.HTTPError as e:
            raise EngineError(f"Status {e.code}: {e.read().decode(errors='replace')[:2000]}") from e
        except (OSError, ValueError) as e:
            raise EngineError(f"engine unreachable: {e}") from e
        probs = []
        for out, (_, opts) in zip(outs, questions):
            hs = out["meta_info"]["hidden_states"]
            hidden = np.asarray(hs[-1] if isinstance(hs[0], list) else hs, dtype=np.float32)
            probs.append(self.head.probabilities(kind, hidden.reshape(-1), len(opts)))
        return probs

    def run(self, job_type: str, payload: dict) -> tuple[int, dict]:
        try:
            result = run_chain(self.read, payload.get("state") or "", payload["steps"])
        except EngineError as e:
            return 502, {"status": "failed", "error": str(e)}
        except (KeyError, IndexError, TypeError, ValueError) as e:
            return 500, {"status": "failed", "error": f"bad chain: {e.__class__.__name__}: {e}"}
        return 200, {"status": "succeeded", "result": result}
