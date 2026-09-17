import argparse
import json
import urllib.error
import urllib.request

from contract.server import Server

REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def chat_body_for_engine(body: dict) -> dict:
    out = {k: v for k, v in body.items() if k != "reasoning"}
    if body.get("reasoning") in REASONING_EFFORTS:
        out["reasoning_effort"] = body["reasoning"]
    return out


class LlmServer(Server):

    capabilities = ("llm",)
    kinds = ("llm",)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument("--engine-url", required=True,
                            help="base url of the OpenAI-compatible engine")

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.engine_url = args.engine_url.rstrip("/")

    def ready(self) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(f"{self.engine_url}/v1/models", timeout=5) as r:
                models = [m.get("id") for m in json.load(r).get("data", [])]
        except (OSError, ValueError) as e:
            return 503, f"engine not answering ({e.__class__.__name__})"
        return (200, f"serving {models}") if models else (503, "engine lists no models")

    def run(self, job_type: str, payload: dict) -> tuple[int, dict]:
        request = urllib.request.Request(
            f"{self.engine_url}/v1/chat/completions",
            data=json.dumps(chat_body_for_engine(payload.get("body") or {})).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request) as r:
                return 200, {"status": "succeeded", "result": json.load(r)}
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")[:2000]
            return 502, {"status": "failed", "error": f"Status {e.code}: {text}"}
        except (OSError, ValueError) as e:
            return 502, {"status": "failed", "error": f"engine unreachable: {e}"}
