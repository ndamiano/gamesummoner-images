"""The llm adapter: a canonical chat request (OpenAI chat shape plus `reasoning` as an effort
string) forwarded to an OpenAI-compatible engine's /v1/chat/completions. No request timeout: the
worker owns the job's deadline."""

import json
import urllib.error
import urllib.request

REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def chat_body_for_engine(body: dict) -> dict:
    out = {k: v for k, v in body.items() if k != "reasoning"}
    if body.get("reasoning") in REASONING_EFFORTS:
        out["reasoning_effort"] = body["reasoning"]
    return out


class LlmAdapter:
    def __init__(self, engine_url: str):
        self.engine_url = engine_url.rstrip("/")

    def ready(self) -> tuple:
        try:
            with urllib.request.urlopen(f"{self.engine_url}/v1/models", timeout=5) as r:
                models = [m.get("id") for m in json.load(r).get("data", [])]
        except (OSError, ValueError) as e:
            return False, f"engine not answering ({e.__class__.__name__})"
        return (True, f"serving {models}") if models else (False, "engine lists no models")

    def run(self, job_type: str, payload: dict) -> tuple:
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
