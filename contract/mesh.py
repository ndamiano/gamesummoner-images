import argparse
import base64
import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from contract.server import Server

logger = logging.getLogger("image")

TRELLIS_TIMEOUT_SECONDS = 900
TRELLIS_ATTEMPTS = 2


class MeshServer(Server):
    capabilities = ("mesh",)
    kinds = ("trellis_mesh",)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument("--engine-url", required=True, help="the TRELLIS server's base url")

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.engine_url = args.engine_url.rstrip("/")

    def ready(self) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(f"{self.engine_url}/health", timeout=5) as r:
                health = json.load(r)
        except (OSError, ValueError) as e:
            return 503, f"TRELLIS not answering ({e.__class__.__name__})"
        if not health.get("warm"):
            return 503, "TRELLIS loading" if not health.get("loaded") else "TRELLIS warming up"
        return 200, "serving"

    def run(self, job_type: str, payload: dict) -> tuple[int, dict]:
        img = base64.b64decode(payload["image_b64"])
        query = ""
        if payload.get("seed") is not None:
            query = "?" + urllib.parse.urlencode({"seed": payload["seed"]})
        last = ""
        for attempt in range(1, TRELLIS_ATTEMPTS + 1):
            request = urllib.request.Request(f"{self.engine_url}/generate{query}", data=img,
                                             method="POST", headers={"Content-Type": "image/png"})
            try:
                with urllib.request.urlopen(request, timeout=TRELLIS_TIMEOUT_SECONDS) as r:
                    return 200, {"status": "succeeded", "result": {
                        "glb_b64": base64.b64encode(r.read()).decode("ascii"),
                        "stage_seconds": r.headers.get("X-Stage-Seconds"),
                        "load_seconds": r.headers.get("X-Load-Seconds"),
                        "generate_seconds": r.headers.get("X-Generate-Seconds")}}
            except urllib.error.HTTPError as e:
                last = f"Status {e.code}: {e.read().decode(errors='replace')[:2000]}"
            except OSError as e:
                last = str(e)
            logger.warning("trellis generate failed (attempt %d): %s", attempt, last)
        return 502, {"status": "failed", "error": f"TRELLIS failed: {last}"}
