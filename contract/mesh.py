import argparse
import base64
import io
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

from contract import safety
from contract.server import Server

logger = logging.getLogger("image")

TRELLIS_TIMEOUT_SECONDS = 900
SHEET_TIMEOUT_SECONDS = 1800
TRELLIS_ATTEMPTS = 2


class MeshServer(Server):
    capabilities = ("mesh",)
    kinds = ("trellis_mesh", "sprite_sheet")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument("--engine-url", required=True, help="the TRELLIS server's base url")
        parser.add_argument("--sprite-url", default=None,
                            help="the sprite server's base url; sprite_sheet jobs fail without")

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.engine_url = args.engine_url.rstrip("/")
        self.sprite_url = args.sprite_url.rstrip("/") if args.sprite_url else None

    def ready(self) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(f"{self.engine_url}/health", timeout=5) as r:
                health = json.load(r)
        except (OSError, ValueError) as e:
            return 503, f"TRELLIS not answering ({e.__class__.__name__})"
        if not health.get("warm"):
            return 503, "TRELLIS loading" if not health.get("loaded") else "TRELLIS warming up"
        if self.sprite_url:
            try:
                with urllib.request.urlopen(f"{self.sprite_url}/health", timeout=5) as r:
                    if not json.load(r).get("loaded"):
                        return 503, "sprite server loading"
            except (OSError, ValueError) as e:
                return 503, f"sprite server not answering ({e.__class__.__name__})"
        return 200, "serving"

    def run(self, job_type: str, payload: dict) -> tuple[int, dict]:
        if job_type == "trellis_mesh":
            return self._mesh(payload)
        return self._sheet(payload)

    def _mesh(self, payload: dict) -> tuple[int, dict]:
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

    def _sheet(self, payload: dict) -> tuple[int, dict]:
        if not self.sprite_url:
            return 500, {"status": "failed", "error": "this box runs no sprite server"}
        request = urllib.request.Request(f"{self.sprite_url}/sheet", method="POST",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=SHEET_TIMEOUT_SECONDS) as r:
                out = json.load(r)
        except urllib.error.HTTPError as e:
            return 502, {"status": "failed",
                         "error": f"Status {e.code}: {e.read().decode(errors='replace')[:2000]}"}
        except (OSError, ValueError) as e:
            return 502, {"status": "failed", "error": f"sprite server unreachable: {e}"}
        if not out.get("fallback"):
            out["safety"] = safety.worst([safety.classify(cell) for cell in _cells(out)]
                                         or [{"error": "the sheet had no cells to score"}])
        return 200, {"status": "succeeded", "result": out}


def _cells(sheet: dict) -> list:
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(base64.b64decode(sheet["sheet_b64"]))).convert("RGB")
    except Exception:  # noqa: BLE001
        return []
    cell = (sheet.get("manifest") or {}).get("cell") or {}
    w, h = int(cell.get("w") or 0), int(cell.get("h") or 0)
    crops = [im] if not w or not h else [
        im.crop((x, y, x + w, y + h))
        for y in range(0, im.height - h + 1, h) for x in range(0, im.width - w + 1, w)]
    out = []
    for crop in crops:
        buf = io.BytesIO()
        crop.save(buf, "PNG")
        out.append(buf.getvalue())
    return out
