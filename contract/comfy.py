import argparse
import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from contract.server import Server

logger = logging.getLogger("image")

WARMUP_TIMEOUT_SECONDS = 240


class ComfyError(Exception):
    pass


class ComfyClient:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def up(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/system_stats", timeout=5):
                return True
        except OSError:
            return False

    def upload(self, name: str, data: bytes) -> str:
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
                f"filename=\"{name}\"\r\nContent-Type: image/png\r\n\r\n").encode() + data + \
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue" \
            f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            f"{self.url}/upload/image", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        return self._json(request, timeout=60, what="/upload/image").get("name", name)

    def submit(self, workflow: dict) -> str:
        request = urllib.request.Request(
            f"{self.url}/prompt", method="POST",
            data=json.dumps({"prompt": workflow, "client_id": str(uuid.uuid4())}).encode(),
            headers={"Content-Type": "application/json"})
        prompt_id = self._json(request, timeout=60, what="/prompt").get("prompt_id")
        if not prompt_id:
            raise ComfyError("ComfyUI returned no prompt_id")
        return prompt_id

    def wait(self, prompt_id: str, timeout: float) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            history = self._json(f"{self.url}/history/{prompt_id}", timeout=30,
                                 what="/history").get(prompt_id)
            if history:
                status = history.get("status") or {}
                if status.get("status_str") == "error":
                    raise ComfyError(f"ComfyUI job {prompt_id} errored: "
                                     f"{json.dumps(status)[:500]}")
                if history.get("outputs"):
                    return history
            time.sleep(1)
        raise ComfyError(f"ComfyUI job {prompt_id} did not finish within {timeout:.0f}s")

    def outputs(self, history: dict) -> list:
        out = []
        for node_output in (history.get("outputs") or {}).values():
            for img in node_output.get("images", []):
                query = urllib.parse.urlencode({"filename": img["filename"],
                                                "subfolder": img.get("subfolder", ""),
                                                "type": img.get("type", "output")})
                try:
                    with urllib.request.urlopen(f"{self.url}/view?{query}", timeout=120) as r:
                        out.append((img["filename"], r.read()))
                except urllib.error.HTTPError as e:
                    raise ComfyError(f"ComfyUI /view status {e.code} for {img['filename']}")
                except OSError as e:
                    raise ComfyError(f"ComfyUI /view unreachable: {e}")
        return out

    def render(self, workflow: dict, timeout: float) -> tuple:
        prompt_id = self.submit(workflow)
        return prompt_id, self.outputs(self.wait(prompt_id, timeout))

    @staticmethod
    def _json(request, timeout: float, what: str) -> dict:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise ComfyError(f"ComfyUI {what} status {e.code}: "
                             f"{e.read().decode(errors='replace')[:2000]}")
        except (OSError, ValueError) as e:
            raise ComfyError(f"ComfyUI {what} failed: {e}")


class ComfyServer(Server):

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument("--engine-url", required=True, help="ComfyUI's base url")

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.comfy = ComfyClient(args.engine_url)
        self._warmup: threading.Thread | None = None
        self._warm: bool | Exception = False

    def warmup_workflow(self) -> dict:
        raise NotImplementedError

    def ready(self) -> tuple[int, str]:
        if not self.comfy.up():
            return 503, "ComfyUI not answering"
        if self._warmup is None:
            self._warmup = threading.Thread(target=self._do_warmup, daemon=True)
            self._warmup.start()
        if isinstance(self._warm, Exception):
            return 500, f"warmup render failed: {self._warm}"
        if not self._warm:
            return 503, "warming up"
        return 200, "serving"

    def _do_warmup(self) -> None:
        started = time.perf_counter()
        try:
            self.comfy.render(self.warmup_workflow(), WARMUP_TIMEOUT_SECONDS)
        except Exception as e:  # noqa: BLE001
            logger.error("warmup render failed: %s", e)
            self._warm = e
            return
        logger.info("warmup render done in %.1fs", time.perf_counter() - started)
        self._warm = True
