import argparse
import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from contract import safety
from contract.server import Server

logger = logging.getLogger("image")

WARMUP_TIMEOUT_SECONDS = 240
POLL_TIMEOUT_SECONDS = 600


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
            for img in node_output.get("images", []) + node_output.get("3d", []):
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
    capabilities = ("comfy",)
    kinds = ("comfy_image", "comfy_mesh")

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
        return {
            "u": {"class_type": "UNETLoader",
                  "inputs": {"unet_name": "qwen_image_2512_fp8_e4m3fn.safetensors",
                             "weight_dtype": "default"}},
            "c": {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                             "type": "qwen_image", "device": "default"}},
            "v": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
            "ms": {"class_type": "ModelSamplingAuraFlow",
                   "inputs": {"shift": 3.1, "model": ["u", 0]}},
            "p": {"class_type": "CLIPTextEncode", "inputs": {"text": "warmup", "clip": ["c", 0]}},
            "n": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["c", 0]}},
            "l": {"class_type": "EmptySD3LatentImage",
                  "inputs": {"width": 256, "height": 256, "batch_size": 1}},
            "k": {"class_type": "KSampler",
                  "inputs": {"seed": 0, "steps": 1, "cfg": 2.5, "sampler_name": "euler",
                             "scheduler": "simple", "denoise": 1.0, "model": ["ms", 0],
                             "positive": ["p", 0], "negative": ["n", 0],
                             "latent_image": ["l", 0]}},
            "d": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["v", 0]}},
            "m": {"class_type": "BiRefNetRMBG",
                  "inputs": {"model": "BiRefNet-general", "mask_blur": 0, "mask_offset": -1,
                             "invert_output": False, "refine_foreground": False,
                             "background": "Alpha", "background_color": "#ffffff",
                             "image": ["d", 0]}},
            "s": {"class_type": "PreviewImage", "inputs": {"images": ["m", 0]}},
        }

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

    def run(self, job_type: str, payload: dict) -> tuple[int, dict]:
        try:
            for up in payload.get("uploads") or []:
                self.comfy.upload(up["name"], base64.b64decode(up["b64"]))
            prompt_id, outputs = self.comfy.render(payload["workflow"], POLL_TIMEOUT_SECONDS)
        except ComfyError as e:
            return 502, {"status": "failed", "error": str(e)}
        except KeyError as e:
            return 500, {"status": "failed", "error": f"payload missing {e}"}
        result = {"prompt_id": prompt_id, "images": []}
        for name, data in outputs:
            if name.endswith(".glb"):
                result["glb_b64"] = base64.b64encode(data).decode("ascii")
            else:
                result["images"].append({"filename": name,
                                         "b64": base64.b64encode(data).decode("ascii"),
                                         "safety": safety.classify(data)})
        return 200, {"status": "succeeded", "result": result}
