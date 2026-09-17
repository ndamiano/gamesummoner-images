import base64

from contract import safety
from contract.comfy import ComfyError, ComfyServer

POLL_TIMEOUT_SECONDS = 600


class ImageServer(ComfyServer):
    capabilities = ("image",)
    kinds = ("comfy_image",)

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

    def run(self, job_type: str, payload: dict) -> tuple[int, dict]:
        try:
            for up in payload.get("uploads") or []:
                self.comfy.upload(up["name"], base64.b64decode(up["b64"]))
            prompt_id, outputs = self.comfy.render(payload["workflow"], POLL_TIMEOUT_SECONDS)
        except ComfyError as e:
            return 502, {"status": "failed", "error": str(e)}
        except KeyError as e:
            return 500, {"status": "failed", "error": f"payload missing {e}"}
        images = [{"filename": name, "b64": base64.b64encode(data).decode("ascii"),
                   "safety": safety.classify(data)} for name, data in outputs]
        return 200, {"status": "succeeded", "result": {"prompt_id": prompt_id, "images": images}}
