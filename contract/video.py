import base64
import copy
import io
import struct
import time
import uuid
import zlib

from contract import safety
from contract.comfy import ComfyError, ComfyServer

POLL_TIMEOUT_SECONDS = 1800


class VideoServer(ComfyServer):
    capabilities = ("video",)
    kinds = ("anim_sheet",)

    def warmup_workflow(self) -> dict:
        still = self.comfy.upload("warmup.png", _white_png(256))
        return {
            "1": {"class_type": "UNETLoader",
                  "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                             "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader",
                  "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                             "type": "minimax"}},
            "3": {"class_type": "VAELoader",
                  "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
            "4": {"class_type": "LoadImage", "inputs": {"image": still}},
            "5": {"class_type": "MiniMaxH3ImageToVideo",
                  "inputs": {"prompt": "warmup", "width": 256, "height": 256, "length": 5,
                             "clip": ["2", 0], "vae": ["3", 0],
                             "first_frame": ["4", 0], "last_frame": ["4", 0]}},
            "6": {"class_type": "MiniMaxH3SigmaShift",
                  "inputs": {"shift_video": 12.0, "shift_audio": 3.0, "model": ["1", 0]}},
            "7": {"class_type": "BasicGuider",
                  "inputs": {"model": ["6", 0], "conditioning": ["5", 0]}},
            "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
            "9": {"class_type": "BasicScheduler",
                  "inputs": {"scheduler": "simple", "steps": 1, "denoise": 1.0,
                             "model": ["6", 0]}},
            "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
            "11": {"class_type": "SamplerCustomAdvanced",
                   "inputs": {"noise": ["10", 0], "guider": ["7", 0], "sampler": ["8", 0],
                              "sigmas": ["9", 0], "latent_image": ["5", 1]}},
            "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 1], "vae": ["3", 0]}},
            "13": {"class_type": "PreviewImage", "inputs": {"images": ["12", 0]}},
        }

    def run(self, job_type: str, payload: dict) -> tuple[int, dict]:
        from PIL import Image

        from contract import anim_sheet as sheets

        job = uuid.uuid4().hex[:8]
        started = time.time()
        still = sheets.prep_still(Image.open(io.BytesIO(base64.b64decode(payload["image_b64"]))))
        try:
            self.comfy.upload(f"{job}-front.png", _png(still))
            facings = {"front": 0}
            stills = {"front": still}
            if payload["turn"] is not None:
                turn = _fill(payload["loop"]["workflow"], f"{job}-front.png",
                             payload["turn"]["prompt"], payload["turn"]["length"],
                             payload["turn"]["steps"])
                _, outputs = self.comfy.render(turn, POLL_TIMEOUT_SECONDS)
                turn_frames = _frames(outputs)
                facings = sheets.pick_facings(turn_frames)
                for d, i in facings.items():
                    if d != "front":
                        stills[d] = turn_frames[i]
                        self.comfy.upload(f"{job}-{d}.png", _png(turn_frames[i]))
            submitted = {}
            for anim, spec in payload["anims"].items():
                for d in payload["dirs"]:
                    facing = f" {sheets.FACING_PHRASES[d]}" if payload["turn"] is not None else ""
                    wf = _fill(payload["loop"]["workflow"], f"{job}-{d}.png",
                               spec["prompt"] + facing, spec["frames"])
                    submitted[(anim, d)] = self.comfy.submit(wf)
            clips = {}
            warnings = []
            for key, prompt_id in submitted.items():
                history = self.comfy.wait(prompt_id, POLL_TIMEOUT_SECONDS)
                frames = _frames(self.comfy.outputs(history))
                if sheets.peak_motion(frames) < sheets.WEAK_MOTION:
                    warnings.append(f"{key[0]}/{key[1]}: weak motion")
                anim = payload["anims"][key[0]]
                clips[key] = [sheets.matte(f)
                              for f in sheets.sample(sheets.trim_pinned(frames), anim["cells"])]
        except ComfyError as e:
            return 502, {"status": "failed", "error": str(e)}
        sheet, manifest = sheets.pack(
            clips, {a: {"fps": s["fps"]} for a, s in payload["anims"].items()},
            dirs=payload["dirs"], warnings=warnings)
        verdicts = [safety.classify(_png(im)) for im in stills.values()]
        return 200, {"status": "succeeded", "result": {
            "sheet_b64": base64.b64encode(_png(sheet)).decode("ascii"),
            "manifest": manifest,
            "facings": facings,
            "safety": safety.worst(verdicts),
            "generate_seconds": round(time.time() - started, 1)}}


def _png(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _frames(outputs) -> list:
    from PIL import Image
    return [Image.open(io.BytesIO(data)).convert("RGB") for _, data in outputs]


def _fill(workflow: dict, still_name: str, prompt: str, length: int, steps=None) -> dict:
    wf = copy.deepcopy(workflow)
    wf["4"]["inputs"]["image"] = still_name
    wf["5"]["inputs"]["prompt"] = prompt
    wf["5"]["inputs"]["length"] = length
    if steps is not None:
        wf["9"]["inputs"]["steps"] = steps
    wf["10"]["inputs"]["noise_seed"] = int(uuid.uuid4().int % (2**32))
    return wf


def _white_png(side: int) -> bytes:
    raw = b"".join(b"\x00" + b"\xff" * (side * 3) for _ in range(side))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
