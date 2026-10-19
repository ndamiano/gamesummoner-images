import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

FOLDERS = {
    "comfy": {
        "checkpoints": ["DreamShaperXL_Turbo_v2_1.safetensors"],
        "diffusion_models": ["qwen_image_2512_fp8_e4m3fn.safetensors",
                             "qwen_image_edit_2511_fp8mixed.safetensors",
                             "trellis_2_bf16.safetensors"],
        "text_encoders": ["qwen_2.5_vl_7b_fp8_scaled.safetensors"],
        "vae": ["qwen_image_vae.safetensors", "trellis_2_shape_vae_bf16.safetensors",
                "trellis_2_texture_vae_bf16.safetensors"],
        "clip_vision": ["dino_v3_vit_l.safetensors"],
    },
}


def tree_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def copy_chunked(src: str, dst: str, chunk: int = 1 << 30, streams: int = 8) -> None:
    size = os.path.getsize(src)
    with open(dst, "wb") as f:
        f.truncate(size)

    def part(off):
        with open(src, "rb", buffering=0) as fin, open(dst, "r+b", buffering=0) as fout:
            fin.seek(off)
            fout.seek(off)
            left = min(chunk, size - off)
            while left:
                buf = fin.read(min(1 << 26, left))
                fout.write(buf)
                left -= len(buf)

    with ThreadPoolExecutor(max_workers=streams) as pool:
        list(pool.map(part, range(0, size, chunk)))


def main(vol: str, stage: str, tree: str, queue: str) -> int:
    folders = FOLDERS[queue]
    rmbg = os.path.join(vol, "RMBG")
    t0 = time.time()
    try:
        files = [(folder, os.path.join(vol, folder, n))
                 for folder, names in folders.items() for n in names]
        need = sum(tree_size(p) for _, p in files) + tree_size(rmbg)
        os.makedirs(stage, exist_ok=True)
        free = shutil.disk_usage(stage).free
        if free < need * 1.1:
            print(f"[stage] skipped: {free/1e9:.1f} GB free at {stage}, need "
                  f"{need*1.1/1e9:.1f} GB — loading off the volume", flush=True)
            return 0
        for folder, src in files:
            os.makedirs(os.path.join(stage, folder), exist_ok=True)
            copy_chunked(src, os.path.join(stage, folder, os.path.basename(src)))
        shutil.copytree(rmbg, os.path.join(stage, "RMBG"), dirs_exist_ok=True)
    except OSError as e:
        print(f"[stage] failed ({e}) — loading off the volume", flush=True)
        return 0
    for name in (*folders, "RMBG"):
        link = os.path.join(tree, name)
        tmp = link + ".new"
        os.symlink(os.path.join(stage, name), tmp)
        os.replace(tmp, link)
    dt = time.time() - t0
    print(f"[stage] staged {need/1e9:.1f} GB to {stage} in {dt:.1f}s ({need/dt/1e6:.0f} MB/s)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:5]))
