#!/usr/bin/env bash
# Populate a RunPod network volume with every model weight the ComfyUI box loads.
#
# Run this ON a pod with the volume mounted at $VOL (default /workspace). Any cheap pod will do —
# this phase is pure download, no GPU touched. Downloads come from HuggingFace at datacenter
# bandwidth; nothing is copied from the home box.
#
#   VOL=/workspace bash runpod/provision_volume.sh
#
# Idempotent: a file already present at its expected byte size is skipped, so a re-run after an
# interrupted pull resumes rather than restarts. Sizes are the bytes observed on the home box —
# a mismatch fails loudly rather than leaving a half file that a backend loads as corrupt.

set -euo pipefail
VOL="${VOL:-/workspace}"

# Pod images ship several interpreters and only some own a pip. Install and run under the SAME one
# or the download imports nothing — and pip's own shebang names that interpreter, which on these
# images is often not any python3 on PATH. hf_transfer is its own package (the extra is gone in 1.x).
PY=""
if command -v pip >/dev/null 2>&1; then
    cand=$(head -1 "$(command -v pip)" | sed 's|^#!||; s| .*||')
    [ -x "$cand" ] && "$cand" -m pip --version >/dev/null 2>&1 && PY="$cand"
fi
for c in python3.13 python3.12 python3.11 python3 python; do
    [ -n "$PY" ] && break
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -m pip --version >/dev/null 2>&1 || continue
    PY="$c"; break
done
if [ -z "$PY" ]; then
    PY=python3
    "$PY" -m ensurepip --upgrade || curl -sS https://bootstrap.pypa.io/get-pip.py | "$PY"
fi
echo "using interpreter: $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
"$PY" -m pip install -q --upgrade huggingface_hub hf_transfer || \
    "$PY" -m pip install -q --upgrade --break-system-packages huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="$VOL/hf-cache"

mkdir -p "$VOL/comfy/models/checkpoints" \
         "$VOL/comfy/models/diffusion_models" "$VOL/comfy/models/text_encoders" \
         "$VOL/comfy/models/vae" "$VOL/comfy/models/clip_vision" \
         "$VOL/comfy/models/RMBG/BiRefNet" "$VOL/hf-cache"

"$PY" - "$VOL" <<'PY'
import os, shutil, sys
from huggingface_hub import hf_hub_download

VOL = sys.argv[1]

# dest dir, repo id, filename in repo, expected bytes (0 = skip the size check), revision,
# and optionally the name it is stored under
#
# The LLM is not here: it lives prepacked on the llm volume in another datacenter (README).
FILES = [
    # The image queue's checkpoints, by kind (see maestro/codegen: sprites, scenes and mesh
    # subjects -> Qwen-Image-2512; tiles + scene-chain terrain -> DreamShaperXL Turbo;
    # the scene embed -> Qwen-Image-Edit-2511). The two Qwen graphs share the VL text encoder
    # and the VAE.
    (f"{VOL}/comfy/models/checkpoints", "Lykon/dreamshaper-xl-v2-turbo",
     "DreamShaperXL_Turbo_v2_1.safetensors", 6939220250, None),
    (f"{VOL}/comfy/models/diffusion_models", "Comfy-Org/Qwen-Image_ComfyUI",
     "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors", 20430679144, None),
    (f"{VOL}/comfy/models/diffusion_models", "Comfy-Org/Qwen-Image-Edit_ComfyUI",
     "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors", 20533762817, None),
    (f"{VOL}/comfy/models/text_encoders", "Comfy-Org/Qwen-Image_ComfyUI",
     "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", 9384670680, None),
    (f"{VOL}/comfy/models/vae", "Comfy-Org/Qwen-Image_ComfyUI",
     "split_files/vae/qwen_image_vae.safetensors", 253806246, None),
    (f"{VOL}/comfy/models/RMBG/BiRefNet", "1038lab/BiRefNet",
     "BiRefNet-general.safetensors", 0, None),
    (f"{VOL}/comfy/models/RMBG/BiRefNet", "1038lab/BiRefNet", "birefnet.py", 0, None),
    (f"{VOL}/comfy/models/RMBG/BiRefNet", "1038lab/BiRefNet", "BiRefNet_config.py", 0, None),
    (f"{VOL}/comfy/models/RMBG/BiRefNet", "1038lab/BiRefNet", "config.json", 0, None),
    (f"{VOL}/comfy/models/diffusion_models", "Comfy-Org/TRELLIS.2",
     "diffusion_models/trellis_2_bf16.safetensors", 10338297616, None),
    (f"{VOL}/comfy/models/vae", "Comfy-Org/TRELLIS.2",
     "vae/trellis_2_shape_vae_bf16.safetensors", 1095844024, None),
    (f"{VOL}/comfy/models/vae", "Comfy-Org/TRELLIS.2",
     "vae/trellis_2_texture_vae_bf16.safetensors", 948461364, None),
    (f"{VOL}/comfy/models/clip_vision", "Comfy-Org/TRELLIS.2",
     "clip_vision/dino_v3_vit_l.safetensors", 1212559776, None),
]

for dest, repo, name, want, rev, *stored in FILES:
    # ComfyUI resolves weights by folder, so a repo that nests its files (Comfy-Org's
    # split_files/) lands FLAT here — the name in the workflow is the basename.
    path = os.path.join(dest, stored[0] if stored else os.path.basename(name))
    if os.path.exists(path) and (want == 0 or os.path.getsize(path) == want):
        print(f"ok (present)  {name}")
        continue
    print(f"downloading   {name}  <- {repo}")
    got = hf_hub_download(repo_id=repo, filename=name, local_dir=dest, revision=rev)
    if got != path:
        os.replace(got, path)
    got = os.path.getsize(path)
    if want and got != want:
        sys.exit(f"SIZE MISMATCH {name}: got {got}, expected {want} — wrong repo or a bad pull")
    print(f"ok            {name}  ({got/1e9:.1f} GB)")

# local_dir downloads leave a .cache/huggingface staging dir beside the weights. The volume holds
# weights and nothing else, so drop them — a re-run re-downloads whole rather than resuming, which
# is the trade this rule buys.
for root, dirs, _ in os.walk(VOL):
    if ".cache" in dirs and not root.startswith(f"{VOL}/hf-cache"):
        shutil.rmtree(os.path.join(root, ".cache"))
        dirs.remove(".cache")

PY

echo
# du only — a RunPod network volume is MooseFS-backed, so df reports the whole cluster, never
# this volume's usage or quota.
echo "== volume contents =="
du -sh "$VOL/comfy/models" "$VOL/hf-cache" "$VOL"
echo "== anything that is not a weight (expect only the two BiRefNet .py) =="
find "$VOL" \( -name .cache -o -name "*.py" -o -name "__pycache__" \) -maxdepth 6
