# RunPod images

RunPod runs containers only, bills per host pull, restarts an exited container, and gives a
pod a network volume mounted at `/workspace` that also answers the S3 API. Everything in this
folder that looks odd is one of those four facts. The row class `runpod/server.py` is empty:
a pod is decommissioned by ending its entrypoint, and the pod itself is the control plane's to
terminate.

| folder | queue | engine | card |
|---|---|---|---|
| `llm/` | llm | Basalt, Qwen3.8 Flash-Next | RTX PRO 6000 |
| `comfy/` | comfy | ComfyUI: images and TRELLIS.2 meshes | 5090 / RTX PRO 4500 |

Every pod gets the same env from the scaler (`CP_URL`, `WORKER_TOKEN`, `WORKER_QUEUE`,
`IDLE_EXIT_SECONDS`, plus the llm's `LLM_*`, `BASALT_ARGS_EXTRA`, `WORKER_SLOTS`) and its
worker id as the pod's args (`--worker-id <id>`); every entrypoint writes those to
`/etc/gamesummoner/env.json` for the worker. Every engine binds `127.0.0.1` and nothing is
exposed: a reachable inference server is an unauthenticated GPU. Every template's env is the
registry auth alone. Every build script stages the pinned worker and bakes a fresh image id;
bump the tag on every push, because RunPod caches images per host.

Two volumes, because a volume pins its datacenter: the **art volume** (EU-RO-1, 5090-class
hosts; `runpod.network_volume_id` in the control plane) holds the ComfyUI
weights, and the **llm volumes** (one per datacenter the RTX PRO 6000 is in;
`queues.llm.network_volume_ids`) hold the Basalt pack. `provision_volume.sh`
is the authority on the art volume: run it on any cheap pod with the volume mounted, no GPU
touched, and it pulls every weight from Hugging Face at datacenter bandwidth, idempotently.
The NSFW classifier at `comfy/models/safety/` (`model.pt` + `config.json`) is not in it: the
control plane's `scripts/export_safety_model.py` produces it, and it is copied over the S3
endpoint. After provisioning a pod never talks to Hugging Face: ComfyUI resolves weights by folder.

## ComfyUI box (comfy)

`comfy/Dockerfile`, built with `--build-arg QUEUE=comfy`: the queue is baked (`QUEUE`), the
cell it runs is `runpod/<queue>/server.py`, and the entrypoint refuses a `WORKER_QUEUE` that
disagrees. The base is `pytorch/pytorch:2.10.0-cuda13.0`,
which matches the home box exactly, the only configuration the sprite workflow has ever run on;
everything installs into that system python on purpose, because a venv would shadow the CUDA
build. ComfyUI is pinned by full SHA to the commit the home box serves from (v0.39.2, which carries
the native TRELLIS.2 and mesh post-processing nodes the mesh graph uses), and the RMBG node is pinned the same way; only its BiRefNet path is
used, so its siblings' 2 GB of dependencies stay out, and `timm` is installed by hand because the
node execs a `birefnet.py` it downloads at inference time that imports it.

The models tree ComfyUI reads is `/opt/comfy-models`, a per-folder symlink tree the entrypoint
builds at boot: every folder is the volume's, except the ones holding weights this pod loads,
which point at a **staged copy on the container disk**. The volume streams bulk reads at
2.7 GB/s, but safetensors' mmap page-faults re-read at ~200 MB/s and the FUSE mount keeps no
page cache, so left alone the first claimed render paid the whole model load that way: 70 s
billed to a user's game for 2 s of sampling (measured 2026-08-02). `stage.py` copies this queue's
weights once, chunked and parallel (a single-stream `cp` of a 17 GB file reads well below what
the volume serves to concurrent readers), in the background while ComfyUI imports torch, and
swaps the symlinks only after every file landed whole. Not tmpfs: `/dev/shm` on a RunPod host
is a fixed ~46 GB, under the queue's weights. A stage that does not fit is skipped with
`[stage] skipped` in the log and a silent 3.5-minute load off the mount, so the templates carry
a 100 GB container disk. The RMBG node ignores `extra_model_paths.yaml` and resolves
`models_dir/RMBG/BiRefNet`, which is why the tree has to be a real path.

Boot (`comfy/entrypoint.sh`): guards, worker id, symlink tree, env file, staging in the
background, ComfyUI, wait for `/system_stats` (it imports torch and every custom node before it
binds), wait for staging, then the contract server and the worker. The contract server's
`/health` is 503 until its **warmup render** has run: the graph the queue's jobs run, one step
at the smallest size the model takes (a 256 px sprite through the Qwen unet and the BiRefNet
matte), so the weight load, the matte's
lazy import and the first-use CUDA kernels are paid before the worker can claim. A warmup that
fails is a 500, which the worker reads as "kill the box": a pod that cannot render dies at boot
instead of failing a user's job.

What the volume holds for these queues (`comfy/models/`):

| queue | weights | GB |
|---|---|---|
| comfy | `checkpoints/DreamShaperXL_Turbo_v2_1` (tiles, scene-chain terrain), `diffusion_models/qwen_image_2512_fp8_e4m3fn` (sprites, scenes, mesh subjects), `diffusion_models/qwen_image_edit_2511_fp8mixed` (the scene embed), the `text_encoders/qwen_2.5_vl_7b_fp8_scaled` and `vae/qwen_image_vae` both Qwen graphs share, `RMBG/BiRefNet` (the matte), `diffusion_models/trellis_2_bf16` with `vae/trellis_2_shape_vae_bf16` and `vae/trellis_2_texture_vae_bf16` and `clip_vision/dino_v3_vit_l` (TRELLIS.2 meshes) | 72 |

An image checkpoint is named by a workflow the control plane sends, so a weight that leaves the
control plane's workflows leaves the volume and `stage.py` with it. Roll the volume forward
first, the template second, and retire the old weight last: the warmup names the weights, so a
pod predating a swap dies at boot on a weight that is not there.

```
runpod/comfy/build.sh ndamiano100/gamesummoner-images:comfy-runpod-v1
```

## llm

Qwen3.8 Flash-Next on the 96 GB RTX PRO 6000 (SM120), served by Basalt from our fork
(`ndamiano/basalt`) at the tag `gamesummoner-v1`: upstream `jesdga95/basalt` plus local
commits, among them a resident-expert batch-doorbell fix in `src/core/verify.cpp`. Basalt builds
for compute capability 12.0 only. Both editions of the card are in `queues.llm.gpu_type_ids`,
Workstation first (faster, more expensive, EU-RO-1 mostly); `allowed_cuda_versions: ["13.0"]` is
the driver floor the CUDA 13 runtime needs, and a pod on an older driver is dead on any card
until the boot deadline reaper collects it.

### The image is the engine; the pack rides the volume

The Dockerfile has two stages. The builder (`nvidia/cuda:13.3.1-devel-ubuntu24.04`) clones the
fork at the tag, builds it with cmake in Release, and makes the venv at `/opt/basalt/.venv` with
Basalt's requirements and `requests` (Basalt's binary runs `<source>/.venv/bin/python`). The
runtime stage (`nvidia/cuda:13.3.1-runtime-ubuntu24.04`) takes `/opt/basalt` from it, plus the
contract server, the worker and the image id. The image is 4.7 GB, pushed as
`ndamiano100/gamesummoner-images:llm-basalt-runpod-v1`; template `2b7rhfi08k` ("LLM Worker
(Basalt)") names it, with a 50 GB container disk and the registry auth.

The weights are one Basalt pack, `Qwen3.8-FN-UD-Q4_K_XL-bf16key-mtptuned.basalt` (106.8 GiB),
served straight from the llm volume at `/workspace/models/basalt/<pack>` (`BASALT_PACK` in the
image env).

### The volume

| path | what |
|---|---|
| `models/basalt/Qwen3.8-FN-UD-Q4_K_XL-bf16key-mtptuned.basalt` | the pack |

The llm volumes are `juqk1aq0ew` (EU-RO-1) and `szjxc7ha34` (US-NC-2).

A pack goes onto a new volume from a CPU pod in its datacenter: `truncate` the file to its final
size on the mount, and have 16 workers each `get-object --range` a 100 MB piece from another
volume's S3 endpoint and `dd … seek=` it into place (~215 MB/s, measured 2026-09-02). Writing it
through the S3 gateway does not work: ~15 MB/s single stream, 413 over ~128 MB parts, 524 at 48
concurrent parts, lost parts by `CompleteMultipartUpload`. A pod launched by hand must not be
named `maestro-<queue>-…`: the scaler reaps a pod under its prefix that never registers a worker.

S3 access: the bucket name is the volume id, the region/endpoint pair is the datacenter.

```
aws s3 ls --profile runpod --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io \
    s3://<volume-id>/ --recursive --human-readable
```

A large `cp` can end in `InvalidPart … N parts missing` with the object already whole; trust
`head-object`'s `ContentLength` against the local size, not the exit code.

### Boot

`entrypoint.sh`, in order.

1. **Guards.** `CP_URL`, `WORKER_TOKEN`, `LLM_MODEL` and `LLM_N_CTX` are set, and `LLM_N_CTX` is
   at most 262144: a budget past the engine window is a prompt the engine rejects, so a mismatch
   dies here rather than on the first turn.
2. **Worker id.** The scaler passes it as the pod's args (`--worker-id <id>`); a template can
   set `WORKER_ID` in the env instead.
3. **Pack.** The pod refuses to boot when the pack is not at `BASALT_PACK`.
4. **Env file.** The scaler's env (`CP_URL`, `WORKER_TOKEN`, `WORKER_QUEUE`,
   `IDLE_EXIT_SECONDS`, `LLM_MODEL`, `LLM_N_CTX`, `BASALT_ARGS_EXTRA`, `WORKER_SLOTS`) plus the
   worker id become `/etc/gamesummoner/env.json`, which the worker reads.
5. **Engine**: `basalt serve --config <json> --alias $LLM_MODEL --host 127.0.0.1 --port 8001
   --parallel $WORKER_SLOTS` plus `BASALT_ARGS_EXTRA`. The entrypoint writes the config: engine
   `expert-cache` auto, `prefill` auto, `spec` 7, `spec-min-p` 0.5, `kv` int8,
   `vram-reserve-mib` 700, `resident-experts` true, `max-context` 262144,
   `conversation-cache-mib` 32000, `conversation-cache-slots` 16; server `vision` dynamic,
   `fit-max-tokens` true. The engine answers to whatever name `LLM_MODEL` gives it.
   `BASALT_ARGS_EXTRA` is the control plane's `llm.basalt_args`, delivered at create, so a
   tuning flag is a settings edit and the next pod, never a new tag. `WORKER_SLOTS` is 1 in
   production: Basalt reuses a slot only on an exact prefix match, an agent's next turn never
   matches one, so with `--parallel` above 1 it re-reads the whole conversation every turn.
   Builds sharing a box take turns on the one slot; conversation parking restores each build's
   conversation in 0.3–0.5 s when its turn comes.
6. **Contract server** (`python -m runpod.llm.server`) and **worker** (baked at
   `worker.ref`), side by side. The engine binds `127.0.0.1` and the worker dials out, so the
   pod exposes no port: an llm server reachable from the internet is an unauthenticated GPU.

Exit: `/shutdown` SIGTERMs the entrypoint, which drains the worker for up to 60 s, ends the
engine and the contract server, and exits 0. A box process dying on its own exits 1. RunPod
restarts an exited container either way, even on exit 0, and keeps billing; the control plane's
scaler terminates the pod behind a worker that deregistered on its next tick (≤15 s), which is
the guarantee. No pod carries a RunPod API key for that reason.

### Build and ship

```
runpod/llm/build.sh ndamiano100/gamesummoner-images:llm-basalt-runpod-v1
docker push ndamiano100/gamesummoner-images:llm-basalt-runpod-v1
```

- Every llm volume holds the pack before the template moves to the tag.
- Bump the tag on every push: RunPod caches images per host, and re-pushing a tag leaves stale
  copies serving on warm hosts.
- The Docker Hub repo is private, so the template carries the registry auth; without it a pod
  exits in one second with no logs. That auth is the template's only env.
