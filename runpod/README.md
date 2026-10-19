# RunPod images

RunPod runs containers only, bills per host pull, restarts an exited container, and gives a
pod a network volume mounted at `/workspace` that also answers the S3 API. Everything in this
folder that looks odd is one of those four facts. The row class `runpod/server.py` is empty:
a pod is decommissioned by ending its entrypoint, and the pod itself is the control plane's to
terminate.

| folder | queue | engine | card |
|---|---|---|---|
| `llm/` | llm | Pennyroyal SGLang, Qwen3.8 Flash-Next | RTX PRO 6000 |
| `image/` | image | ComfyUI (Dockerfile in `comfy/`) | 5090 / RTX PRO 4500 |
| `mesh/` | mesh | TRELLIS.2 | 5090 / RTX PRO 4500 |

Every pod gets the same env from the scaler (`CP_URL`, `WORKER_TOKEN`, `WORKER_QUEUE`,
`IDLE_EXIT_SECONDS`, plus the llm's `LLM_*`, `SGLANG_ARGS_EXTRA`, `WORKER_SLOTS`) and its
worker id as the pod's args (`--worker-id <id>`); every entrypoint writes those to
`/etc/gamesummoner/env.json` for the worker. Every engine binds `127.0.0.1` and nothing is
exposed: a reachable inference server is an unauthenticated GPU. Every template's env is the
registry auth alone. Every build script stages the pinned worker and bakes a fresh image id;
bump the tag on every push, because RunPod caches images per host.

Two volumes, because a volume pins its datacenter: the **art volume** (EU-RO-1, 5090-class
hosts; `runpod.network_volume_id` in the control plane) holds the image and mesh
weights, and the **llm volumes** (one per datacenter the RTX PRO 6000 is in;
`queues.llm.network_volume_ids`) hold the checkpoint and the engine tarball. `provision_volume.sh`
is the authority on the art volume: run it on any cheap pod with the volume mounted, no GPU
touched, and it pulls every weight from Hugging Face at datacenter bandwidth, idempotently.
The NSFW classifier at `comfy/models/safety/` (`model.pt` + `config.json`) is not in it: the
control plane's `scripts/export_safety_model.py` produces it, and it is copied over the S3
endpoint. After provisioning a pod never talks to Hugging Face (`HF_HUB_OFFLINE=1` on the mesh
box; ComfyUI resolves weights by folder).

## ComfyUI box (image)

One Dockerfile, `comfy/Dockerfile`, built per queue with `--build-arg QUEUE=image`:
the queue is baked (`QUEUE`), the cell it runs is `runpod/<queue>/server.py`, and the
entrypoint refuses a `WORKER_QUEUE` that disagrees. The base is `pytorch/pytorch:2.10.0-cuda13.0`,
which matches the home box exactly, the only configuration the sprite workflow has ever run on;
everything installs into that system python on purpose, because a venv would shadow the CUDA
build. ComfyUI is pinned by full SHA to the commit the home box serves from (v0.30.1), and the RMBG node is pinned the same way; only its BiRefNet path is
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
| image | `checkpoints/DreamShaperXL_Turbo_v2_1` (tiles, scene-chain terrain), `diffusion_models/qwen_image_2512_fp8_e4m3fn` (sprites, scenes, mesh subjects), `diffusion_models/qwen_image_edit_2511_fp8mixed` (the scene embed), the `text_encoders/qwen_2.5_vl_7b_fp8_scaled` and `vae/qwen_image_vae` both Qwen graphs share, `RMBG/BiRefNet` (the matte) | 58 |

An image checkpoint is named by a workflow the control plane sends, so a weight that leaves the
control plane's workflows leaves the volume and `stage.py` with it. Roll the volume forward
first, the template second, and retire the old weight last: the warmup names the weights, so a
pod predating a swap dies at boot on a weight that is not there.

```
runpod/image/build.sh ndamiano100/gamesummoner-images:image-runpod-v1
```

## mesh

TRELLIS.2-4B (image to mesh with native PBR textures) behind `engines/trellis/trellis_server.py`.

The Dockerfile's builder stage compiles every CUDA extension TRELLIS's `setup.sh` installs
(nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, the in-tree o-voxel) against `TORCH_CUDA_ARCH_LIST=12.0`,
Blackwell being the only card class this runs on, and ships them as wheels, so the runtime
stage needs no compiler beyond the `gcc` triton wants for flex_gemm's kernels. TRELLIS is pinned
by SHA and patched (`engines/trellis/trellis2-sdpa-dinov3.patch`: sdpa backends for sparse
attention, which upstream only knows through xformers/flash-attn, and the transformers DINOv3
module layout); transformers, timm, einops and kornia are pinned to the home-verified stack
because transformers 5.14 renamed the DINOv3 internals TRELLIS reaches into.

Boot (`mesh/entrypoint.sh`): guards (the volume), worker id, a one-GiB
sequential read probe off the biggest checkpoint (if the pipeline loads at that rate the volume
is the bottleneck; if much slower, the loader's mmap pattern is), env file,
the TRELLIS server, then the contract server and the worker. The TRELLIS server stages its
checkpoints to tmpfs, loads the pipeline and runs one throwaway mesh in the background, and
reports `warm` on `/health`; the contract server's `/health` is 503 until then, because a job
claimed earlier pays the whole cold start (53 s against 13 s, measured) as billed work.

Jobs: `trellis_mesh` posts the PNG to `/generate` and retries once, because the 4B pipeline
degrades across generates (70 s per mesh early, 160 s and CuMesh OOM by ~30) and a 500 is
almost always that; the server's stage/load/generate split rides back on the result, since the
pod's stdout is unreachable.

Volume: `trellis2-weights/` with the encoders beside it (`encoders/dinov3-vitl16`,
`encoders/BiRefNet`, open-licensed mirrors, and the pipeline configs rewritten to those paths so
no hub name is ever resolved), the `hf-cache` pre-seed, and
`flex-gemm-autotune.json`, flex_gemm's autotune picks, kept on the volume because on container
disk every boot re-benchmarked every sparse-conv config (32.8 s of a 41.3 s warmup).

```
runpod/mesh/build.sh ndamiano100/gamesummoner-images:mesh-runpod-v1
```

## llm

Qwen3.8 Flash-Next NVFP4 on the 96 GB RTX PRO 6000 (SM120), served by Pennyroyal, jpezzulli's
SGLang fork for that card, at a pinned tag on our fork (`ndamiano/sglang-rtxpro6000`) that adds
the prepacked loader. It is the only stack measured to serve Flash-Next on that card with
vision, tool calls and a clean reasoning split (2026-09-02: 203 tok/s short-context decode,
~100 at 54K). Both editions of the card are in `queues.llm.gpu_type_ids`, Workstation first
(faster, more expensive, EU-RO-1 mostly); `allowed_cuda_versions: ["13.0"]` is the driver floor
the CUDA 13 wheels need, and a pod on an older driver is dead on any card until the boot
deadline reaper collects it.

### The image is the toolchain; the engine rides the volume

The Dockerfile has three stages.

- `builder` installs the fork, its venv (python 3.12.13 via uv, `--no-build-isolation` so the
  cuda-tile sdist sees the venv), and the sm120 kernel caches, then tars all of it into
  `/env.tar` and writes its sha256 prefix to `/env.id`.
- `env` is `FROM scratch` holding only those two files. `build.sh --env` dumps this stage to
  `runpod/llm/out/` instead of making an image.
- The runtime stage is the CUDA base plus what the JIT link line needs (nvcc, gcc-15, ninja,
  cmake, the CUDA dev headers), the contract server, the worker, and `env.id`. Nothing else.

Why: the venv, fork checkout and kernel caches are ~10 GB of mostly CUDA libraries that do not
compress. As image layers they cost 90-120 s on every host that had not seen the tag, and
RunPod caches per host. Off the volume, `tar -xf` of one uncompressed file is a sequential read
at volume speed (~200 MB/s measured). Cold create-to-serving measured 205 s, warm 110 s
(2026-09-03).

The id ties the two artifacts. The runtime stage is built against that exact venv: its CUDA
libraries are symlinks into `/opt/venv/.../nvidia/cu13/lib/`, a path that only exists after
restore. The entrypoint restores `env/llm-env-<id>.tar` and refuses to boot when it is not on
the volume, so an image and a tarball from different builds never meet, and the mismatch fails
at second zero with a clear message instead of minutes in with a CUDA error. Every llm volume
(one per datacenter, volumes pin their datacenter) must hold the tarball before the template
moves to the tag.

A container restarted in place skips the restore: `/opt/venv/bin/python` is already there.

### Kernel caches

`penny_cache_v6.tgz` is a build input, not source: the FlashInfer/torchinductor/triton caches
captured from the trial pod, plus the serve script and the NIXL config. It lives at
`s3://<llm-volume>/fastboot/` and is fetched into `runpod/llm/` before a build (`build.sh`
checks). Two things the Dockerfile does with it:

- **gcc-15 identity.** The fork pins the JIT identity to gcc-15; a different host compiler
  silently invalidates every baked kernel and the pod re-tunes for ~300 s.
- **AOT promotion.** FlashInfer's JIT trusts only its AOT directory without rebuilding; the
  tarred JIT directory re-ninjas on every fresh container. The `builder` stage copies each
  compiled `.so` into the AOT tree.

HiCache/NIXL are stripped from the serve script: the nixl wheel's backend fails to create.
Persistence returns when a from-source NIXL earns it.

### The volume

| path | what |
|---|---|
| `models/pennyroyal/` | the checkpoint: `config.json`, one safetensors shard, the index, and `prepacked/` |
| `env/llm-env-<id>.tar` | the engine tarball for the image the template names |
| `fastboot/penny_cache_v6.tgz` | the build input above |

The checkpoint is **prepacked**. `RadixArk/Qwen3.8-Flash-Next-NVFP4` (126 GiB) loads through a
CPU-bound repack pass, ~110 s on a fresh pod, NVMe or not. The fork's `--load-format prepacked`
restores the post-repack state from one flat file in 20-55 s. The safetensors shards are gone
from the volume except one, because the 142 GB quota holds one copy, not two, and the fork's
namespace helper raises on a checkpoint with zero shards. The pack's key is the load layout
(quantization backends, tp, config hash), not the path, so a second volume in another
datacenter is a copy, not a re-dump. Keep the `.cache/huggingface` metadata the download
wrote, or the namespace helper hashes every byte.

How to populate a new llm volume (worked 2026-09-02, 130 GB in 9 minutes for $0.04 of CPU pod):

1. `hf download` the checkpoint to `models/pennyroyal/` on any pod with the volume mounted,
   then delete every shard but one plus the index over the volume's S3 endpoint.
2. Rent a CPU pod in the new volume's datacenter, `truncate` the pack file to its final size
   on the mount, and have 16 workers each `get-object --range` a 100 MB piece from the old
   volume's S3 endpoint and `dd … seek=` it into place (~215 MB/s).

What does not work: shards and pack on the volume at once (a full volume fails every write
silently, the mount reports nothing, the S3 gateway keeps answering stale sizes); writing the
pack through the S3 gateway (~15 MB/s single stream, 413 over ~128 MB parts, 524 at 48
concurrent parts, lost parts by `CompleteMultipartUpload`); a GPU pod loading the shards off
the volume to dump a fresh pack (dies to the container cgroup; stage to NVMe first). A pod
launched by hand must not be named `maestro-<queue>-…`: the scaler reaps a pod under its prefix
that never registers a worker.

S3 access: the bucket name is the volume id, the region/endpoint pair is the datacenter.

```
aws s3 ls --profile runpod --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io \
    s3://<volume-id>/ --recursive --human-readable
```

A large `cp` can end in `InvalidPart … N parts missing` with the object already whole; trust
`head-object`'s `ContentLength` against the local size, not the exit code.

### Boot

`entrypoint.sh`, in order.

1. **Guards.** `LLM_MODEL` must be `pennyroyal` and `LLM_N_CTX` at most 524288: the serve
   script bakes the served-model-name and the window, a request naming another model is
   refused, and a budget past the window is a prompt the engine rejects. Both are contracts the
   control plane's settings state, so a mismatch dies here rather than on the first turn.
2. **Worker id.** The scaler passes it as the pod's args (`--worker-id <id>`); a template can
   set `WORKER_ID` in the env instead.
3. **Engine restore** from the volume, as above.
4. **Env file.** The scaler's env (`CP_URL`, `WORKER_TOKEN`, `WORKER_QUEUE`,
   `IDLE_EXIT_SECONDS`, `LLM_MODEL`, `LLM_N_CTX`, `SGLANG_ARGS_EXTRA`, `WORKER_SLOTS`) plus the
   worker id become `/etc/gamesummoner/env.json`, which the worker reads.
5. **Engine launch** through `/root/serve-nohicache.sh` from the tarball, the only launch that
   fits the card: it offloads the PLE table to host RAM (`--ple-offload-embedding`; without it
   the weights alone exceed 96 GB), applies the YaRN rope override, and sets the qwen3
   reasoning and tool-call parsers a build turn depends on. `SGLANG_ARGS_EXTRA` is the control
   plane's `llm.sglang_args`, delivered at create, so a tuning flag is a settings edit and the
   next pod, never a new tag. It is placed between `--load-format prepacked` and the
   `--cuda-graph-bs` list (one entry per worker slot) so the image's own invariants win. A flag
   that changes the FlashInfer autotune key costs one ~300 s re-tune per pod; measure it on one
   pod first. Three attempts: the first launch OOM-SIGKILLs under the 188 GB container cgroup
   about half the time (mmap'd checkpoint plus PLE offload plus load buffers) and a relaunch
   reliably loads. The contract server starts after the engine holds, so the pid it watches is
   the one that serves.
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
runpod/llm/build.sh ndamiano100/gamesummoner-images:llm-runpod-v1 --env
aws s3 cp --profile runpod --region <dc> --endpoint-url https://s3api-<dc>.runpod.io \
    runpod/llm/out/env.tar s3://<llm-volume>/env/llm-env-$(cat runpod/llm/out/env.id).tar
docker push ndamiano100/gamesummoner-images:llm-runpod-v1
```

- `--env` only when the `builder` stage's inputs changed (fork tag, the cache tarball, pip
  deps) or the layer cache is cold; a code-only rebuild reuses the cached tarball and keeps its
  id. The export streams through `$DOCKER_TMPDIR` (default `/tmp`) at its full ~10 GB.
- Upload the tarball to every llm volume before moving the template to the tag.
- Bump the tag on every push: RunPod caches images per host, and re-pushing a tag leaves stale
  copies serving on warm hosts.
- The Docker Hub repo is private, so the template carries the registry auth; without it a pod
  exits in one second with no logs. That auth is the template's only env.
