# gamesummoner-images

Inference images for GameSummoner. An image owns generation end to end: it pulls its weights,
runs the engine, and serves the worker contract. The control-plane side lives in
[gamesummoner-workers](../gamesummoner-workers), which each image pulls at a pinned commit
when it boots.

| path | what |
|---|---|
| `contract/server.py` | `Server`: the image side of the worker contract (`/health`, `/metadata`, `/job`, `/shutdown`), standard library only |
| `contract/llm.py` | `LlmServer`: the llm queue against an OpenAI-compatible engine |
| `contract/comfy.py`, `image.py`, `video.py` | `ComfyServer` and the two queues on ComfyUI: `ImageServer` (a workflow in, images out), `VideoServer` (a still in, a sprite sheet out via `anim_sheet.py`) |
| `contract/mesh.py` | `MeshServer`: TRELLIS meshes and rendered sprite sheets |
| `contract/safety.py` | the NSFW verdict every rendered image carries |
| `engines/` | inference servers we wrote ourselves: `trellis/` (TRELLIS.2), `sprites/` (kimodo sheets); ComfyUI and SGLang are upstream |
| `local/llm/` | the llm box for a local 5090: stock SGLang serving Qwen3.8 27B NVFP4 |
| `runpod/` | the RunPod boxes, one per queue: [runpod/README.md](runpod/README.md) |

The contract servers form a grid. `contract/server.py` is the corner; a queue's
`contract/<queue>.py` is a column (`ready`, `run`, the engine's flags); a provider's
`<provider>/server.py` is a row (what every box on that provider needs on top); and an image's
`<provider>/<queue>/server.py` is the cell, `class X(ProviderServer, QueueServer): pass`. Every
class extends the command line by overriding `add_arguments` and `__init__` and calling super
first, and the MRO threads the chain, so the cell needs no code of its own.

|  | `contract/llm.py` | `contract/image.py` | `contract/video.py` | `contract/mesh.py` |
|---|---|---|---|---|
| `local/server.py` | `local/llm/` | | | |
| `runpod/server.py` | `runpod/llm/` | `runpod/image/` | `runpod/video/` | `runpod/mesh/` |
| `aws/server.py` | | | | |

A column sets two tuples: `capabilities`, the queue names a worker may pull for this box, and
`kinds`, the payload kinds `/job` runs, which is what the worker sends as `type`. They differ
for every queue but llm (`image` runs `comfy_image`; `mesh` runs `trellis_mesh` and
`sprite_sheet`).

One folder per provider, one folder per queue inside it. An image folder holds its `server.py`,
Dockerfile, entrypoint, build script and `worker.ref`. The Dockerfile copies `contract/`, the
row and the cell at their repo paths under `/opt/gamesummoner`, and the entrypoint runs the cell
as `PYTHONPATH=/opt/gamesummoner python3 -m local.llm.server`.

The worker is baked, not pulled: `scripts/stage_worker.sh` (called by every `build.sh`) takes
the commit in the image's `worker.ref` out of a local checkout of
[gamesummoner-workers](../gamesummoner-workers) (`WORKER_REPO_DIR` to point elsewhere) into
`<image>/worker/`, and the Dockerfile copies it to `/opt/gamesummoner/worker`. The repo is
private, so pulling on the box would put a token on every instance; baking costs a megabyte and
boots the same worker every time. Moving an image to a new worker is editing its `worker.ref`
and rebuilding.

Decommissioning is the same everywhere so far: `/shutdown` SIGTERMs the entrypoint
(`--entrypoint-pid`), the entrypoint drains the worker and ends, and the container ends with it.
The instance behind the container is the control plane's to terminate, which it does for any
worker that deregistered, so no image holds a provider credential. Both rows are empty classes
until a provider needs more than that.

### Adding a queue

Write `contract/<queue>.py` with a `Server` subclass: set `capabilities`, implement `ready` (the
`/health` verdict, polled on a thread so the endpoint stays fast) and `run` (one job), and add
the flags its engine needs by extending `add_arguments` and `__init__`.

### Adding a provider

Write `<provider>/server.py` with a `Server` subclass (empty until the provider needs something,
`decommission` being the likely first), then per queue a `<provider>/<queue>/server.py` cell, an
entrypoint that starts the engine and the cell, and a Dockerfile that copies `contract/`, the row
and the cell at their repo paths. The image id is baked at build time and passed as
`--image-id`; the engine's command line is passed as `--cli-params=...` so `/metadata` can
report it.

## Contract

| endpoint | image answers |
|---|---|
| `GET /health` | 2xx when it serves, 429 when it is up but taking no work, 503 while the engine boots, 410 once decommissioning, 500 if the engine process died |
| `GET /metadata` | `{"imageId", "cliParams", "capabilities"}` |
| `POST /job` | `{"jobId", "type", "payload"}` in; 200 `{"status": "succeeded", "result"}` or 5xx `{"status": "failed", "error"}` |
| `POST /shutdown` | 202, health turns 410, then the provider's decommission command runs |

## Local llm box

Boot order in `local/llm/entrypoint.sh`: read env data from `/etc/gamesummoner/env.json`, pull
the weights from Hugging Face, then start SGLang, the contract server and the worker. `/shutdown` and `docker stop` both end the entrypoint,
which ends the container.

```
local/llm/build.sh
local/llm/run.sh path/to/env.json
```

`env.json` follows `local/llm/env.example.json`. The worker id in it must already exist in the
control plane's workers table:

```
cd ai-agent-test/src
../venv/bin/python -c "from db import workers; workers.worker_created('<id>', None, 'llm', None, 0.0)"
```

`run.sh` shares the host's Hugging Face cache, so only the first boot downloads the 18 GB
checkpoint.

Tests: `python -m pytest tests`. The contract tests need only the standard library; the
anim-sheet, sprites3d and video tests also need numpy and Pillow and skip without them.

## RunPod boxes

`runpod/` holds the four RunPod images (llm, image, video, mesh) ported onto the contract.
Everything about them that needs a why (the llm engine tarball on the volume, the prepacked
checkpoint, weight staging on the ComfyUI boxes, the mesh box's two servers, boot orders, build
and ship, volume layouts) is in [runpod/README.md](runpod/README.md).
