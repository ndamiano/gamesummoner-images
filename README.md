# gamesummoner-images

Inference images for GameSummoner. An image owns generation end to end: it pulls its weights,
runs the engine, and serves the worker contract. The control-plane side lives in
[gamesummoner-workers](../gamesummoner-workers), which each image pulls at a pinned commit
when it boots.

| path | what |
|---|---|
| `contract/server.py` | `Server`: the image side of the worker contract (`/health`, `/metadata`, `/job`, `/shutdown`), standard library only |
| `contract/llm.py` | `LlmServer`: the llm queue against an OpenAI-compatible engine |
| `contract/comfy.py` | `ComfyServer`: the comfy queue on ComfyUI, a workflow in, images or a TRELLIS.2 GLB out |
| `contract/decide.py` | `DecideServer`: a chain of JEV System 1 decisions, each read from option-token logprobs through the OpenAI completions API |
| `contract/safety.py` | the NSFW verdict every rendered image carries |
| `engines/` | `jev/head/`: JEV's per-option bias and per-kind temperature, which the decide contract applies; every engine is upstream |
| `local/llm/` | the llm box for a local 5090: stock SGLang serving Qwen3.8 27B NVFP4 |
| `local/decide/` | the decide box for a local 5090: stock vLLM serving JEV-27B-VL's decision LoRA on Qwen3.8 27B NVFP4 |
| `runpod/` | the RunPod boxes, one per queue: [runpod/README.md](runpod/README.md) |
| `aws/` | the EC2 boxes, an AMI per queue: [aws/README.md](aws/README.md) |

The contract servers form a grid. `contract/server.py` is the corner; a queue's
`contract/<queue>.py` is a column (`ready`, `run`, the engine's flags); a provider's
`<provider>/server.py` is a row (what every box on that provider needs on top); and an image's
`<provider>/<queue>/server.py` is the cell, `class X(ProviderServer, QueueServer): pass`. Every
class extends the command line by overriding `add_arguments` and `__init__` and calling super
first, and the MRO threads the chain, so the cell needs no code of its own.

|  | `contract/llm.py` | `contract/comfy.py` | `contract/decide.py` |
|---|---|---|---|
| `local/server.py` | `local/llm/` | `local/comfy/` | `local/decide/` |
| `runpod/server.py` | `runpod/llm/` | `runpod/comfy/` | |
| `aws/server.py` | `aws/llm/` | | `aws/decide/` |

A column sets two tuples: `capabilities`, the queue names a worker may pull for this box, and
`kinds`, the payload kinds `/job` runs, which is what the worker sends as `type`. They differ
for every queue but llm (`comfy` runs `comfy_image` and `comfy_mesh`).

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
(`--entrypoint-pid`), the entrypoint drains the worker and ends, and the container (or, on
EC2, the systemd unit and then the instance) ends with it. A RunPod pod is the control plane's
to terminate, which it does for any worker that deregistered; an EC2 instance terminates itself
on power-off. No image holds a provider credential. Every row is an empty class until a
provider needs more than that.

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

Tests: `python -m pytest tests`. They need only the standard library and pytest.

## Local decide box

Same shape as the llm box: `local/decide/build.sh`, then `local/decide/run.sh env.json` with
`WORKER_QUEUE=decide`. Boot pulls `autotrust/JEV-27B-VL` and `unsloth/Qwen3.8-27B-NVFP4` at pinned
revisions and serves the base with JEV's tokenizer and its `adapter_vllm` as the `jev-decision`
LoRA. `contract/decide.py` asks each decision as one completion with `max_tokens: 1` and logprobs
over the option tokens, then applies the bias and temperature in `engines/jev/head/`.

One prompt per request: JEV's LoRA path moved probabilities by up to 0.085 when several prompts
shared a batch. A decision is one generated token, so the engine runs eager (no torch.compile, no
CUDA graphs), text-only, with streamed weights: ready about 45 s after start on a 5090, from 146 s.

## RunPod boxes

`runpod/` holds the two RunPod images (llm, comfy) ported onto the contract.
Everything about them that needs a why (the llm pack served off the volume, weight staging on
the ComfyUI box, boot orders, build
and ship, volume layouts) is in [runpod/README.md](runpod/README.md).
