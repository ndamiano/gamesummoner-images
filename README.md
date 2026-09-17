# gamesummoner-images

Inference images for GameSummoner. An image owns generation end to end: it pulls its weights,
runs the engine, and serves the worker contract. The control-plane side lives in
[gamesummoner-workers](../gamesummoner-workers), which each image pulls at a pinned commit
when it boots.

| path | what |
|---|---|
| `contract/` | the image side of the worker contract (`/health`, `/metadata`, `/job`, `/shutdown`), standard library only |
| `local/llm/` | the llm box for a local 5090: stock SGLang serving Qwen3.8 27B NVFP4 |

One folder per provider, one folder per queue inside it.

## Contract

| endpoint | image answers |
|---|---|
| `GET /health` | 503 while the engine boots, 2xx when it serves, 410 once decommissioning, 500 if the engine process died |
| `GET /metadata` | `{"imageId", "cliParams", "capabilities"}` |
| `POST /job` | `{"jobId", "type", "payload"}` in; 200 `{"status": "succeeded", "result"}` or 5xx `{"status": "failed", "error"}` |
| `POST /shutdown` | 202, health turns 410, then the provider's decommission command runs |

## Local llm box

Boot order in `local/llm/entrypoint.sh`: read env data from `/etc/gamesummoner/env.json`, pull
the weights from Hugging Face, clone the worker at `local/llm/worker.ref`, then start SGLang,
the contract server and the worker. `/shutdown` and `docker stop` both end the entrypoint,
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
checkpoint, and mounts `~/Documents/gamesummoner-workers` (override with `WORKER_REPO_DIR`)
as the repo the worker is cloned from.

Tests: `python -m pytest tests`.
