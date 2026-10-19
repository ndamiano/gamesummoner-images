# AWS images

EC2 runs machine images, not containers, and gives a g7e a 1.9 TB instance store and S3 in
the same region. The RunPod folder's shape (a container image, weights served off a network
volume) exists for RunPod constraints that do not apply here, so this row starts from the local
box's shape instead: engine baked, weights pulled at boot. The row class `aws/server.py` is
empty: a box is decommissioned by ending its entrypoint, and the instance ends itself.

| folder | queue | engine | instance |
|---|---|---|---|
| `llm/` | llm | Basalt `gamesummoner-v1`, Qwen3.8 Flash-Next UD-Q4_K_XL | g7e.2xlarge (RTX PRO 6000 Server, 96 GB, 64 GiB host) |
| `decide/` | decide | stock SGLang 0.5.21, JEV-27B-VL on Qwen3.8 27B NVFP4 | g7e.2xlarge |

## llm

### The image is an AMI; the pack rides the instance store

`build.sh` launches a builder from the AWS Deep Learning Base AMI (Ubuntu 24.04: driver, SSM
agent, the instance store on an LVM at `/opt/dlami/nvme`), or from one of our own AMIs with
`--base` to keep its built engine, ships the box code through the bucket, and runs
`provision.sh` on it from user-data: `build-essential` and `cmake`, the CUDA 13.3 toolkit from
NVIDIA's apt repo, Basalt cloned from our fork (`ndamiano/basalt`) at the tag `gamesummoner-v1`
into `/opt/basalt` and built with cmake in Release, a venv at `/opt/basalt/.venv` with Basalt's
requirements and `requests` (Basalt's binary runs `<source>/.venv/bin/python`), `s5cmd`, the
contract server, the row, the cell, the worker staged from `worker.ref`, and the systemd unit,
enabled at boot. Basalt builds for compute capability 12.0 only, the RTX PRO 6000 Blackwell.
Docker, containerd, soci-snapshotter, fabricmanager, the CDI refresh, the GPU settings service
and snapd are disabled: together they were most of the 77 s userspace boot and none serves one
GPU. Nothing is pulled from a registry and no credential is involved beyond the instance role.
`bake.sh` wipes the per-boot state and snapshots the builder into an AMI (it runs a box's
`pack_env.sh` first only when the box has one; the llm box does not).

**Weights** are one Basalt pack, `Qwen3.8-FN-UD-Q4_K_XL-bf16key-mtptuned.basalt` (106.8 GiB),
at `s3://gamesummoner-models-<region>-<account>/basalt/<pack>`. The entrypoint pulls it with
`s5cmd` onto the instance store at boot (70 s, measured). The AMI holds no weights: instance
store is never in a snapshot.

### Boot

`entrypoint.sh`, run by `gamesummoner-llm.service` at `multi-user.target`, in order.

1. **Env.** `/etc/gamesummoner/env.json` if it exists, else the instance user-data from IMDS,
   which is how the control plane delivers `CP_URL`, `WORKER_TOKEN`, `WORKER_ID`, `LLM_*`,
   `BASALT_ARGS_EXTRA`, `WORKER_SLOTS` (`launch.sh` passes the env file as the user-data
   verbatim). User-data that is a script is a builder's provisioning, and the entrypoint runs
   it; anything else means not a worker launch, and it exits 0 without powering off. Metadata
   unreachable for 60 s means no network, and the box powers off. cloud-init is disabled in the
   image (it was 15 s of every boot and nothing here needs it), which means it no longer writes
   the netplan config either: the one it wrote on the builder matches the interface by MAC
   address, and a fresh instance has a new MAC, so a box booted that way has no network at all
   (2026-09-18 05:19, one box, no SSM, no worker, would have idled forever). `provision.sh`
   writes a netplan config that matches by interface name instead.
2. **Guards.** `LLM_MODEL` and `LLM_N_CTX` are set and `LLM_N_CTX` is at most 262144, the
   engine window; the instance store is mounted with room for the pack.
3. **Pack.** GPU persistence mode goes on in the background, so the card initialises under the
   download, and `s5cmd` pulls the pack from the region's bucket onto the instance store.
4. **Engine**: `basalt serve --config <json> --alias $LLM_MODEL --host 127.0.0.1 --port 8001
   --parallel $WORKER_SLOTS` plus `BASALT_ARGS_EXTRA`. The entrypoint writes the config: engine
   `expert-cache` auto, `prefill` auto, `spec` 7, `spec-min-p` 0.5, `kv` int8,
   `vram-reserve-mib` 700, `resident-experts` true, `max-context` 262144,
   `conversation-cache-mib` 32000, `conversation-cache-slots` 16; server `vision` dynamic,
   `fit-max-tokens` true. The engine answers to whatever name `LLM_MODEL` gives it. It runs in
   its own session so its process-tree kill on a crash cannot take the entrypoint with it.
   `WORKER_SLOTS` is 1 in production: Basalt reuses a slot only on an exact prefix match, an
   agent's next turn never matches one, so with `--parallel` above 1 it re-reads the whole
   conversation every turn. Builds sharing a box take turns on the one slot; conversation
   parking restores each build's conversation in 0.3–0.5 s when its turn comes.
5. **Contract server** (`python -m aws.llm.server`) and **worker**, side by side, both on the
   venv's python. The engine binds `127.0.0.1` and the security group has no inbound rule.

Exit: `/shutdown` SIGTERMs the entrypoint, which drains the worker for up to 60 s, ends the
engine's process group and the contract server, and exits 0; a box process dying on its own
exits 1. Either way the entrypoint powers the instance off (`GS_NO_POWEROFF=1` for a hand run
on a builder), and the instance was launched with `InstanceInitiatedShutdownBehavior=terminate`,
so it ends without an AWS credential on the box. EC2 does not bill the `shutting-down` state.

### Capacity and quota

Both G-family quotas are in vCPUs, per region: 8 on-demand (`L-DB2E81BA`) and 8 spot
(`L-3819A6DF`), which is one g7e.2xlarge at a time. Increases to 16 were requested 2026-10-19
and are pending. us-east-1 also hosts the decide box, so an llm rung there shares that one slot
until the increase lands. The scripts walk every zone that offers the type and take the first
that answers. A terminated spot instance's one-time request can hold the quota until it is
cancelled (`MaxSpotInstanceCountExceeded` with nothing running means that, not a missing grant).

### Build, test, bake

The scripts in `aws/` serve every box and take its queue first.

```
aws/build.sh llm [--base <ami>]      # builder up and provisioned
aws/ssm.sh <instance> "echo $(base64 -w0 env.json) | base64 -d > /etc/gamesummoner/env.json; \
    GS_NO_POWEROFF=1 nohup setsid /opt/gamesummoner/entrypoint.sh > /var/log/gamesummoner-llm.log 2>&1 < /dev/null &"
aws/ssm.sh <instance> 'grep "^\[boot" /var/log/gamesummoner-llm.log'
aws/bake.sh llm <instance>           # once a job has gone through: the AMI
aws/launch.sh llm <ami> env.json     # a box the way the scaler will start one
```

On the builder the entrypoint is started by hand with `GS_NO_POWEROFF=1`, because the builder
has to survive its own decommission to be baked.

`env.json` follows `env.example.json`; the worker id must exist in the control plane's workers
table and `CP_URL` must be reachable from EC2 (a Tailscale funnel for a local control plane).
Queue a job before the box is healthy or set `IDLE_EXIT_SECONDS` past the boot: the worker's
idle clock starts when it starts, not when the image turns healthy, so a box that boots into an
empty queue with a short window exits on the spot (the scaler only launches on pending work, so
production never sees this).
Standing infrastructure, per region: bucket `gamesummoner-models-<region>-<account>`
(the llm pack under `basalt/`; decide's engine parts under `env/`, its weights under
`decide/`), security group `gs-gpu-worker` (no inbound) in the default VPC, an S3 gateway
endpoint on its route tables. Global: role and instance profile `gs-gpu-worker` (S3 on every
`gamesummoner-models-*` bucket, SSM). A box knows its region from IMDS and its account from
`/opt/gamesummoner/account` (written at provision), so one AMI copied with `copy-image` serves
any region whose bucket holds what the box pulls; nothing in the image names a region. Regions
so far, each bucket holding the llm pack: eu-north-1 (`vpce-006e37cedf2225ed7`), us-west-2
(`vpce-0bb567546d88e900f`), us-east-1 (`vpce-09883550ad82c6d4f`). The scaler launches
these boxes through its EC2 provider (`src/scaler/ec2_client.py` in the control plane repo):
the env as user-data, by zone in the default VPC, the group and profile by name.

AMIs (image id `ece6d46a-cdfb-4f28-9107-0b472b1a23ea`, built 2026-10-19): us-east-1
`ami-07097b9b8a513ac36`, eu-north-1 `ami-057b5f4f5500ff178`, us-west-2
`ami-0fd3668cab20dc240`. Boot-tested on the builder: tool calls, the `reasoning_content` split,
vision and a 48K prompt.

Region choice, 2026-09-18: nine regions offer g7e, two AZs each (us-west-2 four). Seven-day
spot averages for g7e.2xlarge: eu-north-1a $1.16, us-west-2a $1.25, us-east-1d $1.58,
ap-south-1b $1.66; the rest $1.9-2.8 and with a 5-vCPU G spot quota, below one box. The
cheap zone of a region is the contended one (us-west-2a $1.25 against 2.4-3.4 in b, c, d),
so a launch walks the zones of a region before the next region. Cross-region copy of the llm
pack is 106.8 GiB, about $2.30 in transfer, and holding it in a second bucket costs ~$2.60 a
month.

## decide

JEV-27B-VL's decision chains (`contract/decide.py`) on stock SGLang, the same engine, adapter split
and launch line as `local/decide/`, so the boxes answer alike. Not the container: the local image
is 46 GB, so `provision.sh` installs SGLang 0.5.21 natively into `/opt/venv` from `pins.txt`, the
local image's `pip freeze`, installed whole (`-r`) and forced (`--overrides`). Installed from
`sglang==0.5.21` alone, the venv imports on a box with a system CUDA and dies on the DLAMI
(`libcupti.so.13`): the CUDA runtime wheels reach the image through `cuda-toolkit` extras, not
torch's own requirements. `flashinfer-cubin` comes from flashinfer's root index, not its `cu130`
one. The venv is 20 GB.

`pack_env.sh` tars the venv, the CUDA 13.0 toolkit, the uv python and the kernel caches to
`env/decide-env-<image id>.parts/`, and the entrypoint fetches every part at once and untars
them onto the NVMe through a pipe in order as they land. Weights live in the bucket under `decide/<repo>@<revision>/`, plain files the
builder copied from Hugging Face at provision; a box whose bucket lacks them pulls from Hugging
Face instead. The entrypoint preps the decision head on every boot (`engines/jev/prep.py`, 2 s).

The engine runs one request at a time (`--max-running-requests 1`, `WORKER_SLOTS` 1), for the
reason in the top-level README.

Measured 2026-10-16, us-east-1b: unit to engine files 27 s (24 GB), weights 20 s (23.9 GB), prep
2 s, then the engine load; job claimed 134 s after EC2 `running`. The frog chain (22 steps) ran
16.5 s and matched the builder's run of the same AMI exactly. Against the local 5090 every answer
matched but the mechanics list, where four mechanics within 0.035 of the 0.6 line changed side.

```
aws/build.sh decide [--on-demand]
aws/bake.sh decide <instance>
aws/launch.sh decide <ami> env.json
```

AMIs: `ami-04366b4063158fcfe` (us-east-1).
