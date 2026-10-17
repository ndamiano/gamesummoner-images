# AWS images

EC2 runs machine images, not containers, and gives a g7e a 1.9 TB instance store and S3 in
the same region. The RunPod folder's oddities (engine tarball on a volume, image diet, per-host
pull) exist for RunPod constraints that do not apply here, so this row starts from the local
box's shape instead: engine baked, weights pulled at boot. The row class `aws/server.py` is
empty: a box is decommissioned by ending its entrypoint, and the instance ends itself.

| folder | queue | engine | instance |
|---|---|---|---|
| `llm/` | llm | Pennyroyal SGLang, Qwen3.8 Flash-Next NVFP4 | g7e.2xlarge (RTX PRO 6000 Server, 96 GB, 64 GiB host) |
| `decide/` | decide | stock SGLang 0.5.21, JEV-27B-VL on Qwen3.8 27B NVFP4 | g7e.2xlarge |

## llm

### The image is an AMI; the engine rides the instance store

`build.sh` launches a builder from the AWS Deep Learning Base AMI (Ubuntu 24.04: driver, SSM
agent, the instance store on an LVM at `/opt/dlami/nvme`), or from one of our own AMIs with
`--base` to keep its tuned caches, ships the box code through the bucket, and runs
`provision.sh` on it from user-data: gcc-15 (the fork's JIT identity, from the toolchain PPA),
the CUDA 13.3 toolkit from NVIDIA's apt repo, the fork cloned from GitHub at the tag RunPod runs
plus `prepacked-parts.patch`, its venv built from source with `--no-build-isolation` (the
cuda-tile sdist needs the venv's torch), `s5cmd`, the contract server, the row, the cell, the
worker staged from `worker.ref`, and the systemd unit, enabled at boot. Docker, containerd,
soci-snapshotter, fabricmanager, the CDI refresh, the GPU settings service and snapd are
disabled: together they were most of the 77 s userspace boot and none serves one GPU. Nothing is pulled from a registry and no credential is
involved beyond the instance role. `bake.sh` runs `pack_env.sh` on the builder, wipes the
per-boot state and snapshots it into an AMI.

**Why the engine files leave the AMI.** A volume created from a snapshot lazy-loads every block
from S3 on first touch, and Python importing torch, sglang and flashinfer touches tens of
thousands of small files and their metadata. Measured 2026-09-18 on the first launch from an
AMI: 336 s from `sglang serve` to its first log line, the process in disk wait at 3 MiB/s with
IO pressure at 92%. Reading the files ahead in parallel made it worse (328 s, and the metadata
still cold). Fast Snapshot Restore fixes it for $0.75 an hour per AZ, so instead `pack_env.sh`
tars `/opt/venv`, `/opt/penny`, the CUDA toolkit, the uv python and the kernel caches (12 GB after the toolkit diet)
to `s3://<bucket>/env/llm-env-<image id>.parts/` in 1 GiB parts at bake, and the entrypoint
pulls the parts onto the NVMe with `s5cmd` (23 s down and untarred) and bind-mounts the
directories over their AMI paths. Same idea as
the RunPod engine tarball, for a different reason: there it dodged a billed image pull, here
it dodges a lazy EBS restore. The bake itself would pay the same lazy restore reading the
toolkit off a `--base` builder's root, so `pack_env.sh` stages the previous image's tarball on
the NVMe and rsyncs only what changed over it before tarring. It also copies every JIT-built
kernel into FlashInfer's AOT tree, which FlashInfer trusts without ninja: a rebuilt venv gives
the headers new mtimes and ninja would otherwise recompile every kernel (35 min) on the next
start.

**Weights** come from the prepacked pack in `s3://<bucket>/prepacked/` when the `complete`
marker is there, and from Hugging Face (`RadixArk/Qwen3.8-Flash-Next-NVFP4`, pinned by
revision, 126 GiB, ~95 s) otherwise. The pack is the fork's post-repack snapshot of the loaded
model: one flat file per model (target 121 GiB, draft 4 GiB) that `--load-format prepacked`
restores with sequential `preadv` instead of the safetensors load and its CPU repack (181 s
plus 54 s here on 8 vCPUs). It is keyed to the fork's load layout, so it is dumped by this exact
build and never copied from another. A box that finds no pack loads normally, dumps one to the
NVMe as a side effect (the loader does that on a miss), and once healthy splits each `.bin`
into 2 GiB parts under `<key>.parts/` and uploads them with the manifests and the marker. The
next box pulls the small files, then streams the parts in order through a tmpfs window of
three (`s5cmd` writes each object to a temp name and renames it whole), and the patched loader
(`prepacked-parts.patch`) reads each part as soon as it has its full size and deletes it once
every chunk in it is done (`SGLANG_PREPACKED_CONSUME=1`), so the pack goes NIC to RAM to card
and never touches the instance store. A restore chunk is per storage and not aligned to the
2 GiB parts, so the patch reads a chunk across a part boundary; the two boxes that missed that
died on a short read. A tmpfs frees a deleted file only when its last descriptor closes, and
the loader's worker threads kept one descriptor per part they had touched, so consumed parts
stayed resident (an 8 GiB tmpfs full with 4 GiB visible) until the copies died on ENOSPC;
each worker now closes the parts behind it, the window counts only landed `NNNN` names (an
in-flight temp name was counted twice, once as a file and once as a job), and a failed copy
removes its temp file and retries. The AMI holds no weights: instance store is never in a
snapshot, and EBS cannot serve them at speed.

What the stream replaced, measured 2026-09-18: downloading every part to the NVMe first ran
at ~3 GB/s alone but 1.2 GB/s once the restore read the same disk (the 1.9 GB/s instance
store took both the writes and the reads); left in page cache next to the 48 GiB pinned table
the pack thrashed the box (222 s restore); started before the engine tarball it starved that
download instead (tarball 46 s, untar 76 s); and mounting the bucket with mountpoint-s3 and
restoring straight off it ran at 400 MB/s with the FUSE daemon pinned on one core.

### Boot

`entrypoint.sh`, run by `gamesummoner-llm.service` at `multi-user.target`, in order.

1. **Env.** `/etc/gamesummoner/env.json` if it exists, else the instance user-data from IMDS,
   which is how the control plane delivers `CP_URL`, `WORKER_TOKEN`, `WORKER_ID`, `LLM_*`,
   `SGLANG_ARGS_EXTRA`, `WORKER_SLOTS` (`launch.sh` passes the env file as the user-data
   verbatim). User-data that is a script is a builder's provisioning, and the entrypoint runs
   it; anything else means not a worker launch, and it exits 0 without powering off. Metadata
   unreachable for 60 s means no network, and the box powers off. cloud-init is disabled in the
   image (it was 15 s of every boot and nothing here needs it), which means it no longer writes
   the netplan config either: the one it wrote on the builder matches the interface by MAC
   address, and a fresh instance has a new MAC, so a box booted that way has no network at all
   (2026-09-18 05:19, one box, no SSM, no worker, would have idled forever). `provision.sh`
   writes a netplan config that matches by interface name instead.
2. **Guards.** `LLM_MODEL` is `pennyroyal` and `LLM_N_CTX` at most 524288, the contracts the
   control plane's settings state; the instance store is mounted with room; a 64 GB swapfile
   goes on it.
3. **Engine files**: the tarball lives in S3 as 1 GiB parts (`env/llm-env-<image id>.parts/`,
   the count beside the key in `/opt/gamesummoner/env-tar`); every part is fetched at once and
   `tar` reads them through a pipe in order as they land, so the untar overlaps the download and
   the download runs as ~14 objects (one object tops out near 1 GB/s on this NIC, v11b: 16 s
   down then 16 s untar). Unpacked onto the NVMe and bind-mounted.
4. **Weights**: pack manifests and marker from S3, then the parts in the background and the
   checkpoint's config and tokenizer from Hugging Face; or the whole checkpoint when there is
   no pack.
5. **The engine**, with `pinned_max_round_threshold_mb:1024` in
   `PYTORCH_CUDA_ALLOC_CONF`. The engine's first act is one 47.68 GiB pinned host allocation
   for the PLE table (`--ple-offload-embedding`; the model does not fit a 96 GB card without
   it, and the table is already fp8). Torch's pinned caching allocator rounds every request up
   to the next power of two unless that threshold caps it, so the table asks the driver for
   64 GiB, which a 64 GiB box refuses with `cudaErrorMemoryAllocation` however much is free.
   Both EC2 boots of the RunPod-shaped container (2026-09-16, `labs/ec2-g7e` in the control
   plane repo) and the first boot of this box died there; with the threshold set, the exact
   allocation succeeds in 14 s and leaves 11 GiB free (measured 2026-09-17). RunPod's 188 GB
   cgroup never showed it because 64 GiB fit. On the Hugging Face path the page cache is
   dropped first (`drop_caches`, root on the host can, a container could not): a pin needs
   free pages and the download just filled the cache. On the pack path nothing is dropped: the
   only thing cached is the venv just unpacked, the imports want exactly that, and the pin
   comes after the imports and reclaims it then.

   The table leaves ~16 GiB of the host for everything else, and a fresh build's first
   FlashInfer autotune JIT-compiles kernels through nvcc: at 8 parallel jobs the kernel
   OOM-killed the scheduler mid-autotune (2026-09-17, second boot). So the swapfile (pinned
   memory cannot swap, compile jobs and everything pageable can) and a JIT cap of 4 jobs. The
   one-time JIT took 2071 s at that cap; with the caches in the tarball the autotune reads them
   in 5 s.
6. **Engine** through `sglang serve` with the fork's Flash-Next launch line
   (`configs/pennyroyal/serve-flash-next.sh`), minus HiCache and NIXL, plus
   `--load-format prepacked`, `--max-running-requests` and `--cuda-graph-bs` from
   `WORKER_SLOTS`, plus `SGLANG_ARGS_EXTRA`. The engine runs in its own session so its
   process-tree kill on a crash cannot take the entrypoint with it.
7. **Contract server** (`python -m aws.llm.server`) and **worker**, side by side, both on the
   venv's python. The engine binds `127.0.0.1` and the security group has no inbound rule. A
   box that dumped a pack seeds the bucket in the background once it is healthy.

Exit: `/shutdown` SIGTERMs the entrypoint, which drains the worker for up to 60 s, ends the
engine's process group and the contract server, and exits 0; a box process dying on its own
exits 1. Either way the entrypoint powers the instance off (`GS_NO_POWEROFF=1` for a hand run
on a builder), and the instance was launched with `InstanceInitiatedShutdownBehavior=terminate`,
so it ends without an AWS credential on the box. EC2 does not bill the `shutting-down` state.

### Where the time goes

Measured launches from an AMI (launch = EC2 `running`), g7e.2xlarge, eu-north-1a:

| launch, 2026-09-18 | to unit | engine files | weights | engine load | autotune + graphs | serving |
|---|---|---|---|---|---|---|
| 01:01, first AMI, safetensors from HF, EBS venv | 97 s | 336 s (EBS lazy-load) | 110 s pull, 181 + 54 s load | | 15 s | ~14 min |
| 04:06, v3: tarball, pack parts, IMDS env | 59 s | 22 s | 40 s pull, 222 s restore (thrash) | 29 s imports + init | 28 s | 425 s |
| 04:41, v4: cache dropped during the fetch | 56 s | 27 s | 101 s pull (NVMe-bound), 69 s restore | 31 s | 24 s | 213 s |
| 05:54, v6: cloud-init off, parts fetched first | 33 s | 122 s (starved) | 57 s restore | 29 s | 24 s | ~267 s |
| 06:40, v7: parts streamed through tmpfs | 21 s | 5 s | window race: every copy ENOSPC | | | never |
| 07:09, v8: window counted jobs | | 13 s | fds on deleted parts filled the tmpfs, ENOSPC | 35 s | | never |
| 07:37, v9: workers close parts behind them | 37 s | 15 + 12 s | 73 s restore off the stream | 22 s imports, 18 s init + pin | 17 s | 199 s |
| 08:12, v10: idle units off, 48 ranges of 32 MiB per part | | 333 s (`s5cmd cat` piped into tar) | 59 s restore | 22 s, 17 s | 17 s | ~510 s |
| 09:16, v11b: v10 stream settings, tarball copied again | 34 s | 16 + 16 s | 66 s restore | 27 s, 16 s | 15 s | ~191 s |
| 09:48, v11c: 1 GiB parts, window of six | 36 s | 17 + 12 s | 46 s restore (2.85 GB/s) | 37 s (26 s before the first engine line), 16 s | 15 s | ~184 s |
| 10:20, v12: tarball parts, driver libs pre-read | | 108 s (pre-read saturated the root volume) | 46 s restore | 26 s + 7 s, 16 s | 15 s | ~270 s |
| 11:00, v13: persistence mode on at boot | 36 s | 23 s (parts, piped) | 56 s restore (8 ENOSPC retries) | 5 s + 9 s, 16 s | 15 s | ~170 s |
| 11:39, v14: workers close parts before waiting, 32 MiB ranges | 36 s | 23 s | 40 s restore, no retries | 5 s + 8 s, 16 s | 15 s | **153 s** |

v9 from the box's own clocks: kernel at +15 s, unit at +37 s (kernel 6 s, userspace 18 s:
modules-load 5.5 s for the EFA stack, hibinit-agent 4.7 s, apport 4.2 s, networkd-dispatcher
3.5 s, dlami-nvme 3.2 s), tarball +52 s, engine start +69 s, weight load +91 s, restore +109 s
to +182 s at 1.66 GB/s with the download finishing one second after it, worker pulling +199 s.
Cuts taken after it: the idle DLAMI units and the EFA modules disabled, the tarball piped into
`tar` instead of landing on the NVMe first, and the page cache no longer dropped before the
engine starts (that evicted the venv just unpacked; it was for the NVMe pack path).

The 1.66 GB/s is the download, measured on a builder at 07:56 with nothing consuming: one
2 GiB object at a time runs 1.0 GB/s with 16 ranges of 64 MiB, 1.3 at 32, 1.65 at 64 of
32 MiB; three objects in flight 3.0 GB/s, six 3.65 GB/s. It is the count of ranges in flight
that matters, ~100 for 3 GB/s, and the window of three with one part being consumed keeps only
two objects, 32 ranges, in flight. Reading the same parts off the tmpfs into pinned chunks and
onto the card ran 8 GB/s with 8 threads, so the restore is nowhere near a limit. The window
cannot widen in bytes (the pinned table owns the RAM), so v10 tried 48 ranges of 32 MiB per
part: 59 s, ~2.1 GB/s, two objects still the cap. The parts are 1 GiB since 08:40 (re-split in
place on a builder), so the same 7 GiB window holds six of them and keeps five objects in
flight. env.json can override `PACK_WINDOW`, `PACK_CONCURRENCY`, `PACK_RANGE_MB` and
`PACK_THREADS` (defaults 6, 16, 64, 8). v10 also showed that `s5cmd cat` piped into `tar`
takes 333 s for what `cp` then `tar` do in 27 s, and that the SSM agent is a snap whose
AppArmor profile `snapd.apparmor.service` loads, so that unit stays enabled.

With the 1 GiB parts (v11c) the restore runs 46 s and the stream is no longer the long pole.
What moves now is the engine's silent start, from `sglang serve` to its first log line: 13 s
on v9 and v10, 20 s on v11b, 26 s on v11c. The venv is on the NVMe, so the suspect is the root
volume, an EBS volume lazily populated from the AMI snapshot on first read: the NVIDIA driver
libraries (`libcuda`, `libnvidia-*`) live there and CUDA init reads them. v12 tested that by
reading them all in the background at boot: 104 s for ~700 MB, so the root volume is that
slow, but it saturated the volume and stalled the tarball step to +108 s, and the engine's
silent start stayed 26 s with the libraries warm. The engine maps only `libcuda` and
`libnvidia-ml`. Warm on the box, `import torch` is 4.7 s and the server module 7-10 s of one
core, so about 15 s of the 26 are elsewhere, and the box runs with GPU persistence mode off:
the first CUDA context after boot initialises the card (a 96 GB Blackwell scrubs its memory)
inside that silent start. v13 turns persistence on in the background as the entrypoint's
first act, so the card initialises under the tarball download: 19 s, and the engine's first
line came 5 s after its start instead of 26. v13 also exposed a second descriptor leak: a
worker that finished a 1 GiB part and then waited for a part not yet landed still held the
finished one open, so its space never came back and the copies hit ENOSPC until the worker
moved on (the retries cost the restore ~10 s). Workers now close what is behind them before
they wait. The copies also buffer `concurrency x range` each, six of them, so the default
range is 32 MiB (3 GiB of buffers, not 6; the box was 950 MiB from empty with 2 GiB swapped).

Where v14's 153 s go, from EC2 `running`: 13 s until the kernel starts, 23 s of kernel and
userspace to the unit, 23 s tarball, 5 s of small files, 13 s of engine imports and init,
16 s pinning the 48 GiB table, 40 s restore (3 GB/s, the download), 4 s draft and MTP, 9 s
autotune and graph capture, 7 s until the worker's next probe. What is left to take is small
and spread out: the boot could lose a few seconds more of DLAMI units, the pin is the
driver's, the restore is at the NIC's practical rate for six objects, and the tail is a probe
interval. A wider box (g7e.4xlarge: 16 vCPU, 128 GiB) would allow a window of thirty parts and
a faster pin; the quota request is filed.

### Capacity and quota

Both G-family quotas are in vCPUs, per region: 8 on-demand and 8 spot in us-east-1 and
eu-north-1, which is one g7e.2xlarge at a time. us-east-1 has refused every g7e launch
(`InsufficientInstanceCapacity`); eu-north-1a took one on 2026-09-16 at $1.21 spot. The scripts
walk every zone that offers the type and take the first that answers. A terminated spot
instance's one-time request can hold the quota until it is cancelled
(`MaxSpotInstanceCountExceeded` with nothing running means that, not a missing grant).

### Build, test, bake

The scripts in `aws/` serve every box and take its queue first.

```
aws/build.sh llm [--base <ami>]      # builder up and provisioned (~5 min from the DLAMI, ~3 from --base)
aws/ssm.sh <instance> "echo $(base64 -w0 env.json) | base64 -d > /etc/gamesummoner/env.json; \
    GS_NO_POWEROFF=1 nohup setsid /opt/gamesummoner/entrypoint.sh > /var/log/gamesummoner-llm.log 2>&1 < /dev/null &"
aws/ssm.sh <instance> 'grep "^\[boot" /var/log/gamesummoner-llm.log'
aws/bake.sh llm <instance>           # once a job has gone through: the tarball to S3, then the AMI
aws/launch.sh llm <ami> env.json     # a box the way the scaler will start one
```

On the builder the entrypoint is started by hand with `GS_NO_POWEROFF=1`, because the builder
has to survive its own decommission to be baked. The first engine start on a fresh build pays
the FlashInfer autotune and JIT; every later start reads the caches. A builder from `--base`
already has them.

`env.json` follows `env.example.json`; the worker id must exist in the control plane's workers
table and `CP_URL` must be reachable from EC2 (a Tailscale funnel for a local control plane).
Queue a job before the box is healthy or set `IDLE_EXIT_SECONDS` past the boot: the worker's
idle clock starts when it starts, not when the image turns healthy, so a box that boots into an
empty queue with a short window exits on the spot (the scaler only launches on pending work, so
production never sees this).
Standing infrastructure, per region: bucket `gamesummoner-models-<region>-<account>`
(pack under `prepacked/`, the engine parts under `env/`), security group `gs-gpu-worker` (no
inbound) in the default VPC, an S3 gateway endpoint on its route tables. Global: role and
instance profile `gs-gpu-worker` (S3 on every `gamesummoner-models-*` bucket, SSM). A box
knows its region from IMDS and its account from `/opt/gamesummoner/account` (written at
provision), so one AMI copied with `copy-image` serves any region whose bucket holds the
pack and its engine parts; nothing in the image names a region. Regions so far: eu-north-1
(`vpce-006e37cedf2225ed7`), us-west-2 (`vpce-0bb567546d88e900f`), us-east-1
(`vpce-09883550ad82c6d4f`, decide only so far). The scaler launches
these boxes through its EC2 provider (`src/scaler/ec2_client.py` in the control plane repo):
the env as user-data, by zone in the default VPC, the group and profile by name.

Second region proved 2026-09-18: v15 (`ami-0116c8c4637c00e43`, copied to us-west-2 as
`ami-0fe3c370e158967ca`) served a job from us-west-2c with no change but the region. What it
cost to get there: pack and engine parts copied server-side in 8 min (the CLI's default ten
8 MB part copies ran at 20 MB/s; 128 x 64 MB ran at ~300 MB/s), `copy-image` 28 min for the
100 GB snapshot. A bake writes a new engine tarball under its own image id, so the tarball
to copy is the one the copied AMI names in `/opt/gamesummoner/env-tar`, not the previous
image's: the first us-west-2 box looped on `NoSuchKey` for 12 min until its parts arrived.
Capacity: every us-west-2 zone refused g7e.2xlarge and g7e.4xlarge, spot and on-demand, from
16:14 to 16:50 UTC (spot placement score 1 of 10, the same score eu-north-1 and us-east-1
showed at that hour); us-west-2c then took a spot request. The pack restored at 1.6 GiB/s
there (75.8 s against 40 s in eu-north-1) and the GPU init took 31 s against 19 s. A clean
second launch two hours later (us-west-2c, spot after 5 min of refusals): job picked up 204 s
after launch, engine ready at 212 s, no errors. Tarball 30 s, the same as eu-north-1; the
pack streamed in 98 s (~1.3 GB/s against ~3 GB/s) and restored in 61.4 s, so the whole gap
to 153 s is S3-to-box throughput on the pack. Two runs, one zone, a bucket hours old: a new
bucket's prefixes scale up with use, so this may close on its own; if it does not, the knobs
are `PACK_CONCURRENCY` and `PACK_WINDOW`.

Region choice, 2026-09-18: nine regions offer g7e, two AZs each (us-west-2 four). Seven-day
spot averages for g7e.2xlarge: eu-north-1a $1.16, us-west-2a $1.25, us-east-1d $1.58,
ap-south-1b $1.66; the rest $1.9-2.8 and with a 5-vCPU G spot quota, below one box. The
cheap zone of a region is the contended one (us-west-2a $1.25 against 2.4-3.4 in b, c, d),
so a launch walks the zones of a region before the next region. Cross-region copy of the
pack and engine parts is ~146 GB, about $3 in transfer, and a second bucket costs ~$3.40 a
month at rest.

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
`env/decide-env-<image id>.parts/`, and the entrypoint streams them back onto the NVMe the way the
llm box does. Weights live in the bucket under `decide/<repo>@<revision>/`, plain files the
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
