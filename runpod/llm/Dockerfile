# RunPod llm box: Basalt for Qwen3.8 Flash-Next on an RTX PRO 6000. See runpod/README.md.
FROM nvidia/cuda:13.3.1-devel-ubuntu24.04 AS builder

ARG BASALT_REPO=https://github.com/ndamiano/basalt
ARG BASALT_REF=gamesummoner-v1
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git build-essential cmake python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 -b "$BASALT_REF" "$BASALT_REPO" /opt/basalt \
    && cmake -S /opt/basalt -B /opt/basalt/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/basalt/build -j"$(nproc)" \
    && rm -rf /opt/basalt/build /opt/basalt/.git
RUN python3 -m venv /opt/basalt/.venv \
    && /opt/basalt/.venv/bin/pip install -q -r /opt/basalt/requirements.txt "requests>=2.31" \
    && rm -rf /root/.cache/pip

FROM nvidia/cuda:13.3.1-runtime-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl python3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/basalt /opt/basalt

ARG IMAGE_ID
COPY contract /opt/gamesummoner/contract
COPY runpod/server.py /opt/gamesummoner/runpod/server.py
COPY runpod/llm/server.py /opt/gamesummoner/runpod/llm/server.py
COPY runpod/llm/worker /opt/gamesummoner/worker
COPY runpod/llm/entrypoint.sh /opt/gamesummoner/
RUN test -n "$IMAGE_ID" && echo "$IMAGE_ID" > /opt/gamesummoner/image-id \
    && chmod +x /opt/gamesummoner/entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    BASALT_PACK=/workspace/models/basalt/Qwen3.8-FN-UD-Q4_K_XL-bf16key-mtptuned.basalt

ENTRYPOINT ["/opt/gamesummoner/entrypoint.sh"]
