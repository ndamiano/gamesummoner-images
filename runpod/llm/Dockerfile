# RunPod llm box: Pennyroyal SGLang for Qwen3.8 Flash-Next on an RTX PRO 6000. See runpod/README.md.
FROM nvidia/cuda:13.3.1-cudnn-runtime-ubuntu26.04 AS builder

ARG PENNY_REPO=https://github.com/ndamiano/sglang-rtxpro6000
ARG PENNY_REF=pennyroyal-v2.1.0-prepacked1
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git python3-pip gcc-15 g++-15 libnuma-dev ninja-build cmake \
        cuda-nvcc-13-3 cuda-cudart-dev-13-3 cuda-cccl-13-3 cuda-nvrtc-dev-13-3 \
        cuda-nvml-dev-13-3 libcusparse-dev-13-3 libcublas-dev-13-3 libcusolver-dev-13-3 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/local/cuda/lib64/*_static*.a /usr/local/cuda/lib64/libcublas*static*.a

RUN pip install --break-system-packages -q uv
RUN git clone --depth 1 -b "$PENNY_REF" "$PENNY_REPO" /opt/penny

ENV CUDA_HOME=/usr/local/cuda CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15 \
    PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN uv python install 3.12.13 && uv venv --python 3.12.13 /opt/venv
RUN . /opt/venv/bin/activate && uv pip install setuptools wheel packaging wheel_stub \
    && cd /opt/penny && uv pip install --prerelease=allow --index-strategy unsafe-best-match \
        --extra-index-url https://docs.sglang.ai/whl/cu130/ --no-build-isolation -e python \
    && uv pip install nixl hf_transfer huggingface_hub \
    && uv cache clean && rm -rf /root/.cache/pip /root/.cache/uv

ADD runpod/llm/penny_cache_v6.tgz /root/

# Promote the JIT-compiled ops to AOT artifacts (README: "kernel caches").
RUN set -e; AOT=/opt/venv/lib/python3.12/site-packages/flashinfer/data/aot; \
    for so in /root/cache/flashinfer/.cache/flashinfer/*/*/cached_ops/*/*.so; do \
        op=$(basename "$so" .so); mkdir -p "$AOT/$op"; cp "$so" "$AOT/$op/$op.so"; \
    done; ls "$AOT"

RUN /opt/venv/bin/python -m compileall -q /opt/venv/lib/python3.12/site-packages /opt/penny/python -j 8 || true

RUN rm -rf /opt/venv/lib/python3.12/site-packages/nixl_cu12.libs \
           /opt/venv/lib/python3.12/site-packages/nixl_ep_cu12 \
    && tar -cf /env.tar /opt/venv /opt/penny /root/cache /root/.local/share/uv \
           /root/serve-nohicache.sh /root/nixl-posix.toml \
    && sha256sum /env.tar | cut -c1-12 > /env.id

FROM scratch AS env
COPY --from=builder /env.tar /env.id /

FROM nvidia/cuda:13.3.1-cudnn-runtime-ubuntu26.04
ARG DEBIAN_FRONTEND=noninteractive
# The JIT toolchain only; CUDA libraries become symlinks into the venv the entrypoint restores.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git gcc-15 g++-15 libnuma-dev ninja-build cmake \
        cuda-nvcc-13-3 cuda-cudart-dev-13-3 cuda-cccl-13-3 cuda-nvrtc-dev-13-3 \
        cuda-nvml-dev-13-3 libcusparse-dev-13-3 libcublas-dev-13-3 libcusolver-dev-13-3 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/local/cuda/lib64/*_static*.a /usr/local/cuda/lib64/libcublas*static*.a \
    && rm -f /usr/lib/x86_64-linux-gnu/libcudnn* /usr/lib/x86_64-linux-gnu/libnccl* \
    && rm -f /usr/local/cuda/lib64/libnpp* /usr/local/cuda/lib64/libcufft* \
          /usr/local/cuda/lib64/libcurand* /usr/local/cuda/lib64/libnvjpeg* \
          /usr/local/cuda/lib64/libcusolverMg* \
    && cd /usr/local/cuda/lib64 \
    && SP=/opt/venv/lib/python3.12/site-packages \
    && for pair in libcublas.so.13 libcublasLt.so.13 libcusparse.so.12 libcusolver.so.12 \
                   libnvJitLink.so.13 libnvrtc.so.13; do \
        rm -f "$pair".* "$pair"; ln -s "$SP/nvidia/cu13/lib/$pair" "$pair"; \
    done \
    && rm -f libnvrtc-builtins.so* \
    && ln -s "$SP/nvidia/cu13/lib/libnvrtc-builtins.so.13.0" libnvrtc-builtins.so.13.0

ENV CUDA_HOME=/usr/local/cuda CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15 \
    PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ARG IMAGE_ID
COPY --from=builder /env.id /opt/gamesummoner/env.id
COPY contract /opt/gamesummoner/contract
COPY runpod/server.py /opt/gamesummoner/runpod/server.py
COPY runpod/llm/server.py /opt/gamesummoner/runpod/llm/server.py
COPY runpod/llm/worker /opt/gamesummoner/worker
COPY runpod/llm/entrypoint.sh /opt/gamesummoner/
RUN test -n "$IMAGE_ID" && echo "$IMAGE_ID" > /opt/gamesummoner/image-id \
    && chmod +x /opt/gamesummoner/entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    SGLANG_PORT=8001 \
    PENNY_MODEL_DIR=/workspace/models/pennyroyal \
    PENNY_ENV_DIR=/workspace/env \
    TORCHINDUCTOR_CACHE_DIR=/root/cache/torchinductor \
    TRITON_CACHE_DIR=/root/cache/triton

ENTRYPOINT ["/opt/gamesummoner/entrypoint.sh"]
