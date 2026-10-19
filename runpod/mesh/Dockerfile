# RunPod mesh box: TRELLIS.2. See runpod/README.md.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    MAX_JOBS=4

ARG TRELLIS_SHA=75fbf0183001ed9876c8dbb35de6b68552ee08bd
COPY engines/trellis/trellis2-sdpa-dinov3.patch /tmp/trellis2.patch
RUN git init /opt/TRELLIS2 \
    && cd /opt/TRELLIS2 \
    && git remote add origin https://github.com/microsoft/TRELLIS.2 \
    && git fetch --depth 1 origin ${TRELLIS_SHA} \
    && git checkout FETCH_HEAD \
    && git submodule update --init --depth 1 o-voxel/third_party/eigen \
    && git apply /tmp/trellis2.patch

RUN pip install --no-cache-dir ninja \
    && mkdir /wheels /ext \
    && git clone -b v0.4.0 --depth 1 https://github.com/NVlabs/nvdiffrast.git /ext/nvdiffrast \
    && git clone -b renderutils --depth 1 https://github.com/JeffreyXiang/nvdiffrec.git /ext/nvdiffrec \
    && git clone --recursive --depth 1 https://github.com/JeffreyXiang/CuMesh.git /ext/CuMesh \
    && git clone --recursive --depth 1 https://github.com/JeffreyXiang/FlexGEMM.git /ext/FlexGEMM \
    && pip wheel --no-build-isolation --no-deps -w /wheels \
        /ext/nvdiffrast /ext/nvdiffrec /ext/CuMesh /ext/FlexGEMM /opt/TRELLIS2/o-voxel \
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"


FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends curl gcc \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY --from=builder /opt/TRELLIS2 /opt/TRELLIS2
COPY --from=builder /wheels /wheels

RUN pip install --no-cache-dir --no-deps /wheels/*.whl \
    && pip install --no-cache-dir \
        imageio imageio-ffmpeg tqdm easydict opencv-python-headless trimesh \
        transformers==5.13.0 timm==1.0.27 einops==0.8.2 kornia==0.8.3 \
        pandas lpips zstandard pillow fastapi uvicorn requests \
        plyfile scipy moderngl \
    && rm -rf /wheels

ARG IMAGE_ID
COPY contract /opt/gamesummoner/contract
COPY engines /opt/gamesummoner/engines
COPY runpod/server.py /opt/gamesummoner/runpod/server.py
COPY runpod/mesh/server.py /opt/gamesummoner/runpod/mesh/server.py
COPY runpod/mesh/worker /opt/gamesummoner/worker
COPY runpod/mesh/entrypoint.sh /opt/gamesummoner/
RUN test -n "$IMAGE_ID" && echo "$IMAGE_ID" > /opt/gamesummoner/image-id \
    && chmod +x /opt/gamesummoner/entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    TRELLIS_PORT=8189 \
    TRELLIS_REPO=/opt/TRELLIS2 \
    TRELLIS_WEIGHTS=/workspace/trellis2-weights \
    HF_HOME=/workspace/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TRITON_CACHE_DIR=/workspace/triton-cache \
    FLEX_GEMM_AUTOTUNE_CACHE_PATH=/workspace/flex-gemm-autotune.json

# Build-time smoke: the interpreter imports what the server needs.
RUN python -c "import torch, triton"

ENTRYPOINT ["/opt/gamesummoner/entrypoint.sh"]
