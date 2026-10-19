# RunPod ComfyUI box (--build-arg QUEUE=comfy). See runpod/README.md.
FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1

ARG COMFY_SHA=3c1b7a17fdf239d35ce98cc777756f572d089795
RUN git init /opt/ComfyUI \
    && cd /opt/ComfyUI \
    && git remote add origin https://github.com/comfyanonymous/ComfyUI \
    && git fetch --depth 1 origin ${COMFY_SHA} \
    && git checkout FETCH_HEAD
RUN pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt

ARG RMBG_SHA=7f0668e2f5127c0fbf11e346b33ce45d6dc34bf7
RUN git init /opt/ComfyUI/custom_nodes/ComfyUI-RMBG \
    && cd /opt/ComfyUI/custom_nodes/ComfyUI-RMBG \
    && git remote add origin https://github.com/1038lab/ComfyUI-RMBG \
    && git fetch --depth 1 origin ${RMBG_SHA} \
    && git checkout FETCH_HEAD \
    && pip install --no-cache-dir opencv-python-headless "huggingface-hub>=0.19.0" timm

RUN rm -rf /opt/ComfyUI/models && ln -s /opt/comfy-models /opt/ComfyUI/models
RUN pip install --no-cache-dir requests

ARG QUEUE
ARG IMAGE_ID
COPY contract /opt/gamesummoner/contract
COPY runpod/server.py /opt/gamesummoner/runpod/server.py
COPY runpod/${QUEUE}/server.py /opt/gamesummoner/runpod/${QUEUE}/server.py
COPY runpod/${QUEUE}/worker /opt/gamesummoner/worker
COPY runpod/comfy/entrypoint.sh runpod/comfy/stage.py /opt/gamesummoner/
RUN test -n "$QUEUE" && test -n "$IMAGE_ID" \
    && echo "$IMAGE_ID" > /opt/gamesummoner/image-id \
    && chmod +x /opt/gamesummoner/entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    QUEUE=${QUEUE} \
    COMFY_PORT=8188 \
    SAFETY_MODEL_DIR=/opt/comfy-models/safety

ENTRYPOINT ["/opt/gamesummoner/entrypoint.sh"]
