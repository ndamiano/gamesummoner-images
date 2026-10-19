# RunPod decide box: stock SGLang plus the JEV prep, the contract server and the worker at
# runpod/decide/worker.ref (staged by build.sh). Weights come off the art volume. See runpod/README.md.
FROM lmsysorg/sglang:v0.5.21-cu130-runtime

ARG IMAGE_ID
COPY contract /opt/gamesummoner/contract
COPY engines/jev/prep.py /opt/gamesummoner/engines/jev/prep.py
COPY runpod/server.py /opt/gamesummoner/runpod/server.py
COPY runpod/decide/server.py /opt/gamesummoner/runpod/decide/server.py
COPY runpod/decide/worker /opt/gamesummoner/worker
COPY runpod/decide/entrypoint.sh /opt/gamesummoner/
RUN test -n "$IMAGE_ID" && echo "$IMAGE_ID" > /opt/gamesummoner/image-id \
    && chmod +x /opt/gamesummoner/entrypoint.sh

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/opt/gamesummoner/entrypoint.sh"]
