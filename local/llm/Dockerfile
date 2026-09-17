# Local llm box: stock SGLang plus the contract server and the worker at local/llm/worker.ref
# (staged by build.sh). Weights are pulled at boot.
FROM lmsysorg/sglang:v0.5.19-cu130-runtime

ARG IMAGE_ID

COPY contract /opt/gamesummoner/contract
COPY local/server.py /opt/gamesummoner/local/server.py
COPY local/llm/server.py /opt/gamesummoner/local/llm/server.py
COPY local/llm/worker /opt/gamesummoner/worker
COPY local/llm/entrypoint.sh /opt/gamesummoner/
RUN test -n "$IMAGE_ID" && echo "$IMAGE_ID" > /opt/gamesummoner/image-id \
    && chmod +x /opt/gamesummoner/entrypoint.sh \
    && mkdir -p /cache/home && chmod 1777 /cache/home

ENTRYPOINT ["/opt/gamesummoner/entrypoint.sh"]
