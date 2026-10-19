# Local decide box: stock vLLM plus the contract server and the worker at local/decide/worker.ref
# (staged by build.sh). Weights are pulled at boot.
FROM vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34

ARG IMAGE_ID

COPY contract /opt/gamesummoner/contract
COPY engines/jev/head /opt/gamesummoner/engines/jev/head
COPY local/server.py /opt/gamesummoner/local/server.py
COPY local/decide/server.py /opt/gamesummoner/local/decide/server.py
COPY local/decide/worker /opt/gamesummoner/worker
COPY local/decide/entrypoint.sh /opt/gamesummoner/
RUN test -n "$IMAGE_ID" && echo "$IMAGE_ID" > /opt/gamesummoner/image-id \
    && chmod +x /opt/gamesummoner/entrypoint.sh \
    && mkdir -p /cache/home && chmod 1777 /cache/home

ENTRYPOINT ["/opt/gamesummoner/entrypoint.sh"]
