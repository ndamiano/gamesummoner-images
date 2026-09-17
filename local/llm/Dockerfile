# Local llm box: stock SGLang plus the contract server. Weights and the worker are pulled at boot.
FROM lmsysorg/sglang:v0.5.19-cu130-runtime

ARG IMAGE_ID
ENV WORKER_REPO=/src/gamesummoner-workers

COPY contract /opt/gamesummoner/contract
COPY local/llm/entrypoint.sh local/llm/worker.ref /opt/gamesummoner/
RUN test -n "$IMAGE_ID" && echo "$IMAGE_ID" > /opt/gamesummoner/image-id \
    && chmod +x /opt/gamesummoner/entrypoint.sh \
    && mkdir -p /cache/home && chmod 1777 /cache/home

ENTRYPOINT ["/opt/gamesummoner/entrypoint.sh"]
