# Local decide box: stock SGLang plus the JEV prep, the contract server and the worker at
# local/decide/worker.ref (staged by build.sh). Weights are pulled at boot.
FROM lmsysorg/sglang:v0.5.21-cu130-runtime

ARG IMAGE_ID

COPY contract /opt/gamesummoner/contract
COPY engines/jev/prep.py /opt/gamesummoner/engines/jev/prep.py
COPY local/server.py /opt/gamesummoner/local/server.py
COPY local/decide/server.py /opt/gamesummoner/local/decide/server.py
COPY local/decide/worker /opt/gamesummoner/worker
COPY local/decide/entrypoint.sh /opt/gamesummoner/
RUN test -n "$IMAGE_ID" && echo "$IMAGE_ID" > /opt/gamesummoner/image-id \
    && chmod +x /opt/gamesummoner/entrypoint.sh \
    && mkdir -p /cache/home && chmod 1777 /cache/home

ENTRYPOINT ["/opt/gamesummoner/entrypoint.sh"]
