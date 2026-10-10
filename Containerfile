FROM docker.io/library/python:3.12.14-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

LABEL org.opencontainers.image.source="https://github.com/SecAI-Hub/ai-quarantine" \
      org.opencontainers.image.description="AI artifact admission-control core"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/ai-quarantine

COPY requirements.lock /opt/ai-quarantine/requirements.lock

RUN python -m pip install --require-hashes -r requirements.lock && \
    install -d -m 0700 -o 65532 -g 65532 /quarantine /var/lib/secure-ai/logs && \
    install -d -m 0750 -o 65532 -g 65532 /registry

COPY --chown=0:0 quarantine/ /opt/ai-quarantine/quarantine/

# External scanners are intentionally not installed with mutable, best-effort
# commands. Production images must add reviewed scanners at pinned versions and
# keep scanner_missing=fail-closed.
RUN python -m compileall -q /opt/ai-quarantine/quarantine && \
    find /opt/ai-quarantine -type d -exec chmod 0555 {} + && \
    find /opt/ai-quarantine -type f -exec chmod 0444 {} +

USER 65532:65532
ENTRYPOINT ["python", "-I", "-c", "import sys; sys.path.insert(0, '/opt/ai-quarantine'); from quarantine.watcher import main; main()"]
