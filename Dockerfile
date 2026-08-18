FROM cgr.dev/chainguard/python:latest-dev@sha256:4c58eb78b47d5a7cee9ffe0d1df16c56778ee40b067d983fb48d677a035abf8c AS python

USER 0
RUN uv python install --install-dir /opt/python --no-bin --no-progress 3.12.13 \
    && rm -rf \
        /opt/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/site-packages/msgpack* \
        /opt/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/site-packages/pip* \
        /opt/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/site-packages/setuptools*

FROM cgr.dev/chainguard/wolfi-base:latest@sha256:30f03343947c7ae3581fda727a6e2aa7b8ce7009b7bfc2ab8d5c9483ace5812f

ARG BUILD_DATE
ARG VERSION=dev
ARG VCS_REF

LABEL org.opencontainers.image.created=$BUILD_DATE \
    org.opencontainers.image.description="PaddleOCR service for RecipeMonster" \
    org.opencontainers.image.licenses="Apache-2.0" \
    org.opencontainers.image.revision=$VCS_REF \
    org.opencontainers.image.source="https://github.com/recipemonster/recipemonster-ocr" \
    org.opencontainers.image.title="RecipeMonster OCR" \
    org.opencontainers.image.version=$VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/models/home \
    PATH=/opt/python/cpython-3.12.13-linux-x86_64-gnu/bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/usr/sbin:/sbin:/bin \
    PADDLE_OCR_BASE_DIR=/models/paddleocr \
    PADDLE_PDX_CACHE_HOME=/models/paddlex \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    XDG_CACHE_HOME=/models/cache

RUN apk update \
    && apk add --no-cache glib libglvnd libgomp libstdc++

COPY --from=python /opt/python /opt/python
COPY --from=python /usr/bin/uv /usr/bin/uv

WORKDIR /app

COPY requirements-base.txt requirements-cpu.txt requirements-gpu.txt ./
COPY bootstrap.py license_inventory.py main.py ./
COPY api ./api
COPY models ./models
COPY utils ./utils
COPY LICENSE /usr/share/licenses/recipemonster-ocr/LICENSE

RUN python -m py_compile bootstrap.py main.py api/*.py models/*.py utils/*.py \
    && python license_inventory.py \
        --spdx-directory /var/lib/db/sbom \
        --output /usr/share/licenses/recipemonster-ocr \
    && mkdir -p /models \
    && chown -R 65532:65532 /app /models

USER 65532:65532

VOLUME ["/models"]

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=20m --retries=80 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"]

ENTRYPOINT ["python"]
CMD ["bootstrap.py"]
