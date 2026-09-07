ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONPATH=/app

WORKDIR /app

# This image is the lightweight CPAM-LLM web/runtime container. Fine-tuned LLM
# servers and IBM CP Optimizer are external dependencies; see docs/.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

RUN useradd --create-home --shell /usr/sbin/nologin --uid 1000 appuser

COPY --chown=appuser:appuser . .

RUN mkdir -p output kb_store \
    && chown -R appuser:appuser output kb_store
USER appuser

# Reuse the prebuilt RAG-FCA knowledge base tracked in the repository. The
# command exits without reading a source workbook when the JSON store exists,
# so a clean clone does not need the optional training/source spreadsheet.
RUN python build_kb.py --no-review

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/api/kbinfo', timeout=4).status == 200 else 1)" || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "webapp.app:app"]
