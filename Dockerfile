
# SCAI Economic Data Assistant — API image.
#
# One image serves two roles:
#   - default CMD: the FastAPI app (uvicorn)
#   - `run-etl`:   the one-shot CSV loader (same deps, same code, no second image)
#
# The image contains NO model weights and NO data. It talks to vLLM and TEI over
# the Docker network and to Postgres over the compose network; the CSVs are
# mounted, not baked in, so re-loading data never requires a rebuild.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# curl: container healthcheck. postgresql-client: lets you re-apply etl/schema.sql
# by hand without a psql install on the VM host.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl postgresql-client \
 && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.runtime.txt ./requirements.runtime.txt
RUN pip install -r requirements.runtime.txt

COPY app ./app
COPY etl ./etl
COPY scripts ./scripts
COPY docker/run_etl.sh /usr/local/bin/run-etl

RUN chmod +x /usr/local/bin/run-etl \
 && useradd --create-home --uid 10001 scai

USER scai

EXPOSE 8000

# /health is a static dict — it comes up green before Postgres or the models are
# reachable. That is deliberate: it proves the process is alive, not that the
# stack is wired. Use the smoke-test curl in the README for the latter.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
