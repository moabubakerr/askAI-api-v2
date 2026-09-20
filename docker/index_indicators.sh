#!/bin/sh
# Embeds the indicator catalogue into Postgres so a restart does not re-embed
# 1,288 texts before answering. Run AFTER run-etl, and only once the embedding
# server is reachable.
set -eu
if [ -z "${ETL_DSN:-}" ]; then
    echo "index-indicators: ETL_DSN is not set (needs a WRITE-capable role)." >&2
    exit 2
fi
exec python scripts/index_indicators.py "$@"
