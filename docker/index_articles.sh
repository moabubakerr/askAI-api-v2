#!/bin/sh
# Chunks and embeds the SCAI articles. Run AFTER run-etl, and only once the
# embedding server is reachable — unlike the data load, this step needs TEI.
set -eu
if [ -z "${ETL_DSN:-}" ]; then
    echo "index-articles: ETL_DSN is not set (needs a WRITE-capable role)." >&2
    exit 2
fi
exec python scripts/index_articles.py "$@"
