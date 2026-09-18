#!/bin/sh
# One-shot loader: stage the CSVs into a single flat directory, then run the ETL.
#
# Why staging is needed at all — etl/load_data.py reads every file from ONE
# --csv-dir, but the repo ships them across two directories:
#     data/          Articles.csv, Champions.csv, Sectors.csv, "General Entities.csv"
#     data/cms/      the Item_* and P0*–P17 exports
# and one filename does not match what the loader asks for: the file on disk is
# "General Entities.csv" (space) while load_data.py:360 reads "General_Entities.csv"
# (underscore). Renaming the source file would be the other fix; staging a copy
# keeps the delivered data exactly as SCAI sent it.
#
# Staging happens in a tmpdir inside the container, so the mounted data/ is only
# ever read. The ETL itself TRUNCATEs every table first (load_data.py:423) — it
# is a full reload, not an incremental merge. Running it twice is safe; running
# it against a DB someone is querying is not.
set -eu

CSV_SRC="${CSV_SRC:-/srv/data}"

if [ -z "${ETL_DSN:-}" ]; then
    echo "run-etl: ETL_DSN is not set. It must point at a WRITE-capable role —" >&2
    echo "         the app's scai_ro role cannot create or truncate anything." >&2
    exit 2
fi

if [ ! -d "$CSV_SRC" ]; then
    echo "run-etl: no CSV directory at $CSV_SRC — is ./data mounted?" >&2
    exit 2
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

find "$CSV_SRC" -type f -name '*.csv' -exec cp {} "$STAGE"/ \;

if [ -f "$STAGE/General Entities.csv" ]; then
    mv "$STAGE/General Entities.csv" "$STAGE/General_Entities.csv"
fi

echo "run-etl: staged $(find "$STAGE" -name '*.csv' | wc -l) CSV files from $CSV_SRC"

exec python etl/load_data.py --csv-dir "$STAGE" --dsn "$ETL_DSN"
