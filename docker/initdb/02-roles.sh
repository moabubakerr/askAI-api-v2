#!/bin/bash
# Creates the app's read-only role. Runs once, on first init of an empty data
# volume, AFTER 01-schema.sql (docker-entrypoint-initdb.d executes in filename
# order) — the ordering matters, because GRANT SELECT ON ALL TABLES only covers
# tables that already exist at grant time.
#
# This is the guardrail the whole design leans on: the app connects as scai_ro
# and therefore physically cannot write, no matter what SQL any agent emits.
# app/db/executor.py's keyword blocklist is the second layer, not the first.
set -euo pipefail

: "${SCAI_RO_PASSWORD:?SCAI_RO_PASSWORD must be set for the postgres service}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE ROLE scai_ro WITH LOGIN PASSWORD '${SCAI_RO_PASSWORD}';
    GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO scai_ro;
    GRANT USAGE ON SCHEMA public TO scai_ro;
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO scai_ro;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO scai_ro;
EOSQL

echo "02-roles.sh: scai_ro created (SELECT-only on schema public)"
