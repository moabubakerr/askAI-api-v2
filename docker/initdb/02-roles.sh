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

    -- The single exception, granted on one table by name. The read-only rule
    -- exists so the app cannot modify SCAI's data; user feedback is not SCAI's
    -- data, and it has to be written by the process the user is talking to.
    --
    -- INSERT only: the app can record a rating and can never edit or delete
    -- one, so the record of what users said is append-only from its side.
    GRANT INSERT ON TABLE message_feedback TO scai_ro;
    GRANT INSERT ON TABLE chat_messages TO scai_ro;
    -- Which rows each answer was built from. Written by the same best-effort
    -- logger as chat_messages, and append-only for the same reason: the point
    -- of keeping provenance is that nobody can revise it afterwards.
    GRANT INSERT ON TABLE message_citations TO scai_ro;
EOSQL

echo "02-roles.sh: scai_ro created (SELECT-only, plus INSERT on message_feedback, chat_messages and message_citations)"
