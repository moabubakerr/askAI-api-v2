-- SCAI Economic Data Assistant — Postgres schema
-- Generated from actual inspection of the SCAI export files.
-- Run this before etl/load_data.py.

-- ============ Reference / lookup ============

CREATE TABLE countries (
    country_id      TEXT PRIMARY KEY,     -- source GUID, kept as-is (no numeric surrogate needed)
    name_en         TEXT NOT NULL,
    name_ar         TEXT,
    country_code    TEXT,                 -- ISO-ish 2-letter code, e.g. 'QA', 'TW'
    latitude        NUMERIC,
    longitude       NUMERIC
);

CREATE TABLE intervals (
    interval_id     TEXT PRIMARY KEY,
    name_en         TEXT NOT NULL,        -- 'Monthly' | 'Quarterly' | 'Yearly'
    name_ar         TEXT,
    unique_code     TEXT                  -- 'monthly' | 'quarterly' | 'yearly'
);

-- ============ Indicators (merged source + published) ============
-- Item_2/Item_4/Item_1 are the full working dataset (531 indicators).
-- P01/P02/P03 are the curated subset (189 indicators) that's been reviewed
-- and published, with richer metadata (unit, polarity, precomputed YoY, etc).
-- is_published + published_* columns bridge the two.

CREATE TABLE indicators (
    indicator_id            TEXT PRIMARY KEY,      -- Item_2.Id
    name_en                 TEXT NOT NULL,
    name_ar                 TEXT,
    indicator_type_id       TEXT,
    is_active                BOOLEAN,
    is_deleted               BOOLEAN,
    is_published             BOOLEAN NOT NULL DEFAULT FALSE,
    published_indicator_id   TEXT,                  -- P01.PublishedIndicatorId, if published
    indicator_type_en        TEXT,                  -- from P01, only present if published
    indicator_type_ar        TEXT,
    priority_type_en         TEXT,
    priority_ranking         INTEGER
);

CREATE TABLE indicator_details (
    indicator_detail_id      TEXT PRIMARY KEY,       -- Item_4.IndicatorDetailId
    indicator_id             TEXT REFERENCES indicators(indicator_id),
    name_en                  TEXT,
    name_ar                  TEXT,
    definition_en             TEXT,
    definition_ar             TEXT,
    format                    TEXT,                  -- display format string, e.g. '0.00', 'bn0.0'
    baseline_value            NUMERIC,
    baseline_year             INTEGER,
    target_value              NUMERIC,
    target_year               INTEGER,
    is_published              BOOLEAN NOT NULL DEFAULT FALSE,
    published_detail_id       TEXT UNIQUE,            -- P02.PublishedIndicatorDetailId, if published
    is_main                   BOOLEAN,                -- P02.IsMain: the headline reading for its
                                                      -- indicator, vs a sub-breakdown of it. 189 True
                                                      -- (1:1 with published indicators), 100 False.
                                                      -- NULL for unpublished details.
    unit_en                   TEXT,                   -- only present if published (Item_4 lacks unit)
    unit_ar                   TEXT,
    polarity_en               TEXT,                   -- 'Increase' | 'Decrease' — whether higher is better
    value_type_en             TEXT,
    data_source_en            TEXT,
    aggregation_type_en       TEXT,
    input_method_en           TEXT
);

-- Raw/broad indicator values — covers ALL 531 indicators (published + unpublished),
-- includes Qatar (country_id NULL = domestic) and benchmark-country values.
-- Use this table for anything involving unpublished/draft indicators.
CREATE TABLE indicator_values (
    value_id                  TEXT PRIMARY KEY,        -- Item_1.DataPointId
    indicator_detail_id       TEXT REFERENCES indicator_details(indicator_detail_id),
    period_label               TEXT,                    -- original string, e.g. '2024-Q3'
    period_date                DATE,                    -- normalized first-of-period date
    granularity                TEXT,                    -- 'monthly' | 'quarterly' | 'yearly'
    country_en                 TEXT,                    -- NULL/blank = Qatar (domestic)
    actual                     NUMERIC,
    target                     NUMERIC
);

-- Curated/published data points — only the 189 published indicators, but with
-- precomputed MoM/QoQ/YoY, so PREFER this table for published indicators —
-- it saves the analyst agent from doing arithmetic and is what SCAI has vetted.
CREATE TABLE published_data_points (
    published_data_point_id    TEXT PRIMARY KEY,
    published_indicator_detail_id TEXT REFERENCES indicator_details(published_detail_id),
    published_indicator_id     TEXT,
    interval_en                 TEXT,
    period_label                 TEXT,
    period_date                  DATE,
    granularity                  TEXT,
    country_id                   TEXT,
    country_en                   TEXT,                  -- blank = Qatar
    actual                        NUMERIC,
    target                        NUMERIC,
    outlook                       NUMERIC,
    monthly_mom_percent           NUMERIC,
    monthly_mom_pp                NUMERIC,
    monthly_yoy_percent           NUMERIC,
    monthly_yoy_pp                NUMERIC,
    quarterly_qoq_percent         NUMERIC,
    quarterly_qoq_pp              NUMERIC,
    quarterly_yoy_percent         NUMERIC,
    quarterly_yoy_pp              NUMERIC,
    yearly_yoy_percent            NUMERIC,
    yearly_yoy_pp                 NUMERIC
);

-- Merged analyst commentary (Item_1_DataPointAnalysis + P04). `source` tells
-- you which table it came from; published rows have richer bilingual +
-- SRO/NPC/benchmark commentary.
CREATE TABLE indicator_analysis (
    analysis_id                TEXT PRIMARY KEY,
    ref_data_point_id          TEXT,                    -- IndicatorDetailDataPointId or PublishedDataPointId
    source                      TEXT NOT NULL,           -- 'item' | 'published'
    summary_en                  TEXT,
    detailed_analysis_en        TEXT,
    npc_analysis_en              TEXT,
    benchmark_en                  TEXT,
    -- P04 carries all four in Arabic as well and they were never loaded, so
    -- SCAI's own commentary came back in English to an Arabic reader. Coverage
    -- is partial (667 of 1031 rows have SummaryAR), which is why every reader
    -- falls back to the English text rather than being shown nothing.
    summary_ar                  TEXT,
    detailed_analysis_ar        TEXT,
    npc_analysis_ar              TEXT,
    benchmark_ar                  TEXT
);

CREATE TABLE benchmark_countries (
    published_benchmark_country_id TEXT PRIMARY KEY,
    published_indicator_detail_id  TEXT,
    published_indicator_id          TEXT,
    country_en                       TEXT,
    country_code                     TEXT
);

-- Generic reference lookups (P12) — one table covering many small enum-like
-- sets: DataSources, Units, ChartTypes, IndicatorTypes, AggregationTypes,
-- InputMethods, Intervals, Polarities, ValueTypes, ValidationChecks.
-- Use lookup_type to filter to the set you need.
CREATE TABLE lookups (
    lookup_type      TEXT NOT NULL,
    lookup_id        TEXT NOT NULL,
    name_en          TEXT,
    name_ar          TEXT,
    unique_code      TEXT,
    PRIMARY KEY (lookup_type, lookup_id)
);

CREATE TABLE priority_types (
    priority_type_id  TEXT PRIMARY KEY,
    name_en            TEXT,
    name_ar            TEXT,
    unique_code        TEXT,
    color_class        TEXT,
    ranking            INTEGER
);

-- Links a published indicator to the dashboard(s) it appears on AND to the
-- Sector / General_Entity / other CMS record it belongs to. This is the
-- bridge between the numeric indicator world and the CMS world: e.g.
-- "which indicators track the Tourism sector" joins through here.
CREATE TABLE indicator_dashboards (
    published_indicator_dashboard_id TEXT PRIMARY KEY,
    published_indicator_id            TEXT,
    dashboard_id                       TEXT,
    dashboard_name_en                   TEXT,      -- 'Executive Dashboard' | 'Leadership Dashboard'
    sort_order                           INTEGER,
    entity_classification_name            TEXT,     -- 'Sectors' | 'SpecialEntities' | 'SpecialProjects' |
                                                      -- 'DiversificationTargets' | 'Enablers' | 'Drivers' |
                                                      -- 'NationalIndicators'
    entity_id                              TEXT      -- when entity_classification_name = 'Sectors', joins
                                                      -- to sectors.sector_id; when 'SpecialEntities' or similar,
                                                      -- joins to general_entities.entity_id. No single FK possible
                                                      -- since it's polymorphic — join manually based on the
                                                      -- classification value.
);

-- ============ Chart/display metadata (lower priority for Q&A, useful for
-- the Chart Agent to match SCAI's own chosen chart types) ============

CREATE TABLE chart_sub_indicators (
    published_indicator_chart_id  TEXT,
    published_indicator_id         TEXT,
    list_type                       TEXT,
    member_indicator_detail_id       TEXT
);

CREATE TABLE chart_alternate_views (
    published_indicator_chart_id  TEXT,
    published_indicator_id         TEXT,
    alternate_view_id               TEXT,
    alternate_view_en                TEXT,     -- e.g. 'Monthly MoM %', 'Yearly YoY %'
    alternate_view_code               TEXT
);

CREATE TABLE additional_charts (
    published_additional_chart_id  TEXT PRIMARY KEY,
    published_indicator_id          TEXT,
    published_indicator_detail_id    TEXT,
    chart_type_en                     TEXT,     -- e.g. 'Geo Map'
    interval_en                        TEXT
);

CREATE TABLE additional_chart_countries (
    published_additional_chart_id  TEXT,
    published_indicator_id          TEXT,
    published_indicator_detail_id    TEXT,
    country_en                        TEXT,
    country_code                       TEXT
);

-- ============ CMS / entity monitoring ============

CREATE TABLE champions (
    champion_id     TEXT PRIMARY KEY,
    name_en         TEXT,
    name_ar         TEXT,
    description_en   TEXT,
    description_ar   TEXT,
    is_active        BOOLEAN,
    is_ministry      BOOLEAN
);

CREATE TABLE sectors (
    sector_id        TEXT PRIMARY KEY,
    champion_id      TEXT REFERENCES champions(champion_id),
    name_en          TEXT,
    name_ar          TEXT,
    description_en    TEXT,          -- decoded from base64 where applicable
    description_ar    TEXT,
    achievement_en     TEXT,
    achievement_ar     TEXT,
    challenges_en       TEXT,
    challenges_ar       TEXT,
    highlight_en         TEXT,
    highlight_ar         TEXT,
    insight_en            TEXT,
    insight_ar            TEXT,
    is_active              BOOLEAN,
    -- opaque IDs — no lookup table was provided for these; kept for future joins
    stage_of_monitoring_id  TEXT,
    status_id                TEXT,
    workflow_id                TEXT
);

CREATE TABLE general_entities (
    entity_id         TEXT PRIMARY KEY,
    champion_id       TEXT REFERENCES champions(champion_id),
    name_en           TEXT,
    name_ar           TEXT,
    description_en     TEXT,
    description_ar     TEXT,
    achievement_en       TEXT,
    achievement_ar       TEXT,
    challenges_en         TEXT,
    challenges_ar         TEXT,
    highlight_en           TEXT,
    highlight_ar           TEXT,
    insight_en              TEXT,
    insight_ar              TEXT,
    entity_classification    TEXT,
    is_active                 BOOLEAN,
    stage_of_monitoring_id     TEXT,
    status_id                   TEXT,
    workflow_id                   TEXT
);

CREATE TABLE articles (
    article_id        TEXT PRIMARY KEY,
    title_en           TEXT,
    title_ar           TEXT,
    content_en          TEXT,         -- HTML stripped to plain text for LLM grounding
    content_ar          TEXT,
    published            BOOLEAN,
    featured              BOOLEAN,
    is_active              BOOLEAN,
    article_date            DATE,
    published_date           DATE
);

-- ============ Indexes for common query patterns ============

CREATE INDEX idx_indicator_values_detail_period ON indicator_values(indicator_detail_id, period_date);
CREATE INDEX idx_published_dp_detail_period ON published_data_points(published_indicator_detail_id, period_date);
CREATE INDEX idx_published_dp_country ON published_data_points(country_en);
CREATE INDEX idx_indicators_name_en ON indicators USING gin (to_tsvector('english', name_en));
CREATE INDEX idx_sectors_name_en ON sectors USING gin (to_tsvector('english', name_en));
CREATE INDEX idx_entities_name_en ON general_entities USING gin (to_tsvector('english', name_en));
CREATE INDEX idx_articles_title_en ON articles USING gin (to_tsvector('english', title_en));
CREATE INDEX idx_indicator_dashboards_entity ON indicator_dashboards(entity_classification_name, entity_id);
CREATE INDEX idx_indicator_dashboards_indicator ON indicator_dashboards(published_indicator_id);
CREATE INDEX idx_lookups_type ON lookups(lookup_type);

-- ============ Article text retrieval (semantic search over SCAI articles) ============
--
-- Indicators answer "what is the number". Articles answer "what has SCAI
-- written about X", which is a different retrieval problem: the answer is a
-- paragraph inside one of 84 documents averaging ~14,000 characters, not a
-- row. So articles are chunked, each chunk is embedded with the same bge-m3
-- model already serving indicator matching, and the vectors are stored here
-- rather than in a second datastore.
--
-- In Postgres rather than Chroma because Postgres is already running, already
-- backed up, and already the source of the articles themselves — a separate
-- vector store would be a second thing to deploy and a second place for the
-- corpus to drift out of sync with the CMS export.
--
-- Vectors are persisted rather than rebuilt at startup. The indicator catalog
-- re-embeds on every restart (~583 names, ~10s); doing the same for ~2,600
-- article chunks would make a restart cost minutes.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE article_chunks (
    chunk_id       TEXT PRIMARY KEY,
    article_id     TEXT NOT NULL REFERENCES articles(article_id) ON DELETE CASCADE,
    language       TEXT NOT NULL,          -- 'en' | 'ar' — each article is chunked in both
    chunk_index    INTEGER NOT NULL,       -- position within the article, for ordering
    content        TEXT NOT NULL,          -- the passage itself, quoted verbatim in answers
    char_count     INTEGER,
    embedding      vector(1024),           -- BAAI/bge-m3 dimensionality
    UNIQUE (article_id, language, chunk_index)
);

-- Cosine distance, matching how the indicator resolver scores similarity.
-- HNSW rather than IVFFlat: IVFFlat needs a populated table to build a useful
-- index, which makes it order-dependent during a reload.
CREATE INDEX idx_article_chunks_embedding ON article_chunks
    USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_article_chunks_language ON article_chunks(language);

-- Catalogue vectors, so a restart does not re-embed 1,288 texts against a
-- CPU-only embedding server before it can answer its first question.
-- Keyed by the exact text embedded, because an indicator is embedded twice:
-- once as its bare name and once as name-plus-definition.
--
-- `model` is not decoration. These vectors are only comparable with others
-- from the SAME model, and swapping the model while stale rows remain would
-- score a question against vectors from a different embedding space — wrong
-- answers, silently. The loader writes it and the reader filters on it.
CREATE TABLE indicator_embeddings (
    text_key    TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    embedding   vector(1024)
);
CREATE INDEX idx_indicator_embeddings_model ON indicator_embeddings(model);

-- ============ Application data (not SCAI's) ============

-- What a reader thought of an answer. The only table the app itself writes to,
-- and the reason scai_ro is granted INSERT on this one table and nothing else:
-- the read-only guarantee exists to stop the app modifying SCAI's data, and
-- this is not SCAI's data.
--
-- The question and answer are copied in rather than referenced. The transcript
-- lives in memory and is evicted after two hours, so a rating that only
-- carried an id would be unreadable by the time anyone came to read it — and a
-- rating you cannot tie to what was said is not feedback, it is a number.
--
-- Deliberately NOT truncated by the ETL: a data reload must not delete what
-- users have told us.
CREATE TABLE message_feedback (
    feedback_id   TEXT PRIMARY KEY,
    message_id    TEXT NOT NULL,
    session_id    TEXT,
    rating        SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    -- Required by the API when rating <= 2. Enforced here too, so the rule
    -- survives a second caller that does not know about it.
    comment       TEXT CHECK (rating > 2 OR (comment IS NOT NULL AND length(btrim(comment)) > 0)),
    question      TEXT,
    answer        TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_message_feedback_rating ON message_feedback(rating);
CREATE INDEX idx_message_feedback_created ON message_feedback(created_at DESC);

-- Every exchange, for the admin dashboard. Also application data, also never
-- truncated by the ETL.
--
-- Written best-effort: a failure here must never cost the user their answer,
-- so the logger swallows its own errors. That means this table can under-count
-- and a dashboard should treat it as a log, not as a ledger.
--
-- `answer_shape` is derived from what the answer actually CONTAINS rather than
-- from the route that produced it. The routing taxonomy has grown to twenty-odd
-- computation types and keeps changing; what a reader received — a value, a
-- series, a ranking, a refusal — does not.
CREATE TABLE chat_messages (
    message_id     TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    asked_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    endpoint       TEXT NOT NULL,            -- 'chat' | 'read'
    language       TEXT,                     -- 'en' | 'ar'
    question       TEXT,
    answer         TEXT,
    answered       BOOLEAN,                  -- false when the reply was a refusal
    answer_shape   TEXT,
    indicator      TEXT,                     -- the indicator the answer was about, if one
    period_label   TEXT,
    verified       BOOLEAN,                  -- the numeric verifier accepted the wording
    readable       BOOLEAN,
    latency_ms     INTEGER
);
CREATE INDEX idx_chat_messages_session ON chat_messages(session_id, asked_at);
CREATE INDEX idx_chat_messages_asked ON chat_messages(asked_at DESC);
CREATE INDEX idx_chat_messages_answered ON chat_messages(answered);

-- Which rows an answer was built from. app/compute/citations.py already
-- computes this for every exchange, deterministically and outside the LLM, to
-- render the "Sources:" footer; until now it was rendered and thrown away.
-- Keeping it is what lets the admin panel go from an answer back to the exact
-- published_data_points row, and from there to the indicator and the export
-- file it was loaded from.
--
-- Application data, like the two tables above: written by the app, never
-- truncated by the ETL.
--
-- No foreign key to chat_messages, for the same reason message_feedback has
-- none: both writes are best-effort and the message insert can be the one that
-- fails. A citation for a message that was never logged is a partial record; a
-- rejected insert is no record at all.
--
-- record_id is NOT a foreign key either, and that is the more interesting
-- case: the ETL truncates and reloads the source layer, so a data point cited
-- last month may not exist today. The row keeps enough denormalised context —
-- indicator name, period, original data source — to stay readable after the
-- id it points at has gone. A resolving join is a bonus, not the record.
CREATE TABLE message_citations (
    message_id     TEXT NOT NULL,
    -- Position in the answer's citation list. Part of the key so a retried
    -- insert cannot duplicate, and so the panel can show them in the order the
    -- footer did.
    position       SMALLINT NOT NULL,
    source_table   TEXT NOT NULL,     -- published_data_points | indicator_values | ...
    record_id      TEXT,              -- the cited row's primary key, when it has one
    indicator      TEXT,
    data_source    TEXT,              -- original source, e.g. 'World Bank'
    period_label   TEXT,
    country        TEXT,
    PRIMARY KEY (message_id, position)
);
CREATE INDEX idx_message_citations_message ON message_citations(message_id);
CREATE INDEX idx_message_citations_record ON message_citations(source_table, record_id);

-- What the last ETL run actually read, and where each table came from.
--
-- The CSV-to-table mapping used to exist only as hardcoded filenames inside
-- the loader functions in etl/load_data.py, which meant nothing could answer
-- "which export file is this number from?" without reading Python. This table
-- is written by the loader from what it observed on disk, not from a
-- hand-maintained manifest, so it cannot drift from the load that produced it.
--
-- One row per (table, source file). A loader that merges two exports into one
-- table — indicators, from Item_2 and P01 — gets two rows, both carrying that
-- table's final row count.
--
-- Appended, not truncated: the history of loads is how you answer "the numbers
-- changed, when did that happen and which file changed with them".
CREATE TABLE etl_load_log (
    load_id           TEXT NOT NULL,        -- one id per ETL run
    table_name        TEXT NOT NULL,
    source_file       TEXT NOT NULL,
    loader            TEXT NOT NULL,        -- the function in etl/load_data.py
    started_at        TIMESTAMPTZ NOT NULL,
    finished_at       TIMESTAMPTZ NOT NULL,
    -- Of the file as read. A changed digest with an unchanged filename is the
    -- only reliable signal that an export was re-cut, because these filenames
    -- carry a timestamp that the CMS does not always bump.
    file_sha256       TEXT,
    file_bytes        BIGINT,
    file_modified_at  TIMESTAMPTZ,
    rows_in_file      INTEGER,              -- data rows read from the CSV
    rows_in_table     INTEGER,              -- rows present in the table after the load
    PRIMARY KEY (load_id, table_name, source_file)
);
CREATE INDEX idx_etl_load_log_finished ON etl_load_log(finished_at DESC);

-- Convenience for the dashboard. A session is not a table: it is whatever
-- messages share a session_id, so deriving it avoids two sources of truth that
-- can disagree.
CREATE VIEW v_session_summary AS
SELECT session_id,
       min(asked_at)                                   AS started_at,
       max(asked_at)                                   AS last_seen_at,
       count(*)                                        AS messages,
       count(*) FILTER (WHERE answered IS FALSE)       AS refusals,
       count(DISTINCT language)                        AS languages_used,
       max(language)                                   AS language,
       round(avg(latency_ms))                          AS avg_latency_ms
FROM chat_messages
GROUP BY session_id;

CREATE VIEW v_message_feedback AS
SELECT f.feedback_id, f.message_id, f.session_id, f.rating, f.comment,
       f.created_at,
       coalesce(f.question, m.question)                AS question,
       coalesce(f.answer, m.answer)                    AS answer,
       m.answer_shape, m.indicator, m.language, m.answered, m.latency_ms
FROM message_feedback f
LEFT JOIN chat_messages m ON m.message_id = f.message_id;

-- The next two views are MIRRORED in etl/load_data.py, which re-runs them as
-- CREATE OR REPLACE on every load so an existing deployment picks them up
-- without hand-run SQL. Change one, change the other.
--
-- The mirror is partly self-checking: Postgres refuses CREATE OR REPLACE VIEW
-- if the column list, order or types differ, so a load against a database
-- built from this file fails loudly rather than silently installing a
-- different shape. It does NOT check the joins, so divergence in the body is
-- still on you.

-- The most recent load only. The log is append-only so that history survives,
-- but "where does this table come from" almost always means "right now".
CREATE VIEW v_etl_current_load AS
SELECT l.*
FROM etl_load_log l
JOIN (SELECT load_id FROM etl_load_log ORDER BY finished_at DESC LIMIT 1) cur
  ON cur.load_id = l.load_id;

-- An answer, the rows it cited, and the export file each of those rows came
-- from. The chain the admin panel is for, in one place.
--
-- LEFT JOINs throughout: a citation whose record has since been reloaded away,
-- or one against a table with no per-row id (the indicator catalogue, analyst
-- commentary), still appears with everything that is known about it. Dropping
-- those rows would quietly under-report what an answer was built from, which
-- is the one thing this view exists to show.
CREATE VIEW v_message_provenance AS
SELECT c.message_id, c.position, c.source_table, c.record_id,
       c.indicator, c.data_source, c.period_label, c.country,
       m.asked_at, m.question, m.answer, m.answered, m.answer_shape, m.language,
       -- Did the cited row survive to today? Computed from the join rather
       -- than inferred from whether some column came back null: `actual` is
       -- legitimately null on a target-only point, so "no number" and "no row"
       -- are different answers and must not be confused.
       (p.published_data_point_id IS NOT NULL)       AS record_found,
       p.published_indicator_detail_id,
       p.actual, p.target, p.outlook, p.period_date, p.granularity,
       d.indicator_id, d.indicator_detail_id, d.unit_en, d.is_main,
       i.name_en AS indicator_name_en, i.name_ar AS indicator_name_ar,
       e.source_file, e.file_sha256, e.finished_at AS loaded_at
FROM message_citations c
LEFT JOIN chat_messages m ON m.message_id = c.message_id
LEFT JOIN published_data_points p
       ON c.source_table = 'published_data_points'
      AND p.published_data_point_id = c.record_id
LEFT JOIN indicator_details d
       ON d.published_detail_id = p.published_indicator_detail_id
LEFT JOIN indicators i ON i.indicator_id = d.indicator_id
LEFT JOIN v_etl_current_load e ON e.table_name = c.source_table;

-- Lexical fallback for exact names and acronyms ("Tawteen", "ICV"), which
-- embeddings match poorly — a rare token carries little semantic signal.
CREATE INDEX idx_article_chunks_fts ON article_chunks
    USING gin (to_tsvector('simple', content));
