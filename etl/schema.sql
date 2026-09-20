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

-- Lexical fallback for exact names and acronyms ("Tawteen", "ICV"), which
-- embeddings match poorly — a rare token carries little semantic signal.
CREATE INDEX idx_article_chunks_fts ON article_chunks
    USING gin (to_tsvector('simple', content));
