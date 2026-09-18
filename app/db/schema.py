"""
Schema description fed to the Text-to-SQL agent's prompt.

This reflects the REAL SCAI schema (etl/schema.sql), built from actual
inspection of the SCAI export files — not a generic placeholder. Keep this
in sync with etl/schema.sql; if you add/rename a column there, update it here
too, since a stale description degrades SQL accuracy silently.
"""

SCHEMA_DESCRIPTION = """
=== INDICATOR TABLES (numeric economic data) ===

Table: indicators
  - indicator_id (text, PK)
  - name_en, name_ar (text)
  - is_published (boolean)         TRUE if this indicator has been reviewed/published by SCAI
  - published_indicator_id (text)  only set if is_published
  - indicator_type_en (text)       e.g. 'Sector Indicator', 'National Indicator',
                                    'Economic Diversification Targets', 'Enablers', 'Drivers'
                                    (only populated for published indicators)
  - priority_type_en, priority_ranking     (only for published indicators)

Table: indicator_details
  - indicator_detail_id (text, PK)
  - indicator_id (FK -> indicators.indicator_id)
  - name_en, name_ar (text)          the specific metric name, e.g. 'Export Cost', 'Airport Utilization'
  - definition_en, definition_ar (text)
  - baseline_value, baseline_year, target_value, target_year (numeric/int)
  - is_published (boolean)
  - published_detail_id (text)       only set if is_published; joins to published_data_points
  - unit_en (text)                   e.g. '%', 'QAR', 'bn QAR' — ONLY populated if is_published
  - polarity_en (text)               'Increase' | 'Decrease' — whether a HIGHER value is the good direction
  - data_source_en (text)            e.g. 'World Bank', 'Qatar Tourism', 'Ministry Of Commerce And Industry'

Table: indicator_values   -- RAW/BROAD coverage: ALL indicators (published + unpublished)
  - value_id (text, PK)
  - indicator_detail_id (FK -> indicator_details.indicator_detail_id)
  - period_label (text)              original period string, e.g. '2024-Q3', '2024', '2024-03'
  - period_date (date)               normalized to first day of the period — ALWAYS use this for filtering/sorting
  - granularity (text)               'monthly' | 'quarterly' | 'yearly'
  - country_en (text)                blank/NULL = Qatar (domestic); otherwise a benchmark country
  - actual (numeric), target (numeric)

Table: published_data_points   -- CURATED, only the ~189 published indicators, PREFER this table
                                   when the indicator is published: it has vetted, precomputed
                                   period-over-period changes so you don't need to compute them yourself.
  - published_data_point_id (text, PK)
  - published_indicator_detail_id (FK -> indicator_details.published_detail_id)
  - period_date (date), granularity (text), period_label (text)
  - country_en (text)                 blank/NULL = Qatar; else benchmark country
  - actual, target, outlook (numeric)
  - monthly_mom_percent, monthly_yoy_percent           month-over-month / year-over-year % change
  - quarterly_qoq_percent, quarterly_yoy_percent       quarter-over-quarter / year-over-year % change
  - yearly_yoy_percent                                  year-over-year % change for yearly data
  (there are also *_pp sibling columns = percentage-POINT change instead of percent change,
   e.g. monthly_mom_pp — use *_pp columns when the underlying value is itself already a
   percentage/rate, so "change" should be in points not a percent-of-a-percent)

Table: indicator_analysis   -- analyst-written narrative commentary, keyed to a data point
  - analysis_id (text, PK)
  - ref_data_point_id (text)          matches EITHER indicator_values.value_id OR
                                       published_data_points.published_data_point_id, depending on `source`
  - source (text)                     'item' (raw) | 'published' (curated, richer commentary)
  - summary_en (text), detailed_analysis_en (text)
  - npc_analysis_en, benchmark_en (text)     only populated when source = 'published'

Table: benchmark_countries
  - published_indicator_detail_id, published_indicator_id (text)
  - country_en, country_code (text)   which countries SCAI benchmarks this indicator against

Table: countries (country_id, name_en, name_ar, country_code, latitude, longitude)
Table: intervals (interval_id, name_en, unique_code)   reference/lookup only, rarely needed directly

Table: lookups   -- generic reference table covering many small enum-like sets in one place
  - lookup_type (text)   one of: DataSources, Units, ChartTypes, IndicatorTypes, AggregationTypes,
                          InputMethods, Intervals, Polarities, ValueTypes, ValidationChecks
  - lookup_id, name_en, name_ar, unique_code (text)
  - Use WHERE lookup_type = '<type>' to filter to the set you need; this table is mainly for
    resolving *_Id columns to labels, not something users ask about directly.

Table: priority_types
  - priority_type_id (text, PK), name_en (text)   'Non-Priority' | 'Priority' | 'Confidential'
  - ranking (integer)     lower number = higher priority (Confidential=1, Priority=2, Non-Priority=3)
  - color_class (text)    display-only, ignore for Q&A

Table: indicator_dashboards   -- BRIDGE between numeric indicators and the CMS/sector world.
                                  Use this to answer "which indicators track sector/entity X" or
                                  "what sectors does indicator Y appear under".
  - published_indicator_id (text)         joins to indicators.published_indicator_id
  - dashboard_name_en (text)              'Executive Dashboard' | 'Leadership Dashboard'
  - entity_classification_name (text)     'Sectors' | 'SpecialEntities' | 'SpecialProjects' |
                                           'DiversificationTargets' | 'Enablers' | 'Drivers' |
                                           'NationalIndicators'
  - entity_id (text)                      POLYMORPHIC: when entity_classification_name = 'Sectors',
                                           joins to sectors.sector_id; otherwise joins to
                                           general_entities.entity_id. Check the classification
                                           before deciding which table to join.

=== CHART/DISPLAY METADATA (rarely needed for direct Q&A; mainly for the Chart Agent) ===

Table: chart_sub_indicators (published_indicator_chart_id, list_type, member_indicator_detail_id)
Table: chart_alternate_views (published_indicator_chart_id, alternate_view_en)   e.g. 'Monthly MoM %'
Table: additional_charts (published_additional_chart_id, published_indicator_id, chart_type_en)
                                                                                    e.g. 'Geo Map'
Table: additional_chart_countries (published_additional_chart_id, country_en, country_code)

=== CMS / ENTITY MONITORING TABLES (qualitative content, not numeric time series) ===

Table: champions       -- organizations/ministries responsible for sectors or entities
  - champion_id (text, PK), name_en, name_ar, description_en, is_ministry (boolean)

Table: sectors         -- economic sectors being monitored (e.g. Tourism, Health)
  - sector_id (text, PK), champion_id (FK -> champions), name_en, name_ar
  - description_en, achievement_en, challenges_en, highlight_en, insight_en (free text, plain text — HTML stripped)
  - is_active (boolean)
  - stage_of_monitoring_id, status_id, workflow_id (opaque IDs — no lookup table available; do not
    attempt to join or interpret these as human-readable labels)

Table: general_entities   -- specific initiatives/projects being monitored within a sector
  - entity_id (text, PK), champion_id (FK -> champions), name_en, name_ar
  - description_en, achievement_en, challenges_en, highlight_en, insight_en (free text)
  - entity_classification (text), is_active (boolean)
  - stage_of_monitoring_id, status_id, workflow_id (opaque — same caveat as sectors)

Table: articles        -- published thought-leadership / research articles
  - article_id (text, PK), title_en, content_en (free text, HTML stripped)
  - published (boolean), featured (boolean), article_date, published_date (date)

=== RULES FOR QUERY GENERATION ===
  - For a numeric/economic question about a specific indicator: check indicators.is_published first.
    If TRUE, prefer published_data_points (via indicator_details.published_detail_id) for its
    precomputed period-over-period changes. If FALSE, use indicator_values.
  - country_en blank/NULL means Qatar/domestic — do not filter it out by accident with
    "WHERE country_en IS NOT NULL" unless the user explicitly wants only benchmark countries.
  - Always JOIN to indicators/indicator_details to resolve human-readable names — never expose
    raw GUIDs in a query result meant for the analyst step.
  - Always ORDER BY period_date for any time series.
  - For qualitative questions ("what is being done about X", "who champions Y sector", "what
    challenges does Z entity face") — query sectors/general_entities/champions with ILIKE on
    name_en or full-text search, not the indicator tables.
  - For cross-domain questions ("which indicators track sector X", "what sector does indicator Y
    belong to") — go through indicator_dashboards, remembering entity_id is polymorphic (check
    entity_classification_name to know whether to join sectors or general_entities).
  - Never SELECT *  — name columns explicitly.
  - Only SELECT statements are allowed. No INSERT/UPDATE/DELETE/DDL.
  - LIMIT results to {max_rows} rows unless the user asks for the full series.
"""


def get_schema_prompt(max_rows: int) -> str:
    return SCHEMA_DESCRIPTION.format(max_rows=max_rows)
