"""
Loads all SCAI CSV exports into Postgres per etl/schema.sql.

Usage:
    python etl/load_data.py --csv-dir /path/to/csvs --dsn postgresql://scai_rw:pw@localhost:5432/scai_indicators

Run schema.sql against the target DB first:
    psql "$DSN" -f etl/schema.sql

Notes:
- Uses a WRITE-capable role for loading. The app itself later connects with
  a separate READ-ONLY role (see README) — never point the app at this DSN.
- Idempotent-ish: truncates each target table before loading, so it's safe
  to re-run during development. Do not run against prod without adjusting
  that behavior once this is a live system with its own writes.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).parent.parent))
from etl.utils import (
    clean_null, fix_mojibake, html_to_text, maybe_b64_decode, normalize_guid,
    parse_period, strip_html, to_bool, to_float,
)


def read_csv(csv_dir: Path, filename: str) -> pd.DataFrame:
    df = pd.read_csv(csv_dir / filename, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    # normalize the literal-string 'NULL' cells that appear across these exports
    # (DataFrame.applymap was removed in pandas 3.0; .map is the replacement),
    # then case-normalize GUIDs so ids written upper-case in the CMS files join
    # against the same ids written lower-case in the P0* files.
    return df.map(clean_null).map(normalize_guid)


def load_countries(csv_dir, engine):
    """P14 is the canonical, richer country reference (adds ISO code + lat/long,
    which the older Item_6_Countries lacked) — used instead of Item_6."""
    df = read_csv(csv_dir, "P14_Ref_Countries-20260811_151010.csv")
    out = pd.DataFrame({
        "country_id": df["Id"],
        "name_en": df["NameEN"],
        "name_ar": df["NameAR"],
        "country_code": df["Code"],
        "latitude": df["Latitude"].apply(to_float),
        "longitude": df["Longitude"].apply(to_float),
    })
    out.to_sql("countries", engine, if_exists="append", index=False)
    print(f"countries: {len(out)} rows")


def load_intervals(csv_dir, engine):
    df = read_csv(csv_dir, "Item_1_Intervals-20260811_151007.csv")
    out = pd.DataFrame({
        "interval_id": df["Id"],
        "name_en": df["NameEN"],
        "name_ar": df["NameAR"],
        "unique_code": df["UniqueId"],
    })
    out.to_sql("intervals", engine, if_exists="append", index=False)
    print(f"intervals: {len(out)} rows")


def load_indicators(csv_dir, engine):
    item2 = read_csv(csv_dir, "Item_2_Indicators_Catalog-20260811_151007.csv")
    p01 = read_csv(csv_dir, "P01_Published_Indicators-20260811_151007.csv")

    p01_by_source = p01.set_index("SourceIndicatorId")

    rows = []
    skipped_nameless = 0
    for _, r in item2.iterrows():
        # 61 of the 531 Item_2 rows carry no NameEN and no NameAR — empty stubs
        # in the source export. They cannot be loaded (indicators.name_en is
        # NOT NULL) and they would be useless if they could: the resolver matches
        # on name_en, so a nameless indicator is unreachable by any question and
        # uncitable in any answer. Verified safe to drop: none are published
        # (0 appear in P01) and no indicator_details row references any of them,
        # so nothing downstream loses its parent.
        if not str(r["NameEN"] or "").strip():
            skipped_nameless += 1
            continue
        pub = p01_by_source.loc[r["Id"]] if r["Id"] in p01_by_source.index else None
        rows.append({
            "indicator_id": r["Id"],
            "name_en": r["NameEN"],
            "name_ar": r["NameAR"],
            "indicator_type_id": r["IndicatorTypeId"],
            "is_active": to_bool(r["IsActive"]),
            "is_deleted": to_bool(r["IsDeleted"]),
            "is_published": pub is not None,
            "published_indicator_id": pub["PublishedIndicatorId"] if pub is not None else None,
            "indicator_type_en": pub["IndicatorTypeEN"] if pub is not None else None,
            "indicator_type_ar": pub["IndicatorTypeAR"] if pub is not None else None,
            "priority_type_en": pub["PriorityTypeEN"] if pub is not None else None,
            "priority_ranking": to_float(pub["PriorityRanking"]) if pub is not None else None,
        })
    out = pd.DataFrame(rows)
    out.to_sql("indicators", engine, if_exists="append", index=False)
    print(f"indicators: {len(out)} rows ({out['is_published'].sum()} published), "
          f"{skipped_nameless} nameless stubs skipped")


def _existing_ids(engine, table: str, column: str) -> set:
    """Reads back the keys actually present after the parent load. Used to drop
    orphaned child rows before insert rather than letting Postgres reject the
    whole batch — pandas' to_sql sends one multi-row INSERT, so a single bad FK
    aborts every row with it."""
    with engine.connect() as conn:
        return {r[0] for r in conn.execute(text(f"SELECT {column} FROM {table}"))}


def load_indicator_details(csv_dir, engine):
    item4 = read_csv(csv_dir, "Item_4_IndicatorDetails-20260811_151007.csv")
    p02 = read_csv(csv_dir, "P02_Published_IndicatorDetails-20260811_151007.csv")

    p02_by_source = p02.set_index("SourceIndicatorDetailId")

    # 61 of the 644 Item_4 rows name an IndicatorId (52 distinct) that does not
    # exist anywhere in Item_2 — almost certainly indicators deleted upstream
    # without cascading to their details. They cannot be inserted: indicator_id
    # is a FK to indicators. None of them are published (0 appear in P02), so
    # the SCAI-vetted layer is untouched; what is lost is unpublished working
    # data. See the ETL summary — this is worth confirming with SCAI.
    known_indicators = _existing_ids(engine, "indicators", "indicator_id")
    skipped_orphans = 0

    rows = []
    for _, r in item4.iterrows():
        if r["IndicatorId"] not in known_indicators:
            skipped_orphans += 1
            continue
        pub = p02_by_source.loc[r["IndicatorDetailId"]] if r["IndicatorDetailId"] in p02_by_source.index else None
        rows.append({
            "indicator_detail_id": r["IndicatorDetailId"],
            "indicator_id": r["IndicatorId"],
            "name_en": r["NameEN"],
            "name_ar": r["NameAR"],
            "definition_en": r["DefinitionEN"],
            "definition_ar": r["DefinitionAR"],
            "format": r["Format"],
            "baseline_value": to_float(r["BaseLineValue"]),
            "baseline_year": int(to_float(r["BaseLineYear"])) if to_float(r["BaseLineYear"]) else None,
            "target_value": to_float(r["TargetValue"]),
            "target_year": int(to_float(r["TargetYear"])) if to_float(r["TargetYear"]) else None,
            "is_published": pub is not None,
            "published_detail_id": pub["PublishedIndicatorDetailId"] if pub is not None else None,
            "unit_en": pub["UnitEN"] if pub is not None else None,
            "unit_ar": pub["UnitAR"] if pub is not None else None,
            "polarity_en": pub["PolarityEN"] if pub is not None else None,
            "value_type_en": pub["ValueTypeEN"] if pub is not None else None,
            "data_source_en": pub["DataSourceEN"] if pub is not None else None,
            "aggregation_type_en": pub["AggregationTypeEN"] if pub is not None else None,
            "input_method_en": pub["InputMethodEN"] if pub is not None else None,
        })
    out = pd.DataFrame(rows)
    out.to_sql("indicator_details", engine, if_exists="append", index=False)
    print(f"indicator_details: {len(out)} rows ({out['is_published'].sum()} published), "
          f"{skipped_orphans} orphans skipped (parent indicator missing from Item_2)")


def load_indicator_values(csv_dir, engine):
    df = read_csv(csv_dir, "Item_1_IndicatorValues_ByCountry-20260811_151007.csv")

    # Values whose detail was dropped above have to go too — same FK problem one
    # level down. This is the visible cost of the orphaned-details gap: ~1,806 of
    # 11,303 raw data points, all under unpublished details.
    known_details = _existing_ids(engine, "indicator_details", "indicator_detail_id")
    before = len(df)
    df = df[df["IndicatorDetailId"].isin(known_details)].reset_index(drop=True)
    skipped_orphans = before - len(df)

    period_parsed = df["Period"].apply(parse_period)
    out = pd.DataFrame({
        "value_id": df["DataPointId"],
        "indicator_detail_id": df["IndicatorDetailId"],
        "period_label": df["Period"],
        "period_date": [p[0] for p in period_parsed],
        "granularity": [p[1] for p in period_parsed],
        "country_en": df["Country"],
        "actual": df["Actual"].apply(to_float),
        "target": df["Target"].apply(to_float),
    })
    out.to_sql("indicator_values", engine, if_exists="append", index=False)
    print(f"indicator_values: {len(out)} rows, {skipped_orphans} skipped (detail row missing)")


def load_published_data_points(csv_dir, engine):
    df = read_csv(csv_dir, "P03_Published_DataPoints-20260811_151007.csv")
    period_parsed = df["Period"].apply(parse_period)
    out = pd.DataFrame({
        "published_data_point_id": df["PublishedDataPointId"],
        "published_indicator_detail_id": df["PublishedIndicatorDetailId"],
        "published_indicator_id": df["PublishedIndicatorId"],
        "interval_en": df["IntervalEN"],
        "period_label": df["Period"],
        "period_date": [p[0] for p in period_parsed],
        "granularity": [p[1] for p in period_parsed],
        "country_id": df["CountryId"],
        "country_en": df["CountryEN"],
        "actual": df["Actual"].apply(to_float),
        "target": df["Target"].apply(to_float),
        "outlook": df["Outlook"].apply(to_float),
        "monthly_mom_percent": df["MonthlyMoMPercent"].apply(to_float),
        "monthly_mom_pp": df["MonthlyMoMpp"].apply(to_float),
        "monthly_yoy_percent": df["MonthlyYoYPercent"].apply(to_float),
        "monthly_yoy_pp": df["MonthlyYoYpp"].apply(to_float),
        "quarterly_qoq_percent": df["QuarterlyQoQPercent"].apply(to_float),
        "quarterly_qoq_pp": df["QuarterlyQoQpp"].apply(to_float),
        "quarterly_yoy_percent": df["QuarterlyYoYPercent"].apply(to_float),
        "quarterly_yoy_pp": df["QuarterlyYoYpp"].apply(to_float),
        "yearly_yoy_percent": df["YearlyYoYPercent"].apply(to_float),
        "yearly_yoy_pp": df["YearlyYoYpp"].apply(to_float),
    })
    out.to_sql("published_data_points", engine, if_exists="append", index=False)
    print(f"published_data_points: {len(out)} rows")


def load_indicator_analysis(csv_dir, engine):
    item1 = read_csv(csv_dir, "Item_1_DataPointAnalysis-20260811_151006.csv")
    p04 = read_csv(csv_dir, "P04_Published_DataPointAnalysis-20260811_151008.csv")

    rows = []
    for _, r in item1.iterrows():
        rows.append({
            "analysis_id": r["IndicatorDetailDataPointId"],
            "ref_data_point_id": r["IndicatorDetailDataPointId"],
            "source": "item",
            "summary_en": html_to_text(fix_mojibake(r["Summary_EN"])),
            "detailed_analysis_en": html_to_text(fix_mojibake(r["DetailedAnalysis_EN"])),
            "npc_analysis_en": None,
            "benchmark_en": None,
        })
    for _, r in p04.iterrows():
        rows.append({
            "analysis_id": r["PublishedDataPointAnalysisId"],
            "ref_data_point_id": r["PublishedDataPointId"],
            "source": "published",
            "summary_en": html_to_text(fix_mojibake(r["SummaryEN"])),
            "detailed_analysis_en": html_to_text(fix_mojibake(r["DetailedAnalysisEN"])),
            "npc_analysis_en": html_to_text(fix_mojibake(r["NPCAnalysisEN"])),
            "benchmark_en": html_to_text(fix_mojibake(r["BenchmarkEN"])),
        })
    out = pd.DataFrame(rows)
    out.to_sql("indicator_analysis", engine, if_exists="append", index=False)
    print(f"indicator_analysis: {len(out)} rows")


def load_benchmark_countries(csv_dir, engine):
    df = read_csv(csv_dir, "P05_Published_BenchmarkCountries-20260811_151009.csv")
    out = pd.DataFrame({
        "published_benchmark_country_id": df["PublishedBenchmarkCountryId"],
        "published_indicator_detail_id": df["PublishedIndicatorDetailId"],
        "published_indicator_id": df["PublishedIndicatorId"],
        "country_en": df["CountryEN"],
        "country_code": df["CountryCode"],
    })
    out.to_sql("benchmark_countries", engine, if_exists="append", index=False)
    print(f"benchmark_countries: {len(out)} rows")


def load_lookups(csv_dir, engine):
    df = read_csv(csv_dir, "P12_Ref_Lookups_Common-20260811_151009.csv")
    out = pd.DataFrame({
        "lookup_type": df["LookupType"],
        "lookup_id": df["Id"],
        "name_en": df["NameEN"],
        "name_ar": df["NameAR"],
        "unique_code": df["UniqueId"],
    })
    out.to_sql("lookups", engine, if_exists="append", index=False)
    print(f"lookups: {len(out)} rows")


def load_priority_types(csv_dir, engine):
    df = read_csv(csv_dir, "P13_Ref_IndicatorPriorityTypes-20260811_151009.csv")
    out = pd.DataFrame({
        "priority_type_id": df["Id"],
        "name_en": df["NameEN"],
        "name_ar": df["NameAR"],
        "unique_code": df["UniqueId"],
        "color_class": df["ColorClass"],
        "ranking": df["Ranking"].apply(lambda x: int(to_float(x)) if to_float(x) is not None else None),
    })
    out.to_sql("priority_types", engine, if_exists="append", index=False)
    print(f"priority_types: {len(out)} rows")


def load_indicator_dashboards(csv_dir, engine):
    df = read_csv(csv_dir, "P09_Published_Dashboards-20260811_151009.csv")
    out = pd.DataFrame({
        "published_indicator_dashboard_id": df["PublishedIndicatorDashboardId"],
        "published_indicator_id": df["PublishedIndicatorId"],
        "dashboard_id": df["DashboardId"],
        "dashboard_name_en": df["DashboardNameEN"],
        "sort_order": df["SortOrder"].apply(lambda x: int(to_float(x)) if to_float(x) is not None else None),
        "entity_classification_name": df["EntityClassificationName"],
        "entity_id": df["EntityId"],
    })
    out.to_sql("indicator_dashboards", engine, if_exists="append", index=False)
    print(f"indicator_dashboards: {len(out)} rows")


def load_chart_sub_indicators(csv_dir, engine):
    df = read_csv(csv_dir, "P07a_Published_Charts_SubIndicators_Split-20260811_151009.csv")
    out = pd.DataFrame({
        "published_indicator_chart_id": df["PublishedIndicatorChartId"],
        "published_indicator_id": df["PublishedIndicatorId"],
        "list_type": df["ListType"],
        "member_indicator_detail_id": df["MemberIndicatorDetailId"],
    })
    out.to_sql("chart_sub_indicators", engine, if_exists="append", index=False)
    print(f"chart_sub_indicators: {len(out)} rows")


def load_chart_alternate_views(csv_dir, engine):
    df = read_csv(csv_dir, "P07b_Published_Charts_AlternateViews_Split-20260811_151009.csv")
    out = pd.DataFrame({
        "published_indicator_chart_id": df["PublishedIndicatorChartId"],
        "published_indicator_id": df["PublishedIndicatorId"],
        "alternate_view_id": df["AlternateViewId"],
        "alternate_view_en": df["AlternateViewEN"],
        "alternate_view_code": df["AlternateViewCode"],
    })
    out.to_sql("chart_alternate_views", engine, if_exists="append", index=False)
    print(f"chart_alternate_views: {len(out)} rows")


def load_additional_charts(csv_dir, engine):
    df = read_csv(csv_dir, "P08_Published_AdditionalCharts-20260811_151009.csv")
    out = pd.DataFrame({
        "published_additional_chart_id": df["PublishedAdditionalChartId"],
        "published_indicator_id": df["PublishedIndicatorId"],
        "published_indicator_detail_id": df["PublishedIndicatorDetailId"],
        "chart_type_en": df["ChartTypeEN"],
        "interval_en": df["IntervalEN"],
    })
    out.to_sql("additional_charts", engine, if_exists="append", index=False)
    print(f"additional_charts: {len(out)} rows")


def load_additional_chart_countries(csv_dir, engine):
    df = read_csv(csv_dir, "P08a_Published_AdditionalCharts_Countries_Split-20260811_151009.csv")
    out = pd.DataFrame({
        "published_additional_chart_id": df["PublishedAdditionalChartId"],
        "published_indicator_id": df["PublishedIndicatorId"],
        "published_indicator_detail_id": df["PublishedIndicatorDetailId"],
        "country_en": df["CountryEN"],
        "country_code": df["CountryCode"],
    })
    out.to_sql("additional_chart_countries", engine, if_exists="append", index=False)
    print(f"additional_chart_countries: {len(out)} rows")


def load_champions(csv_dir, engine):
    df = read_csv(csv_dir, "Champions.csv")
    out = pd.DataFrame({
        "champion_id": df["Id"],
        "name_en": df["NameEN"],
        "name_ar": df["NameAR"],
        "description_en": df["DescriptionEN"].apply(lambda x: strip_html(maybe_b64_decode(x)) if x else x),
        "description_ar": df["DescriptionAR"],
        "is_active": df["IsActive"].apply(to_bool),
        "is_ministry": df["IsMinistry"].apply(to_bool),
    })
    out.to_sql("champions", engine, if_exists="append", index=False)
    print(f"champions: {len(out)} rows")


def _decode_rich(series):
    return series.apply(lambda x: strip_html(maybe_b64_decode(x)) if x else x)


def load_sectors(csv_dir, engine):
    df = read_csv(csv_dir, "Sectors.csv")
    out = pd.DataFrame({
        "sector_id": df["Id"],
        "champion_id": df["ChampionId"],
        "name_en": df["NameEN"],
        "name_ar": df["NameAR"],
        "description_en": _decode_rich(df["DescriptionEN"]),
        "description_ar": df["DescriptionAR"],
        "achievement_en": _decode_rich(df["AchievementEN"]),
        "achievement_ar": df["AchievementAR"],
        "challenges_en": _decode_rich(df["ChallengesEN"]),
        "challenges_ar": df["ChallengesAR"],
        "highlight_en": _decode_rich(df["HighlightEN"]),
        "highlight_ar": df["HighlightAR"],
        "insight_en": _decode_rich(df["InsightEN"]),
        "insight_ar": df["InsightAR"],
        "is_active": df["IsActive"].apply(to_bool),
        "stage_of_monitoring_id": df["StageOfMonitoringId"],
        "status_id": df["StatusId"],
        "workflow_id": df["WorkflowId"],
    })
    out.to_sql("sectors", engine, if_exists="append", index=False)
    print(f"sectors: {len(out)} rows")


def load_general_entities(csv_dir, engine):
    df = read_csv(csv_dir, "General_Entities.csv")
    out = pd.DataFrame({
        "entity_id": df["Id"],
        "champion_id": df["ChampionId"],
        "name_en": df["NameEN"],
        "name_ar": df["NameAR"],
        "description_en": _decode_rich(df["DescriptionEN"]),
        "description_ar": df["DescriptionAR"],
        "achievement_en": _decode_rich(df["AchievementEN"]),
        "achievement_ar": df["AchievementAR"],
        "challenges_en": _decode_rich(df["ChallengesEN"]),
        "challenges_ar": df["ChallengesAR"],
        "highlight_en": _decode_rich(df["HighlightEN"]),
        "highlight_ar": df["HighlightAR"],
        "insight_en": _decode_rich(df["InsightEN"]),
        "insight_ar": df["InsightAR"],
        "entity_classification": df["EntityClassification"],
        "is_active": df["IsActive"].apply(to_bool),
        "stage_of_monitoring_id": df["StageOfMonitoringId"],
        "status_id": df["StatusId"],
        "workflow_id": df["WorkflowId"],
    })
    out.to_sql("general_entities", engine, if_exists="append", index=False)
    print(f"general_entities: {len(out)} rows")


def load_articles(csv_dir, engine):
    df = read_csv(csv_dir, "Articles.csv")
    out = pd.DataFrame({
        "article_id": df["Id"],
        "title_en": df["TitleSimpleEN"].where(df["TitleSimpleEN"].notna(), df["TitleEN"].apply(strip_html)),
        "title_ar": df["TitleSimpleAR"],
        "content_en": df["ContentEN"].apply(lambda x: strip_html(fix_mojibake(x)) if x else x),
        "content_ar": df["ContentAR"],
        "published": df["Published"].apply(to_bool),
        "featured": df["Featured"].apply(to_bool),
        "is_active": df["IsActive"].apply(to_bool),
        "article_date": pd.to_datetime(df["ArticleDate"], errors="coerce").dt.date,
        "published_date": pd.to_datetime(df["PublishedDate"], errors="coerce").dt.date,
    })
    out.to_sql("articles", engine, if_exists="append", index=False)
    print(f"articles: {len(out)} rows")


TABLES_IN_LOAD_ORDER = [
    "chart_alternate_views", "chart_sub_indicators", "additional_chart_countries",
    "additional_charts", "indicator_dashboards", "benchmark_countries",
    "published_data_points", "indicator_analysis",
    "indicator_values", "indicator_details", "indicators",
    "general_entities", "sectors", "champions", "articles",
    "lookups", "priority_types", "intervals", "countries",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", required=True, type=Path)
    parser.add_argument("--dsn", required=True)
    args = parser.parse_args()

    engine = create_engine(args.dsn)

    with engine.begin() as conn:
        for t in TABLES_IN_LOAD_ORDER:
            conn.execute(text(f"TRUNCATE TABLE {t} CASCADE"))

    # Order matters: parents before children (FK constraints)
    load_countries(args.csv_dir, engine)
    load_intervals(args.csv_dir, engine)
    load_priority_types(args.csv_dir, engine)
    load_lookups(args.csv_dir, engine)
    load_indicators(args.csv_dir, engine)
    load_indicator_details(args.csv_dir, engine)
    load_indicator_values(args.csv_dir, engine)
    load_published_data_points(args.csv_dir, engine)
    load_indicator_analysis(args.csv_dir, engine)
    load_benchmark_countries(args.csv_dir, engine)
    load_indicator_dashboards(args.csv_dir, engine)
    load_champions(args.csv_dir, engine)
    load_sectors(args.csv_dir, engine)
    load_general_entities(args.csv_dir, engine)
    load_articles(args.csv_dir, engine)
    load_chart_sub_indicators(args.csv_dir, engine)
    load_chart_alternate_views(args.csv_dir, engine)
    load_additional_charts(args.csv_dir, engine)
    load_additional_chart_countries(args.csv_dir, engine)

    print("\nETL complete.")


if __name__ == "__main__":
    main()
