"""
Retriever — every query here is a fixed, parameterized statement written by
a person, not generated per-request by an LLM. This is the core change from
the earlier design: the old Text-to-SQL agent could construct an arbitrary
join/filter combination per question, which is exactly the kind of freedom
that let the previous system's failures happen (wrong period picked, wrong
country set, wrong frequency). These functions take already-resolved,
validated parameters (an indicator_detail_id, explicit dates, an explicit
country list) and can only return what they're written to return.
"""
from datetime import date
from typing import Optional

from sqlalchemy import text
from app.db.executor import engine


def get_available_granularities(published_indicator_detail_id: Optional[str], indicator_detail_id: str) -> set[str]:
    with engine.connect() as conn:
        if published_indicator_detail_id:
            rows = conn.execute(text("""
                SELECT DISTINCT granularity FROM published_data_points
                WHERE published_indicator_detail_id = :pdid AND (country_en IS NULL OR country_en = '')
            """), {"pdid": published_indicator_detail_id}).fetchall()
            if rows:
                return {r[0] for r in rows if r[0]}
        rows = conn.execute(text("""
            SELECT DISTINCT granularity FROM indicator_values
            WHERE indicator_detail_id = :did
        """), {"did": indicator_detail_id}).fetchall()
        return {r[0] for r in rows if r[0]}


def get_series(indicator_detail_id: str, published_indicator_detail_id: Optional[str],
                granularity: str, start_date: Optional[date] = None, end_date: Optional[date] = None,
                country_en: Optional[str] = None) -> list[dict]:
    """Returns the raw time series for one indicator/country/granularity,
    preferring published_data_points (has vetted precomputed change fields)
    and falling back to indicator_values when the indicator isn't published."""
    params = {"gran": granularity, "country": country_en or ""}
    date_clause = ""
    if start_date:
        params["start"] = start_date
        date_clause += " AND period_date >= :start"
    if end_date:
        params["end"] = end_date
        date_clause += " AND period_date <= :end"

    with engine.connect() as conn:
        if published_indicator_detail_id:
            rows = conn.execute(text(f"""
                SELECT published_data_point_id AS record_id,
                       period_date, period_label, granularity, country_en,
                       actual, target, outlook,
                       monthly_mom_percent, monthly_yoy_percent,
                       quarterly_qoq_percent, quarterly_yoy_percent,
                       yearly_yoy_percent
                FROM published_data_points
                WHERE published_indicator_detail_id = :pdid
                  AND granularity = :gran
                  AND COALESCE(country_en, '') = :country
                  {date_clause}
                ORDER BY period_date ASC
            """), {**params, "pdid": published_indicator_detail_id}).fetchall()
            if rows:
                result = [dict(r._mapping) for r in rows]
                for r in result:
                    r["source_table"] = "published_data_points"
                return result

        rows = conn.execute(text(f"""
            SELECT value_id AS record_id,
                   period_date, period_label, granularity, country_en, actual, target
            FROM indicator_values
            WHERE indicator_detail_id = :did
              AND granularity = :gran
              AND COALESCE(country_en, '') = :country
              {date_clause}
            ORDER BY period_date ASC
        """), {**params, "did": indicator_detail_id}).fetchall()
        result = [dict(r._mapping) for r in rows]
        for r in result:
            r["source_table"] = "indicator_values"
        return result


def get_series_multi_country(indicator_detail_id: str, published_indicator_detail_id: Optional[str],
                              granularity: str, countries: list[str],
                              start_date: Optional[date] = None, end_date: Optional[date] = None,
                              include_qatar: bool = True) -> dict[str, list[dict]]:
    """One series per requested country (plus Qatar if requested). Countries
    with no data come back as an empty list — the caller is responsible for
    turning that into an honest 'no approved data for X' line, never for
    silently dropping the row or inventing one."""
    result = {}
    if include_qatar:
        result["Qatar"] = get_series(indicator_detail_id, published_indicator_detail_id,
                                      granularity, start_date, end_date, country_en=None)
    for c in countries:
        result[c] = get_series(indicator_detail_id, published_indicator_detail_id,
                                granularity, start_date, end_date, country_en=c)
    return result


def get_analysis(ref_data_point_id: str) -> Optional[dict]:
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT summary_en, detailed_analysis_en, npc_analysis_en, benchmark_en, source
            FROM indicator_analysis WHERE ref_data_point_id = :rid
            ORDER BY (source = 'published') DESC LIMIT 1
        """), {"rid": ref_data_point_id}).fetchone()
    return dict(row._mapping) if row else None


def get_sector_indicators(sector_name_query: str) -> list[dict]:
    """Cross-domain lookup via the indicator_dashboards bridge table.
    Fixes: 'which indicators track sector X' style questions that previously
    had no path at all since indicators and sectors were disconnected."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT i.name_en, i.indicator_id AS record_id, s.name_en AS sector_name,
                             s.sector_id AS sector_record_id
            FROM indicator_dashboards dd
            JOIN sectors s ON s.sector_id = dd.entity_id AND dd.entity_classification_name = 'Sectors'
            JOIN indicators i ON i.published_indicator_id = dd.published_indicator_id
            WHERE s.name_en ILIKE :q
            ORDER BY i.name_en
        """), {"q": f"%{sector_name_query}%"}).fetchall()
    return [dict(r._mapping) for r in rows]


def count_indicators_by_type(indicator_type_en: str) -> list[dict]:
    """Deterministic count+list — fixes F-032 ('how many indicators in
    diversification target') where the old system gave a 'random' answer
    instead of a real count."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT name_en, indicator_id AS record_id FROM indicators
            WHERE indicator_type_en ILIKE :t AND is_published = TRUE
            ORDER BY name_en
        """), {"t": f"%{indicator_type_en}%"}).fetchall()
    return [dict(r._mapping) for r in rows]
