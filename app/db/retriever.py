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
import re
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
                       yearly_yoy_percent,
                       -- The percentage-POINT columns, which were never
                       -- selected. preferred_change_field(as_points=True)
                       -- therefore looked up a key that was not in the row and
                       -- always returned None, everywhere it was called.
                       -- It matters for ratios: Public Debt as a Percentage of
                       -- GDP publishes quarterly_yoy_pp and leaves
                       -- quarterly_yoy_percent empty, so "is public debt
                       -- improving?" was answered "no published change is
                       -- available" about an indicator whose change SCAI
                       -- publishes.
                       monthly_mom_pp, monthly_yoy_pp,
                       quarterly_qoq_pp, quarterly_yoy_pp,
                       yearly_yoy_pp
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


ANALYSIS_FIELDS = (("summary", "summary"), ("detailed_analysis", "detailed"),
                    ("npc_analysis", "npc_analysis"), ("benchmark", "benchmark"))


def analysis_text(analysis: dict, language: str = "en") -> dict:
    """SCAI's commentary in the reader's language, falling back to English.

    P04 carries all four fields in Arabic and the ETL never loaded them, so an
    Arabic question about inflation was answered with SCAI's English analysis —
    the one part of the reply that is quoted verbatim and cannot be rephrased
    by the Composer.

    Coverage is partial: 667 of 1031 published rows have an Arabic summary and
    626 have the detail. Falling back per FIELD rather than per row means a row
    with Arabic commentary and an English benchmark shows both, instead of
    dropping whichever the reader cannot have.
    """
    arabic = str(language).lower().startswith("ar")
    out = {}
    for base, key in ANALYSIS_FIELDS:
        value = ""
        if arabic:
            value = (analysis.get(f"{base}_ar") or "").strip()
        if not value:
            value = (analysis.get(f"{base}_en") or "").strip()
        if value and value != "-":
            out[key] = value
    return out


def get_analysis(ref_data_point_id: str) -> Optional[dict]:
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT summary_en, detailed_analysis_en, npc_analysis_en, benchmark_en,
                   summary_ar, detailed_analysis_ar, npc_analysis_ar, benchmark_ar, source
            FROM indicator_analysis WHERE ref_data_point_id = :rid
            ORDER BY (source = 'published') DESC LIMIT 1
        """), {"rid": ref_data_point_id}).fetchone()
    return dict(row._mapping) if row else None


# Words that carry no retrieval signal and, because the 'simple' text-search
# config does no stopword removal, would otherwise be required to match.
_LEXICAL_STOPWORDS = {
    "the", "and", "for", "what", "has", "have", "about", "written", "wrote",
    "said", "say", "says", "tell", "all", "any", "with", "from", "that", "this",
    "does", "did", "council", "scai",
    "عن", "في", "من", "ما", "هل", "ماذا", "على", "الى", "إلى", "هو", "هي", "المجلس",
}


def _lexical_tsquery(query_text: Optional[str]) -> Optional[str]:
    """Builds an OR tsquery from the question's meaningful words.

    plainto_tsquery ANDs every token, which made the lexical arm almost
    useless: "In-Country Value programme" required the literal token
    'programme' and the article says "Program", so it matched nothing; and
    "the trade war" required the literal 'the', because the 'simple' config
    strips no stopwords. Matching ANY term and ranking by how many hit is far
    more forgiving, and ranking is what decides the order anyway.
    """
    if not query_text:
        return None
    tokens = [t for t in re.findall(r"[\w؀-ۿ]{3,}", query_text.lower())
              if t not in _LEXICAL_STOPWORDS]
    # Deduplicate, keep order, and cap so one long question cannot build a
    # pathological query.
    seen, terms = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            terms.append(t)
    return " | ".join(terms[:12]) or None


def search_article_chunks(query_embedding: list[float], language: str = "en",
                           limit: int = 6, query_text: Optional[str] = None) -> list[dict]:
    """Passages from SCAI articles most relevant to a question.

    Hybrid, not purely semantic. Embeddings handle the paraphrase case that is
    the whole point here ("what does SCAI think about trade wars" finding an
    article titled "Financial Stability Implications of Tariffs"), but they are
    weak on rare exact tokens — a programme name like "Tawteen" or an acronym
    like "ICV" carries little semantic signal, and cosine similarity will
    happily rank a thematically-similar passage above the one that actually
    names it. Full-text search covers exactly that case, so both run and the
    results are merged.

    Distance is returned alongside each passage. The caller needs it: a nearest
    neighbour is always returned, however irrelevant, so "nearest" has to be
    checked against a threshold before anything is claimed from it.
    """
    vector_literal = "[" + ",".join(f"{v:.6f}" for v in query_embedding) + "]"
    results: dict[str, dict] = {}

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT c.chunk_id, c.article_id, c.chunk_index, c.content, c.language,
                   a.title_en, a.title_ar, a.published_date, a.article_date,
                   c.embedding <=> CAST(:qvec AS vector) AS distance
            FROM article_chunks c
            JOIN articles a ON a.article_id = c.article_id
            WHERE c.language = :lang
            ORDER BY c.embedding <=> CAST(:qvec AS vector)
            LIMIT :limit
        """), {"qvec": vector_literal, "lang": language, "limit": limit}).fetchall()
        for r in rows:
            d = dict(r._mapping)
            d["match"] = "semantic"
            results[d["chunk_id"]] = d

        tsquery = _lexical_tsquery(query_text)
        if tsquery:
            # Searches the TITLE as well as the body. "In-Country Value Program"
            # is the article's title, not a phrase in any one chunk, so a
            # body-only search could not find the article the question named.
            lexical = conn.execute(text("""
                SELECT c.chunk_id, c.article_id, c.chunk_index, c.content, c.language,
                       a.title_en, a.title_ar, a.published_date, a.article_date,
                       c.embedding <=> CAST(:qvec AS vector) AS distance
                FROM article_chunks c
                JOIN articles a ON a.article_id = c.article_id
                WHERE c.language = :lang
                  AND to_tsvector('simple',
                        coalesce(a.title_en, '') || ' ' || coalesce(a.title_ar, '') ||
                        ' ' || c.content) @@ to_tsquery('simple', :tsq)
                ORDER BY ts_rank(to_tsvector('simple',
                        coalesce(a.title_en, '') || ' ' || coalesce(a.title_ar, '') ||
                        ' ' || c.content), to_tsquery('simple', :tsq)) DESC
                LIMIT :limit
            """), {"qvec": vector_literal, "lang": language,
                    "tsq": tsquery, "limit": limit}).fetchall()
            for r in lexical:
                d = dict(r._mapping)
                if d["chunk_id"] in results:
                    results[d["chunk_id"]]["match"] = "semantic+lexical"
                else:
                    d["match"] = "lexical"
                    results[d["chunk_id"]] = d

    ordered = sorted(results.values(), key=lambda d: float(d["distance"]))
    return ordered[:limit]


def count_article_chunks() -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM article_chunks")).scalar() or 0


def get_analysis_for_data_points(record_ids: list[str]) -> dict[str, dict]:
    """SCAI's own written commentary for specific data points, keyed by the
    record id of the point it belongs to.

    indicator_analysis.ref_data_point_id holds a published_data_point_id (for
    source='published') or an indicator value id (for source='item'), which are
    exactly the record ids already carried on every citation — so the narration
    layer can attach the Council's text to the precise reading it describes,
    rather than to the indicator in general.

    This text is written by SCAI analysts. It is returned verbatim and must
    never be paraphrased or merged into generated prose: the whole point of the
    /read endpoint is that the reader can tell Council analysis from narration."""
    if not record_ids:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT ref_data_point_id, source, summary_en, detailed_analysis_en,
                   npc_analysis_en, benchmark_en,
                   summary_ar, detailed_analysis_ar, npc_analysis_ar, benchmark_ar
            FROM indicator_analysis
            WHERE ref_data_point_id = ANY(:ids)
        """), {"ids": list(record_ids)}).fetchall()
    out = {}
    for r in rows:
        d = dict(r._mapping)
        # Prefer the published row when a point has both; it carries the extra
        # NPC and benchmark commentary that the working-data row lacks.
        key = d["ref_data_point_id"]
        if key not in out or d["source"] == "published":
            out[key] = d
    return out


def has_country_breakdown(published_indicator_detail_id: Optional[str],
                           indicator_detail_id: str) -> bool:
    """Whether this indicator is reported per country at all.

    Most are not: "Number of International Visitors" has 125 points, every one
    of them a Qatar-level total. Asking it for a per-country figure is not a
    gap in coverage, it is a question the series cannot answer in any period —
    a distinction the answer has to make, because "no approved data found for
    the requested countries" reads as "those countries are missing".
    """
    # A non-empty country_en is not proof of a breakdown. The raw layer uses the
    # literal "National / Overall" to mean "no breakdown" — it is the only such
    # placeholder in the whole dataset, and it is exactly what "Number of
    # International Visitors" carries on all 125 of its rows. Checking for a
    # non-empty string would report that series as broken down by country and
    # then find nothing for any country named.
    with engine.connect() as conn:
        if published_indicator_detail_id:
            found = conn.execute(text("""
                SELECT 1 FROM published_data_points p
                WHERE p.published_indicator_detail_id = :pdid
                  AND p.country_en IS NOT NULL AND p.country_en <> ''
                  AND EXISTS (SELECT 1 FROM countries c
                               WHERE LOWER(c.name_en) = LOWER(p.country_en))
                LIMIT 1
            """), {"pdid": published_indicator_detail_id}).fetchone()
            if found:
                return True
        found = conn.execute(text("""
            SELECT 1 FROM indicator_values v
            WHERE v.indicator_detail_id = :did
              AND v.country_en IS NOT NULL AND v.country_en <> ''
              AND EXISTS (SELECT 1 FROM countries c
                           WHERE LOWER(c.name_en) = LOWER(v.country_en))
            LIMIT 1
        """), {"did": indicator_detail_id}).fetchone()
    return bool(found)


def get_benchmark_countries(published_indicator_detail_id: Optional[str]) -> list[str]:
    """SCAI's own chosen comparison set for one indicator, from P05.

    Used when a ranking question names no countries at all ("which country has
    the lowest inflation") — previously that produced a one-row "ranking" of
    Qatar against itself. The set is SCAI's editorial choice per indicator, not
    a list this code invents, and not a fixed global default: for Inflation it
    is the GCC plus Norway, Singapore and Switzerland.

    Qatar is filtered out because it is stored as a blank country on the
    national series and is added separately by the caller.

    Only 21 of the published indicators define benchmarks, so an empty result
    is normal and means "SCAI has not designated comparators for this one"."""
    if not published_indicator_detail_id:
        return []
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT country_en FROM benchmark_countries
            WHERE published_indicator_detail_id = :pdid
              AND country_en IS NOT NULL AND country_en <> ''
            ORDER BY country_en
        """), {"pdid": published_indicator_detail_id}).fetchall()
    return [r[0] for r in rows if r[0].strip().lower() != "qatar"]


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


def list_indicator_types() -> list[str]:
    """The eight indicator_type_en values, for matching a question against."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT DISTINCT indicator_type_en FROM indicators "
            "WHERE indicator_type_en IS NOT NULL AND indicator_type_en <> '' ORDER BY 1"
        )).fetchall()
    return [r[0] for r in rows]


def list_sector_names() -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT DISTINCT name_en FROM sectors WHERE is_active AND name_en IS NOT NULL ORDER BY 1"
        )).fetchall()
    return [r[0] for r in rows]


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


# Latest actual per published indicator detail, with the target for that same
# period. DISTINCT ON takes the newest row that HAS an actual, for the same
# reason latest_value() does: many series carry target-only rows years ahead,
# and ordering by date alone would report a 2030 target as a current reading.
_SCOPE_PERFORMANCE_SQL = """
WITH scoped AS ({scope_cte}),
latest AS (
    SELECT DISTINCT ON (p.published_indicator_detail_id)
           p.published_indicator_detail_id AS detail_id,
           p.period_label, p.period_date, p.granularity,
           p.actual, p.target AS period_target,
           p.published_data_point_id AS record_id,
           -- SCAI's own vetted year-on-year figures, never a fresh derivation
           -- (F-022..F-026). Which one applies depends on the granularity, so
           -- all three come back and preferred_change_field picks.
           p.monthly_yoy_percent, p.quarterly_yoy_percent, p.yearly_yoy_percent,
           -- And the percentage-point forms, for the ratio indicators that
           -- publish only those.
           p.monthly_yoy_pp, p.quarterly_yoy_pp, p.yearly_yoy_pp
    FROM published_data_points p
    WHERE p.actual IS NOT NULL
      AND (p.country_en IS NULL OR p.country_en = '' OR p.country_en = 'National / Overall')
    ORDER BY p.published_indicator_detail_id, p.period_date DESC
)
SELECT i.name_en AS indicator, i.indicator_id AS indicator_record_id,
       d.unit_en, d.unit_ar, d.format, d.polarity_en,
       -- What the DETAIL is called, and how many the indicator has. IsMain is
       -- not always the headline: Workforce (Economically Active) has four
       -- details and its main one is "High Skilled Blue Collar", a single
       -- component reported under the parent's name as though it were the
       -- total. 0.593054m was being shown as Qatar's workforce; the four
       -- components sum to 2.240506m.
       d.name_en AS detail_name,
       (SELECT COUNT(*) FROM indicator_details sd
         WHERE sd.indicator_id = i.indicator_id AND sd.is_published) AS sibling_count,
       d.target_value, d.target_year, d.baseline_value, d.baseline_year,
       l.period_label, l.actual, l.period_target, l.record_id, l.granularity,
       l.monthly_yoy_percent, l.quarterly_yoy_percent, l.yearly_yoy_percent,
       l.monthly_yoy_pp, l.quarterly_yoy_pp, l.yearly_yoy_pp
FROM scoped s
JOIN indicators i ON i.published_indicator_id = s.published_indicator_id
JOIN indicator_details d ON d.indicator_id = i.indicator_id AND d.is_published = TRUE
-- One row per INDICATOR, not per detail. Several published indicators carry
-- sub-breakdowns as extra details — Real GDP has Hydrocarbon and
-- Non-Hydrocarbon beneath it — and without this filter "which national
-- indicators are increasing?" returned Real GDP three times under one name,
-- listing a total alongside its own components as though they were peers.
-- Matching the detail name against the indicator name instead is NOT a valid
-- substitute, and was tried: indicator names carry a qualifier their detail
-- does not ("Sector Contribution To GDP (Education)" vs "Sector Contribution
-- To GDP"), so that rule disagrees with IsMain on 39 of the 289 published
-- details and drops them. This column has to be loaded.
AND d.is_main IS TRUE
LEFT JOIN latest l ON l.detail_id = d.published_detail_id
ORDER BY i.name_en
"""

_SCOPE_CTE = {
    "sector": """
        SELECT DISTINCT dd.published_indicator_id
        FROM indicator_dashboards dd
        JOIN sectors sec ON sec.sector_id = dd.entity_id
                        AND dd.entity_classification_name = 'Sectors'
        WHERE sec.name_en ILIKE :scope
    """,
    "type": """
        SELECT DISTINCT published_indicator_id
        FROM indicators
        WHERE indicator_type_en ILIKE :scope AND is_published = TRUE
    """,
}


def get_scope_performance(kind: str, scope: str) -> list[dict]:
    """Every published indicator in a sector (or indicator type), with its most
    recent reading, its target, and which DIRECTION counts as good.

    The polarity is the point. Ranking a set of indicators on their raw values
    would be meaningless — this sector alone mixes a cost in thousands of
    riyals, a headcount ratio, a PISA rank and six percentages — and for two of
    them (cost per student, PISA rank) a lower number is the better outcome.
    Comparison is only defensible against each indicator's OWN target, which is
    what this returns the pieces for.
    """
    cte = _SCOPE_CTE.get(kind)
    if not cte:
        return []
    with engine.connect() as conn:
        rows = conn.execute(text(_SCOPE_PERFORMANCE_SQL.format(scope_cte=cte)),
                            {"scope": f"%{scope}%"}).fetchall()
    return [dict(r._mapping) for r in rows]
