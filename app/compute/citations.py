"""
Citations — deterministic source attribution, built entirely outside the
LLM. Every fact the Composer is given comes with a citation; the final
"Sources:" footer on every answer is rendered by render_sources_footer()
below, NOT by asking the LLM to remember to include it. That's the same
principle as the numeric verifier: don't trust the LLM to reliably follow
an instruction that matters this much — enforce it in code.

A citation identifies: which indicator, which underlying data source
(e.g. 'World Bank', 'Qatar Tourism' — from indicator_details.data_source_en),
which table it was read from (published_data_points = SCAI-vetted, curated;
indicator_values = raw/working data), the exact record id, the period, and
the country (Qatar if domestic).
"""
from dataclasses import dataclass, asdict
from typing import Optional


TABLE_LABELS = {
    "published_data_points": "SCAI Approved/Published Data",
    "indicator_values": "SCAI Working Data (not yet published)",
    "indicators": "SCAI Indicator Catalog",
    "sectors": "SCAI Sector Monitoring",
    "general_entities": "SCAI Entity Monitoring",
    "champions": "SCAI Champion Registry",
    "articles": "SCAI Published Articles",
    "indicator_analysis": "SCAI Analyst Commentary",
}


@dataclass
class Citation:
    indicator: Optional[str]
    data_source: Optional[str]     # original source, e.g. 'World Bank' — None if not recorded
    table: str                     # which internal table, see TABLE_LABELS
    record_id: Optional[str]
    period_label: Optional[str] = None
    country: Optional[str] = None  # None/"" treated as Qatar


def build_citation(row: dict, indicator_name: str, data_source_en: Optional[str],
                    table: str, country: Optional[str] = None) -> Citation:
    return Citation(
        indicator=indicator_name,
        data_source=data_source_en,
        table=table,
        record_id=row.get("record_id"),
        period_label=row.get("period_label"),
        country=country or row.get("country_en") or "Qatar",
    )


def citations_for_rows(rows: list[dict], indicator_name: str, data_source_en: Optional[str],
                        country: Optional[str] = None) -> list[Citation]:
    return [
        build_citation(r, indicator_name, data_source_en, r.get("source_table", "unknown"), country)
        for r in rows
    ]


def _group_key(c: Citation):
    return (c.indicator, c.data_source, c.table, c.country)


def format_citation_group(indicator, data_source, table, country, periods: list[str], record_ids: list[str]) -> str:
    table_label = TABLE_LABELS.get(table, table)
    source_part = f" — original source: {data_source}" if data_source else ""
    country_part = f" ({country})" if country and country != "Qatar" else ""
    periods_sorted = periods  # already in retrieval order (chronological)
    if len(periods_sorted) == 1:
        period_part = periods_sorted[0]
    elif len(periods_sorted) <= 3:
        period_part = ", ".join(periods_sorted)
    else:
        period_part = f"{periods_sorted[0]} to {periods_sorted[-1]} ({len(periods_sorted)} data points)"
    return f"{indicator}{country_part} — {table_label}{source_part} — {period_part}"


def render_sources_footer(citations: list[Citation]) -> str:
    """Deduplicates and groups citations into a short, readable footer.
    Always deterministic — never touched by the LLM."""
    if not citations:
        return ""

    groups: dict[tuple, dict] = {}
    for c in citations:
        key = _group_key(c)
        if key not in groups:
            groups[key] = {"periods": [], "record_ids": []}
        if c.period_label and c.period_label not in groups[key]["periods"]:
            groups[key]["periods"].append(c.period_label)
        if c.record_id:
            groups[key]["record_ids"].append(c.record_id)

    lines = []
    for (indicator, data_source, table, country), data in groups.items():
        lines.append("• " + format_citation_group(indicator, data_source, table, country,
                                                    data["periods"], data["record_ids"]))
    return "Sources:\n" + "\n".join(lines)


def citations_to_dicts(citations: list[Citation]) -> list[dict]:
    """Machine-readable form (full record ids included) for API consumers
    who need exact provenance rather than the prose footer — e.g. a
    front-end that wants to link each figure back to its source record."""
    return [asdict(c) for c in citations]
