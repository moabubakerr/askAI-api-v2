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

# The same labels in Arabic. The footer is appended to EVERY answer, so while it
# was English-only an Arabic answer ended in a block of English — the same defect
# answers_in_language() exists to catch in the prose, arriving through the one
# path that never passes through the model. Indicator and country names stay
# verbatim in either language, exactly as composer rule 4 requires of the prose.
TABLE_LABELS_AR = {
    "published_data_points": "بيانات المجلس المعتمدة/المنشورة",
    "indicator_values": "بيانات المجلس قيد العمل (غير منشورة بعد)",
    "indicators": "كتالوج مؤشرات المجلس",
    "sectors": "متابعة قطاعات المجلس",
    "general_entities": "متابعة جهات المجلس",
    "champions": "سجل الجهات الراعية بالمجلس",
    "articles": "مقالات المجلس المنشورة",
    "indicator_analysis": "تحليل محللي المجلس",
}

_FOOTER_STRINGS = {
    "en": {"sources": "Sources:", "original": "original source", "to": "to",
            "points": "data points"},
    "ar": {"sources": "المصادر:", "original": "المصدر الأصلي", "to": "إلى",
            "points": "نقطة بيانات"},
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


def format_citation_group(indicator, data_source, table, country, periods: list[str],
                           record_ids: list[str], language: str = "en") -> str:
    arabic = str(language).lower().startswith("ar")
    words = _FOOTER_STRINGS["ar" if arabic else "en"]
    table_label = (TABLE_LABELS_AR if arabic else TABLE_LABELS).get(table) \
        or TABLE_LABELS.get(table, table)
    source_part = f" — {words['original']}: {data_source}" if data_source else ""
    country_part = f" ({country})" if country and country != "Qatar" else ""
    periods_sorted = periods  # already in retrieval order (chronological)
    if len(periods_sorted) == 1:
        period_part = periods_sorted[0]
    elif len(periods_sorted) <= 3:
        period_part = ", ".join(periods_sorted)
    else:
        period_part = (f"{periods_sorted[0]} {words['to']} {periods_sorted[-1]} "
                       f"({len(periods_sorted)} {words['points']})")
    return f"{indicator}{country_part} — {table_label}{source_part} — {period_part}"


def render_sources_footer(citations: list[Citation], language: str = "en") -> str:
    """Deduplicates and groups citations into a short, readable footer.
    Always deterministic — never touched by the LLM.

    The heading is italicised and the lines are markdown list items, because the
    answer above is now markdown by contract (composer rules 22-28) and a plain
    "Sources:" under a formatted answer reads as though the response ran out
    rather than finished. Record ids stay out of the prose — a policymaker has no
    use for a UUID — and remain available in full through citations_to_dicts().
    """
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

    words = _FOOTER_STRINGS["ar" if str(language).lower().startswith("ar") else "en"]
    lines = []
    for (indicator, data_source, table, country), data in groups.items():
        lines.append("- " + format_citation_group(indicator, data_source, table, country,
                                                    data["periods"], data["record_ids"],
                                                    language))
    return f"*{words['sources']}*\n" + "\n".join(lines)


def citations_to_dicts(citations: list[Citation]) -> list[dict]:
    """Machine-readable form (full record ids included) for API consumers
    who need exact provenance rather than the prose footer — e.g. a
    front-end that wants to link each figure back to its source record."""
    return [asdict(c) for c in citations]
