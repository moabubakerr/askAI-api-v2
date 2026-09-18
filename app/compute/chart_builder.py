"""
Chart Builder — turns already-computed facts into a chart spec, with no
LLM call anywhere in this file. Two things this fixes directly from the
QC report:

  F-004: chart didn't match the indicator's actual unit/significant figures.
         Fixed by reading unit_en and decimal_places directly from
         indicator_details.format/unit_en — the SAME metadata already
         resolved for the text answer, not a separate guess.
  Consistency: since the chart is built from the exact same `rows` list the
         text answer and citations were built from, the chart can never
         show a different number than the prose does — there's no second
         retrieval or recomputation that could drift from the first.

Chart TYPE is chosen by a fixed lookup table keyed on computation_type —
not a model decision, since it's a small, stable mapping (trend -> line,
comparison -> bar, etc.) that doesn't benefit from an LLM call and is one
more place a wrong guess could otherwise creep in.
"""
from typing import Optional


CHART_TYPE_BY_COMPUTATION = {
    "trend": "line",
    "country_comparison": "bar",
    "country_ranking": "bar",
    "period_comparison": "bar",
    "macro_overview": "bar",
}


def _decimal_places_from_format(format_str: Optional[str]) -> int:
    """SCAI's own `Format` field (e.g. '0.00', 'bn0.0', '0') encodes decimal
    places — read it instead of guessing or letting the LLM pick a
    precision, which is exactly what caused F-004."""
    if not format_str:
        return 1
    if "." in format_str:
        return len(format_str.split(".")[-1])
    return 0


def build_chart_spec(computation_type: str, facts: dict, indicator_name: str,
                      unit: Optional[str], format_str: Optional[str] = None) -> Optional[dict]:
    chart_type = CHART_TYPE_BY_COMPUTATION.get(computation_type)
    if not chart_type:
        return None  # a single value (latest_value, min_max, growth_rate...) isn't chart-shaped

    decimals = _decimal_places_from_format(format_str)
    base = {
        "chart_type": chart_type,
        "title": indicator_name,
        "unit": unit or "",
        "decimal_places": decimals,
    }

    if computation_type == "trend":
        series = facts.get("series", [])
        return {**base,
                "x_field": "period_label", "y_field": "actual",
                "data": [{"period_label": p["period_label"], "actual": p["actual"]} for p in series]}

    if computation_type in ("country_comparison", "country_ranking"):
        rows = facts.get("rows") or facts.get("ranked") or []
        return {**base,
                "x_field": "country", "y_field": "actual",
                "data": [{"country": r["country"], "actual": r["actual"], "period_label": r["period_label"]}
                         for r in rows],
                "missing_countries": facts.get("countries_with_no_data", [])}

    if computation_type == "period_comparison":
        return {**base,
                "x_field": "period_label", "y_field": "actual",
                "data": [
                    {"period_label": facts["period_a"], "actual": facts["value_a"]},
                    {"period_label": facts["period_b"], "actual": facts["value_b"]},
                ]}

    if computation_type == "macro_overview":
        overview = facts.get("overview", [])
        return {**base, "title": "Economic Overview",
                "x_field": "indicator", "y_field": "actual",
                "data": [{"indicator": o["indicator"], "actual": o["actual"], "unit": o.get("unit")}
                         for o in overview],
                "note": "Mixed units across indicators — see per-bar unit, not a single shared axis."}

    return None
