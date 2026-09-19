"""
Compute Engine — every arithmetic operation the chatbot can ever perform
lives here, as plain, testable Python. This is the direct fix for the
QC report's core complaint: the old system let the LLM compute growth
rates, differences, and rankings itself, and got them wrong (F-005, F-008,
F-009, F-010, F-011, F-017, F-022 through F-026, F-031).

The rule going forward: an LLM call NEVER appears in this file, and no
function here calls out to one. Every number that reaches the user was
either read directly from the database or produced by one of these
functions — the Composer agent is only allowed to restate these results
in prose, never recompute them.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ComputeResult:
    ok: bool
    facts: dict = field(default_factory=dict)   # every number here is safe to state verbatim
    message: Optional[str] = None                # set when ok=False — an honest explanation


def latest_value(rows: list[dict]) -> ComputeResult:
    if not rows:
        return ComputeResult(False, message="No approved data points were found for this indicator/period.")
    latest = rows[-1]  # rows are ORDER BY period_date ASC from the retriever
    return ComputeResult(True, facts={
        "period_label": latest["period_label"],
        "actual": latest["actual"],
        "target": latest.get("target"),
    })


def min_max(rows: list[dict], which: str) -> ComputeResult:
    """which = 'max' or 'min'. Fixes F-008: the highest value must be found
    by scanning ALL retrieved rows, never by assuming 'latest' means 'highest'."""
    valid = [r for r in rows if r.get("actual") is not None]
    if not valid:
        return ComputeResult(False, message="No approved data points with a value were found.")
    picked = max(valid, key=lambda r: r["actual"]) if which == "max" else min(valid, key=lambda r: r["actual"])
    return ComputeResult(True, facts={"period_label": picked["period_label"], "actual": picked["actual"]})


def difference(rows: list[dict]) -> ComputeResult:
    """Difference between the highest and lowest value in the retrieved set.
    Fixes F-010: subtraction must be a real, checked operation, not an LLM guess."""
    valid = [r for r in rows if r.get("actual") is not None]
    if len(valid) < 2:
        return ComputeResult(False, message="Not enough data points to compute a difference.")
    hi = max(valid, key=lambda r: r["actual"])
    lo = min(valid, key=lambda r: r["actual"])
    return ComputeResult(True, facts={
        "high_period": hi["period_label"], "high_value": hi["actual"],
        "low_period": lo["period_label"], "low_value": lo["actual"],
        "absolute_difference": round(hi["actual"] - lo["actual"], 6),
    })


def period_to_period_change(rows: list[dict], period_label_a: str, period_label_b: str) -> ComputeResult:
    """Explicit two-period comparison, e.g. 'Q1 2025 vs Q4 2025'.
    Fixes F-011: must use the EXACT two periods named, not just 'the latest'."""
    by_label = {r["period_label"]: r for r in rows}
    a, b = by_label.get(period_label_a), by_label.get(period_label_b)
    if a is None or b is None:
        missing = period_label_a if a is None else period_label_b
        return ComputeResult(False, message=f"No approved data point found for {missing}.")
    if a.get("actual") is None or b.get("actual") is None:
        return ComputeResult(False, message="One of the two periods has no recorded value.")
    absolute = round(b["actual"] - a["actual"], 6)
    percent = round((b["actual"] - a["actual"]) / a["actual"] * 100, 4) if a["actual"] != 0 else None
    return ComputeResult(True, facts={
        "period_a": period_label_a, "value_a": a["actual"],
        "period_b": period_label_b, "value_b": b["actual"],
        "absolute_change": absolute, "percent_change": percent,
    })


def growth_rate(rows: list[dict], period_label_start: str, period_label_end: str,
                 method: str = "cagr", periods_per_year: int = 1) -> ComputeResult:
    """CAGR or simple growth between two named periods.
    Fixes F-005/F-017: this is a fixed formula, evaluated once, not something
    the LLM free-associates from a full list of quarterly points."""
    by_label = {r["period_label"]: r for r in rows}
    start, end = by_label.get(period_label_start), by_label.get(period_label_end)
    if start is None or end is None:
        missing = period_label_start if start is None else period_label_end
        return ComputeResult(False, message=f"No approved data point found for {missing}.")
    v0, v1 = start.get("actual"), end.get("actual")
    if v0 is None or v1 is None or v0 <= 0:
        return ComputeResult(False, message="Missing or non-positive values prevent a valid growth-rate calculation.")

    if method == "simple":
        rate = round((v1 - v0) / v0 * 100, 4)
    else:
        # CAGR needs the number of periods between start and end
        n_periods = rows.index(end) - rows.index(start)
        if n_periods <= 0:
            return ComputeResult(False, message="End period must come after the start period.")
        years = n_periods / periods_per_year
        rate = round(((v1 / v0) ** (1 / years) - 1) * 100, 4)

    return ComputeResult(True, facts={
        "period_start": period_label_start, "value_start": v0,
        "period_end": period_label_end, "value_end": v1,
        "method": method, "growth_rate_percent": rate,
    })


def preferred_change_field(row: dict, granularity: str, as_points: bool = False) -> Optional[float]:
    """Pulls SCAI's own pre-vetted change figure instead of deriving one —
    fixes F-022/F-023/F-024/F-025 (QoQ vs YoY confusion): if the user asked
    for a yearly change, use yearly_yoy_percent verbatim; never compute a
    fresh one from raw values when a vetted figure already exists."""
    suffix = "pp" if as_points else "percent"
    field_map = {
        "monthly": f"monthly_yoy_{suffix}" if suffix == "percent" else f"monthly_mom_{suffix}",
        "quarterly": f"quarterly_yoy_{suffix}",
        "yearly": f"yearly_yoy_{suffix}",
    }
    field_name = field_map.get(granularity)
    return row.get(field_name) if field_name else None


def trend(rows: list[dict]) -> ComputeResult:
    """The full chronological series, as-is — no gap-filling, no silent
    skipping. Fixes F-024: 'the system skipped 2025-Q3 and 2025-Q4 without
    explanation' — if periods are missing from the data, that's surfaced,
    not papered over."""
    if not rows:
        return ComputeResult(False, message="No approved data points were found for this indicator.")
    series = [{"period_label": r["period_label"], "actual": r["actual"]} for r in rows]
    return ComputeResult(True, facts={"series": series, "n_points": len(series)})


def period_ranking(rows: list[dict], descending: bool = True) -> ComputeResult:
    """Ranks the retrieved periods by value — F-009: "List Qatar's quarterly GDP
    values for 2024 and 2025 and rank them from highest to lowest."

    Distinct from trend(), which is deliberately chronological, and from
    min_max(), which returns only the single extreme. This sorts the whole
    retrieved set, so the ordering is a real sort over real rows rather than
    the LLM being asked to order numbers itself — the exact operation the QC
    report found it getting wrong."""
    valid = [r for r in rows if r.get("actual") is not None]
    if not valid:
        return ComputeResult(False, message="No approved data points were found for this indicator/period.")
    ranked = sorted(valid, key=lambda r: r["actual"], reverse=descending)
    return ComputeResult(True, facts={
        "ranked_periods": [{"period_label": r["period_label"], "actual": r["actual"]} for r in ranked],
        "order": "highest_to_lowest" if descending else "lowest_to_highest",
        "n_points": len(ranked),
    })


def country_comparison(series_by_country: dict[str, list[dict]], period_label: Optional[str] = None) -> ComputeResult:
    """Builds a comparison across EXACTLY the requested countries.
    Fixes F-001: must never include unrequested benchmark countries.
    Missing countries are reported explicitly, never silently dropped or
    filled with an invented number."""
    result_rows, missing = [], []
    for country, rows in series_by_country.items():
        if period_label:
            match = next((r for r in rows if r["period_label"] == period_label), None)
        else:
            match = rows[-1] if rows else None
        if match and match.get("actual") is not None:
            result_rows.append({"country": country, "period_label": match["period_label"], "actual": match["actual"]})
        else:
            missing.append(country)

    if not result_rows:
        return ComputeResult(False, message="No approved data found for any of the requested countries/period.")

    return ComputeResult(True, facts={
        "rows": result_rows,
        "countries_with_no_data": missing,   # composer MUST state these explicitly, never omit
    })


def country_ranking(series_by_country: dict[str, list[dict]], period_label: Optional[str] = None,
                     ascending: bool = True) -> ComputeResult:
    """Fixes F-027: ranking needs an explicit common period across countries;
    if none is given, use the latest period common to ALL of them and say so."""
    comparison = country_comparison(series_by_country, period_label)
    if not comparison.ok:
        return comparison
    rows = sorted(comparison.facts["rows"], key=lambda r: r["actual"], reverse=not ascending)
    return ComputeResult(True, facts={
        "ranked": rows,
        "countries_with_no_data": comparison.facts["countries_with_no_data"],
        "period_used": rows[0]["period_label"] if rows else None,
    })
