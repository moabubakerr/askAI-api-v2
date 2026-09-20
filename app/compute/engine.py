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
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ComputeResult:
    ok: bool
    facts: dict = field(default_factory=dict)   # every number here is safe to state verbatim
    message: Optional[str] = None                # set when ok=False — an honest explanation


def latest_value(rows: list[dict]) -> ComputeResult:
    """The most recent period that has an ACTUAL reading.

    Not simply rows[-1]. Many series carry target rows years into the future
    with no actual — Percentage of Floating Shares runs to 2030-12 with only a
    target — so taking the chronologically last row answered "what is the
    latest X" with an empty value and a 2030 target. That is how a question
    about solar energy came back "the target in 2030 is set at 50.0%", and how
    Annual Growth in Labor Productivity reported its 2030 target as if it were
    a current reading: a forecast-shaped answer from a system that explicitly
    does not forecast.

    The target for the same period is still returned alongside, so "actual vs
    target" remains answerable. When nothing has an actual, that is stated
    rather than papered over with the target."""
    if not rows:
        return ComputeResult(False, message="No approved data points were found for this indicator/period.")

    actuals = [r for r in rows if r.get("actual") is not None]
    if not actuals:
        targets = [r for r in rows if r.get("target") is not None]
        if targets:
            last = targets[-1]
            return ComputeResult(False, message=(
                f"No actual readings have been recorded for this indicator in the approved data — "
                f"only targets (most recently {last['target']} for {last['period_label']})."
            ))
        return ComputeResult(False, message="No approved data points were found for this indicator/period.")

    latest = actuals[-1]  # rows are ORDER BY period_date ASC from the retriever
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
    # The facts must say that this IS the extreme, and over what. Returning
    # only {period_label, actual} gave the Composer the exact shape of a
    # latest-value answer, so "what was the highest quarterly GDP, and in which
    # quarter?" came back as the flat "Real GDP in 2025-Q3 was 185.99" — the
    # right number, stripped of the one thing that was asked.
    return ComputeResult(True, facts={
        "extremum": "highest" if which == "max" else "lowest",
        "period_label": picked["period_label"],
        "actual": picked["actual"],
        "scanned_points": len(valid),
        "scanned_from": valid[0]["period_label"],
        "scanned_to": valid[-1]["period_label"],
    })


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

    # `actual` is a NUMERIC column, so psycopg2 returns decimal.Decimal, and
    # Decimal ** float raises TypeError — which made every CAGR request a 500.
    # The fractional exponent has no Decimal equivalent, so the arithmetic is
    # done in float. Only the RATE is computed this way; the reported start and
    # end values stay exactly as retrieved, so nothing a user sees is affected
    # by float representation.
    f0, f1 = float(v0), float(v1)

    if method == "simple":
        rate = round((f1 - f0) / f0 * 100, 4)
    else:
        # CAGR needs the number of periods between start and end
        n_periods = rows.index(end) - rows.index(start)
        if n_periods <= 0:
            return ComputeResult(False, message="End period must come after the start period.")
        years = n_periods / periods_per_year
        rate = round(((f1 / f0) ** (1 / years) - 1) * 100, 4)

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
    facts = {"series": series, "n_points": len(series)}

    # The shape of the series, computed here rather than left for the Composer
    # to work out. Narrating 28 points, a model reaches for "rose 11.5% over
    # the period" — a number that was in no payload, so the numeric verifier
    # rejected the whole answer and the user got a template dump instead. The
    # figures it wants are deterministic, so they are supplied: stating them
    # then becomes reporting, which is all the Composer is allowed to do.
    valued = [r for r in rows if r.get("actual") is not None]
    if valued:
        first, last = valued[0], valued[-1]
        facts.update({
            "first_period": first["period_label"], "first_value": first["actual"],
            "last_period": last["period_label"], "last_value": last["actual"],
        })
        high = max(valued, key=lambda r: r["actual"])
        low = min(valued, key=lambda r: r["actual"])
        facts.update({
            "highest_period": high["period_label"], "highest_value": high["actual"],
            "lowest_period": low["period_label"], "lowest_value": low["actual"],
        })
        if first["actual"] not in (None, 0):
            facts["change_percent"] = round(
                (last["actual"] - first["actual"]) / first["actual"] * 100, 4)
            facts["absolute_change"] = round(last["actual"] - first["actual"], 6)
    return ComputeResult(True, facts=facts)


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
        # WHICH END answers the question. The list was sorted correctly and
        # carried no statement of what the sort meant, so a follow-up to "which
        # country had the LOWEST inflation" was written up as "Qatar had the
        # highest" — the right table under the wrong sentence. The first row is
        # the answer; say so rather than leaving it to be inferred.
        "extremum": "lowest" if ascending else "highest",
        "leader": rows[0] if rows else None,
    })


def _attainment(actual: float, target: float, polarity: str) -> Optional[float]:
    """How far a reading has got toward its target, as a percentage.

    Polarity decides the direction, and getting it backwards inverts the whole
    ranking. For 'Decrease' indicators — cost per student, PISA rank — a SMALLER
    number is the better outcome, so a rank of 48 against a target of 35 is
    73% attained, not 137%.

    Returns None rather than a number whenever the arithmetic would not mean
    anything: a zero target, or a negative value on a scale where the ratio
    stops being interpretable.
    """
    if actual is None or target is None:
        return None
    actual, target = float(actual), float(target)
    if target == 0 or actual == 0:
        return None
    if actual < 0 or target < 0:
        return None
    ratio = target / actual if (polarity or "").strip().lower().startswith("decrease") \
        else actual / target
    return round(ratio * 100, 1)


def scope_performance(entries: list[dict], best_first: bool = True,
                       limit: Optional[int] = None) -> ComputeResult:
    """Ranks the indicators of a sector by progress toward their own targets.

    "Which of these are performing best?" cannot be answered by comparing the
    values to each other. The Education Sector's thirteen indicators are a cost
    in thousands of riyals, a students-per-teacher ratio, a PISA world rank and
    ten percentages; sorting those against one another produces an ordering
    that looks authoritative and means nothing. The only comparison the data
    supports is each indicator against the target SCAI set for it, which is a
    common scale by construction.

    Indicators with no target are not ranked and not dropped — they are
    returned separately, because "we cannot assess these five" is part of the
    honest answer and silently showing eight of thirteen is not.
    """
    if not entries:
        return ComputeResult(False, message="No published indicators were found for this group.")

    ranked, unassessable = [], []
    for e in entries:
        # A target filed against the specific reading beats the indicator-level
        # one: it is contemporaneous, where target_value is a standing goal
        # that may sit years out.
        target = e.get("period_target")
        target_basis = "period"
        if target is None:
            target, target_basis = e.get("target_value"), "indicator"
        score = _attainment(e.get("actual"), target, e.get("polarity_en"))
        row = {
            "indicator": (e.get("indicator") or "").strip(),
            "actual": e.get("actual"),
            "period_label": e.get("period_label"),
            "unit": e.get("unit_en"),
            "target": target,
            "polarity": e.get("polarity_en"),
        }
        if score is None:
            # A CODE, not prose: this text is shown to the reader, and the
            # reader may be reading in Arabic. The engine has no language, so
            # it says what happened and the renderer says it in words.
            row["reason_code"] = "no_reading" if e.get("actual") is None else "no_target"
            unassessable.append(row)
        else:
            ranked.append({**row, "attainment_percent": score,
                            "target_basis": target_basis,
                            "target_year": e.get("target_year")})

    if not ranked:
        return ComputeResult(False, message=(
            "None of the indicators in this group have both a reading and a target, "
            "so there is no basis for ranking them by performance."))

    ranked.sort(key=lambda r: r["attainment_percent"], reverse=best_first)
    shown = ranked[:limit] if limit else ranked
    return ComputeResult(True, facts={
        "ranked_indicators": shown,
        "order": "best_first" if best_first else "worst_first",
        "n_ranked": len(ranked),
        "n_total": len(entries),
        "not_assessable": unassessable,
        # Spelled out because the Composer must say it: this is progress
        # against each indicator's own target, not a comparison between them.
        "basis": "percent of each indicator's own target attained, with polarity applied",
    })


def scope_direction(entries: list[dict]) -> ComputeResult:
    """Splits a group of indicators into those rising and those falling,
    year on year.

    "Which national indicators are increasing, and which are declining
    compared with the previous year?" was read as a two-period comparison and
    asked the user to name both periods, then — once they did — sent the whole
    sentence to the indicator resolver, which found no indicator called that.
    It is neither. It is one question about eight indicators, and each one's
    year-on-year change is already stored and vetted.

    Unlike scope_performance this needs no target, so it covers indicators a
    performance ranking cannot reach. It says nothing about whether a movement
    is good: polarity is carried on every row so the caller can, and for
    Inflation or Cost per Student a rise is not an improvement.
    """
    if not entries:
        return ComputeResult(False, message="No published indicators were found for this group.")

    increasing, declining, unchanged, no_comparison = [], [], [], []
    for e in entries:
        change = preferred_change_field(e, e.get("granularity") or "")
        change_kind = "percent"
        if change is None:
            # Ratio indicators publish the percentage-POINT move and leave the
            # percent column empty. Public Debt as a Percentage of GDP was
            # being reported as having no year-on-year figure at all because of
            # it, in a list whose entire purpose is which way things moved.
            change = preferred_change_field(e, e.get("granularity") or "", as_points=True)
            change_kind = "percentage_points"
        row = {
            "indicator": (e.get("indicator") or "").strip(),
            "actual": e.get("actual"),
            "period_label": e.get("period_label"),
            "unit": e.get("unit_en"),
            "polarity": e.get("polarity_en"),
        }
        component = component_label(e)
        if component:
            row["component_name"] = component
            row["component_of_total"] = e.get("sibling_count")
        if change is None:
            row["reason_code"] = ("no_reading" if e.get("actual") is None
                                   else "no_yoy_published")
            no_comparison.append(row)
            continue
        change = round(float(change), 4)
        # Named by what it is: a percentage-point move on a ratio is not a
        # percentage change, and labelling it as one misstates the figure.
        row["change_yoy_percent" if change_kind == "percent" else "change_yoy_pp"] = change
        row["change_kind"] = change_kind
        # Whether a rise is welcome depends on the indicator; whether it IS a
        # rise does not. Only the second is decided here.
        (increasing if change > 0 else declining if change < 0 else unchanged).append(row)

    if not (increasing or declining or unchanged):
        return ComputeResult(False, message=(
            "None of the indicators in this group have a published year-on-year "
            "change, so there is no basis for saying which are rising or falling."))

    def moved(r):
        return r.get("change_yoy_percent", r.get("change_yoy_pp"))

    increasing.sort(key=moved, reverse=True)
    declining.sort(key=moved)
    return ComputeResult(True, facts={
        "increasing": increasing,
        "declining": declining,
        "unchanged": unchanged,
        "no_comparison": no_comparison,
        "n_increasing": len(increasing),
        "n_declining": len(declining),
        "n_total": len(entries),
        "comparison": "year-on-year, at each indicator's most recent reading",
    })


def _is_rank(unit: Optional[str]) -> bool:
    return (unit or "").strip().lower() in ("rank", "ranking", "المرتبة")


def direction_assessment(rows: list[dict], granularity: str,
                          polarity: Optional[str], unit: Optional[str] = None,
                          period_label: Optional[str] = None) -> ComputeResult:
    """Is this indicator getting better or worse? Decided from polarity, not opinion.

    "Improving" is a judgement everywhere else in this system and forbidden to
    the Composer for good reason. Here it is not: SCAI states the desired
    direction in polarity_en, so a fall in Public Debt as a Percentage of GDP
    (polarity Decrease) IS an improvement by the Council's own definition, and
    saying so asserts nothing the data does not.

    Both change forms are tried. A percentage-point move is the right one for a
    ratio — debt/GDP going 40.5 -> 40.6 is +0.1pp, and expressing that as a
    percentage change of itself would be both odd and, for this indicator,
    absent: SCAI publishes yearly_yoy_pp and leaves yearly_yoy_percent empty.
    """
    actuals = [r for r in rows if r.get("actual") is not None]
    if not actuals:
        return ComputeResult(False, message="No approved data points were found for this indicator.")
    # The period the question named, when it named one. "Did competitiveness
    # improve in 2026" is about 2026 against 2025, not about whatever the
    # series happens to end at.
    latest = next((r for r in actuals if r.get("period_label") == period_label), None)         if period_label else None
    latest = latest or actuals[-1]

    change_basis = "published"
    change = preferred_change_field(latest, granularity, as_points=True)
    change_kind = "percentage_points"
    if change is None:
        change = preferred_change_field(latest, granularity)
        change_kind = "percent"

    # Nothing published. Fall back to the reading before this one, which is a
    # subtraction between two published figures — the same operation
    # period_to_period_change already performs, not a new claim.
    #
    # This is the only way ranks are answerable at all: not one of the ten rank
    # indicators publishes a single period-on-period figure, so every "has the
    # ranking improved?" was refused about a series that plainly shows it.
    previous = None
    if change is None:
        index = actuals.index(latest)
        if index > 0:
            previous = actuals[index - 1]
            change = float(latest["actual"]) - float(previous["actual"])
            # Rank movement is counted in PLACES. Calling 9th -> 11th a
            # "22.2% deterioration" is the misleading rank language this must
            # not produce: the gap between 9th and 11th is two places, and a
            # percentage of an ordinal position means nothing.
            change_kind = "places" if _is_rank(unit) else "absolute"
            change_basis = "computed_from_series"
    if change is None:
        return ComputeResult(False, message=(
            "Only one reading is published for this indicator, so there is nothing "
            "to compare it against and its direction of travel cannot be stated."))

    change = round(float(change), 4)
    wants_lower = (polarity or "").strip().lower().startswith("decrease")
    if change == 0:
        verdict = "unchanged"
    elif (change < 0) == wants_lower:
        verdict = "improving"
    else:
        verdict = "deteriorating"

    return ComputeResult(True, facts={
        "actual": latest["actual"],
        "period_label": latest["period_label"],
        "previous_value": previous["actual"] if previous else None,
        "previous_period": previous["period_label"] if previous else None,
        "change": change,
        "change_kind": change_kind,
        "direction": "down" if change < 0 else "up" if change > 0 else "flat",
        "polarity": polarity,
        # Spelled out so the Composer states it rather than deciding it.
        "assessment": verdict,
        # The DIRECTION SCAI wants, as a code. The sentence around it is
        # written by whatever renders the answer, in the reader's language.
        "desired_direction": "decrease" if wants_lower else "increase",
        "granularity": granularity,
        "change_basis": change_basis,
        "unit": unit,
    })


def component_label(entry: dict) -> Optional[str]:
    """The detail's own name, when reporting it under the indicator's name
    would misdescribe it.

    IsMain is meant to pick the headline reading, and for 175 of the 189
    published indicators it does. For fourteen it picks one row out of several,
    and for seven of those the row is a plain component: Workforce
    (Economically Active) -> "High Skilled Blue Collar", Minimum Reserves of
    Strategic Commodities -> "Onions", Self-Sufficiency for Strategic
    Commodities -> "Milk/Dairy". Printing 0.593054m under "Workforce
    (Economically Active)" states that Qatar's workforce is 0.59 million when
    the four published components sum to 2.24 million.

    Where the detail is named "Total ..." it IS the headline and nothing is
    added — that covers the other seven.
    """
    detail = (entry.get("detail_name") or "").strip()
    indicator = (entry.get("indicator") or "").strip()
    if not detail or (entry.get("sibling_count") or 1) <= 1:
        return None
    if detail.lower() == indicator.lower():
        return None
    if "total" in detail.lower():
        return None
    return detail


def scope_snapshot(entries: list[dict]) -> ComputeResult:
    """Current reading for every indicator in a group.

    Deliberately shaped as an "overview", the same structure a multi-metric
    question already produces, so the Composer rule, the template fallback and
    the frontend all handle it without learning a fourth group shape. The only
    difference is how the members were chosen: named by the user there, named
    by belonging to a sector or an indicator type here.

    Each line keeps its OWN period. These twelve report on different schedules
    — 2026-Q1 next to 2023 — and forcing them onto one date would either drop
    the fresher readings or imply a currency the data does not have.
    """
    if not entries:
        return ComputeResult(False, message="No published indicators were found for this group.")

    overview, missing = [], []
    for e in entries:
        if e.get("actual") is None:
            missing.append({"indicator": (e.get("indicator") or "").strip(),
                             "reason_code": "no_reading"})
            continue
        line = {
            "indicator": (e.get("indicator") or "").strip(),
            "actual": e.get("actual"),
            "period_label": e.get("period_label"),
            "unit": e.get("unit_en"),
            "granularity": e.get("granularity"),
            "polarity": e.get("polarity_en"),
        }
        component = component_label(e)
        if component:
            line["component_name"] = component
            line["component_of_total"] = e.get("sibling_count")
        if e.get("decimal_places") is not None:
            line["decimal_places"] = e["decimal_places"]
        change = preferred_change_field(e, e.get("granularity") or "")
        if change is not None:
            line["change_yoy_percent"] = round(float(change), 4)
            line["change_kind"] = "percent"
        else:
            change = preferred_change_field(e, e.get("granularity") or "", as_points=True)
            if change is not None:
                line["change_yoy_pp"] = round(float(change), 4)
                line["change_kind"] = "percentage_points"
        overview.append(line)

    if not overview:
        return ComputeResult(False, message=(
            "None of the indicators in this group have a published reading."))

    facts = {"overview": overview, "n_total": len(entries)}
    if missing:
        facts["not_reported"] = missing
    return ComputeResult(True, facts=facts)


# "Non-Hydrocarbon Exports (share of total exports)" — a share whose whole is
# split into exactly two named parts, one of which is the negation of the
# other. Only then is 100 - x the other part.
#
# The guard is the whole point. Half the percentage indicators in the catalogue
# look superficially similar and have no meaningful complement:
#   Public Debt as a Percentage of GDP (%)   -> 59.4% of GDP is "not debt"?
#   FDI stock (share of GDP)                 -> same
#   Non-Hydrocarbon Govt Revenue as Share of Non-Hydrocarbon GDP
#                                            -> the DENOMINATOR is negated too,
#                                               so the complement is not a
#                                               hydrocarbon anything
# so the pattern requires a "non-" numerator, an un-negated total, and the same
# head noun on both sides.
_COMPLEMENT_NAME = re.compile(
    r"^non[-\s]?(?P<term>[A-Za-z]+)\s+(?P<noun>[A-Za-z]+)\s*"
    r"\(?\s*(share of|as a (share|percentage) of)\s+(the\s+)?total\s+"
    r"(?P<denom>[A-Za-z]+)\s*\)?\s*$",
    re.IGNORECASE)


def complement_of_share(actual, indicator_name: str, question: str) -> ComputeResult:
    """The other side of a two-way share: 100 - 38.589 = 61.411.

    Returns ok=False unless the indicator really is one half of a two-way
    split AND the question asks for the other half by name. Refusing is the
    correct outcome for every percentage that is not that shape, and most are
    not.
    """
    match = _COMPLEMENT_NAME.match((indicator_name or "").strip())
    if not match:
        return ComputeResult(False, message=(
            "This indicator is not a two-way share, so there is no remaining "
            "share to report."))
    term, noun, denom = (match.group("term").lower(), match.group("noun").lower(),
                         match.group("denom").lower())
    # The total must be the total of the SAME thing, and must not itself be
    # negated — "share of total non-hydrocarbon GDP" is a different whole.
    if denom.rstrip("s") != noun.rstrip("s"):
        return ComputeResult(False, message=(
            "This indicator's total is not the total of the same quantity, so a "
            "remaining share cannot be derived from it."))
    # The question has to name the other side, not merely mention the metric.
    asked = re.search(rf"(?<!non-)(?<!non )\b{re.escape(term)}\b", question or "", re.IGNORECASE)
    if not asked:
        return ComputeResult(False, message="No complementary share was asked for.")
    if actual is None:
        return ComputeResult(False, message="No approved reading is available for this indicator.")

    share = float(actual)
    if not 0 <= share <= 100:
        return ComputeResult(False, message=(
            "This reading is not a percentage between 0 and 100, so a remaining "
            "share cannot be derived from it."))
    return ComputeResult(True, facts={
        "reported_share": round(share, 4),
        "reported_share_of": indicator_name.strip(),
        "complement_share": round(100 - share, 4),
        "complement_of": f"{match.group('term')} {match.group('noun')}",
        "derivation": f"100 - {round(share, 4)} = {round(100 - share, 4)}",
        "unit": "%",
    })
