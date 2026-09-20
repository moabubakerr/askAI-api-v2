"""
Numeric Verifier — a regex-based, non-LLM check that every number appearing
in the Composer's output text actually came from the facts payload. This is
the hard enforcement mechanism behind "the solution should not generate any
number" — the Composer prompt asks nicely, this makes sure.

If a stray number is found, the caller should NOT ship the LLM's text —
fall back to a template-rendered answer built directly from the facts
payload (see render_template_fallback), which has no LLM in the loop at all.
"""
import re
from decimal import Decimal

from app.compute.formatting import trim_zeros


# The lookbehind stops a hyphen being read as a minus sign when it follows a
# word character, which made period labels parse asymmetrically: the payload's
# "2025-Q4" yielded ["2025", "4"], but the same quarter written the way SCAI and
# the Composer write it, "Q4-2025", yielded ["4", "-2025"]. "-2025" was in
# neither the allowed set nor the 1900-2100 year exemption, so any answer naming
# a quarter that way was rejected as containing an invented number.
# Genuine negatives ("fell by -3.2%") still parse, since the hyphen there
# follows a space.
NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*")


def _extract_numbers_from_payload(payload) -> set[str]:
    """Walks the facts payload and collects every number as a normalized
    string (no commas, trimmed trailing zeros) so it can be compared against
    numbers found in the LLM's text."""
    found = set()

    def walk(value):
        if isinstance(value, bool):
            return  # bool is a subclass of int; "True" is not a data point
        # Decimal matters here: `actual`/`target` are NUMERIC columns, so
        # psycopg2 hands back decimal.Decimal, not float. Decimal matches none
        # of the branches below, so every retrieved value used to be skipped —
        # which meant the verifier saw an EMPTY set of allowed numbers and
        # rejected the Composer's correct figure as invented. The visible
        # symptom was every numeric answer arriving as the raw template dump
        # ("actual: 185.17") with verified=false.
        if isinstance(value, (int, float, Decimal)):
            found.add(_normalize(value))
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)
        elif isinstance(value, str):
            for m in NUMBER_PATTERN.findall(value):
                found.add(_normalize(m))

    walk(payload)
    return found


def _normalize(value) -> str:
    try:
        f = float(str(value).replace(",", ""))
    except ValueError:
        return str(value)
    # Normalize both "1935" and "1935.0" to the same key; keep reasonable precision
    return f"{f:.4f}".rstrip("0").rstrip(".")


def verify_numbers(answer_text: str, facts_payload: dict, rounding_tolerance_decimals: int = 1) -> tuple[bool, list[str]]:
    """Returns (is_clean, offending_numbers). Allows for the headline
    rounding convention (payload may carry full precision, e.g. 177.429,
    while the answer rounds to 177.4) by checking the LLM's number against
    the payload's number rounded to `rounding_tolerance_decimals` as well."""
    payload_numbers = _extract_numbers_from_payload(facts_payload)
    payload_rounded = set()
    for n in payload_numbers:
        try:
            payload_rounded.add(_normalize(round(float(n), rounding_tolerance_decimals)))
        except ValueError:
            pass

    # A negative value is routinely stated as a magnitude plus a direction
    # word: the payload holds -23.4946 and the answer says "revenues fell
    # 23.5%". That is correct, and it was being rejected, which discarded a
    # good answer in favour of a template. The magnitude does trace to the
    # data, so it is allowed.
    #
    # The limit this accepts: the verifier cannot tell "fell 23.5%" from "rose
    # 23.5%". It never could — direction lives in the prose, not the number,
    # and no version of this check has ever verified a direction word. What it
    # still guarantees is that no figure appears that is not in the data.
    payload_absolute = set()
    for n in payload_numbers | payload_rounded:
        try:
            payload_absolute.add(_normalize(abs(float(n))))
        except ValueError:
            pass

    allowed = payload_numbers | payload_rounded | payload_absolute
    text_numbers = {_normalize(m) for m in NUMBER_PATTERN.findall(answer_text)}

    # ignore tiny integers that are almost certainly not data (years handled
    # separately since 2019-2030 range appears constantly and legitimately)
    offending = []
    for n in text_numbers:
        try:
            f = float(n)
        except ValueError:
            continue
        if 1900 <= f <= 2100 and f == int(f):
            continue  # looks like a year — years are allowed to appear as period labels
        if n not in allowed:
            offending.append(n)

    return (len(offending) == 0, offending)


def render_template_fallback(facts_payload: dict, language: str = "en") -> str:
    """A zero-LLM, template-only rendering used when the Composer's text
    fails verification. Deliberately plain — correctness over eloquence."""
    if facts_payload.get("ok") is False:
        return facts_payload.get("message", "No approved data is available for this request.")

    facts = facts_payload.get("facts", {})
    unit = facts.get("unit") or ""
    indicator = facts.get("indicator") or "This indicator"

    # Readable sentences for the shapes that actually occur, rather than a dump
    # of "period_label: 2025-Q4 / actual: 37.972". This text is what the user
    # sees whenever the Composer's wording fails verification, so it is a normal
    # answer, not a debug view — the previous version looked like the system had
    # broken even though the data behind it was correct.
    def fmt(value):
        # trim_zeros, not rounding: Decimal('3.680000') printed "a change of
        # 3.680000 QAR" — six decimals of apparent precision on a figure that
        # has two. Padding only; the digits themselves are untouched.
        return f"{trim_zeros(value)} {unit}".strip() if value is not None else "—"

    if "complement_share" in facts:
        return (f"{facts['complement_of']} account for approximately "
                f"{facts['complement_share']}% of the total. "
                f"{facts['reported_share_of']} was {facts['reported_share']}% in "
                f"{facts.get('period_label')}, so the remainder is "
                f"{facts['derivation']}.")
    if "assessment" in facts:
        word = {"improving": "an improvement", "deteriorating": "a deterioration",
                "unchanged": "no change"}[facts["assessment"]]
        moved = "percentage points" if facts.get("change_kind") == "percentage_points" else "%"
        return (f"{indicator} was {fmt(facts.get('actual'))} in {facts.get('period_label')}, "
                f"{'up' if facts.get('direction') == 'up' else 'down' if facts.get('direction') == 'down' else 'unchanged'} "
                f"{abs(facts.get('change', 0))} {moved} year on year — {word}. "
                f"{facts.get('assessment_basis', '')}".strip())
    if "actual" in facts and "period_label" in facts:
        line = f"{indicator} was {fmt(facts['actual'])} in {facts['period_label']}."
        if facts.get("target") is not None:
            line += f" The target for that period was {fmt(facts['target'])}."
        return line
    if "growth_rate_percent" in facts:
        return (f"{indicator} moved from {fmt(facts.get('value_start'))} in "
                f"{facts.get('period_start')} to {fmt(facts.get('value_end'))} in "
                f"{facts.get('period_end')} — a {facts['growth_rate_percent']}% "
                f"{facts.get('method', '')} growth rate.".replace("  ", " "))
    if "absolute_difference" in facts:
        return (f"The highest {indicator} was {fmt(facts.get('high_value'))} in "
                f"{facts.get('high_period')} and the lowest {fmt(facts.get('low_value'))} in "
                f"{facts.get('low_period')} — a difference of "
                f"{fmt(facts['absolute_difference'])}.")
    if "percent_change" in facts:
        return (f"{indicator} was {fmt(facts.get('value_a'))} in {facts.get('period_a')} and "
                f"{fmt(facts.get('value_b'))} in {facts.get('period_b')} — a change of "
                f"{fmt(facts.get('absolute_change'))} ({facts['percent_change']}%).")
    if "definition" in facts:
        return facts["definition"]
    if "overview" in facts:
        # One line per indicator, each with its own period — not a repr of the
        # list, which is what a reader was being shown, Decimal() wrappers and
        # all.
        lines = []
        for e in facts["overview"]:
            value = e.get("actual")
            unit = e.get("unit") or ""
            if e.get("report_as_growth") and e.get("change_yoy_percent") is not None:
                lines.append(f"{e.get('indicator')}: {e['change_yoy_percent']}% YoY "
                             f"({e.get('period_label')})")
            else:
                shown = f"{trim_zeros(value)} {unit}".strip() if value is not None else "no reading"
                line = f"{e.get('indicator')}: {shown} ({e.get('period_label')})"
                # The movement, not just the level. "How is the economy doing"
                # answered with four current values is a list of readings, not
                # an answer — the direction is the question.
                if e.get("previous_value") is not None:
                    line += (f", from {trim_zeros(e['previous_value'])} {unit}".rstrip()
                             + f" in {e.get('previous_period')}")
                if e.get("change_yoy_percent") is not None:
                    line += f" ({e['change_yoy_percent']}% YoY)"
                lines.append(line)
        if facts.get("change_ranking"):
            ordered = facts["change_ranking"]
            lines.append("Largest decline: " + ", then ".join(
                f"{e['indicator']} ({e['change_yoy_percent']}%)" for e in ordered) + ".")
        for e in facts.get("no_data_in_period") or []:
            covers = (f" Its published series runs from {e['covers_from']} to {e['covers_to']}."
                      if e.get("covers_from") else "")
            lines.append(f"{e['indicator']} has no reading in the period asked about.{covers}")
        if facts.get("not_found"):
            lines.append("Not found in the approved data: " + ", ".join(facts["not_found"]) + ".")
        return "\n".join(lines)
    if "increasing" in facts and "declining" in facts:
        scope = facts.get("scope") or "These indicators"
        lines = [f"{scope}, compared with a year earlier:"]
        for label, key in (("Increasing", "increasing"), ("Declining", "declining"),
                            ("Unchanged", "unchanged")):
            group = facts.get(key) or []
            if not group:
                continue
            lines.append(f"\n{label} ({len(group)}):")
            for e in group:
                value = f"{trim_zeros(e.get('actual'))} {e.get('unit') or ''}".strip()
                # A percentage-point move is not a percentage change. Printing
                # "+0.1%" for a ratio that moved 40.5 -> 40.6 misstates it.
                moved = (f"{e['change_yoy_percent']}%" if e.get("change_yoy_percent") is not None
                         else f"{e.get('change_yoy_pp')} pp")
                lines.append(f"- {e.get('indicator')}: {moved} "
                             f"({value} in {e.get('period_label')})")
        skipped = facts.get("no_comparison") or []
        if skipped:
            lines.append(f"\nNo year-on-year comparison available for {len(skipped)} of "
                         f"{facts.get('n_total')}: "
                         + "; ".join(f"{e.get('indicator')} ({e.get('reason')})" for e in skipped)
                         + ".")
        return "\n".join(lines)
    if "ranked_indicators" in facts:
        scope = facts.get("scope") or "this group"
        order = "closest to target first" if facts.get("order") == "best_first" \
            else "furthest from target first"
        lines = [f"{scope}, ranked by progress against each indicator's own target "
                 f"({order}):"]
        for i, e in enumerate(facts["ranked_indicators"], 1):
            value = f"{trim_zeros(e.get('actual'))} {e.get('unit') or ''}".strip()
            lines.append(f"{i}. {e.get('indicator')} — {e.get('attainment_percent')}% of target "
                         f"({value} in {e.get('period_label')}, target {e.get('target')})")
        skipped = facts.get("not_assessable") or []
        if skipped:
            lines.append(f"{len(skipped)} of {facts.get('n_total')} could not be ranked: "
                         + "; ".join(f"{e.get('indicator')} ({e.get('reason')})" for e in skipped)
                         + ".")
        return "\n".join(lines)
    if "count" in facts and "names" in facts:
        scope = facts.get("scope")
        # A sector is a container, a type is a label: "101 published Sector
        # Indicators" is right, "13 published Education Sectors" is not.
        if not scope:
            what = "indicators"
        elif facts.get("scope_kind") == "sector":
            what = f"published indicators in the {scope}"
        else:
            what = f"published {scope}s"
        line = f"There are {facts['count']} {what} in the approved data."
        # The names are rendered in full beside this text, so enumerating them
        # here prints them twice. But answering "give me all the names" with
        # "for example: ..." reads as a refusal of the question that was asked,
        # so say plainly that all of them are there. Examples only earn their
        # space on a list too long to take in at a glance.
        if facts["count"] > 15:
            sample = facts.get("names_sample") or facts["names"][:8]
            if sample:
                line += " For example: " + "; ".join(str(n) for n in sample) + "."
        return line + f" All {facts['count']} are listed below."
    if "series" in facts:
        # Summarise; do NOT enumerate. The series is already rendered as a chart
        # and a table beside this text, so listing all 28 points printed the
        # same data three times on one screen.
        parts = [f"{indicator}: {facts.get('n_points', len(facts['series']))} readings"]
        if facts.get("first_period"):
            parts.append(f"from {fmt(facts.get('first_value'))} in {facts['first_period']} "
                         f"to {fmt(facts.get('last_value'))} in {facts.get('last_period')}")
        line = ", ".join(parts) + "."
        if facts.get("change_percent") is not None:
            line += f" That is a change of {facts['change_percent']}% over the period."
        if facts.get("highest_period"):
            line += (f" The highest reading was {fmt(facts.get('highest_value'))} in "
                     f"{facts['highest_period']}, the lowest {fmt(facts.get('lowest_value'))} "
                     f"in {facts.get('lowest_period')}.")
        return line

    lines = []
    for key, value in facts.items():
        if key in ("unit", "indicator", "n_points"):
            continue
        if key == "series":
            for point in value:
                lines.append(f"{point.get('period_label')}: {point.get('actual')}")
        elif key == "rows":
            for row in value:
                lines.append(f"{row.get('country')} ({row.get('period_label')}): {row.get('actual')}")
        elif key == "ranked":
            for i, row in enumerate(value, 1):
                lines.append(f"{i}. {row.get('country')}: {row.get('actual')} ({row.get('period_label')})")
        elif key == "countries_with_no_data" and value:
            lines.append("No approved data for: " + ", ".join(value))
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) if lines else "No approved data is available for this request."
