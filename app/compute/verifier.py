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

    allowed = payload_numbers | payload_rounded
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
        return f"{value} {unit}".strip() if value is not None else "—"

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
    if "count" in facts and "names" in facts:
        scope = facts.get("scope")
        what = f"published {scope}s" if scope else "indicators"
        line = f"There are {facts['count']} {what} in the approved data."
        sample = facts.get("names_sample") or facts["names"][:8]
        if sample:
            line += " For example: " + "; ".join(str(n) for n in sample) + "."
        if facts["count"] > len(sample):
            line += f" The full list of {facts['count']} is shown alongside."
        return line
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
