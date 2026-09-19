"""
Period / frequency resolver — deterministic, no LLM.

Directly targets:
  F-006: "how are prices doing lately, monthly" must use monthly data when
         monthly data exists — never silently substitute quarterly.
  F-015: a monthly indicator asked about by month must not fall back to
         "only annual data available" when monthly data exists.
  F-022/023: "last N years" must resolve to N consecutive calendar years of
         YoY comparisons, not quarterly QoQ data.
  F-024/025: "show the trend" must not guess a starting interval, must not
         skip periods silently, and must default to the indicator's own
         GRANULARITY THAT ACTUALLY EXISTS rather than picking one at random.

Design: never guess a granularity the LLM "feels" like using. If the user
names one explicitly, use it (and error out honestly if that indicator has
no data at that granularity, rather than silently switching). If they don't,
default to the FINEST granularity the indicator actually has data at
(monthly > quarterly > yearly) — this matches "lately"/"trend"-style
questions wanting the most detailed honest picture, and is a fixed,
inspectable rule rather than a per-question LLM guess.
"""
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

GRANULARITY_ORDER = ["monthly", "quarterly", "yearly"]

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


@dataclass
class PeriodResolution:
    kind: str                      # "single_year" | "single_quarter" | "single_month" |
                                    # "range" | "last_n_years" | "unspecified"
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    explicit_granularity: Optional[str] = None   # only set if user stated it explicitly
    n_years: Optional[int] = None
    raw: str = ""


def parse_explicit_frequency(explicit_frequency: Optional[str]) -> Optional[str]:
    if explicit_frequency and explicit_frequency.lower() in GRANULARITY_ORDER:
        return explicit_frequency.lower()
    return None


# Every way a single period is written in these questions, in canonical
# period_label form: 'YYYY', 'YYYY-Q#', 'YYYY-MM'.
_PERIOD_TOKEN = re.compile(
    r"(?P<q1>\d{4})[\s-]*q(?P<q1n>[1-4])"          # 2025-Q1 / 2025 q1
    r"|q(?P<q2n>[1-4])[\s-]*(?P<q2>\d{4})"          # Q1 2025 / Q1-2025
    r"|(?P<m1>\d{4})-(?P<m1n>0[1-9]|1[0-2])"         # 2025-05
    r"|(?P<mname>january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+(?P<my>\d{4})"    # May 2025
    r"|(?P<y>\d{4})",                                 # 2025
    re.IGNORECASE,
)


def parse_period_pair(expr: Optional[str]):
    """Two period labels from one expression, for an explicit A-vs-B comparison.

    parse_period_expression() only ever describes ONE period, so "between Q1
    2025 and Q4 2025" fell through its range branch (which needs two bare
    years) into the single-quarter branch, retrieved Q1 alone, and answered
    "Could not find two distinct periods to compare" — F-011's exact question.

    Returns (label_a, label_b) in the order written, in period_label form, or
    None when the expression does not name two distinct periods.
    """
    if not expr:
        return None
    found = []
    for m in _PERIOD_TOKEN.finditer(expr.lower()):
        if m.group("q1"):
            label = f"{m.group('q1')}-Q{m.group('q1n')}"
        elif m.group("q2"):
            label = f"{m.group('q2')}-Q{m.group('q2n')}"
        elif m.group("m1"):
            label = f"{m.group('m1')}-{m.group('m1n')}"
        elif m.group("mname"):
            label = f"{m.group('my')}-{_MONTH_NAMES[m.group('mname')]:02d}"
        else:
            label = m.group("y")
        if label not in found:
            found.append(label)
    if len(found) < 2:
        return None
    # Mixed granularities ("2024 and Q1 2025") are not a like-for-like
    # comparison, so they are declined rather than guessed at.
    kinds = {("q" if "-Q" in f else "m" if "-" in f else "y") for f in found[:2]}
    if len(kinds) > 1:
        return None
    return found[0], found[1]


def label_bounds(label: str):
    """First and last day covered by a period label ('2025-Q1', '2025-05', '2025')."""
    if "-Q" in label:
        year, q = label.split("-Q", 1)
        month = {"1": 1, "2": 4, "3": 7, "4": 10}[q]
        return date(int(year), month, 1), _end_of_month(int(year), month + 2)
    if "-" in label:
        year, month = label.split("-", 1)
        return date(int(year), int(month), 1), _end_of_month(int(year), int(month))
    return date(int(label), 1, 1), date(int(label), 12, 31)


def _end_of_month(year: int, month: int) -> date:
    """Last day of the given month. Needed because a single month or quarter
    previously set only start_date, and start_date is an inclusive lower bound:
    "inflation in May 2025" therefore returned every period FROM May 2025
    onwards, and the caller took the newest of them. The user got the latest
    reading, silently, instead of the month they asked about — the F-014/F-015
    shape of failure ("How many tourists arrived in May 2025")."""
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def parse_period_expression(expr: Optional[str]) -> PeriodResolution:
    if not expr:
        return PeriodResolution(kind="unspecified", raw="")

    text_l = expr.lower().strip()

    # TWO named periods first. Without this, "between Q1 2025 and Q4 2025" fell
    # into the single-quarter branch below, retrieval was scoped to Q1 alone,
    # and the answer described one quarter as though it were the whole span —
    # Q4 2025 never appeared in it at all. Checked before the single-period
    # branches precisely because those match the FIRST period they see.
    pair = parse_period_pair(expr)
    if pair:
        start, _ = label_bounds(pair[0])
        _, end = label_bounds(pair[1])
        if start > end:
            start, end = label_bounds(pair[1])[0], label_bounds(pair[0])[1]
        return PeriodResolution(kind="range", start_date=start, end_date=end, raw=expr)

    # "last N years" / "last N months"
    m = re.search(r"last\s+(\d+)\s+year", text_l)
    if m:
        return PeriodResolution(kind="last_n_years", n_years=int(m.group(1)), raw=expr)

    # "YYYY-MM" or a named month + year, e.g. "May 2025"
    m = re.search(r"(\d{4})-(\d{2})", text_l)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return PeriodResolution(kind="single_month", start_date=date(y, mo, 1),
                                 end_date=_end_of_month(y, mo), raw=expr)

    for name, num in _MONTH_NAMES.items():
        m = re.search(rf"{name}\s+(\d{{4}})", text_l)
        if m:
            y = int(m.group(1))
            return PeriodResolution(kind="single_month", start_date=date(y, num, 1),
                                     end_date=_end_of_month(y, num), raw=expr)

    # "Q1 2025" / "2025-Q1"
    m = re.search(r"q(\d)\s*(\d{4})", text_l) or re.search(r"(\d{4})[\s-]*q(\d)", text_l)
    if m:
        groups = m.groups()
        year, q = (int(groups[1]), int(groups[0])) if len(groups[0]) == 1 else (int(groups[0]), int(groups[1]))
        month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
        return PeriodResolution(kind="single_quarter", start_date=date(year, month, 1),
                                 end_date=_end_of_month(year, month + 2), raw=expr)

    # "between 2022 and 2025" / "2022 to 2025" / "from 2022 to 2025"
    m = re.search(r"(\d{4}).{0,10}(?:to|and|-)\s*(\d{4})", text_l)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        return PeriodResolution(kind="range", start_date=date(min(y1, y2), 1, 1), end_date=date(max(y1, y2), 12, 31), raw=expr)

    # bare year
    m = re.search(r"\b(\d{4})\b", text_l)
    if m:
        y = int(m.group(1))
        return PeriodResolution(kind="single_year", start_date=date(y, 1, 1), end_date=date(y, 12, 31), raw=expr)

    return PeriodResolution(kind="unspecified", raw=expr)


def choose_granularity(explicit: Optional[str], available_granularities: set[str]) -> Optional[str]:
    """Never guesses freely — either honors the explicit ask (if the data
    supports it) or picks the single finest granularity that actually has
    data, per a fixed rule. Returns None if nothing is available at all."""
    if explicit:
        return explicit if explicit in available_granularities else None
    for g in GRANULARITY_ORDER:
        if g in available_granularities:
            return g
    return None
