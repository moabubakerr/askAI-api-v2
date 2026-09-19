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
