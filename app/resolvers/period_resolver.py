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
                                    # "range" | "last_n_years" | "previous_year" |
                                    # "unspecified"
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    explicit_granularity: Optional[str] = None   # only set if user stated it explicitly
    n_years: Optional[int] = None
    raw: str = ""


def parse_explicit_frequency(explicit_frequency: Optional[str]) -> Optional[str]:
    if explicit_frequency and explicit_frequency.lower() in GRANULARITY_ORDER:
        return explicit_frequency.lower()
    return None


# The three granularities this data has, named. Longest alternatives first so
# "quarterly" is not matched as "quarter" inside a different phrase.
_FREQUENCY_WORDS = (
    # Arabic attaches the definite article between the two words of the
    # compound — "ربع السنوي" as often as "ربع سنوي" — and checking only the
    # bare form let the trailing "السنوي" match the YEARLY pattern below, so a
    # question asking for quarters was read as asking for years.
    ("quarterly",
     r"quarter(?:ly|s)?\b|per\s+quarter|by\s+quarter|ربع\s*(?:ال)?سنوي(?:ة|ا)?|ربعي(?:ة|ا)?"),
    ("monthly", r"month(?:ly|s)?\b|per\s+month|by\s+month|شهري(?:ة|ا)?"),
    ("yearly", r"year(?:ly)?\b|annual(?:ly)?\b|per\s+year|by\s+year|سنوي(?:ة|ا)?"),
)
_FREQUENCY_IN_TEXT = [(gran, re.compile(pattern, re.IGNORECASE))
                      for gran, pattern in _FREQUENCY_WORDS]


def frequency_in_text(text: Optional[str]) -> Optional[str]:
    """The frequency the QUESTION names, read from the question.

    parse_explicit_frequency reads the intent agent's field and nothing else, so
    when the model left it empty the word in the message was invisible to the
    whole pipeline. "List Qatar's quarterly GDP values for 2024 and 2025" was
    answered with two yearly figures: the periods implied the yearly series, the
    word "quarterly" was never consulted, and the user got the opposite of what
    they had asked for in the second word of the sentence.

    A closed vocabulary, which is what makes reading it here legitimate where a
    pattern would be wrong for an open-ended one. There are exactly three
    granularities in this data, GRANULARITY_ORDER names them, and "quarterly"
    means one of the three or nothing at all.

    Checked finest-first so "quarterly figures for the year" reads as quarterly:
    the finer word is the one that says which SERIES, the coarser one usually
    says which window.
    """
    if not text:
        return None
    for granularity, pattern in _FREQUENCY_IN_TEXT:
        if pattern.search(text):
            return granularity
    return None


# Every way a single period is written in these questions, in canonical
# period_label form: 'YYYY', 'YYYY-Q#', 'YYYY-MM'.
# Arabic ordinals for quarters, and Arabic month names. Without these the
# Arabic half of the product had no period parsing at all: "الربع الأول 2026"
# scanned as the bare year 2026, so a question about one quarter was answered
# about the whole year, silently.
_AR_QUARTERS = {"الأول": 1, "الاول": 1, "الثاني": 2, "الثالث": 3, "الرابع": 4,
                "أول": 1, "اول": 1, "ثاني": 2, "ثالث": 3, "رابع": 4}
_AR_MONTHS = {
    "يناير": 1, "كانون الثاني": 1, "فبراير": 2, "شباط": 2, "مارس": 3, "آذار": 3,
    "أبريل": 4, "إبريل": 4, "ابريل": 4, "نيسان": 4, "مايو": 5, "أيار": 5,
    "يونيو": 6, "يونيه": 6, "حزيران": 6, "يوليو": 7, "يوليه": 7, "تموز": 7,
    "أغسطس": 8, "اغسطس": 8, "آب": 8, "سبتمبر": 9, "أيلول": 9,
    "أكتوبر": 10, "اكتوبر": 10, "تشرين الأول": 10, "نوفمبر": 11, "تشرين الثاني": 11,
    "ديسمبر": 12, "كانون الأول": 12,
}

# Longest first, so "تشرين الأول" is not matched as "تشرين" plus a stray ordinal.
_AR_Q_ALT = "|".join(sorted(map(re.escape, _AR_QUARTERS), key=len, reverse=True))
_AR_M_ALT = "|".join(sorted(map(re.escape, _AR_MONTHS), key=len, reverse=True))

_PERIOD_TOKEN = re.compile(
    r"(?P<q1>\d{4})[\s-]*q(?P<q1n>[1-4])"          # 2025-Q1 / 2025 q1
    r"|q(?P<q2n>[1-4])[\s-]*(?P<q2>\d{4})"          # Q1 2025 / Q1-2025
    r"|(?P<m1>\d{4})-(?P<m1n>0[1-9]|1[0-2])"         # 2025-05
    r"|(?P<mname>january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+(?P<my>\d{4})"    # May 2025
    # الربع الأول 2026 / الربع الأول من عام 2026
    r"|الربع\s+(?P<arq>" + _AR_Q_ALT + r")(?:\s+(?:من|في|لعام|لسنة|عام|سنة))*\s*(?P<ary>\d{4})"
    # مايو 2025 / أيار 2025
    r"|(?P<armname>" + _AR_M_ALT + r")\s+(?:من\s+)?(?:عام\s+|سنة\s+)?(?P<army>\d{4})"
    r"|(?P<y>\d{4})",                                 # 2025
    re.IGNORECASE,
)


def period_kind(label: str) -> str:
    """'q' | 'm' | 'y' — which kind of period a label denotes."""
    return "q" if "-Q" in str(label) else "m" if "-" in str(label) else "y"


def scan_period_labels(expr: Optional[str]) -> list[str]:
    """Every period this expression names, in the order written, deduplicated."""
    if not expr:
        return []
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
        elif m.group("arq"):
            label = f"{m.group('ary')}-Q{_AR_QUARTERS[m.group('arq')]}"
        elif m.group("armname"):
            label = f"{m.group('army')}-{_AR_MONTHS[m.group('armname')]:02d}"
        else:
            label = m.group("y")
        if label not in found:
            found.append(label)
    return found


def single_period_label(expr: Optional[str]) -> Optional[str]:
    """The one period this expression names, or None if it names none or several.

    Used to anchor a relative comparison on the period the user actually said:
    "real GDP in Q4 2025, and what was it one year earlier" is anchored on
    Q4 2025, not on the series' latest reading.
    """
    found = scan_period_labels(expr)
    return found[0] if len(found) == 1 else None


# No \b before the Arabic: word boundaries do not behave on Arabic script, and
# the word arrives prefixed ("بنفس الربع"), so an anchored match never fired.
_SAME_PERIOD = re.compile(
    r"\bsame\s+(quarter|month|period)\b|نفس\s*ال?(ربع|شهر|فترة)",
    re.IGNORECASE)


def parse_same_period_pair(expr: Optional[str]):
    """"Q1 2026 compared with the same quarter of 2025" -> ('2025-Q1', '2026-Q1').

    The phrase names two periods but writes the second one as a bare year,
    borrowing its quarter from the first. parse_period_pair saw '2026-Q1' and
    '2025', called them mixed granularities — which they are, read literally —
    and declined, so the question was answered "I need two specific periods to
    compare" when it had given both.

    With no year named ("the same quarter a year earlier") it steps back one
    year from the quarter that was named.
    """
    if not expr or not _SAME_PERIOD.search(expr):
        return None
    labels = scan_period_labels(expr)
    anchor = next((l for l in labels if period_kind(l) in ("q", "m")), None)
    if not anchor:
        return None
    year = next((l for l in labels if period_kind(l) == "y"), None)
    other = f"{year}-{anchor.split('-', 1)[1]}" if year else year_earlier_label(anchor)
    if not other or other == anchor:
        return None
    return tuple(sorted((anchor, other)))


def parse_period_pair(expr: Optional[str]):
    """Two period labels from one expression, for an explicit A-vs-B comparison.

    parse_period_expression() only ever describes ONE period, so "between Q1
    2025 and Q4 2025" fell through its range branch (which needs two bare
    years) into the single-quarter branch, retrieved Q1 alone, and answered
    "Could not find two distinct periods to compare" — F-011's exact question.

    Returns (label_a, label_b) in the order written, in period_label form, or
    None when the expression does not name two distinct periods.
    """
    found = scan_period_labels(expr)
    if len(found) < 2:
        return None
    # Mixed granularities ("2024 and Q1 2025") are not a like-for-like
    # comparison, so they are declined rather than guessed at.
    if len({period_kind(f) for f in found[:2]}) > 1:
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

    # "last year" / "the year before" — no digit, so the "last N years" pattern
    # below never matched them and the expression fell through to
    # "unspecified", which means NO window at all. The F-030 follow-up "WHAT
    # about the last year?" therefore returned the same latest reading it had
    # just given, which reads as the system ignoring the question.
    if re.search(r"\b(last|previous|prior)\s+year\b", text_l) or        re.search(r"\b(the\s+)?year\s+(before|earlier|ago)\b", text_l) or        re.search(r"السنة الماضية|العام الماضي|السنة السابقة|العام السابق", expr):
        return PeriodResolution(kind="previous_year", n_years=1, raw=expr)

    # "last N years" / "last N months"
    m = re.search(r"last\s+(\d+)\s+year", text_l)
    if m:
        return PeriodResolution(kind="last_n_years", n_years=int(m.group(1)), raw=expr)

    # ONE named period — month or quarter — from the same scanner the pair
    # parser uses. This was three separate English-only regexes duplicating
    # that scanner, so an Arabic quarter fell past all of them to the bare-year
    # branch at the bottom: "الربع الأول 2026" was answered about the whole of
    # 2026, with nothing in the answer to show the quarter had been dropped.
    labels = scan_period_labels(expr)
    if len(labels) == 1 and period_kind(labels[0]) in ("q", "m"):
        start, end = label_bounds(labels[0])
        return PeriodResolution(
            kind="single_quarter" if period_kind(labels[0]) == "q" else "single_month",
            start_date=start, end_date=end, raw=expr)

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


def year_earlier_label(period_label: Optional[str]) -> Optional[str]:
    """'2025-Q4' -> '2024-Q4', '2026-04' -> '2025-04', '2025' -> '2024'.

    Built by subtracting from the label rather than by stepping back four rows
    in the series, because a series with a gap in it would make "four rows
    back" a different quarter with nothing in the output to show it happened.
    """
    if not period_label:
        return None
    match = re.match(r"^(\d{4})(.*)$", str(period_label).strip())
    if not match:
        return None
    return f"{int(match.group(1)) - 1}{match.group(2)}"


# Relative comparisons, which carry no year to match on. "this year and the
# year before it" named no period the token scanner could see, so it was
# answered "I need two specific periods to compare" — asking the user to
# restate, in the system's vocabulary, something they had already said clearly.
_REL_PREV_YEAR = re.compile(
    r"\b(this|current|latest)\s+(year|quarter|month)\b.{0,30}?\b(year|one)\s+before\b"
    r"|\b(compared|versus|vs\.?|against)\b.{0,20}?\b(the\s+)?(previous|prior|last|preceding)\s+year\b"
    r"|\byear[\s-]on[\s-]year\b|\byear[\s-]over[\s-]year\b|\byoy\b"
    r"|\b(this|current)\s+year\s+(and|vs\.?|versus|against)\s+(the\s+)?(previous|prior|last)\s+year\b"
    # "…and what was it one year earlier?" — the phrase that produced the wrong
    # answer. It matched the "previous_year" WINDOW rule instead, which narrowed
    # retrieval to 2024 and left the comparison to fall back on the first and
    # last rows of that window: 2024-Q1 vs 2024-Q4, neither one asked for.
    r"|\b(a|one)\s+year\s+(earlier|ago|before|prior)\b"
    r"|\b(the\s+)?(same)\s+(quarter|month|period)\s+(a\s+year|last\s+year|of\s+last\s+year)\b"
    r"|\b12\s+months\s+(earlier|ago|before)\b"
    r"|قبل\s*(عام|سنة)|مقارنة\s*ب?العام\s*(الماضي|السابق)|على\s*أساس\s*سنوي",
    re.IGNORECASE)

_REL_PREV_TWO = re.compile(
    r"\b(last|previous|prior)\s+year\b.{0,30}?\bthe\s+year\s+before\s+(that|it)\b",
    re.IGNORECASE)

_REL_PREV_PERIOD = re.compile(
    r"\b(this|current|latest)\s+(quarter|month)\b.{0,30}?\b(previous|prior|last|preceding|one\s+before)\b"
    r"|\b(compared|versus|vs\.?|against)\b.{0,20}?\b(the\s+)?(previous|prior|preceding)\s+(quarter|month|period)\b"
    r"|\bquarter[\s-]on[\s-]quarter\b|\bqoq\b|\bmonth[\s-]on[\s-]month\b|\bmom\b",
    re.IGNORECASE)


def parse_relative_pair(expr: Optional[str]) -> Optional[str]:
    """Names a comparison the user expressed relative to now, not by date.

    Returns "prev_year", "prev_two_years" or "prev_period" — a description of
    WHICH two readings, to be resolved against the series that is actually
    available. Resolving it here against the calendar instead would ask for
    2026 on an indicator whose latest yearly reading is 2023, and report that
    as missing data when the user's question was answerable all along.
    """
    if not expr:
        return None
    # Checked first: "last year and the year before that" also matches the
    # prev_year pattern, and the more specific reading is the right one.
    if _REL_PREV_TWO.search(expr):
        return "prev_two_years"
    if _REL_PREV_PERIOD.search(expr):
        return "prev_period"
    if _REL_PREV_YEAR.search(expr):
        return "prev_year"
    return None


# A label the data could actually use. Anything else is a model invention.
_CANONICAL_LABEL = re.compile(r"^\d{4}(-Q[1-4]|-(0[1-9]|1[0-2]))?$")


def validate_period_labels(labels, available: Optional[set] = None,
                            want: int = 2) -> Optional[tuple]:
    """Accepts the intent agent's canonical period labels, or rejects them.

    This is the guard that makes an LLM fallback safe to use for periods at
    all. A wrong period is the one error class nothing downstream can catch:
    the numeric verifier confirms every figure traces to the data, and a figure
    for the wrong quarter does trace to the data. It passes every check and
    reads as a normal answer — which is exactly how "Q4 2025 vs one year
    earlier" once came back as 2024-Q1 vs 2024-Q4.

    So the model's output is never used as given. It must be well formed, the
    right count, the same kind of period on both sides, and — when `available`
    is supplied — actually present in the series being read. A label that fails
    any of those is discarded and the caller falls back to refusing, which is
    the outcome the user can see and correct.
    """
    if not labels or not isinstance(labels, (list, tuple)):
        return None
    clean = []
    for label in labels:
        text = str(label).strip().upper().replace("_", "-")
        if not _CANONICAL_LABEL.match(text):
            return None          # malformed: reject the whole set, not just this one
        if text not in clean:
            clean.append(text)
    if len(clean) != want:
        return None
    if len({period_kind(c) for c in clean}) > 1:
        return None              # "2024 and Q1 2025" is not like-for-like
    if available is not None and not all(c in available for c in clean):
        return None              # names a period this indicator does not report
    return tuple(clean)


# What KIND of period the question named, as a granularity. A year names a
# yearly figure, a quarter a quarterly one.
_LABEL_GRANULARITY = {"y": "yearly", "q": "quarterly", "m": "monthly"}


def granularity_from_labels(expr: Optional[str]) -> Optional[str]:
    """The granularity implied by the periods a question names.

    Without this the finest available series always won, so "what is GDP for
    2023" was answered 170.55 — Real GDP in 2023-Q4 — while the yearly row for
    2023 reads 696.697. A quarter reported as if it were the year is wrong by a
    factor of four and looks entirely plausible.

    "Compare 2023 to 2024" failed outright for the same reason: the quarterly
    series has no row labelled "2023", so both endpoints were missing.

    Returns None when the question names no period, or names several kinds —
    a mixed set is not evidence of anything and the usual rule should decide.
    """
    kinds = {period_kind(label) for label in scan_period_labels(expr)}
    if len(kinds) != 1:
        return None
    return _LABEL_GRANULARITY.get(kinds.pop())
