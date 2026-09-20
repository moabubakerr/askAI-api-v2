"""
How a figure should be displayed, read from SCAI's own `Format` field.

The catalogue splits presentation across two columns and the app was only ever
reading one of them. `unit_en` says "QAR"; `Format` says "bn0.0", meaning
billions to one decimal. Show the unit alone and 185.17 reads as 185 riyals
rather than 185.2 billion - which is what the tiles did while the prose beside
them said "QAR 185.2 billion", because the model inferred the scale from
context that the payload never carried.

This is systematic, not incidental: of the 30 published indicators whose
Format carries a scale, NONE repeat that scale in unit_en. It is the only
place the information exists.

  Real GDP                         Format "bn0.0"   Unit "QAR"    -> 185.2 billion QAR
  Number of International Visitors Format "0.000m"  Unit "Count"  -> 2.297 million
  Inflation                        Format "0.0"     Unit "%"      -> 2.6 %

Nothing here changes a stored value. It decides how the same number is written.
"""
import re
from typing import Optional

# "bn0.00", "0.000m", "0.0k" — the scale may lead or trail the digits.
_SCALE = re.compile(r"^(?P<lead>bn|m|k)?[\d.]*(?P<trail>bn|m|k)?$", re.IGNORECASE)

_SCALE_WORDS = {"bn": "billion", "m": "million", "k": "thousand"}

# Units that name no dimension of their own, so the scale word stands alone
# rather than being appended: "2.297 million", not "2.297 Count m".
_DIMENSIONLESS = {"", "count", "number", "na", "n/a", "none", "rank"}


def scale_from_format(format_str: Optional[str]) -> Optional[str]:
    """'bn' | 'm' | 'k' | None."""
    if not format_str:
        return None
    match = _SCALE.match(format_str.strip())
    if not match:
        return None
    scale = match.group("lead") or match.group("trail")
    return scale.lower() if scale else None


def decimals_from_format(format_str: Optional[str], default: int = 1) -> int:
    """Decimal places SCAI specifies. 'bn0.00' -> 2, '.1' -> 1, '0' -> 0."""
    if not format_str:
        return default
    digits = re.sub(r"[^\d.]", "", format_str)
    if "." not in digits:
        return 0
    return len(digits.split(".")[-1])


# Placeholders SCAI uses for "this figure has no unit". Printed literally they
# read as a unit called NA, which 48 published indicators were doing.
# "Count" and "Number" are here for the same reason: they name no dimension, so
# "1,234 Count" reads as a unit when the figure is just 1,234. Where a scale
# applies they become the scale word instead — "2.297 million".
_NO_UNIT = {"na", "n/a", "none", "-", "null", "count", "number"}


def _expand_scale_words(unit: str) -> tuple[str, Optional[str]]:
    """Turns SCAI's abbreviated scale into a word: 'bn QAR' -> ('billion QAR', 'bn').

    Returns the rewritten unit and which scale it already carried, or None.
    Whole tokens only, so 'months' is not read as 'm' and 'M3/MT' is left alone.
    Checked against the catalogue: every bare 'm', 'k' and 'bn' unit is a scale
    (Total Population in millions, Sector Jobs in thousands) — none is metres.
    """
    tokens = unit.replace(",", " ").split()
    found = None
    for i, token in enumerate(tokens):
        key = token.lower()
        if key in _SCALE_WORDS:
            found = key
            tokens[i] = _SCALE_WORDS[key]
    return " ".join(tokens), found


def display_unit(unit_en: Optional[str], format_str: Optional[str]) -> str:
    """The unit as it should appear beside the number, scale included.

    Spelled out and scale-first — "billion QAR", not "QAR bn". Both choices
    follow SCAI's own catalogue, where 64 units are written "bn QAR", "m QAR",
    "m USD", "m MT" with the scale leading, and one is already spelled out in
    full as "million ton-km". Abbreviating what they spell out, or reordering
    what they order, would make the same quantity read two different ways
    depending on which column it happened to come from.
    """
    unit = (unit_en or "").strip()
    if unit.lower() in _NO_UNIT:
        unit = ""
    unit, existing_scale = _expand_scale_words(unit)
    scale = scale_from_format(format_str)
    if not scale:
        return unit
    # The unit already says it — don't say it twice.
    if existing_scale:
        return unit
    if unit.lower() in _DIMENSIONLESS:
        return _SCALE_WORDS[scale]
    return f"{_SCALE_WORDS[scale]} {unit}"


def trim_zeros(value) -> str:
    """'3.680000' -> '3.68', '185.17' -> '185.17', '9.0' -> '9'.

    Postgres NUMERIC preserves the scale it was stored with, so a subtraction
    of two 2-decimal values arrives as Decimal('3.680000') and printed
    "a change of 3.680000 QAR bn" — six decimals of apparent precision on a
    figure that has two. Only padding is removed; no digit that carries
    information is touched, so the number still matches the payload exactly and
    still passes the numeric verifier.
    """
    if value is None:
        return ""
    text = str(value)
    if "." not in text or "e" in text.lower():
        return text
    text = text.rstrip("0").rstrip(".")
    return text or "0"
