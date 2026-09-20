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

  Real GDP                         Format "bn0.0"   Unit "QAR"    -> 185.17 Bn QAR
  Number of International Visitors Format "0.000m"  Unit "Count"  -> 2.297 Mn
  Inflation                        Format "0.0"     Unit "%"      -> 2.6 %

Nothing here changes a stored value. It decides how the same number is written.
"""
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Optional

# "bn0.00", "0.000m", "0.0k" — the scale may lead or trail the digits.
_SCALE = re.compile(r"^(?P<lead>bn|m|k)?[\d.]*(?P<trail>bn|m|k)?$", re.IGNORECASE)

# Abbreviated and capitalised, scale first: "Bn QAR". SCAI writes "bn QAR" in
# the units it spells out; this keeps their order and their abbreviation and
# just capitalises it, so a figure reads the same whether its scale came from
# unit_en or from the Format column.
_SCALE_WORDS = {"bn": "Bn", "m": "Mn", "k": "K"}

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
    """Normalises the scale in a unit: 'bn QAR' -> ('Bn QAR', 'bn').

    Returns the rewritten unit and which scale it already carried, or None.
    Whole tokens only, so 'months' is not read as 'm' and 'M3/MT' is left alone.
    Checked against the catalogue: every bare 'm', 'k' and 'bn' unit is a scale
    (Total Population in millions, Sector Jobs in thousands) — none is metres.
    """
    tokens = unit.replace(",", " ").split()
    found = None
    for i, token in enumerate(tokens):
        key = token.lower()
        # "million ton-km" is spelled out in the catalogue; normalise it to the
        # same abbreviation everything else uses.
        key = {"billion": "bn", "million": "m", "thousand": "k"}.get(key, key)
        if key in _SCALE_WORDS:
            found = key
            tokens[i] = _SCALE_WORDS[key]
    return " ".join(tokens), found


# The Arabic side of the same three decisions. The catalogue carries unit_ar
# for every published detail, so an Arabic answer saying "185.17 Bn QAR"
# was reading a column it did not have to.
_SCALE_WORDS_AR = {"bn": "مليار", "m": "مليون", "k": "ألف"}
_NO_UNIT_AR = {"غير متاح", "لا ينطبق", "لا يوجد", "العدد", "عدد", "-"}
_DIMENSIONLESS_AR = {"", "العدد", "عدد", "غير متاح", "الترتيب"}


def _expand_scale_words_ar(unit: str) -> tuple[str, Optional[str]]:
    """Whether the Arabic unit already names its scale. Substring, not token:
    Arabic attaches prefixes, so "بمليار" carries مليار without standing alone.
    """
    for key, word in _SCALE_WORDS_AR.items():
        if word in unit:
            return unit, key
    return unit, None


def display_unit(unit_en: Optional[str], format_str: Optional[str],
                  language: str = "en", unit_ar: Optional[str] = None) -> str:
    """The unit as it should appear beside the number, scale included.

    Abbreviated and scale-first — "Bn QAR", not "QAR bn". Both choices
    follow SCAI's own catalogue, where 64 units are written "bn QAR", "m QAR",
    "m USD", "m MT" with the scale leading. Reordering what they order, or
    writing the scale one way here and another there, would make the same
    quantity read two different ways depending on which column it came from.
    """
    arabic = str(language).lower().startswith("ar")
    # Fall back to English when the Arabic unit is absent, rather than printing
    # nothing: a missing translation should cost the reader a language, not the
    # unit itself.
    if arabic and (unit_ar or "").strip():
        unit = (unit_ar or "").strip()
        no_unit, dimensionless = _NO_UNIT_AR, _DIMENSIONLESS_AR
        words, expand = _SCALE_WORDS_AR, _expand_scale_words_ar
    else:
        unit = (unit_en or "").strip()
        no_unit, dimensionless = _NO_UNIT, _DIMENSIONLESS
        words, expand = _SCALE_WORDS, _expand_scale_words
        arabic = False

    if unit.lower() in no_unit or unit in no_unit:
        unit = ""
    unit, existing_scale = expand(unit)
    scale = scale_from_format(format_str)
    if not scale:
        return unit
    # The unit already says it — don't say it twice.
    if existing_scale:
        return unit
    if unit.lower() in dimensionless or unit in dimensionless:
        return words[scale]
    return f"{words[scale]} {unit}"


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


def trim_decimal(value):
    """Removes stored padding from a numeric WITHOUT changing its value.

    trim_zeros fixes the text; this fixes the payload, which is what the
    frontend renders into its own tiles. Postgres NUMERIC keeps the scale it
    was stored with, so subtracting two 2-decimal values yields
    Decimal('3.680000') and the tile printed "+3.680000 Bn QAR" beside
    prose that correctly said 3.68.

    Decimal.normalize() alone is not safe here: it turns Decimal('100') into
    Decimal('1E+2'), which renders as "1E+2". Integers are quantized back to a
    plain form instead.
    """
    if not isinstance(value, Decimal):
        return value
    normalized = value.normalize()
    exponent = normalized.as_tuple().exponent
    if isinstance(exponent, int) and exponent > 0:
        return normalized.quantize(Decimal(1))
    return normalized


# The floor on displayed precision. SCAI's Format specifies one decimal for
# Real GDP, which is fine for a headline and too coarse beside a change of
# 19.062818 — and the raw value carries six decimals of stored noise. Two is
# the requested minimum; an indicator whose Format asks for MORE keeps it, so
# "2.297 million visitors" does not become "2.30 million".
MIN_DISPLAY_DECIMALS = 2


def round_for_display(value, decimals: Optional[int] = None):
    """Rounds a figure to the precision it should be read at.

    Padding is removed afterwards, so a whole number stays whole: a rank of 11
    is "11", not "11.00". Non-numeric values pass through untouched.
    """
    if isinstance(value, bool) or value is None:
        return value
    if not isinstance(value, (int, float, Decimal)):
        return value
    places = max(int(decimals) if decimals is not None else 0, MIN_DISPLAY_DECIMALS)
    try:
        quantum = Decimal(1).scaleb(-places)
        rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return value
    return trim_decimal(rounded)
