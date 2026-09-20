"""
How a figure should be displayed, read from SCAI's own `Format` field.

The catalogue splits presentation across two columns and the app was only ever
reading one of them. `unit_en` says "QAR"; `Format` says "bn0.0", meaning
billions to one decimal. Show the unit alone and 185.17 reads as 185 riyals
rather than 185.2 billion — which is what the tiles did while the prose beside
them said "QAR 185.2 billion", because the model inferred the scale from
context that the payload never carried.

This is systematic, not incidental: of the 30 published indicators whose
Format carries a scale, NONE repeat that scale in unit_en. It is the only
place the information exists.

  Real GDP                         Format "bn0.0"   Unit "QAR"    -> 185.2 QAR bn
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
_NO_UNIT = {"na", "n/a", "none", "-", "null"}


def display_unit(unit_en: Optional[str], format_str: Optional[str]) -> str:
    """The unit as it should appear beside the number, scale included."""
    unit = (unit_en or "").strip()
    if unit.lower() in _NO_UNIT:
        unit = ""
    scale = scale_from_format(format_str)
    if not scale:
        return unit
    # Some units already spell it out ("bn QAR", "m QAR") — don't say it twice.
    if scale in unit.lower().replace(",", " ").split():
        return unit
    if unit.lower() in _DIMENSIONLESS:
        return _SCALE_WORDS[scale]
    return f"{unit} {scale}"
