"""
Cleaning helpers shared across the ETL. Every one of these was written against
the ACTUAL SCAI export files (checked by hand), not guessed generically:

- fix_mojibake: several free-text fields (e.g. Item_1_DataPointAnalysis) were
  UTF-8 bytes mis-decoded as cp1252 then re-saved as UTF-8, producing garbage
  like "QCBâ€™s" instead of "QCB's". Confirmed round-trip fix: encode('cp1252')
  -> decode('utf-8').
- maybe_b64_decode: General_Entities/Sectors *DescriptionEN/AR columns store
  some rows as base64-encoded HTML and other rows as plain text, inconsistently.
  Detect and decode per-row rather than assuming one format for the whole column.
- parse_period: SCAI periods appear in three shapes: 'YYYY', 'YYYY-Q#', 'YYYY-MM'.
  Normalize all to a first-of-period date plus an explicit granularity label,
  so the SQL agent can do date range filters and ORDER BY reliably.
- clean_null: the exports use the literal string 'NULL' instead of an empty
  cell in several tables (e.g. Champions.csv) — pandas won't catch this on its
  own without na_values configured, so it's handled explicitly.
"""
import base64
import math
import re
from datetime import date


def clean_null(value):
    if value is None:
        return None
    # pandas' newer string dtype can yield a float NaN for empty cells even
    # with keep_default_na=False — catch that before the isinstance(str) check.
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and value.strip().upper() in ("NULL", ""):
        return None
    return value


def fix_mojibake(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    try:
        fixed = text.encode("cp1252").decode("utf-8")
        return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def maybe_b64_decode(text: str) -> str:
    """Rows in these columns are inconsistently base64-encoded HTML or plain
    text. Try decoding; only use the decoded result if it round-trips to
    printable text, otherwise assume it was already plain text."""
    if not text or not isinstance(text, str):
        return text
    stripped = text.strip()
    if len(stripped) < 8 or len(stripped) % 4 != 0:
        return text
    if not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", stripped):
        return text
    try:
        decoded_bytes = base64.b64decode(stripped, validate=True)
        decoded = decoded_bytes.decode("utf-8")
        # printable check: reject if it decodes to mostly control characters
        printable_ratio = sum(c.isprintable() or c in "\n\r\t" for c in decoded) / max(len(decoded), 1)
        if printable_ratio > 0.95:
            return decoded
        return text
    except Exception:
        return text


_QUARTER_START_MONTH = {1: 1, 2: 4, 3: 7, 4: 10}


def parse_period(period: str):
    """Returns (period_date: date, granularity: str) or (None, None) if unparseable.
    Handles the three formats seen in the SCAI exports: 'YYYY', 'YYYY-Q#', 'YYYY-MM'."""
    if not period or not isinstance(period, str):
        return None, None
    period = period.strip()

    m = re.fullmatch(r"(\d{4})", period)
    if m:
        return date(int(m.group(1)), 1, 1), "yearly"

    m = re.fullmatch(r"(\d{4})-Q(\d)", period)
    if m:
        year, q = int(m.group(1)), int(m.group(2))
        return date(year, _QUARTER_START_MONTH[q], 1), "quarterly"

    m = re.fullmatch(r"(\d{4})-(\d{2})", period)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        return date(year, month, 1), "monthly"

    return None, None


def strip_html(text: str) -> str:
    """Rough HTML stripper for feeding article/entity content to the LLM as
    plain text context. Not meant for display — only for grounding prompts."""
    if not text or not isinstance(text, str):
        return text
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def to_float(value):
    value = clean_null(value)
    if value is None or value == "":
        return None
    try:
        result = float(str(value).replace(",", "").strip())
        return None if math.isnan(result) else result
    except ValueError:
        return None
