"""
Country / country-group resolver — deterministic, no LLM.

Directly targets:
  F-001: "compare Qatar vs Singapore" must return EXACTLY those two, never a
         default benchmark set of 6 countries.
  F-026: "compare" must not silently switch to a country-ranking function.
  F-027: "which country had the lowest inflation" needs an explicit set of
         countries to rank across, or an honest request for a period.
  Test-sheet expectations: "GCC" must resolve to exactly the 6 GCC members
  (minus any explicitly excluded), never mixed up with a generic benchmark
  list, and a missing country must produce an honest "No approved data for X"
  line rather than a silently dropped row or an invented figure.

Country names here are matched against the exact strings used in the SCAI
data (P03/P05 etc: 'KSA', 'UAE', 'Bahrain', 'Oman', 'Singapore', 'Kuwait',
'Norway', 'Egypt', 'Malaysia', 'Jordan', 'South Korea', 'Netherlands',
'Spain', 'Germany', ...) — user phrasing is normalized to those via the
alias map below.
"""
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Optional

from sqlalchemy import text
from app.db.executor import engine
from app.core.messages import _ARABIC_CHARS, _normalize

# GCC membership as used in SCAI's own country strings. Qatar itself is
# "domestic" (blank country_en in the data) and is handled separately by
# the retriever — it is not itself in this list.
GCC_COUNTRIES = ["KSA", "UAE", "Kuwait", "Bahrain", "Oman"]

COUNTRY_GROUPS = {
    "gcc": GCC_COUNTRIES,
    "gulf": GCC_COUNTRIES,
    "gulf cooperation council": GCC_COUNTRIES,
    # Arabic. Without these, "قارن التضخم بين قطر ودول مجلس التعاون" resolved no
    # group at all and compared Qatar against nothing.
    "دول مجلس التعاون": GCC_COUNTRIES,
    "مجلس التعاون الخليجي": GCC_COUNTRIES,
    "دول الخليج": GCC_COUNTRIES,
    "الخليج": GCC_COUNTRIES,
}

# Common user phrasings that don't exact-match the data's own strings.
ALIASES = {
    "saudi": "KSA", "saudi arabia": "KSA", "ksa": "KSA",
    "uae": "UAE", "emirates": "UAE", "united arab emirates": "UAE",
    "qatar": None,  # None = domestic, handled specially by the retriever
    # Arabic forms of the two that need an alias rather than a lookup, plus
    # Qatar's domestic sentinel. Everything else is translated from the
    # countries table (see _arabic_to_english), which already holds name_ar for
    # all 235 countries — it was simply never consulted, so an Arabic question
    # naming a country resolved nothing and the comparison silently became a
    # one-country answer.
    "قطر": None, "دولة قطر": None,
    "السعودية": "KSA", "المملكة العربية السعودية": "KSA",
    "الامارات": "UAE", "الإمارات": "UAE", "الامارات العربية المتحدة": "UAE",
}


@dataclass
class CountryResolution:
    resolved_countries: list        # exact country_en strings to query for
    unresolved_names: list          # user-named countries with no match in the data
    includes_qatar: bool
    is_group: bool
    group_name: Optional[str] = None


def _fetch_country_names() -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT country_en FROM published_data_points WHERE country_en IS NOT NULL AND country_en != ''")).fetchall()
    return [r[0] for r in rows]


def _arabic_to_english(name: str) -> Optional[str]:
    """Maps an Arabic country name onto its English name via the countries
    table, which carries name_ar for every country and was never used."""
    key = _normalize(name)
    if not key:
        return None
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT name_en, name_ar FROM countries WHERE name_ar IS NOT NULL AND name_ar <> ''"
        )).fetchall()
    best, best_score = None, 0.0
    for name_en, name_ar in rows:
        score = SequenceMatcher(None, key, _normalize(name_ar)).ratio()
        if score > best_score:
            best, best_score = name_en, score
    return best if best_score >= 0.85 else None


@lru_cache(maxsize=1)
def _arabic_names() -> dict:
    """Every country's Arabic name, keyed by the English one the data uses.

    The reverse of _arabic_to_english, and needed for the same reason in the
    other direction: an Arabic answer named every country in English —
    "سجّلت Bahrain أدنى Inflation" — because the countries table was consulted
    when reading a question and never when writing an answer.

    The data's own country_en strings are not always the table's name_en: SCAI
    writes "KSA" where the table says "Saudi Arabia". Both are keyed, so a
    lookup by either works.
    """
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT name_en, name_ar FROM countries "
                "WHERE name_ar IS NOT NULL AND name_ar <> ''"
            )).fetchall()
    except Exception:
        # A display convenience, never a reason to fail a request: without it
        # the answer keeps the English names it has always had.
        return {}
    names = {}
    for name_en, name_ar in rows:
        names[(name_en or "").strip().lower()] = name_ar.strip()
    # SCAI's own shorthands, which the countries table does not carry.
    for alias, target in ALIASES.items():
        if target and target.strip().lower() not in names:
            arabic = names.get(alias.strip().lower())
            if arabic:
                names[target.strip().lower()] = arabic
    names.setdefault("qatar", "قطر")
    names.setdefault("ksa", "المملكة العربية السعودية")
    names.setdefault("uae", "الإمارات العربية المتحدة")
    return names


def display_country(name: Optional[str], language: str = "en") -> str:
    """A country's name in the language being written.

    Falls back to the English name where SCAI records no Arabic one, so a
    missing translation costs a language rather than the country.
    """
    text_name = (name or "").strip()
    if not text_name or not str(language).lower().startswith("ar"):
        return text_name
    return _arabic_names().get(text_name.lower(), text_name)


def _match_one(name: str, known: list[str]) -> Optional[str]:
    key = name.strip().lower()
    # Arabic first: translate to the English name, then fall through to the
    # usual alias/fuzzy pipeline so "السعودية" -> "Saudi Arabia" -> "KSA".
    if _ARABIC_CHARS.search(key):
        if _normalize(key) in {_normalize(k) for k in ALIASES if _ARABIC_CHARS.search(k)}:
            for alias, value in ALIASES.items():
                if _ARABIC_CHARS.search(alias) and _normalize(alias) == _normalize(key):
                    return value
        english = _arabic_to_english(key)
        if english:
            key = english.strip().lower()
    if key in ALIASES:
        return ALIASES[key]  # may be None for Qatar
    if key in ("qatar",):
        return None
    best, best_score = None, 0.0
    for k in known:
        score = SequenceMatcher(None, key, k.lower()).ratio()
        if score > best_score:
            best, best_score = k, score
    return best if best_score >= 0.75 else None  # None here means "not found", ambiguous with Qatar sentinel — see resolve_countries


def resolve_countries(countries_mentioned: list[str], group_mentioned: Optional[str]) -> CountryResolution:
    known = _fetch_country_names()

    if group_mentioned:
        key = group_mentioned.strip().lower()
        if key in COUNTRY_GROUPS:
            group_countries = [c for c in COUNTRY_GROUPS[key] if c in known]
            return CountryResolution(group_countries, [], includes_qatar=True, is_group=True, group_name=group_mentioned)
        # unknown group name — don't invent a membership list
        return CountryResolution([], [group_mentioned], includes_qatar=True, is_group=True, group_name=group_mentioned)

    resolved, unresolved = [], []
    includes_qatar = False
    for name in countries_mentioned:
        key = name.strip().lower()
        # Qatar is domestic — stored as a blank country and added by the
        # retriever, never queried by name. The check was `key == "qatar"`,
        # so the Arabic "قطر" fell past it, failed to match any country_en, and
        # was reported as a country with no approved data. The user was told
        # there is no data for Qatar, in an answer built from Qatar's data.
        if key in ALIASES and ALIASES[key] is None:
            includes_qatar = True
            continue
        if key in ALIASES and ALIASES[key] is not None:
            resolved.append(ALIASES[key])
            continue
        match = _match_one(name, known)
        if match:
            resolved.append(match)
        else:
            unresolved.append(name)

    return CountryResolution(resolved, unresolved, includes_qatar, is_group=False)
