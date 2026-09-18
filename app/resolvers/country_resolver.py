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
from typing import Optional

from sqlalchemy import text
from app.db.executor import engine

# GCC membership as used in SCAI's own country strings. Qatar itself is
# "domestic" (blank country_en in the data) and is handled separately by
# the retriever — it is not itself in this list.
GCC_COUNTRIES = ["KSA", "UAE", "Kuwait", "Bahrain", "Oman"]

COUNTRY_GROUPS = {
    "gcc": GCC_COUNTRIES,
    "gulf": GCC_COUNTRIES,
    "gulf cooperation council": GCC_COUNTRIES,
}

# Common user phrasings that don't exact-match the data's own strings.
ALIASES = {
    "saudi": "KSA", "saudi arabia": "KSA", "ksa": "KSA",
    "uae": "UAE", "emirates": "UAE", "united arab emirates": "UAE",
    "qatar": None,  # None = domestic, handled specially by the retriever
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


def _match_one(name: str, known: list[str]) -> Optional[str]:
    key = name.strip().lower()
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
        if key == "qatar":
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
