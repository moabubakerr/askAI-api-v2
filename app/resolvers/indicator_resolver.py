"""
Indicator Resolver — deterministic matching, no LLM free-generation involved.

Directly targets:
  F-014: "tourists arrived" must resolve to "Number of international visitors"
         even without an exact string match. TESTED: plain string similarity
         (difflib) scores this pair at 0.167 — far below any usable threshold.
         This is a genuine synonym/paraphrase problem ("tourist" ≈ "visitor"),
         which is why matching here is embedding-based (see embeddings.py),
         not character-similarity-based. Difflib is kept only as a secondary
         signal that gives exact/near-exact phrasing a confidence boost.
  F-003, F-012: must NOT silently substitute a plausible-sounding wrong indicator
         (VC deals instead of GDP; Nominal GDP when only Real GDP exists) — if
         confidence is low, ambiguous, or a named contrastive term doesn't
         match, refuse honestly instead of guessing.
  F-021: matching should be confident enough not to require the user's exact
         phrase to hit the catalog string.

This is a similarity RETRIEVAL step, not text generation — it only ever
returns a name/ID that is a real row in indicator_details, so it doesn't
reintroduce the "LLM invents a number" risk; it can still pick the WRONG
real indicator, which is why the confidence/gap/contradiction checks below
exist and matter.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from difflib import SequenceMatcher
from typing import Optional

from sqlalchemy import text
from app.db.executor import engine
from app.resolvers.embeddings import (get_embedding, cosine_similarity, cosine_matrix,
                                       embed_keyed)
from app.core.messages import msg
from app.nlu.indicator_picker import pick_indicator

# Embedding cosine similarity is the primary score; difflib contributes a
# small bonus for exact/near-exact phrasing so "GDP" still resolves fast
# and confidently without relying on the embedding model's exact calibration.
EMBED_WEIGHT = 0.8
STRING_WEIGHT = 0.2

# ...and below this, difflib contributes NOTHING. The comment above has always
# described a bonus for near-exact phrasing, but the code applied a flat 0.2
# weight to every score including meaningless ones, so character noise decided
# between candidates the embedding had not separated.
#
# F-014 is exactly that failure. "tourists arrived" against the catalogue
# scores 0.34 for "Number of Visitors on the Visit Qatar website (M)", 0.34 for
# "Number of Medical Tourists" and 0.25 for "Number of International Visitors"
# — differences that mean nothing, since tourist/visitor is a synonym problem
# no character comparison can see. Weighted at 0.2 they were enough to put a
# website metric above the right answer.
#
# Gated, a near-exact name still gets its boost and noise cannot outvote the
# embedding. Answers become slightly more likely to be refused than before,
# since scores no longer get a free 0.05-0.08 from noise — the safer direction
# for a system whose first rule is not to answer with the wrong indicator.
STRING_MIN_TO_COUNT = 0.6

# Raised from 0.55 on measured evidence, not judgement. A calibration run
# against the real bge-m3 over known-answer questions gave:
#
#   highest FALSE positive   0.602   "number of penguins in Qatar" -> Number of Qatari Jobs
#   lowest  TRUE  positive   0.664   "GDP"                          -> Real GDP
#
# 0.68 was set from an earlier run whose scores were inflated by ungated string
# noise; once that was removed every score fell and 0.68 began refusing "GDP"
# at 0.664. 0.63 sits between the highest false positive and the lowest true
# one on the CURRENT scoring.
#
# At 0.55 the penguin question was answered, and so were "solar energy share"
# (0.568 -> "Energy") and the solar/teachers pair (0.597). Every correct match
# in that run scored 0.754 or above, so this sits in the gap and costs none of
# them. Re-run scripts/calibrate_resolver.py after any scoring change and move
# this number to whatever the new distribution says.
MIN_CONFIDENCE = 0.63
MIN_GAP_TO_RUNNER_UP = 0.05

# How much better than the threshold an UNPUBLISHED indicator must score before
# it may answer a data question. Published rows are SCAI's reviewed layer;
# unpublished ones are working data that happens to be in the same table.
UNPUBLISHED_MARGIN = 0.10

# Known contrastive term pairs in economic indicator naming. If the user's
# phrase names one side of a pair and the top-matched indicator's name only
# contains the OTHER side, treat it as a mismatch rather than a match —
# this is precisely how F-012 happened (asked for "Nominal GDP", the catalog
# only has "Real GDP", and the system answered anyway instead of refusing).
CONTRASTIVE_PAIRS = [
    ("nominal", "real"),
    ("gross", "net"),
    ("domestic", "foreign"),
    ("import", "export"),
    ("public", "private"),
    ("direct", "indirect"),
]


# A qualifier the user attached to the measure. Dropping one does not narrow an
# answer, it changes what is being measured — Non-Hydrocarbon GDP and GDP are
# different quantities, and so are GDP and GDP per capita.
#
# Kept separate from CONTRASTIVE_PAIRS because the failure is a different shape.
# A pair catches a match that says the OPPOSITE of the question; this catches one
# that says nothing about it at all, which the pair test cannot see.
_NEGATED = re.compile(r"\bnon[\s-]?(\w+)|\bexcluding\s+(\w+)|\bغير\s+(\S+)", re.IGNORECASE)


@lru_cache(maxsize=1)
def _discriminating_terms() -> frozenset:
    """Words the CATALOGUE itself uses to tell one indicator from another.

    This began as a hand-written list — "per capita", "hydrocarbon",
    "seasonally adjusted" — which is wrong for the same reason a list of
    follow-up phrasings is wrong: it only covers what someone thought of, and
    SCAI publishes 400-odd indicators whose names carry distinctions nobody
    writing this file would predict. A question using one of the others would
    sail past.

    So the distinctions are read from the names. A word that appears in SOME
    indicator names and not others is a word this catalogue discriminates on —
    that is what makes "Real" meaningful in "Real GDP", and it is equally what
    makes "Greenhouses" meaningful in "Crop Yield - Vegetables, Greenhouses".
    A word in nearly every name, or in none, discriminates nothing.

    Derived once and cached with the catalogue it came from, so it follows the
    data rather than this file. clear_catalog_cache() drops it.
    """
    from collections import Counter
    counts, total = Counter(), 0
    try:
        catalog = _fetch_catalog()
    except Exception:
        # This check used to be pure, and callers are entitled to assume it
        # cannot fail. An empty set means the catalogue-derived rule simply
        # finds nothing; the contrastive pairs and the negation rule still
        # apply, so a database hiccup costs a check rather than the request.
        return frozenset()
    for row in catalog:
        total += 1
        for word in set(re.findall(r"[\w؀-ۿ]{3,}", (row.get("name_en") or "").lower())):
            counts[word] += 1
    if not total:
        return frozenset()
    # Present in at least one name and at most a third of them. The upper bound
    # drops words like "rate", "total" and "share" that head half the
    # catalogue and separate nothing.
    ceiling = max(1, total // 3)
    return frozenset(w for w, n in counts.items() if n <= ceiling)


@lru_cache(maxsize=1)
def _catalogue_vocabulary() -> frozenset:
    """Every word that appears in ANY indicator name.

    Different question from _discriminating_terms, which asks which words tell
    indicators APART. This asks whether a word belongs to the catalogue's
    vocabulary at all — whether it could be part of some indicator's name,
    however common.
    """
    words = set()
    try:
        catalog = _fetch_catalog()
    except Exception:
        return frozenset()
    for row in catalog:
        for name in (row.get("name_en"), row.get("name_ar")):
            words.update(re.findall(r"[\w؀-ۿ]{3,}", (name or "").lower()))
    return frozenset(words)


def names_nothing_in_catalogue(phrase: str) -> bool:
    """Whether a phrase attempts no indicator name this catalogue could hold.

    "What is the latest recorded value?" — the canonical follow-up this system
    was built for, quoted in conversation.py as QC finding F-030 — was refused
    as the name of an indicator. has_identifying_content passes it, because
    "latest" and "recorded" are real words that survive the noise filter; it
    just has no way to know they are not the sort of word an indicator is
    called.

    The catalogue knows. A phrase whose every word is absent from all 583
    indicator names is not a failed attempt at naming one, it is scaffolding
    around a question about whatever is already being discussed.

    This is deliberately NOT a list of filler words. "Solar energy share" is
    also not in the catalogue as a name, but "energy" and "share" are
    catalogue words, so it reads as a genuine attempt and still earns the
    refusal that says SCAI publishes no such indicator — which is the answer
    that failure needs, and the opposite of silently answering about the
    previous one.
    """
    vocabulary = _catalogue_vocabulary()
    if not vocabulary:
        return False
    cleaned = normalize_indicator_phrase(phrase or "").lower()
    # Every token, at any length, so an empty phrase can be told apart from one
    # made entirely of fragments. "ir" — the typo for "it" — has no word long
    # enough to be a catalogue name, which is exactly the point: it names
    # nothing, and requiring three letters before counting it made it look as
    # though it named something.
    tokens = re.findall(r"[\w؀-ۿ]+", cleaned)
    if not tokens:
        return False
    candidates = [w for w in tokens
                  if len(w) >= 3 and w not in _CONTENTLESS and not w.isdigit()]
    return not any(w in vocabulary for w in candidates)


def _same_word(a: str, b: str) -> bool:
    """Whether two words are the same word, allowing for inflection.

    Exact matching made "Qatari" and "qataris" different words, so a question
    about "economically active Qataris" read the catalogue's "Qatari Nationals
    (Economically Active)" as dropping a distinction it plainly carries. Two
    such near-misses in one phrase were enough to refuse a correct match — and
    did, silently turning a specific refusal that named both indicators into a
    generic one that named neither.

    A prefix test rather than a stemmer: four characters of agreement separates
    plural from singular without pulling unrelated words together, and adding a
    stemming dependency for one suffix would be the larger change.
    """
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    return len(shorter) >= 4 and longer.startswith(shorter)


def _dropped_discriminators(phrase: str, matched_name: str) -> list[str]:
    """Words the question used, the catalogue discriminates on, and the match
    does not carry."""
    matched_words = set(re.findall(r"[\w؀-ۿ]{3,}", matched_name.lower()))
    terms = _discriminating_terms()
    return [w for w in re.findall(r"[\w؀-ۿ]{3,}", (phrase or "").lower())
            if w in terms and not any(_same_word(w, m) for m in matched_words)]


def _has_contradiction(phrase: str, matched_name: str) -> bool:
    phrase_l, matched_l = phrase.lower(), matched_name.lower()
    for a, b in CONTRASTIVE_PAIRS:
        if a in phrase_l and a not in matched_l and b in matched_l:
            return True
        if b in phrase_l and b not in matched_l and a in matched_l:
            return True

    # A NEGATED subject the match does not carry. "What does non-hydrocarbon GDP
    # mean?" resolved to "Real GDP" and returned Real GDP's definition — a
    # fluent, confident answer to a question about a different quantity, with
    # nothing to tell the reader the subject had been changed. The pair test
    # above could not catch it: it fires only when the match names the OPPOSITE
    # side, and "Real GDP" names neither side, it simply drops the restriction.
    #
    # General rather than a list of known qualifiers — "non-oil", "non-financial",
    # "non-resident" and whatever the catalogue grows next all behave the same,
    # and enumerating them would be one short every time.
    for match in _NEGATED.finditer(phrase_l):
        subject = next((g for g in match.groups() if g), "")
        if subject and subject not in matched_l:
            return True

    # And a qualifier that is not a negation but narrows the measure just as
    # much: GDP per capita is not GDP. Which words those are is read from the
    # catalogue rather than listed here — see _discriminating_terms.
    #
    # Two of them, not one. A single dropped word is ordinary: names are long,
    # questions are short, and "what is Qatar's GDP" legitimately matches "Real
    # GDP" while dropping "Qatar". Two independent distinctions the catalogue
    # draws, both absent from the match, is the match being a different measure.
    return len(_dropped_discriminators(phrase, matched_name)) >= 2


@dataclass
class IndicatorMatch:
    indicator_detail_id: str
    indicator_id: str
    name_en: str
    is_published: bool
    # P02's PublishedIndicatorDetailId — a DIFFERENT GUID from
    # indicator_detail_id, and the only key published_data_points can be
    # queried by (schema.sql:86). Carried here because the caller cannot
    # derive it from indicator_detail_id.
    published_detail_id: Optional[str]
    is_active: Optional[bool]
    unit_en: Optional[str]
    unit_ar: Optional[str]
    polarity_en: Optional[str]
    data_source_en: Optional[str]
    format: Optional[str]
    definition_en: Optional[str]
    definition_ar: Optional[str]
    confidence: float
    # SCAI's own Arabic name for this indicator. It was already being fetched
    # and already used to MATCH an Arabic question — and then dropped, so an
    # Arabic answer named the indicator in English. Carried now for the same
    # reason unit_ar and definition_ar are: the catalogue holds the translation,
    # so nothing has to invent one.
    name_ar: Optional[str] = None

    def display_name(self, language: str = "en") -> str:
        """The name to SHOW, in the language being written.

        Falls back to English when SCAI has recorded no Arabic name, rather than
        showing nothing: a missing translation should cost the reader a
        language, not the name of the thing they asked about.
        """
        if str(language).lower().startswith("ar") and (self.name_ar or "").strip():
            return self.name_ar.strip()
        return (self.name_en or "").strip()


@dataclass
class ResolutionResult:
    match: Optional[IndicatorMatch]
    status: str          # "resolved" | "ambiguous" | "not_found" | "inactive"
    candidates: list      # for "ambiguous" — the top few near-tied matches
    message: Optional[str] = None   # honest, user-facing explanation when not resolved


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


# Words that appear in the QUESTION but carry no signal about WHICH indicator is
# meant, because they are already extracted into their own slots and because
# every indicator here is a Qatari one.
_PHRASE_NOISE = re.compile(
    r"\b(in|for|of|the|a|an)?\s*(qatar'?s?|qataris?|gcc|gulf|"
    r"quarterly|monthly|yearly|annual(ly)?|per\s+(quarter|month|year)|"
    # "growth" and "change" describe the COMPUTATION, not the indicator. "GDP
    # growth" failed to resolve at all — pulled toward "Annual Growth in Labor
    # Productivity" and "Debt to GDP Ratio" — while plain "GDP" resolves to
    # Real GDP immediately. The growth itself comes from SCAI's vetted YoY
    # field, which the same word already triggers downstream.
    # Safe because the original phrase is scored alongside the stripped one, so
    # an indicator genuinely named "...Growth..." still matches itself.
    r"growth|change)\b",
    re.IGNORECASE,
)

# The Arabic equivalents. "نسبة" (rate/percentage) is Arabic's "Qatar": it
# prefixes a great many indicator names, so it separates none of them, yet it
# is long enough to dominate character similarity. "نسبة التضخم" came back as a
# three-way tie between التضخم (Inflation), نسبة السمنة (obesity rate) and
# نسبة الدين إلى الناتج المحلي (debt-to-GDP) — two of which share only the
# word the user did not mean.
_PHRASE_NOISE_AR = re.compile(
    "|".join([
        "نسبة", "معدل", "مؤشر", "قيمة",
        "في قطر", "لقطر", "بقطر", "قطر", "القطرية", "القطري", "دولة قطر",
        "دول مجلس التعاون", "مجلس التعاون", "الخليجية", "الخليج",
        "سنوي", "سنوياً", "سنويا", "شهري", "شهرياً", "شهريا",
        "ربع سنوي", "ربعي", "السنوي", "الشهري",
    ])
)


# Period expressions are extracted into period_expression and then left in the
# indicator phrase as well, where they are pure noise. "tourists arrived into
# Qatar in May 2025" was matched as a whole string and lost to "Number of
# Visitors on the Visit Qatar website (M)" — F-014, the paraphrase case this
# resolver exists for. Stripped, the phrase is "tourists arrived", which is
# what has to reach the embedding.
_PHRASE_PERIOD = re.compile(
    r"\b(in|for|during|of|as of|at)?\s*("
    r"q[1-4][\s-]*\d{4}|\d{4}[\s-]*q[1-4]|"
    r"\d{4}-\d{2}|"
    r"(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+\d{4}|"
    r"last\s+\d+\s+(years?|months?|quarters?)|"
    r"last\s+year|this\s+year|\d{4}"
    r")\b",
    re.IGNORECASE,
)


# Interrogative openings. Pure noise for matching an indicator NAME, and they
# matter more now that the whole question is used when the intent agent does
# not isolate a phrase.
_PHRASE_QUESTION = re.compile(
    r"^\s*(what('s| is| was| are| were)?|how (many|much|fast|is|are|did|does)|"
    r"show( me)?|give me|tell me|list|which|when (was|is)|who)\b[\s,:]*",
    re.IGNORECASE,
)


# Everyday vocabulary that no embedding reliably connects to the catalogue's
# own wording. Measured against the real model, "How fast are prices rising"
# ranks Inflation FIRST — the definition embedding got the ranking right — but
# at 0.457, far below any threshold that also refuses "number of penguins in
# Qatar" at 0.602. Ranking is not the problem; absolute similarity is, and no
# cutoff fixes it.
#
# Each entry is an extra PHRASING of the question, not an answer: the aliased
# form is scored alongside the user's own words and still has to clear
# MIN_CONFIDENCE, survive the contradiction check and win its tie. So a wrong
# alias cannot force a wrong answer, it can only offer a candidate.
#
# Every target below was checked to exist, be published, and have data. This
# table is editorial and belongs with SCAI: it encodes which everyday term maps
# to which published metric, which is their judgement to make, not the model's.
INDICATOR_ALIASES = {
    "prices": "Inflation",
    "price level": "Inflation",
    "cost of living": "Inflation",
    "inflation rate": "Inflation",
    "how fast prices are rising": "Inflation",
    "الأسعار": "Inflation",
    "غلاء المعيشة": "Inflation",

    "economic output": "Real GDP",
    "economic growth": "Real GDP",
    "size of the economy": "Real GDP",
    "gdp": "Real GDP",
    "الناتج المحلي": "Real GDP",

    "non-oil economy": "Non-Hydrocarbon Real GDP",
    "non-hydrocarbon economy": "Non-Hydrocarbon Real GDP",

    # "non-oil" is how everyone outside the catalogue says "non-hydrocarbon".
    # Nothing bridged the two, so "How much of Qatar's exports are non-oil?"
    # was refused as an indicator that does not exist — while the indicator
    # sat there, published, with the answer in it.
    # "diversification of X" is how the strategy documents say it; the
    # catalogue says "Non-Hydrocarbon X as a share of total X". Nothing bridged
    # them, so "compare the diversification of exports with the diversification
    # of government revenues" matched neither side.
    "diversification of exports": "Non-Hydrocarbon Exports (share of total exports)",
    "export diversification": "Non-Hydrocarbon Exports (share of total exports)",
    "diversification of government revenues":
        "Non-Hydrocarbon Government Revenues as Share of Government Revenues",
    "diversification of government revenue":
        "Non-Hydrocarbon Government Revenues as Share of Government Revenues",
    "revenue diversification":
        "Non-Hydrocarbon Government Revenues as Share of Government Revenues",
    "fiscal diversification":
        "Non-Hydrocarbon Government Revenues as Share of Government Revenues",
    "تنويع الصادرات": "Non-Hydrocarbon Exports (share of total exports)",
    "تنويع الإيرادات": "Non-Hydrocarbon Government Revenues as Share of Government Revenues",

    # Deliberately NOT a bare "exports". That alias was tried and made things
    # worse: "how much of Qatar's exports are non-oil?" does not contain the
    # contiguous phrase "non-oil exports", so only the bare alias matched and a
    # question about the non-hydrocarbon share was pointed at total exports in
    # riyals. Bare "exports" is genuinely ambiguous — between Export Cost and
    # four different "Sector Exports" — and ambiguity is now resolved by asking
    # the model to choose from the real candidates, which is what
    # _pick_from_ambiguous does.
    "total exports": "Total Exports (Goods and Services)",
    "إجمالي الصادرات": "Total Exports (Goods and Services)",

    # The catalogue calls it "World Competitiveness Index Rank"; nobody asks
    # for it by that name.
    "competitiveness": "World Competitiveness Index Rank",
    "global competitiveness": "World Competitiveness Index Rank",
    "competitiveness ranking": "World Competitiveness Index Rank",
    "how competitive": "World Competitiveness Index Rank",
    "التنافسية": "World Competitiveness Index Rank",

    "non-oil exports": "Non-Hydrocarbon Exports (share of total exports)",
    "non oil exports": "Non-Hydrocarbon Exports (share of total exports)",
    "nonoil exports": "Non-Hydrocarbon Exports (share of total exports)",
    "non-hydrocarbon exports": "Non-Hydrocarbon Exports (share of total exports)",
    "exports that are not oil": "Non-Hydrocarbon Exports (share of total exports)",
    "الصادرات غير النفطية": "Non-Hydrocarbon Exports (share of total exports)",

    # The catalogue name carries a "(%)" and the words "as a Percentage of
    # GDP"; nobody types that. "public debt", "debt position", "how indebted"
    # all mean this one.
    "public debt": "Public Debt as a Percentage of GDP (%)",
    "public-debt": "Public Debt as a Percentage of GDP (%)",
    "government debt": "Public Debt as a Percentage of GDP (%)",
    "national debt": "Public Debt as a Percentage of GDP (%)",
    "debt position": "Public Debt as a Percentage of GDP (%)",
    "debt to gdp": "Public Debt as a Percentage of GDP (%)",
    "debt-to-gdp": "Public Debt as a Percentage of GDP (%)",
    "الدين العام": "Public Debt as a Percentage of GDP (%)",

    "tourists": "Number of International Visitors",
    "tourist arrivals": "Number of International Visitors",
    "visitors": "Number of International Visitors",
    "السياح": "Number of International Visitors",
    "الزوار": "Number of International Visitors",

    "trade surplus": "Trade Balance (Goods & Services)",
    "trade deficit": "Trade Balance (Goods & Services)",
    "trade balance": "Trade Balance (Goods & Services)",
    "الميزان التجاري": "Trade Balance (Goods & Services)",

    "government income": "Government Revenues",
    "state revenues": "Government Revenues",
    "إيرادات الحكومة": "Government Revenues",
}


# Words that cannot identify an indicator on their own. What is left after
# these and the noise patterns is the substance of the question.
_CONTENTLESS = {
    "for", "the", "a", "an", "of", "in", "on", "at", "to", "and", "or",
    "year", "years", "period", "time", "data", "value", "values", "figure",
    "figures", "number", "is", "was", "were", "are", "be", "it", "this", "that",
    "please", "me", "my", "we", "you",
    # Words a follow-up leans on while naming nothing: "what about in 2024",
    # "same for 2025", "and those again".
    "about", "regarding", "concerning", "same", "those", "these", "them",
    "again", "instead", "also", "too", "now", "then", "show", "tell", "give",
    "what", "how", "which", "when",
    "نفس", "أيضا", "ايضا", "كذلك", "عنها", "عنه", "بخصوص",
    "سنة", "عام", "الفترة", "بيانات", "قيمة", "رقم", "في", "من", "عن", "على",
}


def has_identifying_content(phrase: str) -> bool:
    """Whether anything is left to match on once noise is removed.

    "what is the GDP for the year 2034" can reach the resolver as "for the year
    2034": the period is stripped, and what remains is scaffolding. Matching
    that against the catalogue and reporting 'no indicator matching "for the
    year 2034"' quotes a string the user never meant as a name and diagnoses
    the wrong problem — they did not misname an indicator, they named none.
    """
    cleaned = normalize_indicator_phrase(phrase or "")
    words = [w for w in re.findall(r"[\w؀-ۿ]+", cleaned.lower())
             if w not in _CONTENTLESS and len(w) > 1 and not w.isdigit()]
    return bool(words)


def resembles_catalogue_name(phrase: str, cutoff: float = 0.85) -> bool:
    """Whether the WHOLE phrase is close to one indicator's actual name.

    The test that decides if a comma-separated phrase is one indicator or
    several. "Crop Yield - Vegetables, Greenhouses" is a real name and must not
    be torn in half; "GDP growth, inflation, and government revenues" is three
    questions and must be.

    Character similarity, deliberately: this asks whether the user typed a
    NAME, which is a spelling question, not a meaning one. Using confidence
    instead made the answer depend on whatever the embedding and the alias
    table happened to think — and once "gdp" aliased to "Real GDP", the
    three-metric question resolved confidently to one indicator and stopped
    being split at all.
    """
    if not phrase:
        return False
    target = phrase.strip().lower()
    for row in _fetch_catalog():
        for name in (row.get("name_en"), row.get("name_ar")):
            if name and _similarity(target, name) >= cutoff:
                return True
    return False


def alias_forms(phrase: str) -> list:
    """Catalogue names implied by everyday wording in the phrase.

    The most specific alias wins. Every matching alias used to contribute a
    form, so "how much of Qatar's exports are non-oil" matched both "non-oil
    exports" and the bare "exports" — two different indicators, both boosted,
    and a question with an obvious answer came back as a three-way choice.

    A shorter alias contained in a longer one that also matched is dropped:
    "exports" is what "non-oil exports" is made of, not a second reading of
    the question. Aliases that merely happen to both appear ("inflation and
    exports") are unaffected, since neither contains the other.
    """
    text = (phrase or "").lower()
    matched = [term for term in INDICATOR_ALIASES if term in text]
    specific = [term for term in matched
                if not any(other != term and term in other for other in matched)]
    hits = []
    for term in specific:
        target = INDICATOR_ALIASES[term]
        if target not in hits:
            hits.append(target)
    return hits


def normalize_indicator_phrase(phrase: str) -> str:
    """Strips country and frequency words from the user's phrase before matching.

    "Qatar" is the single most damaging word a question can contain. It is in
    the context of every indicator in this dataset, so it discriminates between
    none of them — yet character similarity weights it heavily. Measured on the
    real catalogue, "Qatar quarterly GDP" scored Real GDP 0.444, "FDI value to
    Qatar" 0.417 and "Qatari Nationals" 0.421: three candidates inside 0.03,
    which trips the ambiguity check and asks the user to choose between an
    indicator they meant and two they have never heard of. The same phrase
    reduced to "GDP" scores 0.545 / 0.062 / 0.000.

    The country and the frequency are already captured in countries_mentioned
    and explicit_frequency, so leaving them here counts them twice: once as
    filters, once as evidence about the indicator's NAME.

    Applied only to the query, never to catalogue names — plenty of real
    indicators are named "... in Qatar" and must keep it.
    """
    cleaned = _PHRASE_QUESTION.sub(" ", (phrase or "").strip())
    cleaned = _PHRASE_PERIOD.sub(" ", cleaned)
    cleaned = _PHRASE_NOISE.sub(" ", cleaned)
    cleaned = _PHRASE_NOISE_AR.sub(" ", cleaned)
    # Trailing punctuation survives into the phrase once the whole question is
    # used as the fallback: "What was inflation in May 2025?" reduced to
    # "inflation ?", and that stray token is compared against every catalogue
    # name.
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-?!;:؟")
    # If stripping leaves nothing to match on, the words were the question.
    return cleaned if len(cleaned) >= 3 else (phrase or "").strip()


DEFINITION_CHARS_FOR_EMBEDDING = 240


@lru_cache(maxsize=1)
def _catalog_embeddings_cached(names: tuple, texts: tuple):
    plain = embed_keyed({n: n for n in names})
    enriched = embed_keyed(dict(zip(names, texts)))
    return {name: (plain[name], enriched[name]) for name in names}


def _catalog_embeddings(catalog):
    """Each indicator gets TWO vectors and keeps whichever matches better.

    Embedding only "name + definition" would have cost the exact-name case: a
    question that IS the catalogue name scored a perfect 1.000 against the bare
    name and necessarily less against name-plus-prose. Embedding only the name
    loses the synonym — nothing in "Number of International Visitors" says
    "tourist". Keeping both views costs one extra vector per indicator, paid
    once per process, and neither case has to lose.
    """
    names = tuple(row["name_en"] for row in catalog)
    texts = tuple(_embedding_text(row) for row in catalog)
    return _catalog_embeddings_cached(names, texts)


def _embedding_text(row: dict) -> str:
    """What an indicator is embedded AS: its name, plus the start of its
    definition when there is a real one."""
    name = (row.get("name_en") or "").strip()
    definition = (row.get("definition_en") or "").strip()
    # Strip the markup BEFORE testing for a placeholder: the CMS stores an
    # empty definition as "<p>-</p>", which is not equal to "-" and slipped
    # through, embedding "Number of Medical Tourists. -".
    definition = re.sub(r"<[^>]+>", " ", definition)
    definition = re.sub(r"\s+", " ", definition).strip(" .-")
    if not definition:
        return name
    return f"{name}. {definition[:DEFINITION_CHARS_FOR_EMBEDDING]}"


# The catalogue changes only when the ETL runs, and that query counts data
# points with two correlated subqueries per row — 644 rows, each scanning the
# two largest tables. It was running on EVERY resolution, and recent routing
# work made a three-metric question resolve five times: the multi-metric guard,
# the macro probe, and once per named metric.
#
# Cached for the process instead. The consequence is that a RUNNING api will
# not see a reload until it restarts, so `docker compose run --rm etl` must be
# followed by `docker compose restart api`. That is written down in the README
# rather than left as folklore.
@lru_cache(maxsize=1)
def _fetch_catalog_cached() -> tuple:
    return tuple(_fetch_catalog_uncached())


def _fetch_catalog() -> list[dict]:
    return list(_fetch_catalog_cached())


def clear_catalog_cache() -> None:
    """Drops the cached catalogue and every vector derived from it."""
    _fetch_catalog_cached.cache_clear()
    _catalog_embeddings_cached.cache_clear()
    # Also derived from the catalogue, so it goes stale the same way: an
    # indicator added with a new distinction in its name is not one this can
    # discriminate on until it is re-read.
    _discriminating_terms.cache_clear()
    # Same provenance, same staleness: an indicator added with a word no other
    # name uses is not part of the vocabulary until this is re-read, and a
    # question using that word would read as naming nothing.
    _catalogue_vocabulary.cache_clear()


def _fetch_catalog_uncached() -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT d.indicator_detail_id, d.indicator_id, d.name_en, d.name_ar,
                   d.is_published, d.published_detail_id,
                   i.is_active, d.unit_en, d.unit_ar, d.polarity_en,
                   d.data_source_en, d.format,
                   d.definition_en, d.definition_ar,
                   (SELECT COUNT(*) FROM published_data_points p
                     WHERE p.published_indicator_detail_id = d.published_detail_id)
                 + (SELECT COUNT(*) FROM indicator_values v
                     WHERE v.indicator_detail_id = d.indicator_detail_id) AS data_point_count
            FROM indicator_details d
            JOIN indicators i ON i.indicator_id = d.indicator_id
            WHERE d.name_en IS NOT NULL
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


def has_usable_definition(row) -> bool:
    """A definition of "-" or "" is a placeholder, not a definition — 383 of the
    644 detail rows carry one of those. Accepts either a catalog dict or an
    IndicatorMatch."""
    get = row.get if isinstance(row, dict) else lambda k: getattr(row, k, None)
    for key in ("definition_en", "definition_ar"):
        value = (get(key) or "").strip()
        if value and value != "-":
            return True
    return False


def resolve_indicator(phrase: str, require_data: bool = True,
                      require_definition: bool = False,
                      language: str = "en") -> ResolutionResult:
    """require_data=True (the default, for any question that needs figures)
    ignores catalog entries that carry no data points at all.

    140 of the 644 indicator_details rows have zero data points — empty catalog
    stubs like "GDP" and "GDP Growth Demo" sitting alongside the real,
    fully-populated "Real GDP". Because those stubs carry the shorter, more
    generic name, they beat the real indicator on both embedding and string
    similarity for a query like "GDP", and the answer became
    "no data points in the approved dataset" while 70 real GDP points sat one
    row away. An indicator with no data cannot answer a data question, so it
    should not be a candidate for one.

    Definition lookups pass require_data=False: a stub can still carry a
    perfectly good definition, and no figures are being claimed."""
    if not phrase or not phrase.strip() or not has_identifying_content(phrase):
        return ResolutionResult(None, "not_found", [],
                                 msg("no_indicator_in_question", language))

    catalog = _fetch_catalog()
    if require_data:
        catalog = [r for r in catalog if (r.get("data_point_count") or 0) > 0]
    if require_definition:
        catalog = [r for r in catalog if has_usable_definition(r)]
    if not catalog:
        return ResolutionResult(None, "not_found", [], msg("empty_catalog", language))

    # Match on the phrase with country/frequency noise removed; the original is
    # kept for the messages, so a refusal still quotes what the user typed.
    match_phrase = normalize_indicator_phrase(phrase)
    # Score against BOTH the stripped phrase and what the user actually typed,
    # taking whichever fits better.
    #
    # Stripping helps the natural question ("Qatar quarterly GDP" -> "GDP") and
    # hurts the literal one: 32 catalogue names contain "Qatar" or "Qatari" and
    # 52 Arabic names begin with "نسبة", so typing a name exactly — "Number of
    # Qatari Jobs" — was being stripped into something that also matched
    # "Number of Non-Qatari Jobs". Measured over all 459 names with data,
    # stripping alone cost 2 self-matches in each language. Keeping both forms
    # costs nothing and keeps the gain.
    original_phrase = (phrase or "").strip()
    phrases = [match_phrase] if match_phrase == original_phrase else [match_phrase, original_phrase]
    # Everyday wording gets the catalogue's own name added as a further form to
    # score. It competes; it does not decide.
    phrases += [a for a in alias_forms(phrase) if a not in phrases]

    # Each entry is embedded as its NAME plus the opening of its definition.
    # The name alone carries no synonym: nothing in "Number of International
    # Visitors" says "tourist", so "tourists arrived into Qatar" lost to
    # "Number of Medical Tourists", which contains the word literally and has
    # no definition at all. The definition — "non-resident travelers entering
    # Qatar for leisure, business or other activities" — is the match the
    # question needs.
    # Truncated because only the opening sentence describes WHAT is measured;
    # the rest is methodology, and a longer text dilutes the vector.

    phrase_embeddings = [get_embedding(p) for p in phrases]

    scored = _score_catalog(catalog, phrases, phrase_embeddings)
    # Definition lookups may legitimately land on an unpublished stub; data
    # questions should not.
    if require_data:
        scored = _prefer_published(scored, MIN_CONFIDENCE)
    result = _resolve_scored(scored, phrase, language)
    if result.status == "not_found":
        rescued = _pick_from_near_misses(scored, phrase, language)
        if rescued:
            return rescued
    # Ambiguity is the question the picker exists to answer — "which of these
    # did you mean" — and we were about to put it to the user. Asking the model
    # first, over the same candidates and with the same validation, is strictly
    # better than asking a person to choose between three catalogue names; if
    # it declines, the user is still asked.
    if result.status == "ambiguous" and result.candidates:
        chosen = pick_indicator(phrase, [c.name_en.strip() for c in result.candidates],
                                 language)
        if chosen:
            match = next((c for c in result.candidates
                          if c.name_en.strip() == chosen), None)
            if match:
                return ResolutionResult(match, "resolved", [])
    return result


# The band where the embeddings ranked something sensibly but not confidently.
# Below this the field is noise and there is nothing worth asking about; above
# it the resolver has already answered. Measured: the true answers that were
# being refused scored 0.542 and 0.573, and the worst nonsense reached 0.602.
NEAR_MISS_FLOOR = 0.45
NEAR_MISS_CANDIDATES = 8


def _pick_from_near_misses(scored, phrase, language):
    """Ask the model to choose among names the embeddings already ranked.

    Calibration showed a gap no threshold can close: "public debt" ranks
    Public Debt as a Percentage of GDP (%) FIRST, at 0.573, and "number of
    penguins in Qatar" reaches 0.602 on a Qatari jobs indicator. The ranking is
    right and the confidence is not, so the choice — not the score — has to
    decide it.

    Constrained on both sides. The model only ever sees real, published names
    that already scored plausibly, and its answer is checked back against that
    list, so the worst it can do is return None and leave the refusal in place.
    """
    candidates = [row for row, score in scored
                  if score >= NEAR_MISS_FLOOR and row.get("is_published")][:NEAR_MISS_CANDIDATES]
    if not candidates:
        return None
    names = [row["name_en"].strip() for row in candidates]
    chosen = pick_indicator(phrase, names, language)
    if not chosen:
        return None
    row = next((r for r in candidates if r["name_en"].strip() == chosen), None)
    if not row:
        return None
    # Re-enter the normal path with this row first, so every downstream check —
    # inactive, no-data, the ambiguity tie-break — still applies. The score is
    # kept as it was: this decided WHICH indicator, not how sure we are.
    reordered = ([(r, s) for r, s in scored if r is row]
                 + [(r, s) for r, s in scored if r is not row])
    picked_score = max(reordered[0][1], MIN_CONFIDENCE)
    return _resolve_scored([(reordered[0][0], picked_score)] + reordered[1:], phrase, language)


def _score_catalog(catalog, phrases, phrase_embeddings):
    """Scores every catalogue row. Shared so scripts measure the SAME code the
    resolver runs — calibrate_resolver.py previously reimplemented this loop
    and therefore reported numbers for scoring that was no longer deployed."""
    name_embeddings = _catalog_embeddings(catalog)

    # Every phrase form against every view of every row, in one matrix product
    # rather than a Python loop per pair. Same arithmetic, ~220x faster.
    views = []
    for row in catalog:
        plain, enriched = name_embeddings[row["name_en"]]
        views.append(plain)
        views.append(enriched)
    sims = cosine_matrix(phrase_embeddings, views) if views else []
    best_by_row = {}
    for index, row in enumerate(catalog):
        # The best score across EVERY phrase form and BOTH views — the same
        # nested max the loop computed. Taking the max of per-form pairs would
        # not be: tuples compare on their first element, so a form scoring
        # (0.9, 0.1) would beat one scoring (0.8, 0.95) and the 0.95 would be
        # lost.
        best_by_row[row["name_en"]] = max(
            (max(form[2 * index], form[2 * index + 1]) for form in sims),
            default=0.0)

    def score(row):
        embed_sim = best_by_row[row["name_en"]]
        # Compare against the Arabic name too. Only name_en was ever fetched or
        # scored, so for an Arabic question the string half of the score was
        # dead weight — Arabic script against Latin scores ~0 for every
        # candidate alike, discriminating between none of them and leaving the
        # embedding to carry the whole decision. bge-m3 is cross-lingual so it
        # mostly coped, but "نسبة التضخم" still tied التضخم with نسبة السمنة.
        targets = [row["name_en"]] + ([row["name_ar"]] if row.get("name_ar") else [])
        string_sim = max(_similarity(p, t) for p in phrases for t in targets)
        if string_sim < STRING_MIN_TO_COUNT:
            string_sim = 0.0
        return EMBED_WEIGHT * embed_sim + STRING_WEIGHT * string_sim

    return sorted(((row, score(row)) for row in catalog), key=lambda x: x[1], reverse=True)


def _prefer_published(scored, min_confidence):
    """SCAI's published indicators get first refusal on a data question.

    "tourists arrived" scores 0.715 for "Tourists Number of arrival Tests" —
    unpublished, 12 data points — against 0.519 for "Number of International
    Visitors", which is published with 250. The embedding is not wrong about
    the words: that name literally contains "Tourists" and "arrival". It is
    wrong about which indicator a person means, and no threshold can separate
    them, because the bad match outscores the good one.

    Curation is the signal that does separate them. This product answers from
    SCAI's approved data, so a published indicator that clears the bar wins
    outright; unpublished working rows are considered only when nothing
    published does. That is the same preference the retriever already applies
    when it reads published_data_points before indicator_values.
    """
    published = [(row, score) for row, score in scored if row.get("is_published")]
    if published and published[0][1] >= min_confidence:
        return published

    # Nothing published clears the bar. Unpublished working data may still
    # answer, but it has to be a CLEARLY strong match, not merely the best of
    # a weak field — otherwise it simply inherits the question by default,
    # which is how "tourists arrived" was answered with a 12-point row called
    # "Tourists Number of arrival Tests" at 0.715 while the published
    # 250-point "Number of International Visitors" sat at 0.519.
    strong = [(row, score) for row, score in scored
              if row.get("is_published") or score >= min_confidence + UNPUBLISHED_MARGIN]
    return strong or published or scored


def _resolve_scored(scored, phrase, language="en"):
    """Turns a scored catalogue into a ResolutionResult: threshold, ambiguity
    tie-break, contradiction and inactive checks."""
    top_row, top_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0

    if top_score < MIN_CONFIDENCE:
        return ResolutionResult(
            None, "not_found", [],
            msg("indicator_not_found", language, phrase=phrase),
        )

    if (top_score - second_score) < MIN_GAP_TO_RUNNER_UP and second_score >= MIN_CONFIDENCE:
        # Before calling it ambiguous: if exactly one of the tied candidates is
        # PUBLISHED, that is the answer.
        #
        # "What is the GDP forecast 2026" offered a choice between "Real GDP",
        # "Expected GDP Impact in 2021-6" and "Expected GDP Impact in 2021-1" —
        # asking the user to choose between the indicator they meant and two
        # unpublished working rows they have never heard of, which differ from
        # each other only by a trailing number. Having to type "real GDP" to
        # get past that is the system's problem, not the user's.
        #
        # This is not the silent substitution F-003/F-012 forbid: it does not
        # reach past a better match, it breaks a TIE toward SCAI's own curated
        # layer, exactly as the retriever already prefers published_data_points
        # over raw indicator_values. The low-confidence disclosure still fires
        # if the winner is below CONFIDENT_MATCH.
        tied = [(r, sc) for r, sc in scored
                if top_score - sc < MIN_GAP_TO_RUNNER_UP and sc >= MIN_CONFIDENCE]
        published_tied = [(r, sc) for r, sc in tied if r.get("is_published")]
        if len(published_tied) == 1:
            top_row, top_score = published_tied[0]
        else:
            # Genuinely ambiguous — surface the real candidates rather than picking one.
            candidates = [
                IndicatorMatch(r["indicator_detail_id"], r["indicator_id"], r["name_en"],
                                r["is_published"], r["published_detail_id"],
                                r["is_active"], r["unit_en"], r.get("unit_ar"), r["polarity_en"],
                                r["data_source_en"], r["format"],
                                r["definition_en"], r["definition_ar"], s, r.get("name_ar"))
                for r, s in scored[:3]
            ]
            names = ", ".join(f'"{c.name_en.strip()}"' for c in candidates)
            return ResolutionResult(
                None, "ambiguous", candidates,
                msg("indicator_ambiguous", language, phrase=phrase, names=names),
            )

    match = IndicatorMatch(
        top_row["indicator_detail_id"], top_row["indicator_id"], top_row["name_en"],
        top_row["is_published"], top_row["published_detail_id"],
        top_row["is_active"], top_row["unit_en"], top_row.get("unit_ar"), top_row["polarity_en"],
        top_row["data_source_en"], top_row["format"],
        top_row["definition_en"], top_row["definition_ar"],
        top_score, top_row.get("name_ar"),
    )

    if _has_contradiction(phrase, match.name_en):
        # Before refusing: the catalogue may hold the indicator that was
        # actually asked for, one place down. Scoring is lexical, so a long
        # correct name can lose to a short wrong one — "Non-Hydrocarbon Exports
        # as Share of Total Exports" against "Total Exports" — and refusing then
        # tells the reader SCAI publishes nothing on a subject it does publish.
        #
        # Only a candidate that already cleared MIN_CONFIDENCE on its own and
        # contradicts nothing. This reaches past the top match, which F-003 and
        # F-012 forbid doing silently — but not toward a LOOSER match: the one
        # it skips has been shown to answer a different question, so the choice
        # is between this candidate and a refusal, not between this candidate
        # and a better one.
        rescued = next((r for r, sc in scored[1:]
                        if sc >= MIN_CONFIDENCE
                        and not _has_contradiction(phrase, r["name_en"])), None)
        if rescued is None:
            return ResolutionResult(
                None, "not_found", [],
                msg("indicator_contradiction", language, phrase=phrase,
                     matched=match.name_en.strip()),
            )
        match = IndicatorMatch(
            rescued["indicator_detail_id"], rescued["indicator_id"], rescued["name_en"],
            rescued["is_published"], rescued["published_detail_id"],
            rescued["is_active"], rescued["unit_en"], rescued.get("unit_ar"),
            rescued["polarity_en"], rescued["data_source_en"], rescued["format"],
            rescued["definition_en"], rescued["definition_ar"],
            next(sc for r, sc in scored[1:] if r is rescued), rescued.get("name_ar"),
        )

    if match.is_active is False:
        return ResolutionResult(
            match, "inactive", [],
            f"\"{match.name_en.strip()}\" is marked inactive in the approved data. "
            f"I can still show its historical values if you'd like, but flagging this first "
            f"rather than answering as if it were current.",
        )

    return ResolutionResult(match, "resolved", [])
