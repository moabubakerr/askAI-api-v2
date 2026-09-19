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
from difflib import SequenceMatcher
from typing import Optional

from sqlalchemy import text
from app.db.executor import engine
from app.resolvers.embeddings import get_embedding, cosine_similarity, embed_keyed
from app.core.messages import msg

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


def _has_contradiction(phrase: str, matched_name: str) -> bool:
    phrase_l, matched_l = phrase.lower(), matched_name.lower()
    for a, b in CONTRASTIVE_PAIRS:
        if a in phrase_l and a not in matched_l and b in matched_l:
            return True
        if b in phrase_l and b not in matched_l and a in matched_l:
            return True
    return False


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
    polarity_en: Optional[str]
    data_source_en: Optional[str]
    format: Optional[str]
    definition_en: Optional[str]
    definition_ar: Optional[str]
    confidence: float


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
    cleaned = _PHRASE_PERIOD.sub(" ", phrase or "")
    cleaned = _PHRASE_NOISE.sub(" ", cleaned)
    cleaned = _PHRASE_NOISE_AR.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    # If stripping leaves nothing to match on, the words were the question.
    return cleaned if len(cleaned) >= 3 else (phrase or "").strip()


DEFINITION_CHARS_FOR_EMBEDDING = 240


def _catalog_embeddings(catalog):
    """Each indicator gets TWO vectors and keeps whichever matches better.

    Embedding only "name + definition" would have cost the exact-name case: a
    question that IS the catalogue name scored a perfect 1.000 against the bare
    name and necessarily less against name-plus-prose. Embedding only the name
    loses the synonym — nothing in "Number of International Visitors" says
    "tourist". Keeping both views costs one extra vector per indicator, paid
    once per process, and neither case has to lose.
    """
    plain = embed_keyed({row["name_en"]: row["name_en"] for row in catalog})
    enriched = embed_keyed({row["name_en"]: _embedding_text(row) for row in catalog})
    return {name: (plain[name], enriched[name]) for name in plain}


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


def _fetch_catalog() -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT d.indicator_detail_id, d.indicator_id, d.name_en, d.name_ar,
                   d.is_published, d.published_detail_id,
                   i.is_active, d.unit_en, d.polarity_en,
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
    if not phrase or not phrase.strip():
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
    return _resolve_scored(scored, phrase, language)


def _score_catalog(catalog, phrases, phrase_embeddings):
    """Scores every catalogue row. Shared so scripts measure the SAME code the
    resolver runs — calibrate_resolver.py previously reimplemented this loop
    and therefore reported numbers for scoring that was no longer deployed."""
    name_embeddings = _catalog_embeddings(catalog)

    def score(row):
        embed_sim = max(cosine_similarity(pe, view)
                        for pe in phrase_embeddings
                        for view in name_embeddings[row["name_en"]])
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
                                r["is_active"], r["unit_en"], r["polarity_en"],
                                r["data_source_en"], r["format"],
                                r["definition_en"], r["definition_ar"], s)
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
        top_row["is_active"], top_row["unit_en"], top_row["polarity_en"],
        top_row["data_source_en"], top_row["format"],
        top_row["definition_en"], top_row["definition_ar"],
        top_score,
    )

    if _has_contradiction(phrase, match.name_en):
        return ResolutionResult(
            None, "not_found", [],
            msg("indicator_contradiction", language, phrase=phrase, matched=match.name_en.strip()),
        )

    if match.is_active is False:
        return ResolutionResult(
            match, "inactive", [],
            f"\"{match.name_en.strip()}\" is marked inactive in the approved data. "
            f"I can still show its historical values if you'd like, but flagging this first "
            f"rather than answering as if it were current.",
        )

    return ResolutionResult(match, "resolved", [])
