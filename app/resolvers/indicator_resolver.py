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
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from sqlalchemy import text
from app.db.executor import engine
from app.resolvers.embeddings import get_embedding, cosine_similarity, embed_catalog

# Embedding cosine similarity is the primary score; difflib contributes a
# small bonus for exact/near-exact phrasing so "GDP" still resolves fast
# and confidently without relying on the embedding model's exact calibration.
EMBED_WEIGHT = 0.8
STRING_WEIGHT = 0.2

MIN_CONFIDENCE = 0.55
MIN_GAP_TO_RUNNER_UP = 0.05

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


def _fetch_catalog() -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT d.indicator_detail_id, d.indicator_id, d.name_en,
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


def resolve_indicator(phrase: str, require_data: bool = True) -> ResolutionResult:
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
        return ResolutionResult(None, "not_found", [], "No indicator was mentioned.")

    catalog = _fetch_catalog()
    if require_data:
        catalog = [r for r in catalog if (r.get("data_point_count") or 0) > 0]
    if not catalog:
        return ResolutionResult(None, "not_found", [], "The indicator catalog is empty.")

    names = [row["name_en"] for row in catalog]
    name_embeddings = embed_catalog(names)
    phrase_embedding = get_embedding(phrase)

    def score(row):
        embed_sim = cosine_similarity(phrase_embedding, name_embeddings[row["name_en"]])
        string_sim = _similarity(phrase, row["name_en"])
        return EMBED_WEIGHT * embed_sim + STRING_WEIGHT * string_sim

    scored = sorted(((row, score(row)) for row in catalog), key=lambda x: x[1], reverse=True)

    top_row, top_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0

    if top_score < MIN_CONFIDENCE:
        return ResolutionResult(
            None, "not_found", [],
            f"I couldn't find an indicator in the approved SCAI data matching \"{phrase}\". "
            f"I won't guess at a similar-sounding one — could you confirm the exact metric name?",
        )

    if (top_score - second_score) < MIN_GAP_TO_RUNNER_UP and second_score >= MIN_CONFIDENCE:
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
            f"\"{phrase}\" could match more than one indicator: {names}. Which one did you mean?",
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
            f"You asked about \"{phrase}\", but the approved data only has "
            f"\"{match.name_en.strip()}\" — that's a different measure, so I won't "
            f"substitute it. There's no matching indicator for what you asked.",
        )

    if match.is_active is False:
        return ResolutionResult(
            match, "inactive", [],
            f"\"{match.name_en.strip()}\" is marked inactive in the approved data. "
            f"I can still show its historical values if you'd like, but flagging this first "
            f"rather than answering as if it were current.",
        )

    return ResolutionResult(match, "resolved", [])
