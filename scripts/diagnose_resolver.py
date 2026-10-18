"""
Shows, step by step, how the resolver handles one phrase against the REAL
catalogue and embeddings: the forms scored, the top candidates with their
scores and contradiction kinds, the broader re-search, and the final result.

Run inside the api container (it has the DSN and reaches TEI and vLLM):

    docker compose exec api python scripts/diagnose_resolver.py "non-oil spending"

With --question, the text is a whole user question: the intent model is asked
for its indicator phrase first (as the pipeline does), and THAT is resolved.
A wrong answer can start there, before the resolver sees anything:

    docker compose exec api python scripts/diagnose_resolver.py --question \
        "How many tourists arrived into Qatar in May 2025?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.resolvers import indicator_resolver as r
from app.resolvers.embeddings import get_embedding

TOP = 8


def show(title, phrase, forms, catalog):
    print(f"\n== {title}")
    print(f"forms: {forms}")
    scored = r._prefer_published(
        r._score_catalog(catalog, forms, [get_embedding(f) for f in forms]), r.MIN_CONFIDENCE)
    for row, score in scored[:TOP]:
        kind = r._contradiction_kind(phrase, row["name_en"]) or "fair"
        flags = ("" if row.get("is_published") else " UNPUBLISHED") + \
                (" INACTIVE" if row.get("is_active") is False else "")
        print(f"  {score:.3f}  {kind:18} {row['name_en'].strip()[:48]:48} "
              f"[{(row.get('parent_name_en') or '').strip()[:30]}]{flags}")
    return scored


def main(phrase: str) -> int:
    print(f"phrase: {phrase!r}")
    print(f"MIN_CONFIDENCE={r.MIN_CONFIDENCE}  SUGGESTION_FLOOR={r.SUGGESTION_FLOOR}  "
          f"NEAR_MISS_FLOOR={r.NEAR_MISS_FLOOR}")
    catalog = [row for row in r._fetch_catalog() if (row.get("data_point_count") or 0) > 0]
    print(f"catalogue rows with data: {len(catalog)}")
    print(f"subject words: {sorted(r._subject_words(phrase))}")

    show("main pass", phrase, r.phrase_forms(phrase), catalog)
    if r._NEGATED.search(phrase):
        stripped = r._NEGATED.sub(" ", phrase)
        show("broader pass (qualifier removed)", phrase, r.phrase_forms(stripped), catalog)

    result = r.resolve_indicator(phrase)
    print("\n== result")
    print(f"status: {result.status}")
    if result.match:
        m = result.match
        print(f"match: {m.name_en} [{m.parent_name_en}] confidence={m.confidence:.3f} "
              f"dropped_qualifier={m.dropped_qualifier!r} related={m.related}")
    if result.candidates:
        print(f"candidates: {[c.name_en for c in result.candidates]}")
    if result.message:
        print(f"message: {result.message}")
    return 0


def from_question(question: str) -> int:
    from app.nlu.intent_agent import extract_intent
    intent = extract_intent(question, "")
    print(f"question: {question!r}")
    for key in ("computation_type", "indicator_phrase", "period_expression",
                "explicit_frequency", "countries_mentioned"):
        print(f"  intent.{key}: {intent.get(key)!r}")
    phrase = intent.get("indicator_phrase") or question
    print(f"  aliases matched: {r.alias_forms(phrase)}")
    return main(phrase)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--question"]:
        sys.exit(from_question(" ".join(sys.argv[2:])))
    sys.exit(main(" ".join(sys.argv[1:]) or "non-oil spending"))
