"""
Prints what article retrieval actually scores, so ARTICLE_MAX_DISTANCE can be
set from evidence rather than guessed.

The threshold is the only thing standing between "semantic search is powerful"
and "semantic search confidently answers from an unrelated article". A vector
search ALWAYS returns a nearest neighbour: ask about penguins and the corpus
will hand back whichever of its 2,700 passages is least distant, with no signal
that the answer is nonsense. The cutoff is what turns that into an honest "the
articles don't cover this".

Set it too high and unrelated questions get answered; too low and real
questions are refused. Both failures are invisible without measuring, so:

    docker compose run --rm etl index-articles      # once
    docker compose exec api python scripts/calibrate_articles.py

Read the output as two groups. Questions the corpus SHOULD answer want a small
top distance; questions it should NOT answer want a large one. A threshold
between the two groups separates them. If they overlap, no threshold works and
the retrieval itself needs changing — more overlap between chunks, or titles
weighted differently in the embedded text.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.graph_v2 import ARTICLE_MAX_DISTANCE
from app.db import retriever
from app.resolvers.embeddings import get_embedding

# (question, should the articles be able to answer it?)
CASES = [
    ("What has SCAI written about the trade war?", True),
    ("tariffs and financial stability", True),
    ("In-Country Value programme", True),
    ("Tawteen", True),
    ("knowledge transfer", True),
    ("globalization", True),
    ("What is the Council's view on economic diversification?", True),
    ("ماذا كتب المجلس عن الرسوم الجمركية؟", True),
    # The corpus should NOT be able to answer these.
    ("What has SCAI written about penguins?", False),
    ("how do I bake sourdough bread", False),
    ("the offside rule in football", False),
    ("quantum chromodynamics", False),
]


def main():
    total = retriever.count_article_chunks()
    if total == 0:
        print("No article chunks indexed. Run: docker compose run --rm etl index-articles")
        return 1
    print(f"{total} chunks indexed. Current ARTICLE_MAX_DISTANCE = {ARTICLE_MAX_DISTANCE}\n")
    print(f"{'VERDICT':9} {'TOP':6} {'2nd':6}  {'QUESTION':46}  BEST MATCH")
    print("-" * 118)

    answerable, unanswerable = [], []
    for question, should_answer in CASES:
        language = "ar" if any("؀" <= c <= "ۿ" for c in question) else "en"
        hits = retriever.search_article_chunks(get_embedding(question), language=language,
                                                limit=3, query_text=question)
        if not hits:
            print(f"{'NO HITS':9} {'':6} {'':6}  {question[:46]:46}")
            continue
        top = float(hits[0]["distance"])
        second = float(hits[1]["distance"]) if len(hits) > 1 else float("nan")
        title = (hits[0].get("title_en") or hits[0].get("title_ar") or "")[:42]
        answered = top <= ARTICLE_MAX_DISTANCE or hits[0]["match"] != "semantic"
        ok = answered == should_answer
        (answerable if should_answer else unanswerable).append(top)
        print(f"{'ok' if ok else 'MISS':9} {top:.3f}  {second:.3f}  {question[:46]:46}  "
              f"{title} [{hits[0]['match']}]")

    print("\n" + "=" * 118)
    if answerable:
        print(f"worst distance among questions the corpus SHOULD answer: {max(answerable):.3f}")
    if unanswerable:
        print(f"best  distance among questions it should NOT answer    : {min(unanswerable):.3f}")
    if answerable and unanswerable:
        worst_good, best_bad = max(answerable), min(unanswerable)
        if worst_good < best_bad:
            print(f"\n=> a threshold between {worst_good:.3f} and {best_bad:.3f} separates them. "
                  f"Suggest ARTICLE_MAX_DISTANCE = {(worst_good + best_bad) / 2:.2f}")
        else:
            print(f"\n=> NO threshold separates them: an unanswerable question scores "
                  f"{best_bad:.3f} while a real one scores {worst_good:.3f}. Retrieval needs "
                  f"changing, not the threshold — consider larger chunk overlap, or weighting "
                  f"the title more heavily in the embedded text.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
