"""
Chunks the SCAI articles and stores an embedding per chunk in Postgres.

Run AFTER the ETL (it reads the articles table) and with the embedding server
reachable:

    docker compose run --rm etl index-articles

Separate from load_data.py on purpose. The data load must work with nothing but
Postgres — it is how you get a usable database before any model server exists —
whereas this step depends on TEI being up. Folding them together would mean a
model outage blocked a data reload.

Re-runnable: chunks are deleted and rebuilt per article, so a changed article
is re-indexed rather than duplicated.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from sqlalchemy import create_engine, text

# ~900 characters is about 2-3 paragraphs of these articles. Short enough that a
# retrieved passage is quotable as an answer rather than a page of prose, long
# enough to carry an argument — a claim split across two chunks is retrievable
# from neither.
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
# Chunking the real corpus produced fragments as short as 7 characters — a
# stray heading or caption left behind by the HTML conversion. They embed to
# near-meaningless vectors that can still win a nearest-neighbour search, and
# they are useless as a quoted passage, so they are dropped rather than indexed.
MIN_CHUNK_CHARS = 120
# TEI reports max_client_batch_size 32; going above it is rejected outright.
EMBED_BATCH = 32


def split_paragraphs(article_text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n{1,}", article_text or "") if p.strip()]


def chunk_text(article_text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Packs whole paragraphs up to `size`, so chunks break at paragraph
    boundaries rather than mid-sentence wherever possible.

    A paragraph longer than `size` is split on sentence ends, and only if that
    still leaves an oversized piece is it cut by character count. The overlap
    carries the tail of the previous chunk forward, so a claim that straddles a
    boundary is still retrievable from at least one chunk.
    """
    chunks: list[str] = []
    buffer = ""

    def flush():
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    for para in split_paragraphs(article_text):
        pieces = [para]
        if len(para) > size:
            pieces, current = [], ""
            for sentence in re.split(r"(?<=[.!?؟])\s+", para):
                if len(current) + len(sentence) + 1 > size and current:
                    pieces.append(current.strip())
                    current = sentence
                else:
                    current = f"{current} {sentence}".strip()
            if current:
                pieces.append(current.strip())
            pieces = [p for piece in pieces for p in
                      ([piece] if len(piece) <= size
                       else [piece[i:i + size] for i in range(0, len(piece), size)])]

        for piece in pieces:
            if len(buffer) + len(piece) + 1 > size and buffer:
                tail = buffer[-overlap:] if overlap else ""
                flush()
                buffer = f"{tail} {piece}".strip() if tail else piece
            else:
                buffer = f"{buffer}\n{piece}".strip() if buffer else piece
    flush()
    return chunks


def embed_batch(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    """Batched, unlike the indicator resolver's one-at-a-time loop — 2,600
    single requests against a CPU-bound TEI would take a very long time."""
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default=os.environ.get("ETL_DSN"))
    parser.add_argument("--embedding-url", default=os.environ.get("EMBEDDING_BASE_URL", "http://tei:80/v1"))
    parser.add_argument("--embedding-model", default=os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"))
    parser.add_argument("--languages", default="en,ar")
    parser.add_argument("--published-only", action="store_true",
                        help="index only articles flagged published (71 of 84)")
    args = parser.parse_args()

    if not args.dsn:
        print("index-articles: no DSN. Set ETL_DSN or pass --dsn.", file=sys.stderr)
        return 2

    engine = create_engine(args.dsn)
    client = OpenAI(base_url=args.embedding_url, api_key="not-needed")
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT article_id, title_en, title_ar, content_en, content_ar, published FROM articles"
            + (" WHERE published" if args.published_only else "")
        )).fetchall()
    articles = [dict(r._mapping) for r in rows]
    print(f"index-articles: {len(articles)} articles, languages={languages}")

    # Build every chunk first so the embedding cost is known before any of it is
    # spent, and so a failure mid-way is a failure of one clear step.
    pending = []
    for article in articles:
        for lang in languages:
            body = article.get(f"content_{lang}") or ""
            title = (article.get(f"title_{lang}") or "").strip()
            if not body.strip():
                continue
            kept = [c for c in chunk_text(body) if len(c) >= MIN_CHUNK_CHARS]
            for i, chunk in enumerate(kept):
                # The title rides along in the embedded text: a chunk from deep
                # inside an article otherwise carries no signal about what the
                # article is about, and "what has SCAI written about tariffs"
                # matches the title far more strongly than any single paragraph.
                pending.append({
                    "chunk_id": f"{article['article_id']}:{lang}:{i}",
                    "article_id": article["article_id"],
                    "language": lang,
                    "chunk_index": i,
                    "content": chunk,
                    "embed_text": f"{title}\n\n{chunk}" if title else chunk,
                })

    print(f"index-articles: {len(pending)} chunks to embed")
    if not pending:
        return 0

    started = time.time()
    for start in range(0, len(pending), EMBED_BATCH):
        batch = pending[start:start + EMBED_BATCH]
        vectors = embed_batch(client, args.embedding_model, [c["embed_text"] for c in batch])
        for chunk, vector in zip(batch, vectors):
            chunk["embedding"] = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"
        done = start + len(batch)
        if done % (EMBED_BATCH * 10) == 0 or done == len(pending):
            rate = done / max(time.time() - started, 0.001)
            print(f"  embedded {done}/{len(pending)} ({rate:.0f} chunks/s)", flush=True)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM article_chunks WHERE article_id = ANY(:ids)"),
                     {"ids": list({c["article_id"] for c in pending})})
        for start in range(0, len(pending), 200):
            conn.execute(text("""
                INSERT INTO article_chunks
                    (chunk_id, article_id, language, chunk_index, content, char_count, embedding)
                VALUES
                    (:chunk_id, :article_id, :language, :chunk_index, :content, :char_count,
                     CAST(:embedding AS vector))
            """), [{**c, "char_count": len(c["content"])}
                   for c in pending[start:start + 200]])

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM article_chunks")).scalar()
        per_lang = conn.execute(text(
            "SELECT language, COUNT(*) FROM article_chunks GROUP BY language ORDER BY language"
        )).fetchall()
    print(f"index-articles: {total} chunks stored " + ", ".join(f"{l}={n}" for l, n in per_lang))
    return 0


if __name__ == "__main__":
    sys.exit(main())
