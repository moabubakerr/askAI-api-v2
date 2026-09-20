"""
Embeds the indicator catalogue once and stores the vectors in Postgres.

Run AFTER the ETL and with the embedding server reachable:

    docker compose run --rm etl index-indicators

Why this exists: the resolver embeds every catalogue entry twice — once as its
bare name, once as name-plus-definition — which is 1,288 texts. Those vectors
were held only in process memory, so every restart re-embedded all of them
against a CPU-only TEI before the first question could be answered, and
`docker compose up --build api` restarts the process.

Stored, a restart costs one SELECT.

The model name is written alongside every vector and the reader filters on it.
Vectors from different embedding models are not comparable, and scoring a
question against stale vectors from another model would produce confidently
wrong indicator matches with nothing in the output to show it. Change
EMBEDDING_MODEL_NAME and this simply finds nothing for the new model, falls
back to embedding at runtime, and is corrected by re-running this script.

Re-runnable: rows are upserted per text.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from sqlalchemy import create_engine, text

# TEI reports max_client_batch_size 32 and rejects anything larger outright.
BATCH = 32

# Mirrors indicator_resolver._embedding_text. Kept in step with it: if the two
# disagree, the stored vectors are for texts the resolver never asks about and
# every lookup misses, which shows up as slowness rather than as an error.
DEFINITION_CHARS = 240

CATALOG_SQL = """
SELECT d.name_en, d.definition_en
FROM indicator_details d
JOIN indicators i ON i.indicator_id = d.indicator_id
WHERE d.name_en IS NOT NULL
"""


def embedding_text(name: str, definition: str) -> str:
    name = (name or "").strip()
    definition = (definition or "").strip()
    if not definition or definition.lower() == name.lower():
        return name
    return f"{name}. {definition[:DEFINITION_CHARS]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default=os.environ.get("ETL_DSN"))
    parser.add_argument("--embedding-url",
                        default=os.environ.get("EMBEDDING_BASE_URL", "http://tei:80/v1"))
    parser.add_argument("--model",
                        default=os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"))
    args = parser.parse_args()
    if not args.dsn:
        print("index-indicators: no DSN. Set ETL_DSN or pass --dsn.", file=sys.stderr)
        return 2

    engine = create_engine(args.dsn)
    client = OpenAI(base_url=args.embedding_url, api_key="not-needed")

    with engine.connect() as conn:
        rows = conn.execute(text(CATALOG_SQL)).fetchall()

    # Both views of every entry, deduplicated: a name with no usable definition
    # is embedded once, not twice.
    texts = []
    for row in rows:
        name = (row[0] or "").strip()
        if not name:
            continue
        for value in (name, embedding_text(name, row[1])):
            if value not in texts:
                texts.append(value)

    print(f"catalogue: {len(rows)} rows -> {len(texts)} distinct texts to embed")
    started = time.time()
    written = 0
    with engine.begin() as conn:
        for start in range(0, len(texts), BATCH):
            batch = texts[start:start + BATCH]
            resp = client.embeddings.create(model=args.model, input=batch)
            for item in resp.data:
                conn.execute(text("""
                    INSERT INTO indicator_embeddings (text_key, model, embedding)
                    VALUES (:k, :m, :e)
                    ON CONFLICT (text_key) DO UPDATE
                      SET model = EXCLUDED.model, embedding = EXCLUDED.embedding
                """), {"k": batch[item.index], "m": args.model,
                        "e": str(item.embedding)})
                written += 1
            print(f"  {written}/{len(texts)}", end="\r", flush=True)

    print(f"\nindicator_embeddings: {written} vectors for {args.model} "
          f"in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
