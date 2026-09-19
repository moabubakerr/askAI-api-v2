"""
Embedding client for semantic indicator-name matching.

Why this exists: plain string similarity (difflib) was tested against the
QC report's own failing case — "tourists arrived" vs the catalog's "Number
of international visitors" scores 0.167 on character-level similarity,
nowhere near a usable threshold. That's a synonym/paraphrase problem
("tourist" ≈ "visitor"), which string similarity cannot solve and an
LLM free-generating a match risks the F-003/F-012 problem (confidently
picking the wrong indicator). Embeddings solve this correctly: it's a
similarity *retrieval* step, not text generation, so it doesn't violate
the "never invent a number" rule — it never produces a number or a name
that isn't already a real row in indicator_details.

Requires a SEPARATE embedding-model server (e.g. BAAI/bge-m3 served via
vLLM's embedding endpoint, or Text-Embeddings-Inference). This is an
additional on-prem service beyond the 72B chat model — see README.
"""
import math
from openai import OpenAI
from app.core.config import settings

embedding_client = OpenAI(
    base_url=settings.EMBEDDING_BASE_URL,
    api_key=settings.LLM_API_KEY,
)

_CACHE: dict[str, list[float]] = {}


def get_embedding(text: str) -> list[float]:
    if text in _CACHE:
        return _CACHE[text]
    resp = embedding_client.embeddings.create(model=settings.EMBEDDING_MODEL_NAME, input=text)
    vec = resp.data[0].embedding
    _CACHE[text] = vec
    return vec


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_catalog(names: list[str]) -> dict[str, list[float]]:
    """Batch-embeds and caches every indicator name once. Call this at app
    startup (or lazily on first resolution) rather than per-request — the
    catalog is small (644 rows) and static between ETL runs, so re-embedding
    per request would be pure waste. For production, persist these vectors
    (e.g. a pgvector column) instead of the in-memory cache here, so a
    process restart doesn't force re-embedding everything."""
    uncached = [n for n in names if n not in _CACHE]
    for n in uncached:
        get_embedding(n)
    return {n: _CACHE[n] for n in names}


def embed_keyed(texts: dict) -> dict:
    """Embeds a {key: text} mapping, returning {key: vector}.

    Lets a catalogue entry be embedded as something richer than its name. The
    name alone loses the synonym that the question actually uses: nothing in
    "Number of International Visitors" says "tourist", so "tourists arrived
    into Qatar" ranked below "Number of Medical Tourists", which contains the
    word literally. That indicator's DEFINITION says "non-resident travelers
    entering Qatar for leisure, business or other activities" — the match the
    question needs, and it was being thrown away.
    """
    for key, text in texts.items():
        if text not in _CACHE:
            get_embedding(text)
    return {key: _CACHE[text] for key, text in texts.items()}
