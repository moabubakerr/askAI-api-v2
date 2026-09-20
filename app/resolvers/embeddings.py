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

# TEI reports max_client_batch_size 32 and rejects anything larger outright.
EMBED_BATCH = 32


def get_embedding(text: str) -> list[float]:
    if text in _CACHE:
        return _CACHE[text]
    prime_from_store()
    if text in _CACHE:
        return _CACHE[text]
    resp = embedding_client.embeddings.create(model=settings.EMBEDDING_MODEL_NAME, input=text)
    vec = resp.data[0].embedding
    _CACHE[text] = vec
    return vec


_PRIMED = False


def prime_from_store() -> int:
    """Loads any catalogue vectors already stored in Postgres into the cache.

    Runs once per process, on the first embedding request rather than at import
    time, so the module stays importable without a database — the calibration
    scripts and the unit tests rely on that.

    Returns how many vectors were loaded, for the log line.
    """
    global _PRIMED
    if _PRIMED:
        return 0
    _PRIMED = True
    try:
        from app.db import retriever
        stored = retriever.load_indicator_embeddings(settings.EMBEDDING_MODEL_NAME)
    except Exception:
        return 0
    for key, vector in stored.items():
        _CACHE.setdefault(key, vector)
    return len(stored)


def embed_many(texts) -> None:
    """Embeds and caches a set of texts, a batch per request.

    Everything here used to go through get_embedding one text at a time, so
    priming the catalogue meant 1,288 sequential HTTP round-trips to a CPU-only
    embedding server — paid in full on the first question after every restart,
    including every `docker compose up --build`. Batched at TEI's own limit
    that is 41 requests.

    Identical vectors either way: batching changes how many requests carry the
    texts, not what the model returns for them.
    """
    prime_from_store()
    pending = []
    seen = set()
    for text in texts:
        if text and text not in _CACHE and text not in seen:
            seen.add(text)
            pending.append(text)
    for start in range(0, len(pending), EMBED_BATCH):
        batch = pending[start:start + EMBED_BATCH]
        resp = embedding_client.embeddings.create(
            model=settings.EMBEDDING_MODEL_NAME, input=batch)
        # The API contract orders data by input index, but it also carries the
        # index explicitly — use it rather than trusting position.
        for item in resp.data:
            _CACHE[batch[item.index]] = item.embedding


def cosine_matrix(query_vectors, candidate_vectors):
    """Cosine of every query against every candidate, as a matrix.

    The scorer used to call cosine_similarity in a Python loop: 459 catalogue
    entries x 2 views x up to 3 phrase forms, each a 1024-dimension dot product
    in interpreted code. Measured at ~0.35 s of pure CPU per resolution, and a
    three-metric question resolves five times.

    float64 throughout, deliberately. The arithmetic differs from the loop only
    in summation order — measured at 1.7e-16 over 300 random pairs, against
    scores compared at three decimals and thresholds separated by 0.05. float32
    would also be safe at 1.4e-08, but costs nothing to avoid.

    Returns a list of rows, one per query, each a list of similarities in
    candidate order.
    """
    import numpy as np

    Q = np.asarray(query_vectors, dtype=np.float64)
    C = np.asarray(candidate_vectors, dtype=np.float64)
    if Q.size == 0 or C.size == 0:
        return [[0.0] * len(candidate_vectors) for _ in query_vectors]
    # A zero vector has no direction; its similarity is 0, as in the loop.
    qn = np.linalg.norm(Q, axis=1, keepdims=True)
    cn = np.linalg.norm(C, axis=1, keepdims=True)
    qn[qn == 0] = 1.0
    cn[cn == 0] = 1.0
    return ((Q / qn) @ (C / cn).T).tolist()


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
    embed_many(names)
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
    embed_many(texts.values())
    return {key: _CACHE[text] for key, text in texts.items()}
