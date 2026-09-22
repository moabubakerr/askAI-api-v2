"""
Logging every exchange, for the admin dashboard.

Best-effort by design: a logging failure must never cost a user their answer,
so everything here swallows its own errors and says so. A dashboard reading
this table should treat it as a log that can under-count, not as a ledger.

Application data, like message_feedback — written by the app under an INSERT
grant on this one table, and never truncated by the ETL.
"""
from typing import Optional

from sqlalchemy import text
from app.db.executor import engine

MAX_TEXT = 4000

# An answer citing more rows than this is a long series, and the hundredth
# citation tells an admin nothing the tenth did not. The cap is on what is
# STORED, not on what is cited: the footer the reader sees already groups a
# long series into "2015 to 2024 (40 data points)".
MAX_CITATIONS = 50


def answer_shape(payload: dict) -> str:
    """What the reader actually received.

    Derived from the payload rather than from the route that produced it. The
    routing taxonomy is past twenty computation types and still moving; what an
    answer CONTAINS — a value, a series, a ranking, a refusal — does not, and a
    dashboard built on the route names would need rewriting every time a route
    is added.
    """
    if not payload or payload.get("ok") is False:
        return "refusal"
    facts = payload.get("facts") or {}
    for key, shape in (
        ("candidates", "disambiguation"),
        ("complement_share", "derived_share"),
        ("assessment", "direction"),
        ("ranked_indicators", "performance_ranking"),
        ("increasing", "direction_split"),
        ("ranked_by_level", "level_comparison"),
        ("overview", "overview"),
        ("ranked", "country_ranking"),
        ("rows", "country_comparison"),
        ("series", "series"),
        ("ranked_periods", "period_ranking"),
        ("names", "catalogue"),
        ("passages", "article"),
        ("definition", "definition"),
        ("percent_change", "period_comparison"),
        ("growth_rate_percent", "growth_rate"),
        ("extremum", "min_max"),
        ("actual", "value"),
        ("note", "greeting"),
    ):
        if key in facts:
            return shape
    return "other"


def _citation_rows(message_id: str, payload: dict) -> list[dict]:
    """The rows this answer was built from, ready to insert.

    Read straight off the payload: app/core/graph_v2.py already attaches
    `citations` to every reply so the "Sources:" footer can be rendered in code
    rather than trusted to the LLM. Nothing new is computed here — the
    provenance was always there and was simply being discarded.
    """
    out = []
    for position, c in enumerate((payload or {}).get("citations") or []):
        if not isinstance(c, dict):
            continue
        out.append({
            "mid": message_id, "pos": position,
            # Cited rows always name a table; 'unknown' is what
            # citations_for_rows falls back to when a row carries no
            # source_table, and it is worth recording as such rather than
            # dropping the citation.
            "table": c.get("table") or "unknown",
            "record_id": c.get("record_id"),
            "indicator": c.get("indicator"),
            "data_source": c.get("data_source"),
            "period": c.get("period_label"),
            "country": c.get("country"),
        })
        if len(out) >= MAX_CITATIONS:
            break
    return out


def log(message_id: str, session_id: str, endpoint: str, question: str,
        answer: str, payload: dict, language: Optional[str] = None,
        readable: Optional[bool] = None, latency_ms: Optional[int] = None) -> None:
    """Records one exchange and the rows it cited. Never raises.

    Both writes share one transaction, so the citations can never outlive the
    message they belong to. The other direction is allowed: if this whole block
    fails the user still gets their answer, which is the trade the module
    exists to make.
    """
    try:
        facts = (payload or {}).get("facts") or {}
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO chat_messages
                    (message_id, session_id, endpoint, language, question, answer,
                     answered, answer_shape, indicator, period_label, verified,
                     readable, latency_ms)
                VALUES (:mid, :sid, :endpoint, :language, :question, :answer,
                        :answered, :shape, :indicator, :period, :verified,
                        :readable, :latency)
                ON CONFLICT (message_id) DO NOTHING
            """), {
                "mid": message_id, "sid": session_id, "endpoint": endpoint,
                "language": language,
                "question": (question or "")[:MAX_TEXT],
                "answer": (answer or "")[:MAX_TEXT],
                "answered": bool((payload or {}).get("ok", True)),
                "shape": answer_shape(payload),
                "indicator": facts.get("indicator") or facts.get("scope"),
                "period": facts.get("period_label") or facts.get("period_used"),
                "verified": "_verifier_rejected_numbers" not in (payload or {}),
                "readable": readable,
                "latency": latency_ms,
            })
            rows = _citation_rows(message_id, payload)
            if rows:
                # DO NOTHING on conflict, like the message insert above: a
                # retry of the same exchange re-records nothing rather than
                # failing, and (message_id, position) is stable across retries.
                conn.execute(text("""
                    INSERT INTO message_citations
                        (message_id, position, source_table, record_id,
                         indicator, data_source, period_label, country)
                    VALUES (:mid, :pos, :table, :record_id,
                            :indicator, :data_source, :period, :country)
                    ON CONFLICT (message_id, position) DO NOTHING
                """), rows)
    except Exception:
        # Deliberately silent. The alternative is a 500 on a question that was
        # answered correctly, because a log write failed.
        pass
