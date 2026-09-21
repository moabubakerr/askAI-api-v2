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


def log(message_id: str, session_id: str, endpoint: str, question: str,
        answer: str, payload: dict, language: Optional[str] = None,
        readable: Optional[bool] = None, latency_ms: Optional[int] = None) -> None:
    """Records one exchange. Never raises."""
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
    except Exception:
        # Deliberately silent. The alternative is a 500 on a question that was
        # answered correctly, because a log write failed.
        pass
