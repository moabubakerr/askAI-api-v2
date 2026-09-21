"""
Recording what a reader thought of an answer.

The only place the application writes to Postgres. It connects as scai_ro,
which holds SELECT on everything and INSERT on this one table by name — the
read-only rule exists so the app cannot modify SCAI's data, and a rating is not
SCAI's data. INSERT and nothing else, so the app can record a rating and can
never edit or delete one.

The question and answer are stored alongside the rating rather than referenced.
The transcript lives in memory and is evicted after two hours, so a rating that
carried only an id would be unreadable by the time anyone came to read it, and
a score you cannot tie to what was said is a number, not feedback.
"""
import uuid
from typing import Optional

from sqlalchemy import text
from app.db.executor import engine

# Below this, a comment is required. A low score with no explanation tells you
# something is wrong and nothing about what, which is the least actionable
# feedback there is.
COMMENT_REQUIRED_AT_OR_BELOW = 2

MAX_COMMENT_CHARS = 2000


class FeedbackError(ValueError):
    """Rejected before it reaches the database — the caller can fix it."""


def validate(rating, comment: Optional[str]) -> tuple:
    """Returns (rating, comment) or raises FeedbackError.

    The same rule is a CHECK constraint on the table, so a second caller that
    does not know about it cannot write a bare 1-star either.
    """
    try:
        rating = int(rating)
    except (TypeError, ValueError):
        raise FeedbackError("rating must be a whole number from 1 to 5.")
    if not 1 <= rating <= 5:
        raise FeedbackError("rating must be between 1 and 5.")

    comment = (comment or "").strip()
    if rating <= COMMENT_REQUIRED_AT_OR_BELOW and not comment:
        raise FeedbackError(
            f"A comment is required for a rating of {COMMENT_REQUIRED_AT_OR_BELOW} "
            f"or below — please say what was wrong with the answer.")
    return rating, (comment[:MAX_COMMENT_CHARS] or None)


def record(message_id: str, rating: int, comment: Optional[str],
           session_id: Optional[str] = None, question: Optional[str] = None,
           answer: Optional[str] = None) -> str:
    """Writes one rating. Returns its id."""
    feedback_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO message_feedback
                (feedback_id, message_id, session_id, rating, comment, question, answer)
            VALUES (:fid, :mid, :sid, :rating, :comment, :question, :answer)
        """), {"fid": feedback_id, "mid": message_id, "sid": session_id,
                "rating": rating, "comment": comment,
                "question": question, "answer": answer})
    return feedback_id
