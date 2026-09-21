"""
FastAPI gateway — v2, wired to the deterministic pipeline (app/core/graph_v2.py).

Conversation history and the resolved-slot state both live server-side, keyed
by session_id, in app/core/conversation.py. Callers send a session_id and
nothing else; they are not required to replay the transcript on every request.
Single-process and in-memory — see that module before running more than one
replica.
"""
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.core.graph_v2 import handle_message
from app.core.reading import read_message
from app.core.conversation import conversations
from app.core.messages import detect_language
from app.db import feedback as feedback_store
from app.db import message_log
from app.api.admin import router as admin_router

app = FastAPI(title="SCAI Economic Data Assistant", version="2.0.0")
app.include_router(admin_router)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    # Optional override. Normally left unset: the server keeps the transcript
    # itself, and a caller that sends nothing still gets working follow-ups.
    # Supplying it replaces the stored history for this one request, which is
    # useful for replaying a conversation the server has since evicted.
    conversation_context: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    # Identifies THIS answer, so a rating can name what it is about. Generated
    # per response and recorded with the turn, which is how /feedback can store
    # the question and answer alongside the score.
    message_id: str = ""
    facts_payload: dict
    chart: Optional[dict] = None
    verified: bool = True
    # True only when the answer actually contains readings — a value measured
    # at a period. The frontend shows "Read this for me" on this flag alone:
    # offering it on a greeting, a refusal, a definition or a catalogue listing
    # gives the user a button that reveals nothing they were not already shown.
    readable: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


class ReadResponse(BaseModel):
    """Same readings as /chat, split by who wrote each piece of text.

    council_analysis is SCAI's, verbatim. narration is generated. They are
    separate fields rather than one rendered string precisely so the frontend
    cannot blur them and a reader can always tell which is which.
    """
    ok: bool
    readable: bool = False              # same meaning as on /chat
    message: Optional[str] = None       # only when ok is false
    headline: Optional[dict] = None     # single-reading answers only
    one_liner: Optional[str] = None
    council_analysis: list = []
    evidence: list = []
    narration: Optional[str] = None
    disclaimer: Optional[str] = None
    chart: Optional[dict] = None
    facts_payload: dict = {}
    verified: bool = True


def _language_of(message: str) -> str:
    """What language the ANSWER will be in, which is decided by the question's
    script. Recorded so the dashboard can show the two halves of the product
    separately — an Arabic failure rate hidden inside an overall average is how
    "the Arabic commentary is in English" went unnoticed."""
    return detect_language(message)


def _context_for(req: ChatRequest) -> str:
    """Server-kept transcript, unless the caller explicitly supplied one."""
    if req.conversation_context:
        return req.conversation_context
    return conversations.render_context(req.session_id)


@app.post("/read", response_model=ReadResponse)
def read_endpoint(req: ChatRequest):
    """"Read this for me" — the readings formatted for a person, rather than as
    one paragraph of prose. Takes the same request as /chat."""
    started = time.perf_counter()
    result = read_message(
        user_message=req.message,
        conversation_context=_context_for(req),
        session_state=conversations.get_state(req.session_id),
    )

    message_id = str(uuid.uuid4())
    answer_text = result.get("narration") or result.get("message") or ""
    conversations.record(req.session_id, req.message, answer_text,
                          result["session_state"], message_id=message_id)
    message_log.log(message_id=message_id, session_id=req.session_id, endpoint="read",
                     question=req.message, answer=answer_text,
                     payload=result.get("facts_payload") or {},
                     language=_language_of(req.message),
                     readable=result.get("readable"),
                     latency_ms=int((time.perf_counter() - started) * 1000))
    return ReadResponse(**{k: v for k, v in result.items() if k != "session_state"})


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    started = time.perf_counter()
    result = handle_message(
        user_message=req.message,
        conversation_context=_context_for(req),
        session_state=conversations.get_state(req.session_id),
    )

    message_id = str(uuid.uuid4())
    conversations.record(req.session_id, req.message, result["answer"],
                          result["session_state"], message_id=message_id)

    payload = result["facts_payload"]
    message_log.log(message_id=message_id, session_id=req.session_id, endpoint="chat",
                     question=req.message, answer=result["answer"], payload=payload,
                     language=_language_of(req.message),
                     readable=result.get("readable", False),
                     latency_ms=int((time.perf_counter() - started) * 1000))
    return ChatResponse(
        answer=result["answer"],
        message_id=message_id,
        facts_payload=payload,
        chart=payload.get("chart"),
        verified="_verifier_rejected_numbers" not in payload,
        readable=result.get("readable", False),
    )


@app.get("/session/{session_id}")
def session_info(session_id: str):
    """What the server currently remembers for a session — the transcript it
    will send to the model, and the slots a follow-up would inherit. Exposed
    because "why did it answer about the wrong indicator" is otherwise
    unanswerable from outside the process."""
    return {
        "session_id": session_id,
        "context": conversations.render_context(session_id),
        "state": conversations.get_state(session_id),
    }


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Start a fresh conversation. Worth wiring to a "new chat" button: without
    it, a session id reused across unrelated topics carries the old indicator
    into the new one."""
    return {"session_id": session_id, "cleared": conversations.clear(session_id)}


class FeedbackRequest(BaseModel):
    """A rating for one answer.

    A comment is REQUIRED at 2 or below. A low score with no explanation says
    something is wrong and nothing about what, which is the least actionable
    feedback there is — and the person who could tell us has already moved on
    by the time anyone reads it.
    """
    message_id: str
    rating: int
    comment: Optional[str] = None
    session_id: str = "default"


class FeedbackResponse(BaseModel):
    """Only the fields that mean something for this outcome.

    A success carried "message": null and "comment_required": false, which are
    answers to a question that was not asked — the caller has to read two
    fields to learn nothing. Both are dropped unless they apply, so a success
    is {ok, feedback_id} and a failure says what to do about it.
    """
    ok: bool
    feedback_id: Optional[str] = None
    # Present when the rating was rejected, phrased for a reader rather than a
    # developer — the frontend can show it verbatim beside the comment box.
    message: Optional[str] = None
    comment_required: Optional[bool] = None


@app.post("/feedback", response_model=FeedbackResponse, response_model_exclude_none=True)
def feedback_endpoint(req: FeedbackRequest):
    try:
        rating, comment = feedback_store.validate(req.rating, req.comment)
    except feedback_store.FeedbackError as exc:
        # 422, not 500: the caller can fix this, and the frontend needs to tell
        # the user what to do rather than that something went wrong.
        raise HTTPException(
            status_code=422,
            # True only when the rating itself was fine and the comment was the
            # problem, so the frontend knows to open a comment box rather than
            # ask for a different score.
            detail={"ok": False, "message": str(exc),
                     "comment_required":
                         1 <= req.rating <= feedback_store.COMMENT_REQUIRED_AT_OR_BELOW},
        )

    # The exchange this rating is about, while it is still in memory. Evicted
    # sessions simply record less; a rating is never refused for it.
    turn = conversations.find_turn(req.session_id, req.message_id) or {}
    feedback_id = feedback_store.record(
        message_id=req.message_id, rating=rating, comment=comment,
        session_id=req.session_id,
        question=turn.get("user"), answer=turn.get("assistant"),
    )
    return FeedbackResponse(ok=True, feedback_id=feedback_id)


@app.get("/sessions/stats")
def sessions_stats():
    return conversations.stats()
