"""
FastAPI gateway — v2, wired to the deterministic pipeline (app/core/graph_v2.py).

Conversation history and the resolved-slot state both live server-side, keyed
by session_id, in app/core/conversation.py. Callers send a session_id and
nothing else; they are not required to replay the transcript on every request.
Single-process and in-memory — see that module before running more than one
replica.
"""
import json
import queue
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from app.core.graph_v2 import handle_message
from app.core.reading import read_message
from app.core.conversation import conversations
from app.core.messages import detect_language
from app.db import feedback as feedback_store
from app.db import message_log
from app.api.admin import router as admin_router
from app.api.admin_auth import warn_if_default_password

app = FastAPI(title="SCAI Economic Data Assistant", version="2.0.0")
app.include_router(admin_router)
# Says so on every startup if the placeholder password is still in place.
warn_if_default_password()


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
        history=conversations.recent_turns(req.session_id),
    )

    message_id = str(uuid.uuid4())
    answer_text = result.get("narration") or result.get("message") or ""
    conversations.record(req.session_id, req.message, answer_text,
                          result["session_state"], message_id=message_id,
                          slots=result.get("turn_slots"),
                          language=result.get("language"))
    message_log.log(message_id=message_id, session_id=req.session_id, endpoint="read",
                     question=req.message, answer=answer_text,
                     payload=result.get("facts_payload") or {},
                     language=_language_of(req.message),
                     readable=result.get("readable"),
                     latency_ms=int((time.perf_counter() - started) * 1000))
    # turn_slots and language are for the conversation store, not the caller —
    # excluded explicitly rather than relying on the model to drop unknown keys.
    internal = {"session_state", "turn_slots", "language"}
    return ReadResponse(**{k: v for k, v in result.items() if k not in internal})


def _run_chat(req: ChatRequest, progress=None, endpoint: str = "chat") -> tuple[dict, str, int]:
    """One chat turn, including the logging and the conversation record.

    Shared by both shapes of /chat so they cannot drift: a streamed answer must
    be the same answer, recorded the same way, with the same message_id. The
    only difference is when the caller hears about it.

    `endpoint` is what gets logged. It was hardcoded to "chat" for both, which
    meant that once a frontend moved to streaming there was no way to see the
    change — adoption, latency and failure rate for the two would sit in one
    undifferentiated bucket in /admin/stats.
    """
    started = time.perf_counter()
    result = handle_message(
        user_message=req.message,
        conversation_context=_context_for(req),
        session_state=conversations.get_state(req.session_id),
        history=conversations.recent_turns(req.session_id),
        progress=progress,
    )
    message_id = str(uuid.uuid4())
    # The answer WITHOUT its sources footer. The store feeds both the router's
    # transcript and the Composer's history, and neither needs the block — it is
    # rendered deterministically every turn, and shown to the Composer as an
    # example of a previous answer it copied, writing its own footer beside the
    # real one.
    conversations.record(req.session_id, req.message,
                          result.get("answer_for_history") or result["answer"],
                          result["session_state"], message_id=message_id,
                          slots=result.get("turn_slots"),
                          language=result.get("language"))
    latency_ms = int((time.perf_counter() - started) * 1000)
    message_log.log(message_id=message_id, session_id=req.session_id, endpoint=endpoint,
                     question=req.message, answer=result["answer"],
                     payload=result["facts_payload"],
                     language=_language_of(req.message),
                     readable=result.get("readable", False),
                     latency_ms=latency_ms)
    return result, message_id, latency_ms


def _chat_body(result: dict, message_id: str) -> dict:
    """The response body, built once so the streamed and plain forms are equal.

    Two code paths assembling the same six fields is how they end up disagreeing
    about a seventh.
    """
    payload = result["facts_payload"]
    return {
        "answer": result["answer"],
        "message_id": message_id,
        "facts_payload": payload,
        "chart": payload.get("chart"),
        "verified": "_verifier_rejected_numbers" not in payload,
        "readable": result.get("readable", False),
    }


def _wants_stream(request: Request) -> bool:
    """Whether this caller asked for progress events rather than a JSON body.

    Content negotiation rather than a second URL. The two used to be separate
    endpoints, which gave every caller a choice it mostly should not have to
    make and gave the frontend two integrations to keep in step. What actually
    differs between them is the media type the caller can handle, which is what
    Accept is for.
    """
    return "text/event-stream" in (request.headers.get("accept") or "").lower()


@app.post("/chat", response_model=None)
def chat_endpoint(req: ChatRequest, request: Request):
    """One turn of the conversation, in whichever form the caller can read.

    Send `Accept: text/event-stream` and the reply is Server-Sent Events: a
    `stage` event for each step of the pipeline as it starts, then one `answer`
    event carrying the same body the JSON form returns. Send anything else — or
    nothing — and it is that JSON body, with an ordinary HTTP status code.

    The distinction matters for callers that are not people. A stream commits to
    200 before the work begins, so a failure can only arrive as an `error`
    event; a monitor or a test harness watching status codes would read an
    outage as success. Those callers send no Accept header and keep their status
    codes. A browser with a person waiting sends one and gets the progress.

    STAGES, not tokens, and that is not a limitation waiting to be lifted. The
    composed text is verified against the facts payload only once it is
    complete, and a draft quoting a number the payload does not contain is
    discarded in favour of a template. Streaming the draft as it was written
    would put figures on screen that the verifier then withdraws — and a
    retracted number is worse than a slow one in a product whose whole design is
    that it never states a figure it cannot source.
    """
    if not _wants_stream(request):
        result, message_id, _ = _run_chat(req)
        return ChatResponse(**_chat_body(result, message_id))

    events: "queue.Queue[Optional[dict]]" = queue.Queue()
    outcome: dict = {}

    # The pipeline announces everything through one callback; the event name it
    # gets on the wire is decided here. "delta" and "replace" are about the
    # ANSWER and are their own SSE events, so a client can handle them without
    # string-matching inside a generic stage event.
    _OWN_EVENT = {"delta", "replace"}

    def emit(name: str, detail: dict) -> None:
        if name in _OWN_EVENT:
            events.put({"event": name, "data": dict(detail or {})})
        else:
            events.put({"event": "stage", "data": {"stage": name, **(detail or {})}})

    def work() -> None:
        try:
            result, message_id, _ = _run_chat(req, progress=emit, endpoint="chat-stream")
            outcome["data"] = _chat_body(result, message_id)
        except Exception as exc:  # noqa: BLE001 — reported to the caller, see below
            # The stream has already returned 200, so an exception here cannot
            # become an HTTP error code. It is sent as an event instead: a
            # connection that simply stops tells the frontend nothing it can
            # distinguish from a network failure.
            outcome["error"] = str(exc)
        finally:
            events.put(None)

    worker = threading.Thread(target=work, daemon=True)
    worker.start()

    def stream():
        while True:
            item = events.get()
            if item is None:
                break
            yield f"event: {item['event']}\ndata: {json.dumps(item['data'], default=str)}\n\n"
        if "error" in outcome:
            yield f"event: error\ndata: {json.dumps({'message': outcome['error']})}\n\n"
        else:
            yield f"event: answer\ndata: {json.dumps(outcome.get('data', {}), default=str)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        # nginx buffers proxied responses by default, which holds every event
        # until the response completes and turns this endpoint back into /chat
        # with extra steps.
        "X-Accel-Buffering": "no",
    })


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
