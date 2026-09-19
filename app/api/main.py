"""
FastAPI gateway — v2, wired to the deterministic pipeline (app/core/graph_v2.py).

Conversation history and the resolved-slot state both live server-side, keyed
by session_id, in app/core/conversation.py. Callers send a session_id and
nothing else; they are not required to replay the transcript on every request.
Single-process and in-memory — see that module before running more than one
replica.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from app.core.graph_v2 import handle_message
from app.core.reading import read_message
from app.core.conversation import conversations

app = FastAPI(title="SCAI Economic Data Assistant", version="2.0.0")


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
    facts_payload: dict
    chart: Optional[dict] = None
    verified: bool = True


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


def _context_for(req: ChatRequest) -> str:
    """Server-kept transcript, unless the caller explicitly supplied one."""
    if req.conversation_context:
        return req.conversation_context
    return conversations.render_context(req.session_id)


@app.post("/read", response_model=ReadResponse)
def read_endpoint(req: ChatRequest):
    """"Read this for me" — the readings formatted for a person, rather than as
    one paragraph of prose. Takes the same request as /chat."""
    result = read_message(
        user_message=req.message,
        conversation_context=_context_for(req),
        session_state=conversations.get_state(req.session_id),
    )

    conversations.record(req.session_id, req.message,
                          result.get("narration") or result.get("message") or "",
                          result["session_state"])
    return ReadResponse(**{k: v for k, v in result.items() if k != "session_state"})


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    result = handle_message(
        user_message=req.message,
        conversation_context=_context_for(req),
        session_state=conversations.get_state(req.session_id),
    )

    conversations.record(req.session_id, req.message, result["answer"], result["session_state"])

    payload = result["facts_payload"]
    return ChatResponse(
        answer=result["answer"],
        facts_payload=payload,
        chart=payload.get("chart"),
        verified="_verifier_rejected_numbers" not in payload,
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


@app.get("/sessions/stats")
def sessions_stats():
    return conversations.stats()
